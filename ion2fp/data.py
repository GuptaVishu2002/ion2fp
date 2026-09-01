import random
from pathlib import Path

import matchms
import matchms.filtering as ms_filters
import numpy as np
import pandas as pd
import torch
from matchms.importing import load_from_mgf
from rdkit import DataStructs
from rdkit.Chem import AllChem as Chem
from torch.utils.data import Dataset, IterableDataset, get_worker_info


def default_matchms_transforms(
    spec: matchms.Spectrum,
    n_max_peaks: int = 60,
    mz_from: float = 10,
    mz_to: float = 1000,
) -> matchms.Spectrum:
    spec = ms_filters.select_by_mz(spec, mz_from=mz_from, mz_to=mz_to)
    if n_max_peaks is not None:
        spec = ms_filters.reduce_to_number_of_peaks(spec, n_max=n_max_peaks)
    spec = ms_filters.normalize_intensities(spec)
    return spec


def morgan_fp(mol: Chem.Mol, fp_size=2048, radius=2, to_np=True):
    fp = Chem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=fp_size)
    if to_np:
        fp_np = np.zeros((0,), dtype=np.int32)
        DataStructs.ConvertToNumpyArray(fp, fp_np)
        fp = fp_np
    return fp


def load_spectra(path):
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".mgf":
        return list(load_from_mgf(str(path)))

    if suffix in (".csv", ".tsv"):
        sep = "\t" if suffix == ".tsv" else ","
        df = pd.read_csv(path, sep=sep)
        required = {"mzs", "intensities"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{path} is missing columns {sorted(missing)}, need mzs and intensities")
        spectra = []
        for _, row in df.iterrows():
            mzs = np.array([float(v) for v in str(row["mzs"]).split(",")])
            intensities = np.array([float(v) for v in str(row["intensities"]).split(",")])
            metadata = {}
            for key in ("precursor_mz", "smiles", "identifier", "adduct", "fold"):
                if key in df.columns and pd.notna(row[key]):
                    metadata[key] = row[key]
            spectra.append(matchms.Spectrum(mz=mzs, intensities=intensities, metadata=metadata))
        return spectra

    raise ValueError(f"file must be .mgf, .csv, or .tsv, got {path.suffix}")


def pad_peaks(mzs, intensities, max_n_ions, sort_by_intensity=False):
    n = len(mzs)
    if sort_by_intensity:
        order = np.argsort(intensities)[::-1]
        mzs, intensities = mzs[order], intensities[order]
    if n > max_n_ions:
        if sort_by_intensity:
            mzs, intensities = mzs[:max_n_ions], intensities[:max_n_ions]
        else:
            keep = np.argsort(intensities)[::-1][:max_n_ions]
            keep.sort()
            mzs, intensities = mzs[keep], intensities[keep]
        n = max_n_ions

    mzs_p = np.zeros(max_n_ions, dtype=np.float32)
    ints_p = np.zeros(max_n_ions, dtype=np.float32)
    mask = np.zeros(max_n_ions, dtype=bool)
    mzs_p[:n] = mzs
    ints_p[:n] = intensities
    mask[:n] = True
    return mzs_p, ints_p, mask


def _spectrum_to_item(spec, hp, max_n_ions, require_labels, fallback_identifier, fingerprint_cache=None):
    try:
        spec = default_matchms_transforms(
            spec, n_max_peaks=hp.max_n_ions, mz_from=hp.mz_from, mz_to=hp.mz_to
        )
    except Exception:
        spec = None
    if spec is None or spec.peaks.mz.size == 0:
        return None

    smiles = spec.get("smiles")
    fingerprint = None
    if smiles is not None:
        if fingerprint_cache is not None and smiles in fingerprint_cache:
            fingerprint = fingerprint_cache[smiles]
        else:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                fingerprint = morgan_fp(mol, fp_size=hp.fp_size, radius=hp.fp_radius, to_np=True)
                fingerprint = fingerprint.astype(np.float32)
            if fingerprint_cache is not None:
                fingerprint_cache[smiles] = fingerprint

    if require_labels and fingerprint is None:
        return None

    mzs_p, ints_p, mask = pad_peaks(spec.peaks.mz, spec.peaks.intensities, max_n_ions, sort_by_intensity=hp.sort_ions_by_intensity)
    identifier = spec.get("unique_identifier") or spec.get("identifier") or spec.get("title") or fallback_identifier
    fold = spec.get("fold")
    fold = canonical_fold(fold) if fold is not None else None

    return {
        "mzs": mzs_p,
        "intensities": ints_p,
        "mask": mask,
        "fingerprint": fingerprint,
        "identifier": str(identifier),
        "smiles": smiles,
        "fold": fold,
    }


class SpectrumFingerprintDataset(Dataset):
    def __init__(self, spectra, hp, require_labels):
        self.hp = hp
        self.max_n_ions = hp.padded_max_n_ions()
        self.items = []
        n_skipped = 0
        fingerprint_cache = {}

        for i, spec in enumerate(spectra):
            item = _spectrum_to_item(spec, hp, self.max_n_ions, require_labels, str(i), fingerprint_cache=fingerprint_cache)
            if item is None:
                n_skipped += 1
                continue
            self.items.append(item)

        self.n_skipped = n_skipped

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


class AugmentedTrainDataset(IterableDataset):
    def __init__(self, always_items, aug_paths, n_aug_per_epoch, hp, fold_filter=None, shuffle_buffer=0, resample_every_epoch=True):
        self.always_items = list(always_items)
        self.aug_paths = [Path(p) for p in aug_paths]
        self.n_aug_per_epoch = n_aug_per_epoch
        self.hp = hp
        self.fold_filter = fold_filter
        self.shuffle_buffer = shuffle_buffer
        self.max_n_ions = hp.padded_max_n_ions()
        if n_aug_per_epoch is None and not resample_every_epoch:
            raise ValueError("can't cache the full augmentation pool, need resample_every_epoch=True for that")
        self.resample_every_epoch = resample_every_epoch
        self._cached_augmented = None

    def __len__(self):
        if self.n_aug_per_epoch is None:
            raise TypeError("no known length when using the full augmentation pool")
        return len(self.always_items) + self.n_aug_per_epoch

    def _this_worker_share(self, n):
        worker_info = get_worker_info()
        if worker_info is None:
            return n
        base, remainder = divmod(n, worker_info.num_workers)
        return base + (1 if worker_info.id < remainder else 0)

    def _always_items_for_this_worker(self):
        worker_info = get_worker_info()
        if worker_info is None:
            return self.always_items
        return self.always_items[worker_info.id::worker_info.num_workers]

    def _files_for_this_worker(self):
        worker_info = get_worker_info()
        if worker_info is None:
            return list(self.aug_paths)
        return self.aug_paths[worker_info.id::worker_info.num_workers]

    def _iter_full_pool(self):
        fingerprint_cache = {}
        paths = self._files_for_this_worker()
        random.shuffle(paths)
        for path in paths:
            for i, spec in enumerate(load_from_mgf(str(path))):
                item = _spectrum_to_item(
                    spec, self.hp, self.max_n_ions, True, f"{path.name}:{i}",
                    fingerprint_cache=fingerprint_cache,
                )
                if item is None:
                    continue
                if self.fold_filter is not None and not self.fold_filter(item["fold"]):
                    continue
                yield item

    def _iter_sampled_augmentation(self):
        if self.n_aug_per_epoch is None:
            yield from self._iter_full_pool()
            return

        target = self._this_worker_share(self.n_aug_per_epoch)
        if target <= 0 or not self.aug_paths:
            return

        fingerprint_cache = {}
        yielded = 0
        while yielded < target:
            paths = list(self.aug_paths)
            random.shuffle(paths)
            yielded_this_pass = 0
            for path in paths:
                if yielded >= target:
                    break
                for i, spec in enumerate(load_from_mgf(str(path))):
                    if yielded >= target:
                        break
                    item = _spectrum_to_item(
                        spec, self.hp, self.max_n_ions, True, f"{path.name}:{i}",
                        fingerprint_cache=fingerprint_cache,
                    )
                    if item is None:
                        continue
                    if self.fold_filter is not None and not self.fold_filter(item["fold"]):
                        continue
                    yield item
                    yielded += 1
                    yielded_this_pass += 1
            if yielded_this_pass == 0:
                break

    def _augmented_items_for_this_epoch(self):
        if self.resample_every_epoch:
            yield from self._iter_sampled_augmentation()
            return

        if self._cached_augmented is None:
            self._cached_augmented = list(self._iter_sampled_augmentation())
            yield from self._cached_augmented
            return

        cached = list(self._cached_augmented)
        random.shuffle(cached)
        yield from cached

    def _iter_items(self):
        yield from self._always_items_for_this_worker()
        yield from self._augmented_items_for_this_epoch()

    def __iter__(self):
        if self.shuffle_buffer <= 0:
            yield from self._iter_items()
            return

        buffer = []
        for item in self._iter_items():
            if len(buffer) < self.shuffle_buffer:
                buffer.append(item)
                continue
            idx = random.randrange(len(buffer))
            yield buffer[idx]
            buffer[idx] = item
        random.shuffle(buffer)
        yield from buffer


def canonical_fold(v):
    s = str(v).strip().lower()
    try:
        f = float(s)
    except ValueError:
        return s
    return str(int(f)) if f.is_integer() else s


def _normalize_fold_values(fold_spec):
    values = fold_spec if isinstance(fold_spec, (list, tuple, set)) else [fold_spec]
    return {canonical_fold(v) for v in values}


def split_by_fold(dataset, val_fold="val", test_fold="test"):
    val_values = _normalize_fold_values(val_fold)
    test_values = _normalize_fold_values(test_fold)

    train_idx, val_idx, test_idx = [], [], []
    for i, item in enumerate(dataset.items):
        fold = item["fold"]
        if fold in val_values:
            val_idx.append(i)
        elif fold in test_values:
            test_idx.append(i)
        else:
            train_idx.append(i)
    return train_idx, val_idx, test_idx


def filter_by_fold(dataset, folds):
    keep_values = _normalize_fold_values(folds)
    dataset.items = [item for item in dataset.items if item["fold"] in keep_values]
    return dataset


def train_augmentation_file_paths(augmentation_dir, source_fold, splits=10):
    augmentation_dir = Path(augmentation_dir)
    source_fold = canonical_fold(source_fold)
    paths = []
    for split_count in range(1, splits + 1):
        path = augmentation_dir / f"fragnnet_split{split_count}_fold{source_fold}.mgf"
        if not path.is_file():
            raise FileNotFoundError(f"can't find augmentation file {path}")
        paths.append(path)
    return paths


def collate_fn(batch):
    out = {
        "mzs": torch.as_tensor(np.stack([b["mzs"] for b in batch]), dtype=torch.float32),
        "intensities": torch.as_tensor(np.stack([b["intensities"] for b in batch]), dtype=torch.float32),
        "mask": torch.as_tensor(np.stack([b["mask"] for b in batch]), dtype=torch.bool),
        "identifier": [b["identifier"] for b in batch],
        "smiles": [b["smiles"] for b in batch],
    }
    fingerprints = [b["fingerprint"] for b in batch]
    if all(fp is not None for fp in fingerprints):
        out["fingerprint"] = torch.as_tensor(np.stack(fingerprints), dtype=torch.float32)
    return out
