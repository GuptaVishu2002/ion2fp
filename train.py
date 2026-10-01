import argparse
import json
import os
import sys

for _i, _arg in enumerate(sys.argv):
    _is_cpu_flag = (_arg == "--accelerator" and sys.argv[_i + 1 : _i + 2] == ["cpu"]) or _arg == "--accelerator=cpu"
    if _is_cpu_flag:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
        break

from dataclasses import asdict, fields
from pathlib import Path

import numpy as np
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader, Subset, random_split

from ion2fp.config import Hyperparams
from ion2fp.data import (
    AugmentedTrainDataset,
    SpectrumFingerprintDataset,
    canonical_fold,
    collate_fn,
    load_spectra,
    split_by_fold,
    train_augmentation_file_paths,
)
from ion2fp.lightning_module import FingerprintLightningModule


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--input", required=True, type=str)
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--val_fold", type=str, default="val")
    parser.add_argument("--test_fold", type=str, default="test")
    parser.add_argument("--augmentation_multiplier", type=int, default=1)
    parser.add_argument("--augmentation_file", type=str, default=None)
    parser.add_argument("--augmentation_dir", type=str,default="/Genomics/skinniderlab/msms-triangulation/spectra/spectraverse/aug_fragnet/formula")
    parser.add_argument("--augmentation_splits", type=int, default=10)
    parser.add_argument("--augmentation_folds", type=int, default=10)
    parser.add_argument("--augmentation_source", type=str, default="test", choices=["test", "train"])
    parser.add_argument("--augmentation_exclude", type=str, default="none", choices=["none", "ground_truth", "candidates", "overlap"])
    parser.add_argument("--augmentation_candidates", type=str, default=None)
    parser.add_argument("--augmentation_only", type=lambda v: str(v).lower() in ("y", "yes", "true", "1"), default=False)
    parser.add_argument("--shuffle_buffer", type=int, default=20000)
    parser.add_argument("--augmentation_resample_every_epoch",type=lambda v: str(v).lower() in ("y", "yes", "true", "1"), default=False)

    default_hp = Hyperparams()
    for f in fields(default_hp):
        if f.type is bool or isinstance(f.default, bool):
            parser.add_argument(
                f"--{f.name}",
                type=lambda v: str(v).lower() in ("y", "yes", "true", "1"),
                default=f.default,
            )
        else:
            parser.add_argument(f"--{f.name}", type=type(f.default), default=f.default)
    return parser


def main():
    args = build_arg_parser().parse_args()
    hp = Hyperparams(**{f.name: getattr(args, f.name) for f in fields(Hyperparams)})
    seed_everything(hp.seed, workers=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading data from {args.input}")
    spectra = load_spectra(args.input)
    dataset = SpectrumFingerprintDataset(spectra, hp, require_labels=True)
    print(f"got {len(dataset)} spectra, skipped {dataset.n_skipped}")
    if len(dataset) < 2:
        raise ValueError("Need at least 2 usable (spectrum, SMILES) pairs to train.")

    has_fold = any(item["fold"] is not None for item in dataset.items)
    if args.augmentation_multiplier != 1 and not args.augmentation_file and not has_fold:
        raise ValueError("need fold info in --input to use augmentation_multiplier, unless you pass --augmentation_file")
    if args.augmentation_multiplier < 0:
        raise ValueError(f"--augmentation_multiplier must be >= 0, got {args.augmentation_multiplier}")
    if args.augmentation_only and args.augmentation_multiplier == 1:
        raise ValueError("--augmentation_only needs --augmentation_multiplier other than 1, otherwise there's no training data at all")

    if has_fold:
        val_fold = [v.strip() for v in args.val_fold.split(",")]
        test_fold = [v.strip() for v in args.test_fold.split(",")]
        train_idx, val_idx, test_idx = split_by_fold(dataset, val_fold=val_fold, test_fold=test_fold)
        print(f"train {len(train_idx)} val {len(val_idx)} test {len(test_idx)}")
        if not val_idx:
            raise ValueError(f"no spectra matched --val_fold {args.val_fold!r}, check your fold values")
        if not test_idx:
            raise ValueError(f"no spectra matched --test_fold {args.test_fold!r}, check your fold values")
        val_set = Subset(dataset, val_idx)

        if args.augmentation_multiplier != 1:
            smiles_test = {dataset.items[i]["smiles"] for i in test_idx}
            smiles_train = {dataset.items[i]["smiles"] for i in train_idx}
            source_smiles = smiles_test if args.augmentation_source == "test" else smiles_train
            opposite_smiles = smiles_train if args.augmentation_source == "test" else smiles_test

            if args.augmentation_exclude == "none":
                smiles_exclude = None
            elif args.augmentation_exclude == "ground_truth":
                smiles_exclude = source_smiles
            else:
                if not args.augmentation_candidates:
                    raise ValueError("--augmentation_exclude candidates/overlap needs --augmentation_candidates to point at a candidates json file")
                print(f"loading candidates from {args.augmentation_candidates}")
                with open(args.augmentation_candidates) as f:
                    candidates_json = json.load(f)
                candidate_union = set()
                for smiles in source_smiles:
                    candidate_union.update(candidates_json.get(smiles, []))
                print(f"found candidates for {sum(1 for s in source_smiles if s in candidates_json)}/{len(source_smiles)} source smiles, {len(candidate_union)} unique candidate smiles total")
                if args.augmentation_exclude == "candidates":
                    smiles_exclude = candidate_union
                else:
                    smiles_exclude = candidate_union & opposite_smiles
            print(f"augmentation source is {args.augmentation_source}, excluding {args.augmentation_exclude} ({len(smiles_exclude) if smiles_exclude is not None else 0} smiles)")

            if args.augmentation_file:
                aug_files = [Path(args.augmentation_file)]
                if not aug_files[0].is_file():
                    raise FileNotFoundError(f"--augmentation_file does not exist: {aug_files[0]}")
                if args.augmentation_source == "test":
                    if len(test_fold) != 1:
                        raise ValueError(f"need just one --test_fold value for augmentation, got {args.test_fold!r}")
                    keep_folds = {canonical_fold(test_fold[0])}
                    fold_filter = lambda f, kf=keep_folds: f is None or f in kf
                else:
                    if len(val_fold) != 1 or len(test_fold) != 1:
                        raise ValueError("need just one --val_fold and one --test_fold value for augmentation_source=train")
                    held_out = {canonical_fold(val_fold[0]), canonical_fold(test_fold[0])}
                    fold_filter = lambda f, ho=held_out: f is None or f not in ho
            else:
                if args.augmentation_source == "test":
                    if len(test_fold) != 1:
                        raise ValueError(f"need just one --test_fold value for augmentation, got {args.test_fold!r}")
                    source_folds = [canonical_fold(test_fold[0])]
                else:
                    if len(val_fold) != 1 or len(test_fold) != 1:
                        raise ValueError("need just one --val_fold and one --test_fold value for augmentation_source=train")
                    held_out = {canonical_fold(val_fold[0]), canonical_fold(test_fold[0])}
                    source_folds = [canonical_fold(f) for f in range(1, args.augmentation_folds + 1) if canonical_fold(f) not in held_out]
                aug_files = train_augmentation_file_paths(
                    args.augmentation_dir, source_folds=source_folds,
                    splits=args.augmentation_splits,
                )
                source_folds_set = set(source_folds)
                fold_filter = lambda f, sfs=source_folds_set: f in sfs
            if args.augmentation_only:
                always_items = []
                print("augmentation_only is on, real train-fold rows are excluded")
            else:
                always_items = [dataset.items[i] for i in train_idx]
            resample_every_epoch = args.augmentation_resample_every_epoch
            if args.augmentation_multiplier == 0:
                if not resample_every_epoch:
                    print("multiplier is 0 so resample_every_epoch does not matter, using everything anyway")
                    resample_every_epoch = True
                n_aug_per_epoch = None
                print(f"using multiplier 0, adding every entry from {len(aug_files)} augmentation files each epoch")
            else:
                n_real = len(train_idx)
                n_aug_per_epoch = (args.augmentation_multiplier - 1) * n_real
                print(f"using multiplier {args.augmentation_multiplier}, adding {n_aug_per_epoch} extra rows from {len(aug_files)} files")
            train_set = AugmentedTrainDataset(
                always_items, aug_files, n_aug_per_epoch, hp,
                fold_filter=fold_filter,
                smiles_exclude=smiles_exclude,
                shuffle_buffer=args.shuffle_buffer,
                resample_every_epoch=resample_every_epoch,
            )
            train_is_streaming = True
        else:
            train_set = Subset(dataset, train_idx)
            train_is_streaming = False
    else:
        n_val = max(1, int(round(len(dataset) * hp.val_fraction)))
        n_train = len(dataset) - n_val
        generator = torch.Generator().manual_seed(hp.seed)
        train_set, val_set = random_split(dataset, [n_train, n_val], generator=generator)
        test_idx = []
        train_is_streaming = False
        print(f"no fold info, random split train {n_train} val {n_val}")

    persistent_workers = args.num_workers > 0
    train_loader = DataLoader(
        train_set, batch_size=hp.batch_size, shuffle=not train_is_streaming,
        collate_fn=collate_fn, num_workers=args.num_workers, persistent_workers=persistent_workers,
    )
    val_loader = DataLoader(
        val_set, batch_size=hp.batch_size, shuffle=False, collate_fn=collate_fn,
        num_workers=args.num_workers, persistent_workers=persistent_workers,
    )

    module = FingerprintLightningModule(**asdict(hp))

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(output_dir / "checkpoints"),
        filename="ion2fp-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
        verbose=True,
    )
    logger = CSVLogger(save_dir=str(output_dir), name="logs")

    trainer = Trainer(
        accelerator=args.accelerator,
        max_epochs=hp.max_epochs,
        logger=logger,
        callbacks=[checkpoint_callback],
    )

    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)

    print(f"done, best model at {checkpoint_callback.best_model_path}")
    print(f"last model at {checkpoint_callback.last_model_path}")

    if test_idx:
        print(f"testing on {len(test_idx)} spectra now")
        test_loader = DataLoader(
            Subset(dataset, test_idx), batch_size=hp.batch_size, shuffle=False,
            collate_fn=collate_fn, num_workers=args.num_workers,
        )
        predictions = trainer.predict(module, dataloaders=test_loader)
        preds = torch.cat(predictions, dim=0)
        targets = torch.as_tensor(
            np.stack([dataset.items[i]["fingerprint"] for i in test_idx]), dtype=torch.float32
        )
        test_cos_sim = torch.nn.functional.cosine_similarity(preds, targets).mean().item()
        print(f"test score is {test_cos_sim:.4f}")


if __name__ == "__main__":
    main()
