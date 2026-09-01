import argparse
import os
import sys

for _i, _arg in enumerate(sys.argv):
    _is_cpu_flag = (_arg == "--accelerator" and sys.argv[_i + 1 : _i + 2] == ["cpu"]) or _arg == "--accelerator=cpu"
    if _is_cpu_flag:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
        break

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

from ion2fp.data import SpectrumFingerprintDataset, collate_fn, filter_by_fold, load_spectra
from ion2fp.lightning_module import FingerprintLightningModule


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--input", required=True, type=str)
    parser.add_argument("--output", required=True, type=str)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--fold", type=str, default=None)
    args = parser.parse_args()

    module = FingerprintLightningModule.load_from_checkpoint(args.checkpoint, map_location="cpu")
    hp = module.hp
    module.eval()

    print(f"loading spectra from {args.input}")
    spectra = load_spectra(args.input)
    dataset = SpectrumFingerprintDataset(spectra, hp, require_labels=False)
    print(f"got {len(dataset)} spectra, skipped {dataset.n_skipped}")

    if args.fold is not None:
        n_before = len(dataset)
        folds = [v.strip() for v in args.fold.split(",")]
        dataset = filter_by_fold(dataset, folds)
        print(f"filtered to fold {args.fold!r}: {len(dataset)}/{n_before} spectra")
        if len(dataset) == 0:
            raise ValueError(f"no spectra matched --fold {args.fold!r}, check your fold values")

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    trainer = Trainer(accelerator=args.accelerator, logger=False, enable_checkpointing=False, enable_progress_bar=False)
    predictions = trainer.predict(module, dataloaders=loader)
    preds = torch.cat(predictions, dim=0).numpy()

    all_ids = [item["identifier"] for item in dataset.items]
    all_smiles = [item["smiles"] for item in dataset.items]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix == ".npy":
        np.save(output_path, preds.astype(np.float32))
    else:
        df = pd.DataFrame(preds, columns=[f"fp_{i}" for i in range(preds.shape[1])])
        df.insert(0, "smiles", all_smiles)
        df.insert(0, "identifier", all_ids)
        df.to_csv(output_path, index=False)

    print(f"saved {preds.shape[0]} fingerprints ({preds.shape[1]} bits) to {output_path}")

    fingerprints = [item["fingerprint"] for item in dataset.items]
    if fingerprints and all(fp is not None for fp in fingerprints):
        target = torch.as_tensor(np.stack(fingerprints), dtype=torch.float32)
        sim = torch.nn.functional.cosine_similarity(torch.as_tensor(preds), target).mean().item()
        print(f"mean cosine similarity to true fingerprints is {sim:.4f}")


if __name__ == "__main__":
    main()
