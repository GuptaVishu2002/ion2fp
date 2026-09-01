import argparse
import faulthandler
import json
import multiprocessing
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import DataStructs
from tqdm import tqdm


def random_tiebreak_ranking(scores):
    scores = np.array(scores)
    ranks = np.empty_like(scores, dtype=int)
    unique_scores = np.unique(scores)
    current_rank = 1
    for val in sorted(unique_scores, reverse=True):
        idx = np.where(scores == val)[0]
        if len(idx) > 1:
            np.random.shuffle(idx)
        for i in idx:
            ranks[i] = current_rank
            current_rank += 1
    return ranks


def convert_one_smiles(item):
    smiles, bitvectors = item
    arrays = []
    for bv in bitvectors:
        arr = np.zeros((0,), dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(bv, arr)
        arrays.append(arr)
    return smiles, arrays


def load_candidates(path, needed_smiles, workers=None):
    suffix = Path(path).suffix.lower()

    if suffix == ".json":
        with open(path, "r") as f:
            fp_dict = json.load(f)
        return {s: fps for s, fps in fp_dict.items() if s in needed_smiles}

    if suffix in (".pkl", ".pickle"):
        t0 = time.time()
        size_mb = os.path.getsize(path) / (1024 * 1024)
        with open(path, "rb") as f:
            raw_bytes = f.read()
        t1 = time.time()
        raw_fp_dict = pickle.loads(raw_bytes)
        del raw_bytes
        raw_fp_dict = {s: fps for s, fps in raw_fp_dict.items() if s in needed_smiles}

        spawn_ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=spawn_ctx) as ex:
            converted = tqdm(
                ex.map(convert_one_smiles, raw_fp_dict.items(), chunksize=50),
                total=len(raw_fp_dict), desc="Converting fingerprints",
            )
            result = dict(converted)
        return result

    raise ValueError(f"candidates file must be .json or .pkl/.pickle, got {suffix}")


def main():
    import torch
    import torch.nn.functional as F

    faulthandler.dump_traceback_later(120, repeat=True, exit=False, file=sys.stderr)

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--predictions", required=True, type=str)
    parser.add_argument("--candidates", required=True, type=str)
    parser.add_argument("--scores_out", required=True, type=str)
    parser.add_argument("--ranks_out", required=True, type=str)
    parser.add_argument("--similarity", type=str, default="bce", choices=["bce", "cos"])
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    for out_path in (args.scores_out, args.ranks_out):
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(args.predictions)
    if "smiles" not in df.columns:
        raise ValueError(f"{args.predictions} has no smiles column, re-run predict.py with smiles included")
    fp_cols = sorted(
        [c for c in df.columns if c.startswith("fp_")],
        key=lambda c: int(c.split("_")[1]),
    )
    if not fp_cols:
        raise ValueError(f"{args.predictions} has no fp_* columns")

    print(f"loading candidates from {args.candidates}, {df['smiles'].nunique()} unique smiles to match", flush=True)
    fp_dict = load_candidates(args.candidates, needed_smiles=set(df["smiles"]), workers=args.workers)

    example_len = None
    for smiles in df["smiles"]:
        cand_fps = fp_dict.get(smiles, [])
        if len(cand_fps):
            example_len = len(cand_fps[0])
            break
    if example_len is not None and example_len != len(fp_cols):
        raise ValueError(f"fingerprint size mismatch, predictions are {len(fp_cols)}-bit but candidates are {example_len}-bit")

    eps = 1e-7

    smiles_arr = df["smiles"].to_numpy()
    identifier_arr = df["identifier"].to_numpy()
    fp_arr = df[fp_cols].to_numpy(dtype=np.float32)

    def process_row(i):
        smiles = smiles_arr[i]
        identifier = identifier_arr[i]
        cand_fps = fp_dict.get(smiles, [])
        if len(cand_fps) == 0:
            return identifier, [], []

        cands = torch.tensor(np.array(cand_fps).astype(np.float32))
        row_fp = fp_arr[i]
        if args.similarity == "bce":
            fp_pred = torch.from_numpy(np.clip(row_fp, eps, 1.0 - eps)).unsqueeze(0)
            fp_pred_repeated = fp_pred.repeat(cands.size(0), 1)
            similarities = 1 - torch.mean(
                F.binary_cross_entropy(fp_pred_repeated, cands, reduction="none"), dim=1
            )
        else:  # cos
            fp_pred = torch.from_numpy(row_fp.astype(np.float32)).unsqueeze(0)
            fp_pred_repeated = fp_pred.repeat(cands.size(0), 1)
            similarities = F.cosine_similarity(fp_pred_repeated, cands, dim=1)

        scores = similarities.cpu().numpy()
        ranks = random_tiebreak_ranking(scores)
        return identifier, scores, ranks

    _ = F.binary_cross_entropy(torch.tensor([0.5]), torch.tensor([0.5]))
    _ = F.cosine_similarity(torch.tensor([[0.5]]), torch.tensor([[0.5]]))
    print("torch warmed up, starting thread pool now...", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        results = list(tqdm(executor.map(process_row, range(len(df))), total=len(df), desc="Processing rows"))

    n_missing = sum(1 for _, scores, _ in results if len(scores) == 0)
    if n_missing:
        print(f"{n_missing}/{len(results)} spectra had no candidates found")

    with open(args.scores_out, "w") as score_f, open(args.ranks_out, "w") as rank_f:
        for identifier, scores, ranks in tqdm(results, desc="Writing output"):
            score_f.write(f"{identifier} {' '.join(map(str, scores))}\n")
            rank_f.write(f"{identifier} {' '.join(map(str, ranks))}\n")

    print(f"wrote scores to {args.scores_out} and ranks to {args.ranks_out}")


if __name__ == "__main__":
    main()
