"""Convert pseudo_soundscapes_labels_v10.csv → PseudoDataset npz format.

PseudoDataset expects:
  - kept_files: array of filenames (unique)
  - kept_file_idx: array of file indices for each window
  - kept_end_secs: array of end_sec for each window
  - kept_probs: (n_windows, 234) probability/multi-hot per window

v10 is hard pseudo labels: each row = (filename, start, end, primary_label).
We convert to multi-hot binary target per (filename, end_sec) window.

For each window, classes present in v10 = 1, others = 0.
Tier weights are encoded in the binary as well (we keep all 3 tiers).

Output: experiments/sed/pseudo_v10.npz
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/data/birdclef2026")
V10 = ROOT / "data/birdclef-2026/pseudo_soundscapes_labels_v10.csv"
SAMPLE_SUB = ROOT / "data/birdclef-2026/sample_submission.csv"
OUT_NPZ = ROOT / "experiments/sed/pseudo_v10.npz"


def main():
    print("Loading v10...")
    v10 = pd.read_csv(V10)
    print(f"v10: {v10.shape}")

    # Build class index from sample_submission.csv (canonical 234 ordering)
    sub = pd.read_csv(SAMPLE_SUB, nrows=1)
    PRIMARY_LABELS = list(sub.columns[1:])
    assert len(PRIMARY_LABELS) == 234
    l2i = {c: i for i, c in enumerate(PRIMARY_LABELS)}

    # Group by (filename, end) — collect labels per window
    # v10 has start/end as integers (sec offsets)
    v10["end"] = v10["end"].astype(int)
    v10["filename"] = v10["filename"].astype(str)

    # Map labels to class index
    v10["cls_idx"] = v10["primary_label"].map(l2i)
    if v10["cls_idx"].isna().any():
        missing = v10[v10["cls_idx"].isna()]["primary_label"].unique()
        print(f"WARNING: {len(missing)} primary_label not in PRIMARY_LABELS:")
        print(f"  {missing}")
        v10 = v10.dropna(subset=["cls_idx"])
    v10["cls_idx"] = v10["cls_idx"].astype(int)

    # Group by (filename, end_sec) → multi-hot target
    window_keys = v10.groupby(["filename", "end"])["cls_idx"].apply(list).reset_index()
    print(f"Unique windows: {len(window_keys)}")
    print(f"Mean labels per window: {window_keys['cls_idx'].apply(len).mean():.2f}")
    print(f"Max labels per window: {window_keys['cls_idx'].apply(len).max()}")

    # Build outputs
    kept_files = window_keys["filename"].unique()
    file_to_idx = {f: i for i, f in enumerate(kept_files)}
    kept_file_idx = window_keys["filename"].map(file_to_idx).to_numpy(dtype=np.int32)
    kept_end_secs = window_keys["end"].to_numpy(dtype=np.int32)

    n_windows = len(window_keys)
    kept_probs = np.zeros((n_windows, 234), dtype=np.float32)
    for i, cls_list in enumerate(window_keys["cls_idx"].values):
        for c in cls_list:
            kept_probs[i, c] = 1.0

    print(f"\nOutput shapes:")
    print(f"  kept_files: {kept_files.shape}")
    print(f"  kept_file_idx: {kept_file_idx.shape}")
    print(f"  kept_end_secs: {kept_end_secs.shape}")
    print(f"  kept_probs: {kept_probs.shape}")
    print(f"  Positive density: {kept_probs.mean():.4f} ({100*kept_probs.mean():.2f}%)")

    np.savez_compressed(
        OUT_NPZ,
        files=kept_files,            # NB: train.py expects "files" key, not "kept_files"
        kept_file_idx=kept_file_idx,
        kept_end_secs=kept_end_secs,
        kept_probs=kept_probs,
    )
    print(f"\nSaved: {OUT_NPZ}")
    print(f"Use this in SedConfig.pseudo.pseudo_npz = 'experiments/sed/pseudo_v10.npz'")


if __name__ == "__main__":
    main()
