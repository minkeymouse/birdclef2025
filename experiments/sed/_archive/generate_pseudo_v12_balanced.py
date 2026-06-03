"""generate_pseudo_v12_balanced.py — F6 fix for v62 failure mode.

v62 post-mortem diagnostic (2026-05-18) showed:
  - 65380: 30,659 pseudo chunks → 10 false positives by exp183 student
  - 47158son06: 410 chunks → 9 false positives
  - 516975: 4 chunks → 8 missed true positives
  → Student picks up pseudo class distribution as bias, NOT as supervision

F6 fix: per-class cap + labeled SS exclusion.

Differs from v11_noisy:
  1. **Skip labeled SS files** (66 files leak as direct supervision; ~1.6%
     of pseudo but pure leakage)
  2. **Per-class cap = 1500 chunks**: for chunks where max-prob class has
     already accumulated 1500 chunks, drop. Random keep below cap.
  3. Same selection logic (max prob ≥ 0.5 keep chunk, classes < 0.1 zero out)

Output: experiments/sed/pseudo_v12_balanced.npz
Expected size: ~15-20k chunks (down from 49k in v11)
"""
from __future__ import annotations
import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path("/data/birdclef2026")
sys.path.insert(0, str(ROOT))

from experiments.sed.config import exp175_tucker_actually
from experiments.sed.model import DistilledSED
from experiments.sed.data import load_audio, DATA

SR = 32_000
CHUNK_S = 5
FILE_DUR_S = 60
N_CHUNKS = FILE_DUR_S // CHUNK_S
N_CLS = 234
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TEACHER_CKPTS = [
    ROOT / "experiments/_data_pipelines/exp175_outputs/seed42" / f"fold{i}/best_ckpt.pt"
    for i in range(5)
] + [
    ROOT / "experiments/_data_pipelines/exp176_outputs" / f"fold{i}/best_ckpt.pt"
    for i in range(5)
]

OUT_PATH = ROOT / "experiments/sed/pseudo_v12_balanced.npz"

# F6 hyperparams
PER_CLASS_CAP = 1500
MAX_THR = 0.5
FLOOR_THR = 0.1


class SSFileDataset(Dataset):
    def __init__(self, files, source_dir: Path):
        self.files = files
        self.source_dir = source_dir

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = str(self.files[idx])
        wav = load_audio(self.source_dir / fname, SR)
        target = FILE_DUR_S * SR
        if wav is None:
            wav = np.zeros(target, dtype=np.float32)
        elif len(wav) < target:
            wav = np.pad(wav, (0, target - len(wav)))
        else:
            wav = wav[:target]
        chunks = wav.reshape(N_CHUNKS, CHUNK_S * SR).astype(np.float32)
        return idx, torch.from_numpy(chunks)


def load_teachers():
    cfg = exp175_tucker_actually()
    models = []
    for ck in TEACHER_CKPTS:
        assert ck.exists(), f"missing {ck}"
        m = DistilledSED(cfg, n_cls=N_CLS).to(DEVICE)
        sd = torch.load(ck, map_location=DEVICE, weights_only=False)
        if "model_state_dict" in sd:
            sd = sd["model_state_dict"]
        elif "state_dict" in sd:
            sd = sd["state_dict"]
        m.load_state_dict(sd, strict=False)
        m.eval()
        models.append(m)
    print(f"Loaded {len(models)} teacher ckpts.")
    return models


@torch.inference_mode()
def predict_file(models, chunks):
    chunks = chunks.to(DEVICE, non_blocking=True)
    probs_sum = torch.zeros(N_CHUNKS, N_CLS, device=DEVICE)
    for m in models:
        out = m(chunks)
        if isinstance(out, dict):
            logits = out.get("clip_logits", out.get("logits"))
        elif isinstance(out, (tuple, list)):
            logits = out[0]
        else:
            logits = out
        probs_sum += torch.sigmoid(logits)
    return (probs_sum / len(models)).cpu().numpy()


def get_labeled_ss_files():
    """Return set of filenames that appear in labeled SS data (must be excluded from pseudo)."""
    ss_csv = DATA / "train_soundscapes_labels.csv"
    df = pd.read_csv(ss_csv)
    return set(df["filename"].unique())


def main(limit: int = 0):
    src_dir = DATA / "train_soundscapes"
    all_files = sorted([f.name for f in src_dir.glob("*.ogg")])

    # F6 fix 1: exclude labeled SS files
    labeled_fnames = get_labeled_ss_files()
    files = [f for f in all_files if f not in labeled_fnames]
    print(f"Total files: {len(all_files)}, after labeled-SS exclusion: {len(files)} (-{len(all_files)-len(files)})")
    if limit > 0:
        files = files[:limit]

    models = load_teachers()
    ds = SSFileDataset(files, src_dir)
    loader = DataLoader(ds, batch_size=1, num_workers=4, pin_memory=True)

    # Collect ALL candidate chunks first (before per-class cap)
    cand_file_idx = []
    cand_end_secs = []
    cand_probs = []
    cand_top_class = []  # top-1 class per chunk for cap accounting

    t0 = time.time()
    for batch_i, (file_idx_t, chunks_t) in enumerate(loader):
        file_idx = int(file_idx_t.item())
        probs = predict_file(models, chunks_t[0])  # (12, 234)
        max_per_chunk = probs.max(axis=1)
        top_per_chunk = probs.argmax(axis=1)
        kept_mask = max_per_chunk >= MAX_THR
        # zero out per-class
        out = probs.copy()
        out[out < FLOOR_THR] = 0.0
        for ci in np.where(kept_mask)[0]:
            cand_file_idx.append(file_idx)
            cand_end_secs.append((ci + 1) * CHUNK_S)
            cand_probs.append(out[ci])
            cand_top_class.append(int(top_per_chunk[ci]))

        if (batch_i + 1) % 100 == 0:
            dt = time.time() - t0
            rate = (batch_i + 1) / dt
            eta = (len(files) - batch_i - 1) / rate / 60
            print(f"  [{batch_i+1}/{len(files)}] cand={len(cand_file_idx)} rate={rate:.2f} f/s eta={eta:.1f}min", flush=True)

    print(f"\nDone inference in {(time.time()-t0)/60:.1f} min")
    print(f"Total candidate chunks: {len(cand_file_idx)}")

    # F6 fix 2: per-class cap
    print(f"\n=== Applying per-class cap = {PER_CLASS_CAP} ===")
    cand_top_class = np.array(cand_top_class, dtype=np.int32)
    by_class_counts = np.bincount(cand_top_class, minlength=N_CLS)
    print(f"  Per-class chunk distribution before cap:")
    top10 = np.argsort(-by_class_counts)[:10]
    bot10 = np.argsort(by_class_counts)[:10]
    for c in top10:
        print(f"    class {c:3d}: {by_class_counts[c]:5d} chunks")
    print(f"  ...")
    for c in bot10:
        if by_class_counts[c] > 0:
            print(f"    class {c:3d}: {by_class_counts[c]:5d} chunks")

    # Random select within cap per class
    rng = np.random.default_rng(42)
    selected = np.zeros(len(cand_top_class), dtype=bool)
    for c in range(N_CLS):
        mask = cand_top_class == c
        idxs = np.where(mask)[0]
        if len(idxs) <= PER_CLASS_CAP:
            selected[idxs] = True
        else:
            keep = rng.choice(idxs, PER_CLASS_CAP, replace=False)
            selected[keep] = True

    sel_idx = np.where(selected)[0]
    files_arr = np.array(files, dtype=object)
    kept_file_idx = np.array([cand_file_idx[i] for i in sel_idx], dtype=np.int32)
    kept_end_secs = np.array([cand_end_secs[i] for i in sel_idx], dtype=np.int32)
    kept_probs = np.array([cand_probs[i] for i in sel_idx], dtype=np.float32)

    print(f"\n=== After per-class cap ===")
    print(f"Total chunks: {len(sel_idx)} (kept {100*len(sel_idx)/len(cand_top_class):.1f}% of candidates)")
    final_counts = np.bincount(cand_top_class[selected], minlength=N_CLS)
    print(f"  Max per-class count: {final_counts.max()}")
    print(f"  Min per-class count (non-zero): {final_counts[final_counts > 0].min() if (final_counts > 0).any() else 0}")
    print(f"  Classes with ≥1 chunk: {(final_counts > 0).sum()} / {N_CLS}")
    print(f"  Top10 (post-cap):")
    for c in np.argsort(-final_counts)[:10]:
        print(f"    class {c:3d}: {final_counts[c]} chunks")

    print(f"\nProb stats:")
    p = kept_probs
    print(f"  Mean nonzero classes/chunk: {(p > 0).sum(1).mean():.2f}")
    print(f"  Mean prob value (nonzero): {p[p > 0].mean():.3f}")
    print(f"  Soft signal (frac 0.1<p<0.9): {((p > 0.1) & (p < 0.9)).sum() / max(1, (p > 0).sum()):.3f}")

    np.savez_compressed(OUT_PATH,
        files=files_arr,
        kept_file_idx=kept_file_idx,
        kept_end_secs=kept_end_secs,
        kept_probs=kept_probs,
    )
    print(f"\nSaved → {OUT_PATH}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    main(limit=args.limit)
