"""generate_pseudo_v11_noisy.py — Soft pseudo labels from 10-ckpt ensemble teacher.

Implements Sydorskyi 2025 2nd-place selection logic:
  1. Predict each 5s chunk of every unlabeled SS file with each of 10 teacher ckpts
  2. Average sigmoid(clip_logits) across the 10 ckpts (ensemble teacher)
  3. Per-chunk selection: drop chunk if max(prob) < 0.5
  4. For retained chunks: zero out classes with prob < 0.1, KEEP REMAINING AS SOFT

Output: experiments/sed/pseudo_v11_noisy.npz with keys
  files            (N_files,)   str
  kept_file_idx    (N_chunks,)  int32
  kept_end_secs    (N_chunks,)  int32
  kept_probs       (N_chunks, 234) float32  ← SOFT (not binarized)

Differs from prior v3/v7/v10: those were either v33 single-teacher binarized,
sonotype-cosine binarized, or filter-rule binarized. v11 is ensemble-mean SOFT
per 2nd place recipe (kept_probs[i] is the actual averaged sigmoid output).

Wall time: ~2-4 hr on RTX 5090 (10,658 files × 12 chunks × 10 ckpts).
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path("/data/birdclef2026")
sys.path.insert(0, str(ROOT))

from experiments.sed.config import SedConfig, exp175_tucker_actually
from experiments.sed.model import DistilledSED
from experiments.sed.data import load_audio, DATA

SR = 32_000
CHUNK_S = 5
FILE_DUR_S = 60
N_CHUNKS = FILE_DUR_S // CHUNK_S  # 12 per file
N_CLS = 234
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TEACHER_CKPTS = [
    ROOT / "experiments/_data_pipelines/exp175_outputs/seed42" / f"fold{i}/best_ckpt.pt"
    for i in range(5)
] + [
    ROOT / "experiments/_data_pipelines/exp176_outputs" / f"fold{i}/best_ckpt.pt"
    for i in range(5)
]

OUT_PATH = ROOT / "experiments/sed/pseudo_v11_noisy.npz"


class SSFileDataset(Dataset):
    """Yields (file_idx, 12-chunk batch) per __getitem__."""

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


def load_teachers() -> list[DistilledSED]:
    cfg = exp175_tucker_actually()
    models = []
    for ckpt_path in TEACHER_CKPTS:
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Missing teacher ckpt: {ckpt_path}")
        m = DistilledSED(cfg, n_cls=N_CLS).to(DEVICE)
        sd = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        if "model_state_dict" in sd:
            sd = sd["model_state_dict"]
        elif "state_dict" in sd:
            sd = sd["state_dict"]
        missing, unexpected = m.load_state_dict(sd, strict=False)
        if len(missing) > 5 or len(unexpected) > 5:
            print(f"  WARN {ckpt_path.parent.parent.name}/{ckpt_path.parent.name}: "
                  f"missing={len(missing)} unexpected={len(unexpected)}")
        m.eval()
        models.append(m)
    print(f"Loaded {len(models)} teacher ckpts.")
    return models


@torch.inference_mode()
def predict_file(models: list[DistilledSED], chunks: torch.Tensor) -> np.ndarray:
    """chunks: (12, SR*5). Returns (12, 234) averaged sigmoid probs."""
    chunks = chunks.to(DEVICE, non_blocking=True)
    probs_sum = torch.zeros(N_CHUNKS, N_CLS, device=DEVICE)
    for m in models:
        out = m(chunks)
        # Model returns either logits tensor or dict — handle both.
        if isinstance(out, dict):
            logits = out.get("clip_logits", out.get("logits"))
        elif isinstance(out, (tuple, list)):
            logits = out[0]
        else:
            logits = out
        probs_sum += torch.sigmoid(logits)
    return (probs_sum / len(models)).cpu().numpy()


def apply_selection(probs: np.ndarray, max_thr: float = 0.5,
                    floor_thr: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    """2nd place selection. Returns (kept_chunk_mask, processed_probs).

    Chunk kept iff max(prob) >= max_thr. For retained chunks: zero out classes
    with prob < floor_thr; keep remaining as soft.
    """
    max_per_chunk = probs.max(axis=1)
    kept_mask = max_per_chunk >= max_thr
    out = probs.copy()
    out[out < floor_thr] = 0.0
    return kept_mask, out


def main(limit: int = 0):
    src_dir = DATA / "train_soundscapes"
    files = sorted([f.name for f in src_dir.glob("*.ogg")])
    if limit > 0:
        files = files[:limit]
    print(f"Files to process: {len(files)}")

    models = load_teachers()

    ds = SSFileDataset(files, src_dir)
    loader = DataLoader(ds, batch_size=1, num_workers=4, pin_memory=True)

    all_file_idx = []
    all_end_secs = []
    all_probs = []

    t0 = time.time()
    for batch_i, (file_idx_t, chunks_t) in enumerate(loader):
        file_idx = int(file_idx_t.item())
        chunks = chunks_t[0]  # (12, SR*5)
        probs = predict_file(models, chunks)
        kept_mask, processed = apply_selection(probs)
        for ci in np.where(kept_mask)[0]:
            all_file_idx.append(file_idx)
            all_end_secs.append((ci + 1) * CHUNK_S)
            all_probs.append(processed[ci])

        if (batch_i + 1) % 100 == 0:
            dt = time.time() - t0
            rate = (batch_i + 1) / dt
            eta = (len(files) - batch_i - 1) / rate / 60
            print(f"  [{batch_i+1}/{len(files)}] kept={len(all_file_idx)} "
                  f"rate={rate:.2f} f/s eta={eta:.1f} min", flush=True)

    print(f"\nDone in {(time.time()-t0)/60:.1f} min")
    print(f"Total kept chunks: {len(all_file_idx)} ({100*len(all_file_idx)/(len(files)*N_CHUNKS):.1f}% of "
          f"{len(files)*N_CHUNKS} total)")

    out = dict(
        files=np.array(files, dtype=object),
        kept_file_idx=np.array(all_file_idx, dtype=np.int32),
        kept_end_secs=np.array(all_end_secs, dtype=np.int32),
        kept_probs=np.array(all_probs, dtype=np.float32),
    )

    # Quick sanity print
    p = out["kept_probs"]
    if len(p) > 0:
        print(f"\nProb stats:")
        print(f"  Mean nonzero classes/chunk: {(p > 0).sum(1).mean():.2f}")
        print(f"  Mean prob value (nonzero): {p[p > 0].mean():.3f}")
        print(f"  Soft signal (frac nonzero with 0.1 < p < 0.9): "
              f"{((p > 0.1) & (p < 0.9)).sum() / max(1, (p > 0).sum()):.3f}")

    np.savez_compressed(OUT_PATH, **out)
    print(f"\nSaved → {OUT_PATH}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0,
                    help="Process only first N files (0=all). Use 10 for dry-run.")
    args = ap.parse_args()
    main(limit=args.limit)
