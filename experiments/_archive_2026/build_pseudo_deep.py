"""Build a PROPER pseudo-label npz from the unlabeled-SS Tucker 5-fold scores
(cache/unlabeled_ss_strided.npz). Addresses prior naive attempts: soft labels, threshold tuned to
the actual score distribution (not blind 0.5), trim<0.1. Output format matches PseudoDataset:
files / kept_file_idx / kept_end_secs / kept_probs.  (Stage A teacher = Tucker; Stage B will swap in
a diverse-teacher ensemble.)  Run: uv run python experiments/build_pseudo_deep.py [keep_target]
"""
import numpy as np, sys
from pathlib import Path
R = Path("/data/birdclef2026")
KEEP_TARGET = int(sys.argv[1]) if len(sys.argv) > 1 else 40000
TRIM = 0.1

U = np.load(R / "cache/unlabeled_ss_strided.npz", allow_pickle=True)
sc = U["scores"].astype(np.float32)            # (N,234) Tucker 5-fold mean, sample_submission order
fidx = U["file_idx"].astype(np.int32); esec = U["end_sec"].astype(np.int32); files = U["files"]
mx = sc.max(axis=1)
print(f"windows: {len(mx)} | max-score percentiles:",
      {p: round(float(np.percentile(mx, p)), 3) for p in (50, 70, 80, 90, 95, 99)})

THR = float(np.percentile(mx, 100 * (1 - KEEP_TARGET / len(mx)))) if len(mx) > KEEP_TARGET else 0.0
keep = mx >= THR
probs = sc[keep].copy(); probs[probs < TRIM] = 0.0
conf = probs.sum(axis=1).astype(np.float32)
out = R / "experiments/sed/pseudo_deep_tucker.npz"
np.savez(out, files=files, kept_file_idx=fidx[keep], kept_end_secs=esec[keep],
         kept_probs=probs.astype(np.float32), confidence_weights=conf)
print(f"THR={THR:.3f} kept {keep.sum()}/{len(mx)} ({100*keep.mean():.0f}%) | "
      f"avg active classes/window={(probs > 0).sum(1).mean():.2f} | -> {out}")
