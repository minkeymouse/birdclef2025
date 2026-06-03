"""Generate Noisy Student iter 1 pseudo from v12 balanced base.

Difference vs v12:
  - Power transform: p' = p ** alpha   (Boredom 2025 1st recipe)
    Reshapes the soft probability distribution. alpha < 1 sharpens
    (raises low probs more), alpha > 1 flattens.
  - confidence_weights: sum of (p' per row) saved for weighted sampler.
  - Inherit v12's per-class balancing (no new filtering).

Run:
    uv run python experiments/sed/generate_pseudo_noisy_student.py --power 0.7
"""

from __future__ import annotations
import argparse
import numpy as np
from pathlib import Path

SED_DIR = Path(__file__).parent
V12_PATH = SED_DIR / "pseudo_v12_balanced.npz"
OUT_TEMPLATE = "pseudo_noisy_student_iter1_p{p:.2f}.npz"


def main(power: float):
    z = np.load(V12_PATH, allow_pickle=True)
    files = z["files"]
    kept_file_idx = z["kept_file_idx"]
    kept_end_secs = z["kept_end_secs"]
    kept_probs = z["kept_probs"].astype(np.float32)
    print(f"loaded v12: {kept_probs.shape}")

    # Power transform — reshape distribution
    # alpha < 1: sharpens (raise low confidence), alpha > 1: dulls
    kept_probs_ns = np.clip(kept_probs, 1e-6, 1 - 1e-6) ** power
    print(f"power transform p ** {power}")
    print(f"  original: mean {kept_probs.mean():.4f}, max {kept_probs.max():.4f}, "
          f"95pct {np.percentile(kept_probs, 95):.4f}")
    print(f"  after:    mean {kept_probs_ns.mean():.4f}, max {kept_probs_ns.max():.4f}, "
          f"95pct {np.percentile(kept_probs_ns, 95):.4f}")

    # Confidence-weight per row = sum of pseudo labels (Sydorskyi recipe)
    confidence_weights = kept_probs_ns.sum(axis=1).astype(np.float32)
    print(f"confidence_weights: min {confidence_weights.min():.3f}, "
          f"mean {confidence_weights.mean():.3f}, max {confidence_weights.max():.3f}")

    out_path = SED_DIR / OUT_TEMPLATE.format(p=power)
    np.savez(out_path,
             files=files,
             kept_file_idx=kept_file_idx,
             kept_end_secs=kept_end_secs,
             kept_probs=kept_probs_ns,
             confidence_weights=confidence_weights,
             power_alpha=np.array([power], dtype=np.float32))
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--power", type=float, default=0.7)
    args = ap.parse_args()
    main(args.power)
