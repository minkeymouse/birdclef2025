"""Launch exp185 (full Noisy Student) — single-fold smoke test by default,
add --all-folds to run 5 folds.

Smoke test usage:
    uv run python experiments/sed/run_exp185.py --fold 0

5-fold usage:
    uv run python experiments/sed/run_exp185.py --all-folds
"""

from __future__ import annotations
import argparse
import time
from pathlib import Path
import sys

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from experiments.sed.config import exp185_noisy_student_full
from experiments.sed.train import train_one_fold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0, help="fold id 0..4")
    ap.add_argument("--all-folds", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = exp185_noisy_student_full()
    cfg.seed = args.seed
    print(f"\n=== exp185_noisy_student_full (seed={cfg.seed}) ===")
    print(f"  pseudo_npz: {cfg.pseudo.pseudo_npz}")
    print(f"  pseudo_share: {cfg.pseudo.pseudo_share_per_batch}")
    print(f"  MIXUP_PROB: {cfg.MIXUP_PROB}")
    print(f"  drop_path_rate: {cfg.BACKBONE_DROP_PATH_RATE}")
    print(f"  use_pseudo_weighted_sampler: {cfg.use_pseudo_weighted_sampler}")
    print()

    # CI check
    diffs = cfg.diff_from_tucker()
    print(f"diff from Tucker: {len(diffs)} fields")
    for k, v in diffs.items():
        print(f"  {k}: tucker={v['tucker']} → ours={v['ours']}")
    print()

    folds = list(range(5)) if args.all_folds else [args.fold]
    t0 = time.time()
    for f in folds:
        print(f"\n{'='*60}\n  FOLD {f}  (elapsed {time.time()-t0:.0f}s)\n{'='*60}")
        train_one_fold(cfg, fold_id=f)


if __name__ == "__main__":
    main()
