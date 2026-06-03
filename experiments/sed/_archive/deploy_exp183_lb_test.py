"""deploy_exp183_lb_test.py — ONE-shot LB test for Noisy Student iter 1.

Test: fold_filter = (0..4, 16..19, 30)
  Replaces exp176 fold 0 (slot 15) with exp183 fold 0 (slot 30).
  Same 10-ckpt ensemble size as v53 anchor; isolates Noisy Student effect to
  one fold slot.

Pre-condition: exp183 fold 0 trained (best_ckpt.pt exists at slot 30).
Post-condition: Kaggle dataset re-pushed, kernel pushed, submission ready.

Usage:
  KAGGLE_API_TOKEN=$(cat ~/.kaggle/access_token) uv run python -m experiments.sed.deploy_exp183_lb_test
  # then manually: kaggle competitions submit ... once kernel completes
"""
from __future__ import annotations
import sys
import shutil
from pathlib import Path

ROOT = Path("/data/birdclef2026")
sys.path.insert(0, str(ROOT))

from experiments.sed.config import exp183_noisy_student_iter1
from experiments.sed.deploy_ensemble import (
    collect_members,
    export_all_members,
    upload_dataset,
    SLUG,
    TOKEN,
)
from experiments.sed._common import (
    KERNEL_SLUG,
    META_PATH,
    NB_DIR,
    NB_PATH,
    PERCH_META_DATASET,
    PERCH_ONNX_DATASET,
    push_kernel,
    reset_notebook_to_anchor,
)
from experiments.sed.notebook_state import NotebookState, apply_state
import json

# Custom fold filter: exp175 seed=42 fold 0-4 + exp176 fold 1-4 + exp183 fold 0
# Slot indexing in deploy_ensemble.collect_members():
#   exp175 seed=42 → 0-4
#   exp175 seed=43 → 5-9
#   exp175 seed=44 → 10-14
#   exp176        → 15-19
#   exp177        → 20-24
#   exp178        → 25-29
#   exp183 fold 0 → 30
FOLD_FILTER = (0, 1, 2, 3, 4, 16, 17, 18, 19, 30)


def verify_exp183_ready():
    """Confirm exp183 fold 0 is in collected members at expected slot."""
    members = collect_members()
    print(f"Total members discovered: {len(members)}")
    for i, m in enumerate(members):
        if m[0] == "exp183_noisy_student_iter1":
            print(f"  exp183 fold {m[2]} → slot {i}")
            if i not in FOLD_FILTER:
                print(f"  WARNING: exp183 at slot {i} but FOLD_FILTER doesn't include it!")
    return members


def patch_notebook_with_filter():
    """Patch notebook with our custom fold_filter (replaces exp176 fold 0 with exp183 fold 0)."""
    apply_state(
        NotebookState(
            sed_dataset=SLUG,
            sed_finder_dir_token=TOKEN,
            ulyanov_blend=False,
            konbu_head=False,
            sonotype_mirror=True,
            rare_suppression=True,
            sed_fold_filter=FOLD_FILTER,
        ),
        NB_PATH,
    )
    print(f"Patched notebook: SED_FOLD_FILTER = {FOLD_FILTER}")


def patch_metadata():
    sources = [PERCH_META_DATASET, PERCH_ONNX_DATASET, SLUG]
    if META_PATH.exists():
        meta = json.loads(META_PATH.read_text())
        meta["dataset_sources"] = sources
    else:
        raise FileNotFoundError(f"{META_PATH} missing")
    META_PATH.write_text(json.dumps(meta, indent=2))
    print(f"Patched metadata.")


def main():
    print("=" * 80)
    print("exp183 Noisy Student iter 1 — LB test deploy")
    print("=" * 80)

    members = verify_exp183_ready()
    has_183 = any(m[0] == "exp183_noisy_student_iter1" for m in members)
    if not has_183:
        print("\n❌ exp183 fold 0 not in members! Has the training completed?")
        print("   Expected: experiments/_data_pipelines/exp183_outputs/fold0/best_ckpt.pt")
        sys.exit(1)

    # Export all members to fresh ONNX dir
    onnx_dir = Path("/data/birdclef2026/experiments/_data_pipelines/ensemble_v3/onnx")
    if onnx_dir.exists():
        shutil.rmtree(onnx_dir)
    print(f"\n=== Exporting {len(members)} ckpts → {onnx_dir} ===")
    export_all_members(onnx_dir, members)
    n_onnx = len(list(onnx_dir.glob("*.onnx")))
    print(f"Exported {n_onnx} ONNX files.")

    # Upload Kaggle dataset
    print(f"\n=== Uploading Kaggle dataset {SLUG} ===")
    upload_dataset(onnx_dir)

    # Patch notebook + metadata + push
    print(f"\n=== Patching notebook ===")
    reset_notebook_to_anchor()
    patch_notebook_with_filter()
    patch_metadata()

    print(f"\n=== Pushing kernel {KERNEL_SLUG} ===")
    v = push_kernel()
    print(f"\n✅ Pushed kernel version {v}.")
    print(f"\nNext step: wait for kernel to complete, then submit with:")
    print(f"  uv run kaggle competitions submit -c birdclef-2026 \\")
    print(f"      -f submission.csv \\")
    print(f"      -k {KERNEL_SLUG} -v {v} \\")
    print(f"      -m 'v60 exp183 Noisy Student iter 1 — soft pseudo from 10-ckpt teacher, replace exp176 fold 0'")


if __name__ == "__main__":
    main()
