"""deploy_exp183_5fold_lb.py — ONE-shot LB test for full 5-fold Noisy Student swap.

Test: fold_filter = (0..4, 30..34)
  Replaces ALL exp176 (slots 15-19) with exp183 (slots 30-34).
  Same 10-ckpt ensemble size as v53; isolates Noisy Student framework
  effect at full 5-fold scale.

Comparison to v53 anchor:
  v53: exp175 seed42 fold 0-4 + exp176 fold 0-4 (10 ckpts)
  v62: exp175 seed42 fold 0-4 + exp183 fold 0-4 (10 ckpts)
  Difference = exp176 (per-fold-SS, no pseudo) → exp183 (Noisy Student soft pseudo)

Pre-condition: exp183 fold 0-4 all trained.
Post-condition: Kaggle dataset re-pushed, kernel pushed, submission ready.
"""
from __future__ import annotations
import sys
import shutil
import json
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

# Full 5-fold swap: replace ALL of exp176 (slots 15-19) with exp183 (slots 30-34)
FOLD_FILTER = (0, 1, 2, 3, 4, 30, 31, 32, 33, 34)


def verify_exp183_5fold_ready():
    members = collect_members()
    exp183_count = sum(1 for m in members if m[0] == "exp183_noisy_student_iter1")
    print(f"Total members: {len(members)}")
    print(f"exp183 folds available: {exp183_count}")
    exp183_slots = [i for i, m in enumerate(members) if m[0] == "exp183_noisy_student_iter1"]
    print(f"exp183 slots: {exp183_slots}")
    return members, exp183_count, exp183_slots


def patch_notebook_5fold():
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
    meta = json.loads(META_PATH.read_text())
    meta["dataset_sources"] = sources
    META_PATH.write_text(json.dumps(meta, indent=2))
    print("Patched metadata.")


def main():
    print("=" * 80)
    print("exp183 Noisy Student iter 1 — FULL 5-FOLD LB test deploy")
    print("=" * 80)

    members, exp183_count, exp183_slots = verify_exp183_5fold_ready()
    if exp183_count < 5:
        print(f"\n❌ Only {exp183_count}/5 exp183 folds available. Wait for training.")
        sys.exit(1)
    if not all(s in FOLD_FILTER for s in exp183_slots[:5]):
        print(f"\n❌ exp183 slots {exp183_slots} don't match expected FOLD_FILTER")
        print(f"   Expected exp183 at slots [30, 31, 32, 33, 34]")
        sys.exit(1)

    onnx_dir = Path("/data/birdclef2026/experiments/_data_pipelines/ensemble_v3/onnx")
    if onnx_dir.exists():
        shutil.rmtree(onnx_dir)
    print(f"\n=== Exporting {len(members)} ckpts → {onnx_dir} ===")
    export_all_members(onnx_dir, members)
    n_onnx = len(list(onnx_dir.glob("*.onnx")))
    print(f"Exported {n_onnx} ONNX files.")

    print(f"\n=== Uploading Kaggle dataset {SLUG} ===")
    upload_dataset(onnx_dir)

    print(f"\n=== Patching notebook ===")
    reset_notebook_to_anchor()
    patch_notebook_5fold()
    patch_metadata()

    print(f"\n=== Pushing kernel {KERNEL_SLUG} ===")
    v = push_kernel()
    print(f"\n✅ Pushed kernel version {v}.")
    print(f"\nNext: verify kernel loaded 10 ckpts including slots 30-34, then submit:")
    print(f"  KAGGLE_API_TOKEN=$(cat ~/.kaggle/access_token) uv run kaggle competitions submit \\")
    print(f"      -c birdclef-2026 -f submission.csv \\")
    print(f"      -k {KERNEL_SLUG} -v {v} \\")
    print(f"      -m 'v{v} exp183 Noisy Student iter 1 FULL 5-FOLD (replaces exp176 5-fold). Soft pseudo from 10-ckpt teacher (89% soft signal). Local v61 single-fold = -0.006; tests if 5-fold compounds.'")


if __name__ == "__main__":
    main()
