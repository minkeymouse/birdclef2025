"""deploy_exp175 — full Kaggle deploy pipeline for the exp175 5-fold SED.

End-to-end (idempotent):
  1. Reset notebook.ipynb to v5 anchor (commit c4df217)
  2. Apply NotebookState (SED dataset = bc2026-exp175-sed, finder token = exp175)
  3. Patch kernel-metadata.json (drop konbu/exp174a, add bc2026-exp175-sed)
  4. Export 5 ONNX folds from exp175 ckpts
  5. Upload Kaggle dataset (creates new version, or creates dataset on first run)
  6. Push notebook (creates new Kaggle version)

Usage:
  KAGGLE_API_TOKEN=KGAT_... uv run python -m experiments.sed.deploy_exp175
"""
from __future__ import annotations
import json
import subprocess
from pathlib import Path
from typing import Any

from .config import exp175_tucker_actually
from .export import export_all_folds
from .notebook_state import NotebookState, apply_state
from ._common import (
    EXP175_SED_DATASET,
    KERNEL_SLUG,
    META_PATH,
    NB_DIR,
    NB_PATH,
    PERCH_META_DATASET,
    PERCH_ONNX_DATASET,
    kaggle_env,
    push_kernel,
    reset_notebook_to_anchor,
)

SLUG = EXP175_SED_DATASET
TOKEN = "exp175"


def patch_notebook_for_exp175() -> None:
    print(f"[2] patch SED loader: exp169 → {TOKEN}", flush=True)
    apply_state(
        NotebookState(
            sed_dataset=SLUG,
            sed_finder_dir_token=TOKEN,
            ulyanov_blend=False,
            konbu_head=False,
        ),
        NB_PATH,
    )


def patch_metadata() -> None:
    print(f"[3] update kernel-metadata.json", flush=True)
    sources = [PERCH_META_DATASET, PERCH_ONNX_DATASET, SLUG]
    if META_PATH.exists():
        meta = json.loads(META_PATH.read_text())
    else:
        meta = _default_kernel_metadata()
    meta["dataset_sources"] = sources
    META_PATH.write_text(json.dumps(meta, indent=2))


def _default_kernel_metadata() -> dict[str, Any]:
    return {
        "id": KERNEL_SLUG,
        "title": KERNEL_SLUG.split("/", 1)[-1],
        "code_file": "notebook.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_internet": False,
        "competition_sources": ["birdclef-2026"],
        "kernel_sources": [
            "ashok205/tf-wheels",
            "vyankteshdwivedi/birdclef-2026-onnx-perch-sequence-modeling",
        ],
        "model_sources": [
            "google/bird-vocalization-classifier/TensorFlow2/perch_v2_cpu/1",
        ],
    }


def export_and_upload(cfg) -> None:
    print(f"[4] export ONNX (5 folds)", flush=True)
    onnx_dir = export_all_folds(cfg)

    print(f"[5] upload Kaggle dataset {SLUG}", flush=True)
    md = onnx_dir / "dataset-metadata.json"
    md.write_text(json.dumps({
        "title": "bc2026-exp175-sed",
        "id": SLUG,
        "licenses": [{"name": "CC0-1.0"}],
    }, indent=2))

    proc = subprocess.run(
        ["uv", "run", "kaggle", "datasets", "version", "-p", str(onnx_dir),
         "-m", "exp175 Tucker-actually 5-fold (drop_rate fix + xavier init)",
         "-r", "zip"],
        env=kaggle_env(), capture_output=True, text=True,
    )
    out = proc.stdout + proc.stderr
    print(out[-400:], flush=True)
    if proc.returncode != 0 and ("404" in out or "not exist" in out.lower()):
        print("  dataset doesn't exist, creating new", flush=True)
        proc2 = subprocess.run(
            ["uv", "run", "kaggle", "datasets", "create", "-p", str(onnx_dir), "-r", "zip"],
            env=kaggle_env(), capture_output=True, text=True,
        )
        print((proc2.stdout + proc2.stderr)[-400:], flush=True)


def main():
    cfg = exp175_tucker_actually()

    for f in range(cfg.N_FOLDS):
        ck = cfg.resolved_output_dir() / f"fold{f}" / "best_ckpt.pt"
        if not ck.exists():
            raise FileNotFoundError(f"missing {ck}; training not complete")
    print("all 5 fold ckpts present", flush=True)

    print(f"[1] reset notebook to anchor", flush=True)
    reset_notebook_to_anchor()
    patch_notebook_for_exp175()
    patch_metadata()
    export_and_upload(cfg)
    print(f"[6] push kernel", flush=True)
    v = push_kernel()
    if v is None:
        print(f"⚠️ could not parse pushed version", flush=True)
    else:
        print(f"  → version {v}", flush=True)
    print(f"done. monitor https://kaggle.com/code/{KERNEL_SLUG}", flush=True)


if __name__ == "__main__":
    main()
