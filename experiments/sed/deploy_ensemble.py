"""deploy_ensemble — export multi-seed + multi-arch ensemble to a single ONNX dataset.

Combines all available exp175 seeds + exp177 (B1) folds into one ensemble of
~20 ONNX files. The notebook's SED loader picks them up via `sed_fold*.onnx`
glob pattern.

Naming: sed_fold00.onnx through sed_fold19.onnx (zero-padded so glob sorts
in declared order).

Usage:
  KAGGLE_API_TOKEN=KGAT_... uv run python -m experiments.sed.deploy_ensemble
"""
from __future__ import annotations
import argparse
import json
import shutil
import subprocess
import time
from dataclasses import replace
from pathlib import Path

from .config import exp175_tucker_actually, exp176_per_fold_ss, exp177_b1_backbone, exp178_softauc, exp183_noisy_student_iter1
from .export import export_one
from .notebook_state import NotebookState, apply_state
from ._common import (
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


SLUG = "ultimatumgame/bc2026-ensemble-v3-sed"
TOKEN = "ensemble_v3"  # Python-identifier-safe; was "ensemble-v3" which broke find_<token>_dir() codegen


def _fold_ready(cfg, fold: int) -> bool:
    ck = cfg.resolved_output_dir() / f"fold{fold}" / "best_ckpt.pt"
    hist = cfg.resolved_output_dir() / f"fold{fold}" / "history.json"
    if not (ck.exists() and hist.exists()):
        return False
    try:
        h = json.loads(hist.read_text())
        return h.get("history", [{}])[-1].get("epoch", 0) >= 25
    except Exception:
        return False


def collect_members() -> list[tuple[str, int, int]]:
    """Return list of (config_name, seed, fold) for available trained folds.

    Auto-discovers all trained checkpoints across:
    - exp175 (B0, multi-seed): seeds 42, 43, 44              → fold 00-14
    - exp176 (per-fold SS, B0): seed 42                       → fold 15-19
    - exp177 (B1 backbone, multi-arch): seed 42               → fold 20-24
    - exp178 (B0 SoftAUC, multi-loss): seed 42                → fold 25-29
    """
    members = []
    # exp175 B0 multi-seed
    for seed in (42, 43, 44):
        cfg = replace(exp175_tucker_actually(), seed=seed)
        for f in range(cfg.N_FOLDS):
            if _fold_ready(cfg, f):
                members.append(("exp175_tucker_actually", seed, f))
    # exp176 (per-fold SS) — different EVAL strategy, same B0 arch
    cfg = exp176_per_fold_ss()
    for f in range(cfg.N_FOLDS):
        if _fold_ready(cfg, f):
            members.append(("exp176_per_fold_ss", 42, f))
    # exp177 B1 backbone
    cfg = exp177_b1_backbone()
    for f in range(cfg.N_FOLDS):
        if _fold_ready(cfg, f):
            members.append(("exp177_b1_backbone", 42, f))
    # exp178 SoftAUC (B0, BCE→SoftAUC loss family) — recipe axis #3
    cfg = exp178_softauc()
    for f in range(cfg.N_FOLDS):
        if _fold_ready(cfg, f):
            members.append(("exp178_softauc", 42, f))
    # exp183 Noisy Student iter 1 (Sydorskyi 2025 2nd-place recipe with REAL soft pseudo)
    cfg = exp183_noisy_student_iter1()
    for f in range(cfg.N_FOLDS):
        if _fold_ready(cfg, f):
            members.append(("exp183_noisy_student_iter1", 42, f))
    return members


def export_all_members(out_dir: Path, members: list) -> None:
    """Export each (cfg, seed, fold) to sed_fold{NN}.onnx in out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_funcs = {
        "exp175_tucker_actually": exp175_tucker_actually,
        "exp176_per_fold_ss": exp176_per_fold_ss,
        "exp177_b1_backbone": exp177_b1_backbone,
        "exp178_softauc": exp178_softauc,
        "exp183_noisy_student_iter1": exp183_noisy_student_iter1,
    }
    for i, (cfg_name, seed, fold) in enumerate(members):
        cfg = replace(cfg_funcs[cfg_name](), seed=seed)
        ckpt = cfg.resolved_output_dir() / f"fold{fold}" / "best_ckpt.pt"
        out = out_dir / f"sed_fold{i:02d}.onnx"
        print(f"  [{i:02d}] {cfg_name} seed={seed} fold={fold} → {out.name}", flush=True)
        export_one(cfg, ckpt, out)


def patch_notebook() -> None:
    apply_state(
        NotebookState(
            sed_dataset=SLUG,
            sed_finder_dir_token=TOKEN,
            ulyanov_blend=False,
            konbu_head=False,
            sonotype_mirror=True,    # Path A — needless090 0.946 trick
            rare_suppression=True,   # Path G — needless090 0.946 trick
        ),
        NB_PATH,
    )


def patch_metadata() -> None:
    sources = [PERCH_META_DATASET, PERCH_ONNX_DATASET, SLUG]
    if META_PATH.exists():
        meta = json.loads(META_PATH.read_text())
    else:
        meta = {
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
    meta["dataset_sources"] = sources
    META_PATH.write_text(json.dumps(meta, indent=2))


def upload_dataset(onnx_dir: Path) -> None:
    md = onnx_dir / "dataset-metadata.json"
    md.write_text(json.dumps({
        "title": "bc2026-ensemble-v3-sed",
        "id": SLUG,
        "licenses": [{"name": "CC0-1.0"}],
    }, indent=2))

    print(f"[upload] kaggle datasets create -p {onnx_dir}", flush=True)
    proc = subprocess.run(
        ["uv", "run", "kaggle", "datasets", "create", "-p", str(onnx_dir), "-r", "zip"],
        env=kaggle_env(), capture_output=True, text=True,
    )
    out = proc.stdout + proc.stderr
    print(out[-400:], flush=True)
    # Kaggle error messages vary: "already exists", "already in use", "title ... is taken"
    out_lower = out.lower()
    if any(s in out_lower for s in ("already exists", "already in use", "is taken", "is in use")):
        # Fall back to version
        print("  dataset exists, creating new version", flush=True)
        proc2 = subprocess.run(
            ["uv", "run", "kaggle", "datasets", "version", "-p", str(onnx_dir),
             "-m", f"ensemble v3 ({time.strftime('%Y-%m-%d %H:%M')})", "-r", "zip"],
            env=kaggle_env(), capture_output=True, text=True,
        )
        print((proc2.stdout + proc2.stderr)[-400:], flush=True)
    print("  waiting 90s for Kaggle to index dataset...", flush=True)
    time.sleep(90)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--skip-upload", action="store_true")
    parser.add_argument("--skip-push", action="store_true")
    args = parser.parse_args()

    members = collect_members()
    print(f"=== Found {len(members)} trained ckpts ===", flush=True)
    for m in members:
        print(f"  {m}", flush=True)
    if not members:
        print("no members; abort", flush=True)
        return
    if len(members) < 5:
        print(f"warning: only {len(members)} ckpts; ensemble may be weak", flush=True)

    onnx_dir = Path("/data/birdclef2026/experiments/_data_pipelines/ensemble_v3/onnx")
    if not args.skip_export:
        if onnx_dir.exists():
            shutil.rmtree(onnx_dir)
        export_all_members(onnx_dir, members)

    print(f"\n=== ONNX dir: {onnx_dir} ({len(list(onnx_dir.glob('*.onnx')))} files) ===", flush=True)

    if not args.skip_upload:
        upload_dataset(onnx_dir)

    if not args.skip_push:
        print(f"\n=== reset notebook + push ===", flush=True)
        reset_notebook_to_anchor()
        patch_notebook()
        patch_metadata()
        v = push_kernel()
        print(f"  → version {v}", flush=True)


if __name__ == "__main__":
    main()
