"""sed/deploy.py — Kaggle dataset + notebook push, idempotent.

Replaces exp17{0..5}_deploy.py + exp17X_export_onnx.py + various patch scripts.

Usage:
  uv run python -m experiments.sed.deploy --config exp175
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
from pathlib import Path

from . import config as cfg_mod
from .config import SedConfig
from .export import export_all_folds
from .notebook_state import NotebookState, apply_state

KAGGLE_TOKEN_ENV = "KAGGLE_API_TOKEN"
NB_PATH = Path("/data/birdclef2026/notebooks/birdclef-2026-mattia-fork/notebook.ipynb")
META_PATH = Path("/data/birdclef2026/notebooks/birdclef-2026-mattia-fork/kernel-metadata.json")


def upload_dataset(onnx_dir: Path, slug: str, title: str) -> None:
    md = onnx_dir / "dataset-metadata.json"
    md.write_text(json.dumps({
        "title": title,
        "id": slug,
        "licenses": [{"name": "CC0-1.0"}],
    }, indent=2))
    env = os.environ.copy()
    print(f"uploading {slug}...", flush=True)
    proc = subprocess.run(
        ["uv", "run", "kaggle", "datasets", "version", "-p", str(onnx_dir),
         "-m", f"sed/deploy.py auto-publish {slug}", "-r", "zip"],
        env=env, capture_output=True, text=True,
    )
    out = proc.stdout + proc.stderr
    if proc.returncode != 0 and ("404" in out or "not exist" in out.lower()):
        print(f"  dataset doesn't exist, creating new", flush=True)
        proc = subprocess.run(
            ["uv", "run", "kaggle", "datasets", "create", "-p", str(onnx_dir), "-r", "zip"],
            env=env, capture_output=True, text=True,
        )
        print(proc.stdout[-300:], flush=True)
    else:
        print(out[-300:], flush=True)


def patch_metadata(slug_keep: list[str]):
    """Idempotent: set notebook metadata dataset_sources to exactly slug_keep + standard deps."""
    standard = [
        "jaejohn/perch-meta",
        "rishikeshjani/perch-onnx-for-birdclef-2026",
    ]
    sources = standard + slug_keep
    meta = json.loads(META_PATH.read_text())
    meta["dataset_sources"] = sources
    META_PATH.write_text(json.dumps(meta, indent=2))
    print(f"metadata: {sources}", flush=True)


def patch_notebook(token: str, slug: str):
    state = NotebookState(
        sed_dataset=slug,
        sed_finder_dir_token=token,
        ulyanov_blend=False,
        konbu_head=False,
    )
    apply_state(state, NB_PATH)
    print(f"notebook patched: token={token} slug={slug}", flush=True)


def push_kernel():
    env = os.environ.copy()
    proc = subprocess.run(
        ["uv", "run", "kaggle", "kernels", "push"],
        cwd=NB_PATH.parent, env=env, capture_output=True, text=True,
    )
    print((proc.stdout + proc.stderr)[-400:], flush=True)


def deploy_full(cfg: SedConfig, slug: str, token: str) -> None:
    onnx_dir = export_all_folds(cfg)
    title = f"{cfg.name} SED 5-fold ({slug})"
    upload_dataset(onnx_dir, slug, title)
    patch_metadata([slug])
    patch_notebook(token, slug)
    push_kernel()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="config name in sed.config")
    parser.add_argument("--slug", required=True, help="ultimatumgame/<dataset_slug>")
    parser.add_argument("--token", required=True, help="find_<token>_dir token")
    args = parser.parse_args()

    cfg_fn = getattr(cfg_mod, args.config)
    cfg = cfg_fn()
    deploy_full(cfg, args.slug, args.token)


if __name__ == "__main__":
    main()
