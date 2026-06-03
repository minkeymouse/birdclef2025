"""Slot 5 retune driver.

After applying base ensemble_v3 patch (with optional fold filter), modifies
notebook's mattia_blend(...) call to set sed_w to target value, then pushes
+ submits.

Usage:
  KAGGLE_API_TOKEN=... uv run python -m experiments.sed.slot5_retune <sed_w> [<filter>]

filter is one of: mega25 / multiseed15 / multiarch10 / multirecipe10 / "0,1,2,...,19"
default = mega25 (None = all 25 ckpts).
"""
from __future__ import annotations
import json
import os
import re
import sys
import time

from .notebook_state import NotebookState, apply_state
from ._common import (
    NB_PATH, META_PATH,
    push_kernel, reset_notebook_to_anchor, try_submit,
    PERCH_META_DATASET, PERCH_ONNX_DATASET,
)

ENSEMBLE_SLUG = "ultimatumgame/bc2026-ensemble-v3-sed"
TOKEN = "ensemble_v3"

PRESETS = {
    "mega25": None,
    "multiseed15": tuple(range(15)),
    "multiarch10": tuple(list(range(5)) + list(range(20, 25))),
    "multirecipe10": tuple(list(range(5)) + list(range(15, 20))),
    "B0only20": tuple(range(20)),
}


def parse_filter(spec: str) -> tuple | None:
    if spec in PRESETS:
        return PRESETS[spec]
    if "," in spec:
        return tuple(int(x.strip()) for x in spec.split(",") if x.strip())
    raise ValueError(f"unknown filter spec: {spec}")


def _patch_call(args: str, target_w: float) -> str:
    if "sed_w=" in args:
        new_args = re.sub(r"sed_w\s*=\s*[\d.]+", f"sed_w={target_w}", args)
    else:
        new_args = args + f", sed_w={target_w}"
    return f"mattia_blend({new_args})"


def patch_sed_w(target_w: float):
    """Find SED_W = ... or mattia_blend(...) in notebook, override sed weight.

    Mattia raw notebook uses module-level `SED_W = 0.40` (anchor default).
    Some refactored copies use `DEFAULT_SED_W = 0.30` in helper code.
    Some cells call `mattia_blend(..., sed_w=...)` explicitly.
    Patch all three forms.
    """
    nb = json.loads(NB_PATH.read_text())
    n_modified = 0
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if "mattia_blend(" not in src and "DEFAULT_SED_W" not in src and "SED_W" not in src:
            continue
        new_src = re.sub(
            r"mattia_blend\(([^)]*)\)",
            lambda m: _patch_call(m.group(1), target_w),
            src,
        )
        new_src = re.sub(r"DEFAULT_SED_W\s*=\s*0?\.\d+", f"DEFAULT_SED_W = {target_w}", new_src)
        new_src = re.sub(r"^SED_W\s*=\s*0?\.\d+", f"SED_W = {target_w}", new_src, flags=re.M)
        if new_src != src:
            cell["source"] = [new_src]
            n_modified += 1
    if n_modified == 0:
        raise RuntimeError("no cell with mattia_blend / DEFAULT_SED_W / SED_W found")
    NB_PATH.write_text(json.dumps(nb, indent=1))
    print(f"patched sed_w={target_w} in {n_modified} cell(s)", flush=True)


def patch_metadata():
    sources = [PERCH_META_DATASET, PERCH_ONNX_DATASET, ENSEMBLE_SLUG]
    meta = json.loads(META_PATH.read_text()) if META_PATH.exists() else {
        "id": "ultimatumgame/birdclef-2026-mattia-fork",
        "title": "birdclef-2026-mattia-fork",
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


def main():
    if "KAGGLE_API_TOKEN" not in os.environ:
        os.environ["KAGGLE_API_TOKEN"] = "KGAT_c9d844b39256a18b7e777072c92a9ac1"
    if len(sys.argv) < 2:
        print(f"usage: python -m experiments.sed.slot5_retune <sed_w> [<filter_preset>]")
        print(f"  filter presets: {list(PRESETS.keys())}")
        sys.exit(1)
    sed_w = float(sys.argv[1])
    filter_spec = sys.argv[2] if len(sys.argv) > 2 else "mega25"
    fold_filter = parse_filter(filter_spec)
    n = len(fold_filter) if fold_filter is not None else 25

    print(f"\n=== SLOT 5 retune ===", flush=True)
    print(f"  filter      : {filter_spec} ({n} ckpts)", flush=True)
    print(f"  sed_w       : 0.30 default → {sed_w}", flush=True)
    print(f"  fold_indices: {fold_filter}", flush=True)

    reset_notebook_to_anchor()
    apply_state(NotebookState(
        sed_dataset=ENSEMBLE_SLUG,
        sed_finder_dir_token=TOKEN,
        ulyanov_blend=False,
        konbu_head=False,
        sonotype_mirror=False,
        rare_suppression=False,
        sed_fold_filter=fold_filter,
    ), NB_PATH)
    patch_sed_w(sed_w)
    patch_metadata()
    v = push_kernel()
    if v is None:
        raise RuntimeError("push failed")
    print(f"=== SLOT 5 ({filter_spec}, sed_w={sed_w}) pushed v{v} ===", flush=True)

    msg = (f"S5 retune: filter={filter_spec} ({n} ckpts) + Mattia sed_w={sed_w} "
           f"(default 0.30). Tests if stronger ensemble supports higher SED weight.")
    fails = 0
    while fails < 90:
        ok, out = try_submit(v, msg)
        ts = time.strftime("%H:%M:%S")
        if ok:
            print(f"[{ts}] SLOT 5 v{v} SUBMITTED", flush=True)
            return
        if "401" in out:
            print(f"[{ts}] AUTH FAIL", flush=True)
            return
        fails += 1
        print(f"[{ts}] v{v} not ready (#{fails})  {out[:80]}", flush=True)
        time.sleep(120)


if __name__ == "__main__":
    main()
