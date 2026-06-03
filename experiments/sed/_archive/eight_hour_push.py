"""8-hour LB push driver — 5 orthogonal slots on bc2026-ensemble-v3-sed.

Each slot tests one axis of variance reduction:
  S1 mega25       : all 25 (multi-seed + multi-recipe + multi-arch)
  S2 multiseed15  : exp175 seeds 42/43/44 only (seed axis)
  S3 multiarch10  : exp175(seed42) + exp177-B1 (arch axis)
  S4 multirecipe10: exp175(seed42) + exp176 per-fold-SS (recipe axis)
  S5 retune       : mega + Mattia rank-blend dose tweak (decided after S1-S4)

Usage:
  KAGGLE_API_TOKEN=... uv run python -m experiments.sed.eight_hour_push push <slot_id>
  KAGGLE_API_TOKEN=... uv run python -m experiments.sed.eight_hour_push submit <slot_id> <version>
"""
from __future__ import annotations
import json
import os
import sys
import time

from .notebook_state import NotebookState, apply_state
from ._common import (
    NB_PATH, META_PATH, ROOT,
    push_kernel, reset_notebook_to_anchor, try_submit,
    PERCH_META_DATASET, PERCH_ONNX_DATASET,
)

ENSEMBLE_SLUG = "ultimatumgame/bc2026-ensemble-v3-sed"
TOKEN = "ensemble_v3"

# fold00-04: exp175 seed=42 (B0, BCE Tucker-canonical)
# fold05-09: exp175 seed=43 (B0)
# fold10-14: exp175 seed=44 (B0)
# fold15-19: exp176 seed=42 (B0, per-fold SS)
# fold20-24: exp177 seed=42 (B1 backbone)
# fold25-29: exp178 seed=42 (B0, SoftAUC loss family)  ← added 2026-05-15

SLOT_DEFS = {
    1: {
        "filter": None,
        "name": "mega25",
        "msg": ("S1 mega 25 ckpts: exp175 seed=42/43/44 + exp176 + exp177-B1. "
                "Mattia 0.941 blend, no MirrorRare/konbu. "
                "Tests union ensemble vs anchor 0.938. Wall time tight ~94 min."),
    },
    2: {
        "filter": tuple(range(15)),
        "name": "multiseed15",
        "msg": ("S2 multi-seed 15: exp175 seeds 42/43/44 (B0 same recipe). "
                "Tests seed averaging contribution. Wall time ~72 min."),
    },
    3: {
        "filter": tuple(list(range(5)) + list(range(20, 25))),
        "name": "multiarch10",
        "msg": ("S3 multi-arch 10: exp175 seed=42 (B0) + exp177 (B1 backbone). "
                "Tests backbone arch diversity. Wall time ~62 min."),
    },
    4: {
        "filter": tuple(list(range(5)) + list(range(15, 20))),
        "name": "multirecipe10",
        "msg": ("S4 multi-recipe 10: exp175 seed=42 + exp176 per-fold-SS (B0 different recipes). "
                "Tests recipe diversity within B0. Wall time ~62 min."),
    },
    5: {
        # exp175 seed=42 (5 B0) + exp176 (5 B0 per-fold-SS) + exp177 (5 B1 backbone)
        # = builds on S4 (multi-recipe 10 = 0.941) by adding multi-arch B1.
        "filter": tuple(list(range(5)) + list(range(15, 25))),
        "name": "multirecipe_multiarch15",
        "msg": ("S5 multi-recipe + multi-arch 15: exp175 seed=42 (5 B0 Tucker) + "
                "exp176 (5 B0 per-fold-SS) + exp177 (5 B1 backbone). Builds on "
                "S4 = 0.941 (multi-recipe alone) by adding B1 arch axis. Tests "
                "synergy of recipe + arch diversity. Wall time ~72 min."),
    },
    6: {
        # exp175 seeds 42/43/44 (15) + exp176 (5) = 20 ckpts B0 only
        # Multi-recipe + multi-seed combined (no B1 — B1 disrupted S5 rank cdf).
        "filter": tuple(range(20)),
        "name": "multirecipe_multiseed_20",
        "msg": ("L1 multi-recipe + multi-seed 20: exp175 seeds 42/43/44 (15 B0) "
                "+ exp176 seed=42 (5 B0 per-fold-SS). Builds on v45 S4=0.941 by "
                "adding seed averaging across multi-recipe pair. Hypothesis: "
                "multi-seed within multi-recipe (S2 was 0 within single recipe) "
                "may transfer here. Wall time ~83 min (tight)."),
    },
    7: {
        # exp175 seed=42 (5 B0 BCE Tucker) + exp176 (5 B0 per-fold-SS) + exp178 (5 B0 SoftAUC)
        # Recipe-triple ensemble: 3rd recipe axis = loss family (BCE → SoftAUC).
        "filter": tuple(list(range(5)) + list(range(15, 20)) + list(range(25, 30))),
        "name": "multirecipe_triple_15",
        "msg": ("L2 multi-recipe TRIPLE 15: exp175 seed=42 (5 B0 BCE Tucker) + "
                "exp176 seed=42 (5 B0 per-fold-SS) + exp178 seed=42 (5 B0 SoftAUC, "
                "Babych 2025 1st-place loss). 3rd recipe axis = loss family. "
                "Builds on v45 S4=0.941. Tests whether v45's adapter-axis "
                "diversity mechanism (cos=0.55 distill_head) generalizes from "
                "ckpt-selection-objective axis to loss-family axis. Wall time ~72 min."),
    },
}


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


def push_slot(slot_id: int) -> int:
    sd = SLOT_DEFS[slot_id]
    print(f"\n=== SLOT {slot_id}: {sd['name']} (filter={sd['filter']}) ===", flush=True)
    reset_notebook_to_anchor()
    print(f"[apply] token={TOKEN}, filter={sd['filter']}", flush=True)
    state = NotebookState(
        sed_dataset=ENSEMBLE_SLUG,
        sed_finder_dir_token=TOKEN,
        ulyanov_blend=False,
        konbu_head=False,
        sonotype_mirror=False,
        rare_suppression=False,
        sed_fold_filter=sd["filter"],
    )
    apply_state(state, NB_PATH)
    patch_metadata()
    print("[push] kernel push (Kaggle dev run will start)", flush=True)
    v = push_kernel()
    if v is None:
        raise RuntimeError("push failed")
    print(f"=== SLOT {slot_id} pushed v{v} ===", flush=True)
    return v


def submit_loop(slot_id: int, version: int, max_polls: int = 90):
    """Poll-and-submit. Returns True if submitted."""
    sd = SLOT_DEFS[slot_id]
    msg = sd["msg"]
    fails = 0
    while fails < max_polls:
        ok, out = try_submit(version, msg)
        ts = time.strftime("%H:%M:%S")
        if ok:
            print(f"[{ts}] SLOT {slot_id} v{version} SUBMITTED", flush=True)
            return True
        fails += 1
        if "401" in out:
            print(f"[{ts}] AUTH FAIL: KAGGLE_API_TOKEN missing/expired", flush=True)
            return False
        print(f"[{ts}] v{version} not ready (#{fails})  {out[:80]}", flush=True)
        time.sleep(120)
    print(f"giving up on v{version}", flush=True)
    return False


def main():
    if "KAGGLE_API_TOKEN" not in os.environ:
        os.environ["KAGGLE_API_TOKEN"] = "KGAT_099e311ecb741fe6f3a12493e5d861a5"
    if len(sys.argv) < 3:
        print("usage: python -m experiments.sed.eight_hour_push <push|submit|both> <slot> [<version>]")
        sys.exit(1)
    cmd, slot_id = sys.argv[1], int(sys.argv[2])
    if cmd in ("push", "both"):
        v = push_slot(slot_id)
        print(f"OK pushed slot{slot_id} as v{v}", flush=True)
        if cmd == "both":
            submit_loop(slot_id, v)
    elif cmd == "submit":
        if len(sys.argv) < 4:
            print("submit needs version")
            sys.exit(1)
        version = int(sys.argv[3])
        submit_loop(slot_id, version)
    else:
        print(f"unknown command {cmd}")
        sys.exit(1)


if __name__ == "__main__":
    main()
