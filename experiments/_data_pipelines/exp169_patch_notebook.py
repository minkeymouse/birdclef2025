#!/usr/bin/env python3
"""Patch the mattia-fork notebook so Stream B uses ONLY our exp169 5-fold.

The user's directive (2026-05-04): the leaderboard submission must use
weights we trained from scratch, not Tucker's published checkpoints. So
this patch *replaces* Tucker's 5 fold ONNX sessions with our 5 exp169
sessions. The two share the same Tucker recipe (Perch backbone
distillation + stop-gradient, mel-256, 0.5*BCE_clip + 0.5*BCE_frame_max
+ 1.0*MSE), but the exp169 sessions are loaded from the
ultimatumgame/exp169-distilled-sed dataset.

The Tucker dataset is left attached so its labels.csv etc. don't break
any auxiliary code, but no Tucker weight is consumed during inference.

Patch is idempotent.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

NB_PATH = Path("/data/birdclef2026/notebooks/birdclef-2026-mattia-fork/notebook.ipynb")
KERNEL_META = Path("/data/birdclef2026/notebooks/birdclef-2026-mattia-fork/kernel-metadata.json")
EXP169_DATASET_ID = "ultimatumgame/exp169-distilled-sed"


# Anchor on the original sed_sessions construction.
OLD_LOAD = """sed_dir = find_sed_dir()

sed_fold_paths = sorted(
    sed_dir.glob("sed_fold*.onnx"),
    key=lambda p: int(re.search(r"sed_fold(\\d+)", p.name).group(1))
)

sed_sessions = [make_sed_session(p) for p in sed_fold_paths]

print(f"SED dir: {sed_dir}")
print(f"SED folds loaded: {[p.name for p in sed_fold_paths]}")"""


NEW_LOAD = """# ===== exp169 patch: Stream B = ONLY our exp169 5-fold (no Tucker weights) =====
def find_exp169_dir():
    hits = sorted(Path("/kaggle/input").rglob("sed_fold0.onnx"))
    candidates = [p.parent for p in hits
                  if "exp169" in str(p) or "ultimatumgame" in str(p)]
    return candidates[0] if candidates else None

exp169_dir = find_exp169_dir()
if exp169_dir is None:
    raise FileNotFoundError(
        "exp169 SED ONNX directory not found. Attach "
        "ultimatumgame/exp169-distilled-sed to this notebook."
    )

sed_fold_paths = sorted(
    exp169_dir.glob("sed_fold*.onnx"),
    key=lambda p: int(re.search(r"sed_fold(\\d+)", p.name).group(1))
)
sed_dir = exp169_dir
sed_sessions = [make_sed_session(p) for p in sed_fold_paths]

print(f"SED dir: {sed_dir}  (exp169, our distilled SED)")
print(f"SED folds loaded: {[p.name for p in sed_fold_paths]}")
print(f"Tucker weights are NOT used (per user directive 2026-05-04).")
# ===== end exp169 patch ====="""


def patch_notebook(nb_path: Path):
    nb = json.loads(nb_path.read_text())
    found = False
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if "exp169 patch" in src:
            print("notebook already patched (exp169 marker present); skipping")
            return False
        if OLD_LOAD in src:
            cell["source"] = [src.replace(OLD_LOAD, NEW_LOAD)]
            found = True
            break
    if not found:
        raise RuntimeError("OLD_LOAD anchor missing")
    nb_path.write_text(json.dumps(nb))
    print(f"patched {nb_path.name} (exp169 ONLY, no Tucker weights)")
    return True


def patch_kernel_metadata(meta_path: Path):
    meta = json.loads(meta_path.read_text())
    sources = meta.setdefault("dataset_sources", [])
    if EXP169_DATASET_ID in sources:
        print("kernel metadata already lists exp169; skipping")
        return False
    sources.append(EXP169_DATASET_ID)
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"added {EXP169_DATASET_ID} to kernel sources")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--unpatch", action="store_true")
    args = parser.parse_args()
    if args.unpatch:
        nb = json.loads(NB_PATH.read_text())
        any_change = False
        for cell in nb["cells"]:
            if cell.get("cell_type") != "code":
                continue
            src = "".join(cell.get("source", []))
            if "exp169 patch" in src and NEW_LOAD in src:
                cell["source"] = [src.replace(NEW_LOAD, OLD_LOAD)]
                any_change = True
        if any_change:
            NB_PATH.write_text(json.dumps(nb))
            print("notebook unpatched")
        meta = json.loads(KERNEL_META.read_text())
        if EXP169_DATASET_ID in meta.get("dataset_sources", []):
            meta["dataset_sources"].remove(EXP169_DATASET_ID)
            KERNEL_META.write_text(json.dumps(meta, indent=2))
            print("removed exp169 from kernel sources")
        return
    patch_notebook(NB_PATH)
    patch_kernel_metadata(KERNEL_META)


if __name__ == "__main__":
    main()
