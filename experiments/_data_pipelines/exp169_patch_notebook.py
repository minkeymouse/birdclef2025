#!/usr/bin/env python3
"""Patch the mattia-fork notebook for a Tucker / exp169 weighted Stream B.

This replaces the simple-average patch with an explicit weighted blend:
  p_streamB = (1 - W_EXP169) * p_tucker + W_EXP169 * p_exp169
where p_tucker is the mean over the 5 Tucker folds and p_exp169 is the
mean over the 5 exp169 folds. Default W_EXP169 = 0.25.

Rationale: exp169 5-fold scored val_SS 0.84 (held out) versus Tucker 0.94
on its own training data. exp169 brings Pearson 0.73 decorrelation but is
likely weaker in absolute quality, so we cap its blend weight at 0.25.

Patch is idempotent.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

NB_PATH = Path("/data/birdclef2026/notebooks/birdclef-2026-mattia-fork/notebook.ipynb")
KERNEL_META = Path("/data/birdclef2026/notebooks/birdclef-2026-mattia-fork/kernel-metadata.json")
EXP169_DATASET_ID = "ultimatumgame/exp169-distilled-sed"

W_EXP169 = 0.25  # exp169 weight inside Stream B (rest goes to Tucker)

# We anchor on the section that loads the 5 Tucker folds and the section
# that loops over them. We rewrite both into a Tucker+exp169 split layout.
OLD_LOAD = """sed_dir = find_sed_dir()

sed_fold_paths = sorted(
    sed_dir.glob("sed_fold*.onnx"),
    key=lambda p: int(re.search(r"sed_fold(\\d+)", p.name).group(1))
)

sed_sessions = [make_sed_session(p) for p in sed_fold_paths]

print(f"SED dir: {sed_dir}")
print(f"SED folds loaded: {[p.name for p in sed_fold_paths]}")"""

NEW_LOAD = """# ===== exp169 patch: split Tucker / exp169 sessions =====
sed_dir = find_sed_dir()

sed_fold_paths = sorted(
    sed_dir.glob("sed_fold*.onnx"),
    key=lambda p: int(re.search(r"sed_fold(\\d+)", p.name).group(1))
)

# exp169 5-fold (our Tucker-recipe distilled SED)
def find_exp169_dir():
    hits = sorted(Path("/kaggle/input").rglob("sed_fold0.onnx"))
    candidates = [p.parent for p in hits
                  if "exp169" in str(p) or "ultimatumgame" in str(p)]
    return candidates[0] if candidates else None

exp169_dir = find_exp169_dir()
exp169_fold_paths = []
if exp169_dir is not None and exp169_dir != sed_dir:
    exp169_fold_paths = sorted(
        exp169_dir.glob("sed_fold*.onnx"),
        key=lambda p: int(re.search(r"sed_fold(\\d+)", p.name).group(1))
    )

tucker_sessions = [make_sed_session(p) for p in sed_fold_paths]
exp169_sessions = [make_sed_session(p) for p in exp169_fold_paths]

# Backwards-compat: keep `sed_sessions` symbol alive in case any later cell uses it
sed_sessions = list(tucker_sessions) + list(exp169_sessions)

W_EXP169 = """ + repr(W_EXP169) + """  # exp169 share inside Stream B

print(f"SED dir (Tucker): {sed_dir}")
print(f"  Tucker folds: {[p.name for p in sed_fold_paths]}")
if exp169_sessions:
    print(f"SED dir (exp169): {exp169_dir}")
    print(f"  exp169 folds: {[p.name for p in exp169_fold_paths]}")
    print(f"  Stream B blend: {1.0 - W_EXP169} Tucker + {W_EXP169} exp169")
else:
    print("  exp169: NOT FOUND -> Tucker-only Stream B")
    W_EXP169 = 0.0
# ===== end exp169 patch (load) ====="""


OLD_PREDICT = """    p_sum = np.zeros((len(chunks), N_CLASSES), dtype=np.float32)

    for sess in sed_sessions:
        outs = sess.run(None, {sess.get_inputs()[0].name: mel})

        clip_logits = outs[0]             # (12, 234)
        frame_max   = outs[1].max(axis=1) # (12, 234)

        p_sum += 0.5 * sigmoid_sed(clip_logits) + 0.5 * sigmoid_sed(frame_max)

    p_mean = p_sum / len(sed_sessions)"""

NEW_PREDICT = """    # ===== exp169 patch: weighted Tucker / exp169 averaging =====
    def avg_sessions(sessions):
        if not sessions:
            return None
        p = np.zeros((len(chunks), N_CLASSES), dtype=np.float32)
        for sess in sessions:
            outs = sess.run(None, {sess.get_inputs()[0].name: mel})
            cl = outs[0]
            fm = outs[1].max(axis=1)
            p += 0.5 * sigmoid_sed(cl) + 0.5 * sigmoid_sed(fm)
        return p / len(sessions)

    p_tucker = avg_sessions(tucker_sessions)
    p_exp169 = avg_sessions(exp169_sessions)
    if p_exp169 is not None:
        p_mean = (1.0 - W_EXP169) * p_tucker + W_EXP169 * p_exp169
    else:
        p_mean = p_tucker
    # ===== end exp169 patch (predict) ====="""


def patch_notebook(nb_path: Path):
    nb = json.loads(nb_path.read_text())
    found_load = False
    found_predict = False
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if "exp169 patch" in src:
            print("notebook already patched (exp169 marker present); skipping")
            return False
        if OLD_LOAD in src:
            src = src.replace(OLD_LOAD, NEW_LOAD)
            found_load = True
        if OLD_PREDICT in src:
            src = src.replace(OLD_PREDICT, NEW_PREDICT)
            found_predict = True
        if found_load or found_predict:
            cell["source"] = [src]
        if found_load and found_predict:
            break
    if not (found_load and found_predict):
        raise RuntimeError(f"anchors missing: load={found_load} predict={found_predict}")
    nb_path.write_text(json.dumps(nb))
    print(f"patched {nb_path.name} (W_EXP169={W_EXP169})")
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
            if "exp169 patch" not in src:
                continue
            new_src = src
            if NEW_LOAD in new_src:
                new_src = new_src.replace(NEW_LOAD, OLD_LOAD)
            if NEW_PREDICT in new_src:
                new_src = new_src.replace(NEW_PREDICT, OLD_PREDICT)
            cell["source"] = [new_src]
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
