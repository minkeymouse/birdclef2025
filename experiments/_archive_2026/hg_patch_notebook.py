"""Patch phase4 -> eos8-hgblend: BLEND HGNet (nischaydnk distilled ONNX, same mel interface)
into the Tucker SED stream at W_HG, WITHOUT replacing Tucker. Single-variable diversity test.
Differs from the failed eos8-hgnet (0.937) which let HGNet REPLACE Tucker as the sole SED."""
import json, shutil
from pathlib import Path

SRC = Path("notebooks/birdclef-2026-eos8-phase4")
DST = Path("notebooks/birdclef-2026-eos8-hgblend")
DST.mkdir(exist_ok=True)
shutil.copy(SRC/"notebook.ipynb", DST/"notebook.ipynb")
nb = json.load(open(DST/"notebook.ipynb"))

def as_str(s): return "".join(s) if isinstance(s, list) else s
src = as_str(nb["cells"][11]["source"])

# --- edit 1: find_sed_dir must EXCLUDE hgnet so Tucker stays the main SED ---
old1 = 'hits = sorted(Path("/kaggle/input").rglob("sed_fold0.onnx"))'
new1 = ('hits = sorted(p for p in Path("/kaggle/input").rglob("sed_fold0.onnx") '
        'if "hgnet" not in str(p).lower() and "distilled-onnx" not in str(p).lower())')
assert src.count(old1) == 1, f"find_sed_dir marker count={src.count(old1)}"
src = src.replace(old1, new1)

def insert_after(text, marker, block_lines):
    lines = text.splitlines(keepends=True)
    for i, ln in enumerate(lines):
        if marker in ln:
            indent = ln[:len(ln) - len(ln.lstrip())]
            blk = "".join(indent + b + "\n" for b in block_lines)
            return "".join(lines[:i+1] + [blk] + lines[i+1:]), True
    return text, False

# --- edit 2: load HGNet session (separate) after Tucker sed_sessions ---
hg_load = [
    "HG_SESS = None; HG_IN = None; W_HG = 0.20",
    "try:",
    '    _hg = sorted(p for p in Path("/kaggle/input").rglob("sed_fold0.onnx") if "hgnet" in str(p).lower() or "distilled-onnx" in str(p).lower())',
    "    if _hg:",
    "        HG_SESS = make_sed_session(_hg[0]); HG_IN = HG_SESS.get_inputs()[0].name",
    '        print(f"[HGBLEND] HGNet SED loaded: {_hg[0]} (W_HG={W_HG}); Tucker folds kept = {len(sed_sessions)}")',
    "    else:",
    '        print("[HGBLEND] no HGNet onnx found -> Tucker SED only")',
    "except Exception as _e:",
    '    print("[HGBLEND] HGNet load failed:", _e); HG_SESS = None',
]
src, ok2 = insert_after(src, "sed_sessions = [make_sed_session(p) for p in sed_fold_paths]", hg_load)
assert ok2, "sed_sessions marker not found"

# --- edit 3: blend HGNet into p_mean (after Tucker ensemble mean) ---
hg_blend = [
    "if HG_SESS is not None:",
    "    try:",
    "        _ho = HG_SESS.run(None, {HG_IN: mel})",
    "        _php = 0.5 * sigmoid_sed(_ho[0]) + 0.5 * sigmoid_sed(_ho[1].max(axis=1))",
    "        p_mean = ((1.0 - W_HG) * p_mean + W_HG * _php).astype(np.float32)",
    "    except Exception as _e:",
    '        print("[HGBLEND] infer failed:", _e)',
]
src, ok3 = insert_after(src, "p_mean = p_sum / len(sed_sessions)", hg_blend)
assert ok3, "p_mean marker not found"

nb["cells"][11]["source"] = src
json.dump(nb, open(DST/"notebook.ipynb", "w"), indent=1)

# --- kernel-metadata ---
meta = json.load(open(SRC/"kernel-metadata.json"))
meta["id"] = "ultimatumgame/birdclef-2026-eos8-hgblend"
meta["title"] = "birdclef-2026-eos8-hgblend"
if "nischaydnk/birdclef-2026-distilled-onnx-hgnet" not in meta["dataset_sources"]:
    meta["dataset_sources"].append("nischaydnk/birdclef-2026-distilled-onnx-hgnet")
json.dump(meta, open(DST/"kernel-metadata.json", "w"), indent=2)
print("patched eos8-hgblend; datasets=", len(meta["dataset_sources"]))

import ast
ast.parse(as_str(nb["cells"][11]["source"]))
print("cell 11 AST OK")
