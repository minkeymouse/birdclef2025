"""Patch eos8-phase4 -> eos8-exp187: W-blend the exp187 distilled SED (B1 + SoftAUC + heterogeneous
teacher pseudo) into the Tucker SED stream. exp187 is a DistilledSED with the SAME 256-mel interface
as Tucker, so it runs on the same `mel` tensor (clip+frame outputs) — clean drop-in, mirrors
hg_patch_notebook.py. ONLY run after fold-0 GATE PASSES + full folds exported to ONNX + uploaded as a
Kaggle dataset. W_EXP187 from the local gate (rank-blend non-drag weight, default 0.30).
Run: uv run python experiments/exp187_patch_notebook.py
"""
import json, shutil
from pathlib import Path
SRC = Path("notebooks/birdclef-2026-eos8-phase4")
DST = Path("notebooks/birdclef-2026-eos8-exp187")
DS_SLUG = "ultimatumgame/bc2026-exp187-sed"   # upload exported exp187 ONNX folds here (sed_fold*.onnx under an exp187/ dir)
W_EXP187 = 0.10
DST.mkdir(exist_ok=True)
shutil.copy(SRC / "notebook.ipynb", DST / "notebook.ipynb")
nb = json.load(open(DST / "notebook.ipynb"))
def as_str(s): return "".join(s) if isinstance(s, list) else s
src = as_str(nb["cells"][11]["source"])

def insert_after(text, marker, lines):
    i = text.find(marker)
    if i < 0: return text, False
    j = text.find("\n", i) + 1
    indent = text[i - (len(text[:i]) - len(text[:i].rstrip(" "))):i]  # leading spaces of marker line
    pad = text[text.rfind("\n", 0, i) + 1:i]
    block = "".join(pad + l + "\n" for l in lines)
    return text[:j] + block + text[j:], True

# edit 1: load exp187 SED fold sessions (separate ensemble) after Tucker sed_sessions
load_lines = [
    "EX187_SESS = []; EX187_IN = None; W_EXP187 = %.2f" % W_EXP187,
    "try:",
    '    _ex = sorted(p for p in Path("/kaggle/input").rglob("sed_fold*.onnx") if "exp187" in str(p).lower())',
    "    EX187_SESS = [make_sed_session(p) for p in _ex]",
    "    if EX187_SESS:",
    "        EX187_IN = EX187_SESS[0].get_inputs()[0].name",
    '        print(f"[EXP187] loaded {len(EX187_SESS)} exp187 folds (W={W_EXP187})")',
    "    else:",
    '        print("[EXP187] no exp187 onnx found -> Tucker only")',
    "except Exception as _e:",
    '    print("[EXP187] load failed:", _e); EX187_SESS = []',
]
src, ok1 = insert_after(src, "sed_sessions = [make_sed_session(p) for p in sed_fold_paths]", load_lines)
assert ok1, "sed_sessions marker not found"

# edit 2: blend exp187 ensemble into p_mean (W-blend; gate-validated non-drag)
blend_lines = [
    "if EX187_SESS:",
    "    try:",
    "        _ep = np.zeros_like(p_mean)",
    "        for _s in EX187_SESS:",
    "            _o = _s.run(None, {EX187_IN: mel})",
    "            _ep += 0.5 * sigmoid_sed(_o[0]) + 0.5 * sigmoid_sed(_o[1].max(axis=1))",
    "        _ep /= len(EX187_SESS)",
    "        p_mean = ((1.0 - W_EXP187) * p_mean + W_EXP187 * _ep).astype(np.float32)",
    "    except Exception as _e:",
    '        print("[EXP187] infer failed:", _e)',
]
src, ok2 = insert_after(src, "p_mean = p_sum / len(sed_sessions)", blend_lines)
assert ok2, "p_mean marker not found"

nb["cells"][11]["source"] = src
import ast; ast.parse(as_str(nb["cells"][11]["source"]))  # validate
json.dump(nb, open(DST / "notebook.ipynb", "w"), indent=1)

meta = json.load(open(SRC / "kernel-metadata.json"))
meta["id"] = "ultimatumgame/birdclef-2026-eos8-exp187"
meta["title"] = "birdclef-2026-eos8-exp187"
if DS_SLUG not in meta["dataset_sources"]:
    meta["dataset_sources"].append(DS_SLUG)
json.dump(meta, open(DST / "kernel-metadata.json", "w"), indent=2)
print(f"patched eos8-exp187 (W={W_EXP187}); cell11 AST OK; datasets={len(meta['dataset_sources'])}")
print("PREREQ: gate PASS + export exp187 folds to ONNX (named sed_fold*.onnx) + upload as", DS_SLUG)
