"""Patch eos8-phase4 -> eos8-exp189: blend the exp189 SED (Tucker recipe + EXTERNAL non-Aves data)
into the Tucker SED stream, but ONLY for non-Aves columns (Aves left byte-identical = Tucker).

Why non-Aves-only: exp187 (full blend into p_mean) dragged -0.012, the cleanest mechanism being
perturbation of the Aves-heavy scored mass (84% of blind species are Aves; SED < Perch on Aves). exp189's
value is the external non-Aves supervision; restricting the blend to the 72 non-Aves captures that upside
while leaving the Aves scored-bulk on pure Tucker. exp189 is BCE (calibrated like Tucker) -> clean prob blend
(no exp187 SoftAUC scale distortion). p_mean is (n_windows, 234) in PRIMARY_LABELS order -> mask aligns.

PREREQ: exp189 gate PASS + folds exported to ONNX (sed_fold*.onnx) + uploaded as the DS_SLUG dataset.
Run: uv run python experiments/exp189_patch_notebook.py
"""
import json, shutil, ast
from pathlib import Path
import sys
SRC = Path("notebooks/birdclef-2026-eos8-phase4")
DS_SLUG = "ultimatumgame/bc2026-exp189-sed"   # exp189 ONNX folds (sed_fold*.onnx under an exp189/ dir)
W_EXP189 = float(sys.argv[2]) if len(sys.argv) > 2 else 0.30   # blend weight; finalize from gate
BLEND_MODE = sys.argv[1] if len(sys.argv) > 1 else "nonaves"   # "nonaves" (Aves protected) | "full" (all classes)
assert BLEND_MODE in ("nonaves", "full")
DST = Path(f"notebooks/birdclef-2026-eos8-exp189-{BLEND_MODE}")
DST.mkdir(exist_ok=True)
shutil.copy(SRC / "notebook.ipynb", DST / "notebook.ipynb")
nb = json.load(open(DST / "notebook.ipynb"))
def as_str(s): return "".join(s) if isinstance(s, list) else s
src = as_str(nb["cells"][11]["source"])

def insert_after(text, marker, lines):
    i = text.find(marker)
    if i < 0: return text, False
    j = text.find("\n", i) + 1
    pad = text[text.rfind("\n", 0, i) + 1:i]   # leading whitespace of marker line
    block = "".join(pad + l + "\n" for l in lines)
    return text[:j] + block + text[j:], True

# edit 1: load exp189 SED fold sessions + build the non-Aves column mask (PRIMARY_LABELS order)
load_lines = [
    "EX189_SESS = []; EX189_IN = None; W_EXP189 = %.2f; EXP189_BLEND_MODE = \"%s\"" % (W_EXP189, BLEND_MODE),
    "_NONAVES_COLS = np.array([CLASS_NAME_MAP.get(l, \"Aves\") != \"Aves\" for l in PRIMARY_LABELS], dtype=bool)",
    "try:",
    '    _ex = sorted(p for p in Path("/kaggle/input").rglob("sed_fold*.onnx") if "exp189" in str(p).lower())',
    "    EX189_SESS = [make_sed_session(p) for p in _ex]",
    "    if EX189_SESS:",
    "        EX189_IN = EX189_SESS[0].get_inputs()[0].name",
    '        print(f"[EXP189] loaded {len(EX189_SESS)} exp189 folds (W={W_EXP189}, non-Aves-only n={int(_NONAVES_COLS.sum())})")',
    "    else:",
    '        print("[EXP189] no exp189 onnx found -> Tucker only")',
    "except Exception as _e:",
    '    print("[EXP189] load failed:", _e); EX189_SESS = []',
]
src, ok1 = insert_after(src, "sed_sessions = [make_sed_session(p) for p in sed_fold_paths]", load_lines)
assert ok1, "sed_sessions marker not found"

# edit 2: NON-Aves-only blend of exp189 into p_mean (Aves columns stay = Tucker)
blend_lines = [
    "if EX189_SESS:",
    "    try:",
    "        _ep = np.zeros_like(p_mean)",
    "        for _s in EX189_SESS:",
    "            _o = _s.run(None, {EX189_IN: mel})",
    "            _ep += 0.5 * sigmoid_sed(_o[0]) + 0.5 * sigmoid_sed(_o[1].max(axis=1))",
    "        _ep /= len(EX189_SESS)",
    "        _blended = ((1.0 - W_EXP189) * p_mean + W_EXP189 * _ep).astype(np.float32)",
    "        if EXP189_BLEND_MODE == \"full\":",
    "            p_mean = _blended",
    "        else:",
    "            p_mean[:, _NONAVES_COLS] = _blended[:, _NONAVES_COLS]",
    "    except Exception as _e:",
    '        print("[EXP189] infer failed:", _e)',
]
src, ok2 = insert_after(src, "p_mean = p_sum / len(sed_sessions)", blend_lines)
assert ok2, "p_mean marker not found"

nb["cells"][11]["source"] = src
ast.parse(as_str(nb["cells"][11]["source"]))  # validate
json.dump(nb, open(DST / "notebook.ipynb", "w"), indent=1)

meta = json.load(open(SRC / "kernel-metadata.json"))
meta["id"] = f"ultimatumgame/birdclef-2026-eos8-exp189-{BLEND_MODE}"
meta["title"] = f"birdclef-2026-eos8-exp189-{BLEND_MODE}"
if DS_SLUG not in meta["dataset_sources"]:
    meta["dataset_sources"].append(DS_SLUG)
json.dump(meta, open(DST / "kernel-metadata.json", "w"), indent=2)
print(f"patched eos8-exp189 (W={W_EXP189}, NON-Aves-only); cell11 AST OK; datasets={len(meta['dataset_sources'])}")
print("PREREQ: gate PASS + export exp189 folds to ONNX (sed_fold*.onnx) + upload as", DS_SLUG)
