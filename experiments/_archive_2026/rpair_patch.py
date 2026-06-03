"""Recipe-pair lever on phase4: blend OUR exp175/176 SED (ensemble_v3, soundscape-aware,
comparable ~0.941, decorrelated-from-Tucker, SAME mel interface) into phase4's Tucker SED
stream at W_OURS. The proven +0.003 mechanism (recipe-pair), applied to the EoS8 stack.
Differs from hgblend: ours is soundscape-aware+comparable (no domain gap, not weak)."""
import json, shutil
from pathlib import Path

SRC = Path("notebooks/birdclef-2026-eos8-phase4")
DST = Path("notebooks/birdclef-2026-eos8-rpair")
DST.mkdir(exist_ok=True)
shutil.copy(SRC/"notebook.ipynb", DST/"notebook.ipynb")
nb = json.load(open(DST/"notebook.ipynb"))
src = "".join(nb["cells"][11]["source"]) if isinstance(nb["cells"][11]["source"], list) else nb["cells"][11]["source"]

# find_sed_dir: exclude our ensemble-v3 + hgnet so Tucker stays the MAIN SED
old1 = 'hits = sorted(Path("/kaggle/input").rglob("sed_fold0.onnx"))'
new1 = ('hits = sorted(p for p in Path("/kaggle/input").rglob("sed_fold0.onnx") '
        'if "ensemble-v3" not in str(p).lower() and "hgnet" not in str(p).lower())')
assert src.count(old1) == 1
src = src.replace(old1, new1)

def insert_after(text, marker, lines):
    L = text.splitlines(keepends=True)
    for i, ln in enumerate(L):
        if marker in ln:
            ind = ln[:len(ln)-len(ln.lstrip())]
            return "".join(L[:i+1] + ["".join(ind+b+"\n" for b in lines)] + L[i+1:]), True
    return text, False

# load OUR sed sessions (ensemble_v3) after Tucker sed_sessions
ours_load = [
    "OUR_SESS = []; W_OURS = 0.30",
    "try:",
    '    _keep = {"sed_fold00.onnx","sed_fold01.onnx","sed_fold15.onnx"}',
    '    _ours = sorted(p for p in Path("/kaggle/input").rglob("sed_fold*.onnx") if "ensemble-v3" in str(p).lower() and p.name in _keep)',
    "    OUR_SESS = [make_sed_session(p) for p in _ours]   # 3 folds: exp175(00,01)+exp176(15) recipe-pair, wall-time-safe (8 SED total + sidecars off)",
    '    print(f"[RPAIR] our decorrelated SED folds loaded: {len(OUR_SESS)} (W_OURS={W_OURS}); Tucker folds={len(sed_sessions)}")',
    "except Exception as _e:",
    '    print("[RPAIR] our SED load failed:", _e); OUR_SESS = []',
]
src, ok2 = insert_after(src, "sed_sessions = [make_sed_session(p) for p in sed_fold_paths]", ours_load)
assert ok2

# blend OUR SED stream into p_mean (weighted, Tucker dominant)
ours_blend = [
    "if OUR_SESS:",
    "    _ps = np.zeros_like(p_mean)",
    "    for _s in OUR_SESS:",
    "        _o = _s.run(None, {_s.get_inputs()[0].name: mel})",
    "        _ps += 0.5*sigmoid_sed(_o[0]) + 0.5*sigmoid_sed(_o[1].max(axis=1))",
    "    _po = _ps / len(OUR_SESS)",
    "    p_mean = ((1.0 - W_OURS)*p_mean + W_OURS*_po).astype(np.float32)",
]
src, ok3 = insert_after(src, "p_mean = p_sum / len(sed_sessions)", ours_blend)
assert ok3

nb["cells"][11]["source"] = src

# disable 0-LB sidecars (BirdNET v56=0, PCEN finding=0) to free wall-time for our SED folds
c4 = "".join(nb["cells"][4]["source"]) if isinstance(nb["cells"][4]["source"], list) else nb["cells"][4]["source"]
for a, b in [("RUN_BIRDNET_SIDECAR = True", "RUN_BIRDNET_SIDECAR = False"),
             ("RUN_EXP002_SIDECAR = True", "RUN_EXP002_SIDECAR = False")]:
    assert c4.count(a) == 1, f"sidecar marker {a!r} count={c4.count(a)}"
    c4 = c4.replace(a, b)
nb["cells"][4]["source"] = c4
print("disabled BirdNET + PCEN sidecars (0-LB) for wall-time")

json.dump(nb, open(DST/"notebook.ipynb", "w"), indent=1)

meta = json.load(open(SRC/"kernel-metadata.json"))
meta["id"] = "ultimatumgame/birdclef-2026-eos8-rpair"; meta["title"] = "birdclef-2026-eos8-rpair"
if "ultimatumgame/bc2026-ensemble-v3-sed" not in meta["dataset_sources"]:
    meta["dataset_sources"].append("ultimatumgame/bc2026-ensemble-v3-sed")
json.dump(meta, open(DST/"kernel-metadata.json", "w"), indent=2)

import ast; ast.parse(src)
print("built eos8-rpair; AST OK; datasets=", len(meta["dataset_sources"]))
