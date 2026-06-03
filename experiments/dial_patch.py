"""Build a phase4-derived kernel changing ONE Controls constant (single-variable dial test).
Usage: python dial_patch.py <dst_suffix> <old_substr> <new_substr> [<cell_idx default 4>]"""
import json, shutil, sys
from pathlib import Path

dst_suffix, old, new = sys.argv[1], sys.argv[2], sys.argv[3]
cell_idx = int(sys.argv[4]) if len(sys.argv) > 4 else 4

SRC = Path("notebooks/birdclef-2026-eos8-phase4")
DST = Path(f"notebooks/birdclef-2026-eos8-{dst_suffix}")
DST.mkdir(exist_ok=True)
shutil.copy(SRC/"notebook.ipynb", DST/"notebook.ipynb")
nb = json.load(open(DST/"notebook.ipynb"))

s = nb["cells"][cell_idx]["source"]
s = "".join(s) if isinstance(s, list) else s
assert s.count(old) == 1, f"marker count={s.count(old)} for {old!r} in cell {cell_idx}"
s = s.replace(old, new)
nb["cells"][cell_idx]["source"] = s
json.dump(nb, open(DST/"notebook.ipynb", "w"), indent=1)

m = json.load(open(SRC/"kernel-metadata.json"))
m["id"] = f"ultimatumgame/birdclef-2026-eos8-{dst_suffix}"
m["title"] = f"birdclef-2026-eos8-{dst_suffix}"
json.dump(m, open(DST/"kernel-metadata.json", "w"), indent=2)

import ast
ast.parse("".join(nb["cells"][cell_idx]["source"]) if isinstance(nb["cells"][cell_idx]["source"], list) else nb["cells"][cell_idx]["source"])
print(f"built eos8-{dst_suffix}: {old!r} -> {new!r} (cell {cell_idx}); AST OK")
