"""Fork phase4 -> eos8-ptax2: skip GENUS taxonomy-smoothing for Insecta only.
The 25 sonotypes (47158son01-25) share one scientific-name 'genus' -> the genus smoothing
(TAX_GENUS_ALPHA=0.15) averages 15% of each sonotype's score toward the 25-way mean. But sonotypes are
MUTUALLY-EXCLUSIVE acoustic types -> that averaging CROSS-CONTAMINATES each sonotype's ranking (pulls a
present type toward the absent ones). Skipping genus smoothing for all-Insecta genus groups restores pure
per-sonotype scores. Aves / Amphibia genera keep 0.15 (unchanged) -> Aves byte-identical. Sonotypes are
mostly EVALUABLE (20 Insecta with n_pos>=10) -> rare lever that touches measurable species. RANK-CHANGING.
Run: uv run python experiments/ptax2_patch.py
"""
import json, shutil, ast
from pathlib import Path

SRC = Path("notebooks/birdclef-2026-eos8-phase4")
DST = Path("notebooks/birdclef-2026-eos8-ptax2")
DST.mkdir(exist_ok=True)
shutil.copy(SRC / "notebook.ipynb", DST / "notebook.ipynb")
nb = json.load(open(DST / "notebook.ipynb"))

c13 = "".join(nb["cells"][13]["source"])
old = ("    for members in multi_genus.values():\n"
       "        idx = [col_to_idx[m] for m in members]\n"
       "        group_mean = probs[:, idx].mean(axis=1, keepdims=True)\n"
       "        probs[:, idx] = (1.0 - genus_alpha) * probs[:, idx] + genus_alpha * group_mean")
assert c13.count(old) == 1, f"cell13 genus-loop marker count={c13.count(old)}"
new = ("    for members in multi_genus.values():\n"
       "        idx = [col_to_idx[m] for m in members]\n"
       "        if all(species_to_class.get(m, \"\") == \"Insecta\" for m in members):\n"
       "            continue  # RANK2: no genus smoothing for Insecta sonotypes (mutually-exclusive -> avg corrupts)\n"
       "        group_mean = probs[:, idx].mean(axis=1, keepdims=True)\n"
       "        probs[:, idx] = (1.0 - genus_alpha) * probs[:, idx] + genus_alpha * group_mean")
nb["cells"][13]["source"] = c13.replace(old, new)
ast.parse("".join(nb["cells"][13]["source"]))

json.dump(nb, open(DST / "notebook.ipynb", "w"), indent=1)
m = json.load(open(SRC / "kernel-metadata.json"))
m["id"] = "ultimatumgame/birdclef-2026-eos8-ptax2"
m["title"] = "birdclef-2026-eos8-ptax2"
json.dump(m, open(DST / "kernel-metadata.json", "w"), indent=2)
print("built eos8-ptax2: skip genus tax-smoothing for Insecta sonotypes (cell13); AST OK")
