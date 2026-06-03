"""Fork phase4 -> eos8-ptlam: PER-TAXON prior lambda.
Aves keep POWEROPT_PRIOR_LAMBDA (0.65); non-Aves get POWEROPT_PRIOR_LAMBDA_NONAVES (0.15).
Mechanism: hour/site priors are estimated from labeled SS (68% site-22, Aves-heavy dawn-chorus
structure). That timing prior MISRANKS nocturnal/continuous-calling non-Aves (Insecta/Amphibia).
Weakening lambda ONLY for the 72 non-Aves re-ranks them off the wrong Aves prior. Aves output is
byte-identical (lambda unchanged) -> Aves AUC cannot regress. RANK-CHANGING on non-Aves -> CAN move macro-AUC.
apply_prior does `out += lambda_prior * prior_logit`; a (234,) lambda vector broadcasts -> no fn change.
"""
import json, shutil, ast
from pathlib import Path

SRC = Path("notebooks/birdclef-2026-eos8-phase4")
DST = Path("notebooks/birdclef-2026-eos8-ptlam")
DST.mkdir(exist_ok=True)
shutil.copy(SRC / "notebook.ipynb", DST / "notebook.ipynb")
nb = json.load(open(DST / "notebook.ipynb"))

def get(ci):
    s = nb["cells"][ci]["source"]
    return "".join(s) if isinstance(s, list) else s
def put(ci, s):
    nb["cells"][ci]["source"] = s

# --- Cell 4: add the non-Aves lambda constant ---
c4 = get(4)
old4 = "POWEROPT_PRIOR_LAMBDA = 0.65"
assert c4.count(old4) == 1, f"cell4 marker count={c4.count(old4)}"
new4 = ("POWEROPT_PRIOR_LAMBDA = 0.65\n"
        "POWEROPT_PRIOR_LAMBDA_NONAVES = 0.15  # per-taxon: weaker hour/site prior for non-Aves "
        "(Aves-tuned site-22 prior misranks nocturnal/continuous non-Aves)")
put(4, c4.replace(old4, new4))

# --- Cell 11: build per-class lambda vector once, swap scalar -> vector in BOTH apply_prior calls ---
c11 = get(11)
anchor = "prior_tables   = build_prior_tables(sc, Y_SC)"
assert c11.count(anchor) == 1, f"cell11 anchor count={c11.count(anchor)}"
vec = (anchor + "\n"
       "        _PTLAM_NONAVES = np.array([CLASS_NAME_MAP.get(l, \"Aves\") != \"Aves\" for l in PRIMARY_LABELS], dtype=bool)\n"
       "        _LAMBDA_VEC = np.where(_PTLAM_NONAVES, POWEROPT_PRIOR_LAMBDA_NONAVES, POWEROPT_PRIOR_LAMBDA).astype(np.float32)\n"
       "        print(f\"[PTLAM] per-taxon prior lambda: Aves={POWEROPT_PRIOR_LAMBDA} non-Aves={POWEROPT_PRIOR_LAMBDA_NONAVES} \"\n"
       "              f\"(n_nonaves={int(_PTLAM_NONAVES.sum())})\")")
c11 = c11.replace(anchor, vec)
swap_old = "lambda_prior=POWEROPT_PRIOR_LAMBDA)"
n = c11.count(swap_old)
assert n == 2, f"expected 2 apply_prior calls, found {n}"
c11 = c11.replace(swap_old, "lambda_prior=_LAMBDA_VEC)")
put(11, c11)
ast.parse(get(11))  # validate cell 11 python

json.dump(nb, open(DST / "notebook.ipynb", "w"), indent=1)

m = json.load(open(SRC / "kernel-metadata.json"))
m["id"] = "ultimatumgame/birdclef-2026-eos8-ptlam"
m["title"] = "birdclef-2026-eos8-ptlam"
json.dump(m, open(DST / "kernel-metadata.json", "w"), indent=2)
print("built eos8-ptlam: per-taxon prior lambda (Aves 0.65 / non-Aves 0.15); cell4+cell11 patched; AST OK")
print(f"  kernel id: {m['id']}")
