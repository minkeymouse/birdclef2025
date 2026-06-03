"""Patch phase4 -> new R1 kernel: save emb_te in cell 11 + append probe-blend end cell.
Handles notebook source stored as a single string (phase4 style)."""
import json, shutil
from pathlib import Path

SRC = Path("notebooks/birdclef-2026-eos8-phase4")
DST = Path("notebooks/birdclef-2026-eos8-r1")
DST.mkdir(exist_ok=True)
shutil.copy(SRC/"notebook.ipynb", DST/"notebook.ipynb")
nb = json.load(open(DST/"notebook.ipynb"))

def as_str(s): return "".join(s) if isinstance(s, list) else s

# ---- 1) insert emb_te save in cell 11 after run_perch(test_paths ...) ----
src = as_str(nb["cells"][11]["source"])
lines = src.splitlines(keepends=True)
ins_at = None; indent = ""
for i, line in enumerate(lines):
    if "= run_perch(test_paths" in line:
        ins_at = i + 1
        indent = line[:len(line) - len(line.lstrip())]
        break
assert ins_at is not None, "run_perch(test_paths) not found"
save_block = "".join([
    f"{indent}try:\n",
    f"{indent}    import numpy as _np_r1\n",
    f"{indent}    _np_r1.save('/kaggle/working/r1_emb_te.npy', emb_te.astype('float32'))\n",
    f"{indent}    meta_te[['row_id']].to_csv('/kaggle/working/r1_emb_te_rows.csv', index=False)\n",
    f"{indent}    print('[R1] saved emb_te', emb_te.shape)\n",
    f"{indent}except Exception as _e_r1:\n",
    f"{indent}    print('[R1] emb_te save failed:', _e_r1)\n",
])
nb["cells"][11]["source"] = "".join(lines[:ins_at] + [save_block] + lines[ins_at:])
print(f"inserted emb_te save after cell11 line {ins_at-1} (indent={len(indent)})")

# ---- 2) append probe-blend end cell ----
end_code = r'''# === R1: train_audio Perch probe blend for non-Aves species (site-invariant) ===
try:
    import numpy as np, pandas as pd, os, glob
    _cands = glob.glob("/kaggle/input/**/r1_probe_weights.npz", recursive=True)
    PROBE = _cands[0] if _cands else "/kaggle/input/bc2026-r1-train-audio-probes/r1_probe_weights.npz"
    print("[R1] probe file:", PROBE, "| found:", bool(_cands))
    EMB, ROWS = "/kaggle/working/r1_emb_te.npy", "/kaggle/working/r1_emb_te_rows.csv"
    W_BLEND = 0.25
    if os.path.exists(PROBE) and os.path.exists(EMB) and os.path.exists(ROWS):
        P = np.load(PROBE, allow_pickle=True)
        emb = np.load(EMB).astype(np.float32)
        rows = pd.read_csv(ROWS)["row_id"].astype(str).to_numpy()
        Xz = (emb - P["scaler_mean"]) / P["scaler_scale"]
        Xp = (Xz - P["pca_mean"]) @ P["pca_components"].T
        probe = 1.0 / (1.0 + np.exp(-(Xp @ P["W"].T + P["b"])))
        species = [str(s) for s in P["species"].tolist()]
        pdf = pd.DataFrame(probe, columns=species); pdf["row_id"] = rows
        pdf = pdf.drop_duplicates("row_id").set_index("row_id")
        sub = pd.read_csv("submission.csv")
        order = sub["row_id"].astype(str).to_numpy()
        pdf = pdf.reindex(order)
        def rankpct(a):
            a = np.asarray(a, dtype=float)
            return np.argsort(np.argsort(a)) / (len(a) - 1) if len(a) > 1 else a * 0.0
        nchg = 0
        for s in species:
            if s in sub.columns and pdf[s].notna().all():
                new = (1 - W_BLEND) * rankpct(sub[s].to_numpy()) + W_BLEND * rankpct(pdf[s].to_numpy())
                sub[s] = new.astype(np.float32); nchg += 1
        vals = sub.drop(columns=["row_id"]).to_numpy()
        assert np.isfinite(vals).all() and vals.min() >= 0 and vals.max() <= 1, "R1 invalid probs"
        assert sub["row_id"].is_unique and len(sub) == len(order), "R1 row mismatch"
        sub.to_csv("submission.csv", index=False)
        print(f"[R1] blended {nchg}/{len(species)} non-Aves probe columns (W={W_BLEND}) -> submission.csv")
    else:
        print("[R1] inputs missing -> submission unchanged:", os.path.exists(PROBE), os.path.exists(EMB), os.path.exists(ROWS))
except Exception as _e:
    print("[R1] probe blend FAILED, keeping original submission.csv:", _e)
'''
nb["cells"].append({"cell_type": "code", "execution_count": None,
                    "metadata": {}, "outputs": [], "source": end_code})
print(f"appended probe-blend cell; total cells now {len(nb['cells'])}")
json.dump(nb, open(DST/"notebook.ipynb", "w"), indent=1)

# ---- 3) kernel-metadata ----
meta = json.load(open(SRC/"kernel-metadata.json"))
meta["id"] = "ultimatumgame/birdclef-2026-eos8-r1"
meta["title"] = "birdclef-2026-eos8-r1"
if "ultimatumgame/bc2026-r1-train-audio-probes" not in meta["dataset_sources"]:
    meta["dataset_sources"].append("ultimatumgame/bc2026-r1-train-audio-probes")
json.dump(meta, open(DST/"kernel-metadata.json", "w"), indent=2)
print("wrote kernel-metadata id=", meta["id"], "| n_datasets=", len(meta["dataset_sources"]))
