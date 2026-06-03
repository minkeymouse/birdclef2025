"""Patch eos8-phase4 -> eos8-emcent: add an embedding-CENTROID non-Aves stream.
Mirrors r1_patch_notebook.py exactly (save emb_te in cell 11 + append a rank-blend end cell),
but the extra stream is cosine(test Perch emb, train_audio per-class centroid) for the 44 non-Aves
species that HAVE train_audio — bypassing the zero-signal ProtoSSM entries. Site-invariant
(centroids built from focal train_audio only, no soundscape labels). Local Tier-0: evaluable
non-Aves macro-AUC +0.03..+0.046 (experiments/teen_nonaves_probe.py). Aves untouched.
All file writes are LOCAL; no Kaggle upload/push/submit here.
"""
import json, shutil
from pathlib import Path
import numpy as np

ROOT = Path("/data/birdclef2026")
SRC = ROOT / "notebooks/birdclef-2026-eos8-phase4"
DST = ROOT / "notebooks/birdclef-2026-eos8-emcent"
DS_SLUG = "ultimatumgame/bc2026-emcent-nonaves-protos"
PROTO_NPZ = ROOT / "model-weights/teen_nonaves_prototypes.npz"
W_BLEND = 0.35

DST.mkdir(exist_ok=True)
shutil.copy(SRC / "notebook.ipynb", DST / "notebook.ipynb")
nb = json.load(open(DST / "notebook.ipynb"))

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
assert ins_at is not None, "run_perch(test_paths) not found in cell 11"
save_block = "".join([
    f"{indent}try:\n",
    f"{indent}    import numpy as _np_ec\n",
    f"{indent}    _np_ec.save('/kaggle/working/emcent_emb_te.npy', emb_te.astype('float32'))\n",
    f"{indent}    meta_te[['row_id']].to_csv('/kaggle/working/emcent_emb_te_rows.csv', index=False)\n",
    f"{indent}    print('[EMCENT] saved emb_te', emb_te.shape)\n",
    f"{indent}except Exception as _e_ec:\n",
    f"{indent}    print('[EMCENT] emb_te save failed:', _e_ec)\n",
])
nb["cells"][11]["source"] = "".join(lines[:ins_at] + [save_block] + lines[ins_at:])
print(f"inserted emb_te save after cell11 line {ins_at-1} (indent={len(indent)})")

# ---- 2) append centroid-blend end cell ----
end_code = (
    "# === EMCENT: train_audio Perch-centroid stream for non-Aves species (site-invariant) ===\n"
    "try:\n"
    "    import numpy as np, pandas as pd, os, glob\n"
    "    _cands = glob.glob('/kaggle/input/**/teen_nonaves_prototypes.npz', recursive=True)\n"
    "    NPZ = _cands[0] if _cands else '/kaggle/input/bc2026-emcent-nonaves-protos/teen_nonaves_prototypes.npz'\n"
    "    EMB, ROWS = '/kaggle/working/emcent_emb_te.npy', '/kaggle/working/emcent_emb_te_rows.csv'\n"
    f"    W_BLEND = {W_BLEND}\n"
    "    print('[EMCENT] proto file:', NPZ, '| found:', bool(_cands))\n"
    "    if os.path.exists(NPZ) and os.path.exists(EMB) and os.path.exists(ROWS):\n"
    "        P = np.load(NPZ, allow_pickle=True)\n"
    "        labels = [str(s) for s in P['primary_labels'].tolist()]\n"
    "        cents = P['raw_centroids'].astype(np.float32)          # (234,1536), L2-normalized\n"
    "        cname = [str(s) for s in P['class_name'].tolist()]\n"
    "        nclip = np.asarray(P['n_clip'])\n"
    "        target = [i for i in range(len(labels)) if cname[i] != 'Aves' and nclip[i] > 0]\n"
    "        emb = np.load(EMB).astype(np.float32)\n"
    "        emb /= (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)\n"
    "        cos = emb @ cents[target].T                            # (N_test, n_target) cosine\n"
    "        tcols = [labels[i] for i in target]\n"
    "        rows = pd.read_csv(ROWS)['row_id'].astype(str).to_numpy()\n"
    "        tdf = pd.DataFrame(cos, columns=tcols); tdf['row_id'] = rows\n"
    "        tdf = tdf.drop_duplicates('row_id').set_index('row_id')\n"
    "        sub = pd.read_csv('submission.csv')\n"
    "        order = sub['row_id'].astype(str).to_numpy()\n"
    "        tdf = tdf.reindex(order)\n"
    "        def rankpct(a):\n"
    "            a = np.asarray(a, dtype=float)\n"
    "            return np.argsort(np.argsort(a)) / (len(a) - 1) if len(a) > 1 else a * 0.0\n"
    "        nchg = 0\n"
    "        for s in tcols:\n"
    "            if s in sub.columns and tdf[s].notna().all():\n"
    "                new = (1 - W_BLEND) * rankpct(sub[s].to_numpy()) + W_BLEND * rankpct(tdf[s].to_numpy())\n"
    "                sub[s] = new.astype(np.float32); nchg += 1\n"
    "        vals = sub.drop(columns=['row_id']).to_numpy()\n"
    "        assert np.isfinite(vals).all() and vals.min() >= 0 and vals.max() <= 1, 'EMCENT invalid probs'\n"
    "        assert sub['row_id'].is_unique and len(sub) == len(order), 'EMCENT row mismatch'\n"
    "        sub.to_csv('submission.csv', index=False)\n"
    "        print(f'[EMCENT] blended {nchg}/{len(tcols)} non-Aves centroid columns (W={W_BLEND}) -> submission.csv')\n"
    "    else:\n"
    "        print('[EMCENT] inputs missing -> submission unchanged:', os.path.exists(NPZ), os.path.exists(EMB), os.path.exists(ROWS))\n"
    "except Exception as _e:\n"
    "    print('[EMCENT] centroid blend FAILED, keeping original submission.csv:', _e)\n"
)
nb["cells"].append({"cell_type": "code", "execution_count": None,
                    "metadata": {}, "outputs": [], "source": end_code})
print(f"appended centroid-blend cell; total cells now {len(nb['cells'])}")
json.dump(nb, open(DST / "notebook.ipynb", "w"), indent=1)

# ---- 3) kernel-metadata ----
meta = json.load(open(SRC / "kernel-metadata.json"))
meta["id"] = "ultimatumgame/birdclef-2026-eos8-emcent"
meta["title"] = "birdclef-2026-eos8-emcent"
if DS_SLUG not in meta["dataset_sources"]:
    meta["dataset_sources"].append(DS_SLUG)
json.dump(meta, open(DST / "kernel-metadata.json", "w"), indent=2)
print("wrote kernel-metadata id=", meta["id"], "| n_datasets=", len(meta["dataset_sources"]))

# ---- 4) prepare (local only) Kaggle dataset dir for the centroid npz ----
ds_dir = ROOT / "model-weights/emcent_ds"
ds_dir.mkdir(exist_ok=True)
shutil.copy(PROTO_NPZ, ds_dir / "teen_nonaves_prototypes.npz")
json.dump({"title": "bc2026-emcent-nonaves-protos", "id": DS_SLUG,
           "licenses": [{"name": "CC0-1.0"}]}, open(ds_dir / "dataset-metadata.json", "w"), indent=2)
print(f"prepared dataset dir {ds_dir} (npz {(PROTO_NPZ.stat().st_size/1e6):.1f}MB) — NOT uploaded")
