"""Evaluate IN-DOMAIN non-Aves centroids (built from confident UNLABELED soundscape windows,
gated by Tucker SED) vs the FOCAL train_audio centroid (emcent), on the site-safe eval.
Run AFTER extract_unlabeled_ss.py finishes. CPU. Tests whether closing the focal→SS domain
gap improves the non-Aves stream. Run: uv run python experiments/emcent_indomain_eval.py
"""
import numpy as np, pandas as pd, warnings
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")
R = "/data/birdclef2026"
def l2n(x): return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)

# site-safe eval
pa = np.load(f"{R}/cache/perch_arrays.npz", allow_pickle=True)
SP = pa["primary_labels"].astype(str).tolist(); ssn = l2n(pa["embs"].astype(np.float32))
y = np.load(f"{R}/cache/sidecar_dpo/y_soundscape.npy"); npos = y.sum(0)
tax = pd.read_csv(f"{R}/data/birdclef-2026/taxonomy.csv")
cls = tax.set_index(tax.primary_label.astype(str)).class_name.to_dict()
aves = np.array([cls[s] == "Aves" for s in SP])
# focal centroids (emcent)
foc = np.load(f"{R}/model-weights/teen_nonaves_prototypes.npz", allow_pickle=True)["raw_centroids"].astype(np.float32)
# unlabeled SS extraction
U = np.load(f"{R}/cache/unlabeled_ss_strided.npz", allow_pickle=True)
uemb = l2n(U["embs"].astype(np.float32)); usc = U["scores"].astype(np.float32)
ulab = [str(s) for s in U["labels"].tolist()]            # SED/score column order
sed_col = {s: j for j, s in enumerate(ulab)}
print(f"unlabeled windows: {uemb.shape[0]} | partial={U['partial'] if 'partial' in U.files else '?'}")

def AUC(score, i):
    try: return roc_auc_score(y[:, i], score)
    except ValueError: return np.nan
def rp(a): a = np.asarray(a, float); return np.argsort(np.argsort(a)) / (len(a) - 1)

for TOPK in (50, 100, 200):
    rows = []
    for MP in (10,):
        ev = [i for i in range(234) if (not aves[i]) and npos[i] >= MP and SP[i] in sed_col]
        af, ai, ab = [], [], []
        for i in ev:
            j = sed_col[SP[i]]
            order = np.argsort(-usc[:, j])[:TOPK]                 # top-K confident unlabeled windows
            if len(order) < 10: continue
            indc = l2n(uemb[order].mean(0))                       # in-domain centroid
            s_foc = ssn @ l2n(foc[i]); s_ind = ssn @ indc
            af.append(AUC(s_foc, i)); ai.append(AUC(s_ind, i))
            ab.append(AUC(0.5 * rp(s_foc) + 0.5 * rp(s_ind), i))
        print(f"TOPK={TOPK} non-Aves n_pos>={MP} ({len(af)}sp): "
              f"focal={np.nanmean(af):.4f}  in-domain={np.nanmean(ai):.4f}  blend={np.nanmean(ab):.4f}")
print("\n→ if in-domain or blend > focal by >0.005, rebuild emcent prototypes with in-domain (or blended) centroids")
