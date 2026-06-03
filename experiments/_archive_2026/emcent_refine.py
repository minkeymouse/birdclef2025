"""Pick the best non-Aves embedding-prototype variant for emcent, on the site-safe eval.
Compares (per evaluable non-Aves species, soundscape AUC): single mean-centroid cosine,
multi-prototype k-means (max-cosine, K=2/3/4), and the existing R1 logreg probe on overlap.
CPU only, cached embeddings. Run: uv run python experiments/emcent_refine.py
"""
import numpy as np, pandas as pd, warnings
from sklearn.metrics import roc_auc_score
from sklearn.cluster import KMeans
warnings.filterwarnings("ignore")
ROOT = "/data/birdclef2026"

def l2n(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)

pa = np.load(f"{ROOT}/cache/perch_arrays.npz", allow_pickle=True)
SP = pa["primary_labels"].astype(str).tolist()
ssn = l2n(pa["embs"].astype(np.float32))                       # (708,1536) normalized
y = np.load(f"{ROOT}/cache/sidecar_dpo/y_soundscape.npy"); npos = y.sum(0)
tr = np.load(f"{ROOT}/cache/train_audio_perch_embeddings.npz", allow_pickle=True)
tre = tr["embeddings"].astype(np.float32); trs = tr["species"].astype(str)
tax = pd.read_csv(f"{ROOT}/data/birdclef-2026/taxonomy.csv")
cls = tax.set_index(tax.primary_label.astype(str)).class_name.to_dict()
classes = np.array([cls[s] for s in SP]); aves = classes == "Aves"

# per-class normalized train_audio embeddings
byc = {}
for i, s in enumerate(SP):
    m = trs == s
    if m.sum() > 0:
        byc[i] = l2n(tre[m])

def AUC(score, i):
    try:
        return roc_auc_score(y[:, i], score)
    except ValueError:
        return np.nan

def s_single(i):
    return ssn @ l2n(byc[i].mean(0))

def s_multi(i, K):
    X = byc[i]; k = min(K, len(X))
    if k <= 1:
        return ssn @ l2n(X.mean(0))
    C = l2n(KMeans(k, n_init=4, random_state=0).fit(X).cluster_centers_)
    return (ssn @ C.T).max(1)

def s_lse(i, K, tau=0.05):
    X = byc[i]; k = min(K, len(X))
    if k <= 1:
        return ssn @ l2n(X.mean(0))
    C = l2n(KMeans(k, n_init=4, random_state=0).fit(X).cluster_centers_)
    sims = ssn @ C.T
    return tau * np.log(np.exp(sims / tau).sum(1))

for MP in (5, 10):
    ev = [i for i in range(234) if (not aves[i]) and (i in byc) and npos[i] >= MP]
    print(f"\n── non-Aves n_pos≥{MP}: {len(ev)} species ──")
    variants = {
        "single-mean": lambda i: s_single(i),
        "multi-K2": lambda i: s_multi(i, 2),
        "multi-K3": lambda i: s_multi(i, 3),
        "multi-K4": lambda i: s_multi(i, 4),
        "lse-K3": lambda i: s_lse(i, 3),
    }
    for name, fn in variants.items():
        a = np.nanmean([AUC(fn(i), i) for i in ev])
        print(f"   {name:14s} non-Aves macro-AUC = {a:.4f}")

# R1 probe overlap check
P = np.load(f"{ROOT}/model-weights/r1_probe_weights.npz", allow_pickle=True)
r1sp = [str(s) for s in P["species"].tolist()]
Xz = (pa["embs"].astype(np.float32) - P["scaler_mean"]) / P["scaler_scale"]
Xp = (Xz - P["pca_mean"]) @ P["pca_components"].T
probe = 1.0 / (1.0 + np.exp(-(Xp @ P["W"].T + P["b"])))         # (708,20)
r1map = {s: j for j, s in enumerate(r1sp)}
ev10 = [i for i in range(234) if (not aves[i]) and (i in byc) and npos[i] >= 10]
ev_r1 = [i for i in ev10 if SP[i] in r1map]
if ev_r1:
    ap = np.nanmean([AUC(probe[:, r1map[SP[i]]], i) for i in ev_r1])
    ac = np.nanmean([AUC(s_single(i), i) for i in ev_r1])
    # 0.5/0.5 rank blend of probe+centroid on overlap
    def rp(a): a = np.asarray(a, float); return np.argsort(np.argsort(a)) / (len(a) - 1)
    ab = np.nanmean([AUC(0.5 * rp(probe[:, r1map[SP[i]]]) + 0.5 * rp(s_single(i)), i) for i in ev_r1])
    print(f"\nR1 overlap ({len(ev_r1)} sp): probe={ap:.4f}  centroid={ac:.4f}  blend50={ab:.4f}")
print("\n→ choose the variant with highest non-Aves AUC at both thresholds for emcent v2")
