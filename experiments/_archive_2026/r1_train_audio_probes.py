"""R1 capacity check + probe training: do Perch embeddings (train_audio, site-invariant)
separate the NON-AVES species that the baseline's site-22 SS-probe handles weakly?

NOT an LB predictor — a capacity check: can a probe even learn these species from
focal train_audio embeddings (35k clips, no site-22 bias)? Decides if R1 is worth a slot.
"""
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import pickle, time

ROOT = Path("/data/birdclef2026")
d = np.load(ROOT/"cache/train_audio_perch_embeddings.npz", allow_pickle=True)
emb = d["embeddings"].astype(np.float32)          # (35549, 1536)
sp  = d["species"].astype(str)
tax = pd.read_csv(ROOT/"data/birdclef-2026/taxonomy.csv")
cls_map = tax.set_index(tax["primary_label"].astype(str))["class_name"].to_dict()

# PCA-64 to match baseline probe feature space, standardized
scaler = StandardScaler().fit(emb)
Xz = scaler.transform(emb)
pca = PCA(n_components=64, random_state=0).fit(Xz)
X = pca.transform(Xz).astype(np.float32)
print(f"emb {emb.shape} -> PCA {X.shape}")

non_aves = [s for s in np.unique(sp) if cls_map.get(s,"Aves") != "Aves"]
print(f"non-Aves species in train_audio: {len(non_aves)}")

rows=[]; probes={}
t0=time.time()
for s in non_aves:
    y = (sp==s).astype(int)
    npos = int(y.sum())
    if npos < 10:
        rows.append((s, cls_map.get(s), npos, np.nan)); continue
    # 5-fold CV AUC (one-vs-rest) — capacity, not LB
    aucs=[]
    skf=StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    for tr,va in skf.split(X,y):
        clf=LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
        clf.fit(X[tr],y[tr])
        aucs.append(roc_auc_score(y[va], clf.predict_proba(X[va])[:,1]))
    cv=float(np.mean(aucs))
    rows.append((s, cls_map.get(s), npos, cv))
    # fit final probe on all data for shipping
    clf=LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced").fit(X,y)
    probes[s]=clf

rep=pd.DataFrame(rows, columns=["species","class","n_pos","cv_auc"]).sort_values("cv_auc")
print(f"\ntrained {len(probes)} probes in {time.time()-t0:.0f}s")
print("\n=== per-class train_audio CV AUC (non-Aves, n_pos>=10) ===")
with pd.option_context("display.max_rows", 200):
    print(rep[rep.n_pos>=10].to_string(index=False))
ev = rep[rep.n_pos>=10]
print(f"\nSUMMARY non-Aves probes: n={len(ev)}  mean CV AUC={ev.cv_auc.mean():.3f}  "
      f"median={ev.cv_auc.median():.3f}  frac>0.85={ (ev.cv_auc>0.85).mean():.2f}  frac>0.95={(ev.cv_auc>0.95).mean():.2f}")
by_cls = ev.groupby("class").cv_auc.agg(["count","mean","median"])
print("\nby class:\n", by_cls.to_string())

out = ROOT/"model-weights/r1_train_audio_probes.pkl"
with open(out,"wb") as f:
    pickle.dump({"probes":probes,"scaler":scaler,"pca":pca,"report":rep}, f)
print(f"\nsaved -> {out}")
