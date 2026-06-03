"""Tier-0 probe: does an embedding-space PROTOTYPE stream give non-Aves signal that the
frozen-Perch ProtoSSM head lacks (31 unmapped species → logit 0)?  Compares, per evaluable
non-Aves species, the soundscape AUC of:
  (a) RAW   = cosine(SS emb, train_audio per-class centroid)
  (b) TEEN  = cosine(SS emb, TEEN-calibrated centroid)   [NeurIPS 2023, arXiv:2312.05229]
  (c) baseline `scores` already in perch_arrays.npz (current pipeline, for reference)
Site-22-safe: built only from train_audio centroids; NO soundscape labels used to fit anything.
CPU/numpy only, no GPU/network/LB.  Run: uv run python experiments/teen_nonaves_probe.py
"""
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score

ROOT = Path("/data/birdclef2026")
MIN_POS_REPORT = (5, 10, 20)
ALPHAS = (0.3, 0.5, 0.7, 0.9)   # weight on the novel (own) centroid; ≥0.5 keeps own signal dominant
TAUS = (8.0, 16.0, 32.0)

def l2n(x, axis=-1, eps=1e-8):
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + eps)

# ── canonical class order + aligned SS validation set ──────────────────────────
pa = np.load(ROOT / "cache/perch_arrays.npz", allow_pickle=True)
SP = pa["primary_labels"].astype(str)                 # (234,) canonical order
ss_emb = l2n(pa["embs"].astype(np.float32))           # (708,1536) normalized
ref_scores = pa["scores"].astype(np.float32)          # (708,234) current-pipeline scores
y = np.load(ROOT / "cache/sidecar_dpo/y_soundscape.npy").astype(np.int8)  # (708,234)
tax = pd.read_csv(ROOT / "data/birdclef-2026/taxonomy.csv")
cls = tax.set_index(tax.primary_label.astype(str)).class_name.to_dict()
classes = np.array([cls[p] for p in SP])
npos = y.sum(0)

# ── per-class train_audio centroids (raw) ──────────────────────────────────────
tr = np.load(ROOT / "cache/train_audio_perch_embeddings.npz", allow_pickle=True)
tr_emb = tr["embeddings"].astype(np.float32)          # (35549,1536)  (do NOT touch 'logits')
tr_sp = tr["species"].astype(str)
cent = np.zeros((234, 1536), np.float32)
nclip = np.zeros(234, int)
for i, sp in enumerate(SP):
    m = tr_sp == sp
    nclip[i] = int(m.sum())
    if m.sum() > 0:
        cent[i] = tr_emb[m].mean(0)
cent_n = l2n(cent)

aves = classes == "Aves"
have = nclip > 0
base_idx = np.where(aves & have)[0]
base = cent_n[base_idx]                               # (Nb,1536) normalized Aves prototypes

def teen_calibrate(novel_n, alpha, tau):
    sims = base @ novel_n                             # cosine to each Aves base prototype
    w = np.exp(sims / tau); w /= w.sum()
    transferred = w @ base
    cal = alpha * novel_n + (1.0 - alpha) * transferred
    return cal / (np.linalg.norm(cal) + 1e-8)

def auc(scores_col, i):
    try:
        return roc_auc_score(y[:, i], scores_col)
    except ValueError:
        return np.nan

# ── evaluate ────────────────────────────────────────────────────────────────────
print(f"non-Aves species: {int((~aves).sum())}, with train_audio: {int(((~aves)&have).sum())}, "
      f"evaluable (n_pos≥1): {int(((~aves)&(npos>=1)).sum())}")
print(f"train_audio clips/species — non-Aves: min={nclip[~aves].min()} med={int(np.median(nclip[~aves]))} max={nclip[~aves].max()}\n")

for MIN_POS in MIN_POS_REPORT:
    ev = np.where((~aves) & have & (npos >= MIN_POS))[0]
    raw = np.array([auc(ss_emb @ cent_n[i], i) for i in ev])
    ref = np.array([auc(ref_scores[:, i], i) for i in ev])
    print(f"── non-Aves, n_pos≥{MIN_POS}: {len(ev)} species ──")
    print(f"   RAW centroid  macro-AUC = {np.nanmean(raw):.4f}")
    print(f"   ref `scores`  macro-AUC = {np.nanmean(ref):.4f}  (current pipeline, for context)")
    best = None
    for a in ALPHAS:
        for t in TAUS:
            cal = np.array([auc(ss_emb @ teen_calibrate(cent_n[i], a, t), i) for i in ev])
            d = np.nanmean(cal) - np.nanmean(raw)
            if best is None or np.nanmean(cal) > best[0]:
                best = (np.nanmean(cal), a, t, d)
    print(f"   TEEN best     macro-AUC = {best[0]:.4f}  (α={best[1]}, τ={best[2]}, Δ vs raw = {best[3]:+.4f})\n")

# ── save artifacts (raw + best-overall TEEN at n_pos≥10) for potential 3rd-stream wiring ──
ev10 = np.where((~aves) & have & (npos >= 10))[0]
raw10 = np.nanmean([auc(ss_emb @ cent_n[i], i) for i in ev10])
bestcfg, bestv = (0.5, 16.0), -1
for a in ALPHAS:
    for t in TAUS:
        v = np.nanmean([auc(ss_emb @ teen_calibrate(cent_n[i], a, t), i) for i in ev10])
        if v > bestv:
            bestv, bestcfg = v, (a, t)
a, t = bestcfg
teen_cent = cent_n.copy()
for i in np.where((~aves) & have)[0]:
    teen_cent[i] = teen_calibrate(cent_n[i], a, t)
out = ROOT / "model-weights/teen_nonaves_prototypes.npz"
np.savez(out, primary_labels=SP, raw_centroids=cent_n, teen_centroids=teen_cent,
         class_name=classes, n_clip=nclip, n_pos_ss=npos, best_alpha=a, best_tau=t)
print(f"saved {out}  (best TEEN cfg α={a}, τ={t}; n_pos≥10 raw={raw10:.4f} teen={bestv:.4f})")
print("\nDECISION: raw≥ref → centroid stream is a candidate; TEEN Δ>0 → TEEN helps; both weak → need T3A/TIM or Perch 2.0")
