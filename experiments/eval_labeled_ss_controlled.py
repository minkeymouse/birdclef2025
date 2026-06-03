"""Controlled, fully-aligned labeled-SS eval: run Perch + Tucker 5-fold directly on the labeled
soundscape windows (own window/label construction, no reliance on perch_arrays' unknown order).
Decisive test: does the in-domain centroid (from unlabeled SS) ADD over the production non-Aves
signal (Tucker SED)? Reports Tucker AUC, in-domain AUC, blend(Tucker+indom), and pearson(decorrelation).
GPU ~3min. Run: uv run python experiments/eval_labeled_ss_controlled.py
"""
import numpy as np, pandas as pd, soundfile as sf, librosa, onnxruntime as ort, time
from pathlib import Path
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata, pearsonr
R = Path("/data/birdclef2026"); DATA = R / "data/birdclef-2026"; SSD = DATA / "train_soundscapes"
SR = 32000; WIN = 5 * SR
LABELS = pd.read_csv(DATA / "sample_submission.csv").columns[1:].tolist(); L2I = {s: i for i, s in enumerate(LABELS)}
cls = pd.read_csv(DATA / "taxonomy.csv").set_index(lambda_df := "primary_label") if False else None
tax = pd.read_csv(DATA / "taxonomy.csv"); CLS = tax.set_index(tax.primary_label.astype(str)).class_name.to_dict()
nfocal = pd.read_csv(DATA / "train.csv").groupby(lambda i: None) if False else None
nf = pd.read_csv(DATA / "train.csv"); NFOCAL = nf.groupby(nf.primary_label.astype(str)).size().to_dict()

def t2s(t): p = t.split(":"); return int(p[0]) * 3600 + int(p[1]) * 60 + int(p[2])

df = pd.read_csv(DATA / "train_soundscapes_labels.csv"); df["end_sec"] = df["end"].apply(t2s)
win = {}
for _, r in df.iterrows():
    k = (r["filename"], int(r["end_sec"])); y = win.setdefault(k, np.zeros(234, np.float32))
    for t in str(r["primary_label"]).split(";"):
        t = t.strip()
        if t in L2I: y[L2I[t]] = 1.0
keys = sorted(win.keys()); Y = np.stack([win[k] for k in keys])
print(f"labeled windows: {len(keys)} from {len(set(k[0] for k in keys))} files | total positives={int(Y.sum())}")

prov = ["CUDAExecutionProvider", "CPUExecutionProvider"]
perch = ort.InferenceSession(str(R / "model-weights/perch_v2_onnx/perch_v2.onnx"), providers=prov); pin = perch.get_inputs()[0].name
seds = [ort.InferenceSession(p, providers=prov) for p in sorted((R / "model-weights/tucker_sed").glob("sed_fold*.onnx"))]
def mel_of(x):
    s = librosa.feature.melspectrogram(y=x, sr=SR, n_fft=2048, hop_length=512, n_mels=256, fmin=20, fmax=16000, power=2.0)
    s = librosa.power_to_db(s, top_db=80); s = (s - s.mean()) / (s.std() + 1e-6)
    return np.pad(s, ((0, 0), (0, max(0, 313 - s.shape[1]))))[:, :313].astype(np.float32)
sig = lambda z: 1 / (1 + np.exp(-z))
emb = np.zeros((len(keys), 1536), np.float32); tuck = np.zeros((len(keys), 234), np.float32)
cur_fn, cur_w = None, None; t0 = time.time()
for idx, (fn, es) in enumerate(keys):
    if fn != cur_fn:
        w, _ = sf.read(str(SSD / fn), dtype="float32", always_2d=False); cur_w = w.mean(1) if w.ndim > 1 else w; cur_fn = fn
    seg = cur_w[max(0, (es - 5) * SR):es * SR]; seg = np.pad(seg, (0, max(0, WIN - len(seg))))[:WIN].astype(np.float32)
    emb[idx] = perch.run(None, {pin: seg.reshape(1, -1)})[0][0]
    m = mel_of(seg)[None, None]; acc = np.zeros(234, np.float32)
    for s in seds:
        c, f = s.run(None, {"mel": m})[:2]; acc += 0.5 * sig(c[0]) + 0.5 * sig(f.max(1)[0] if f.ndim == 3 else f[0])
    tuck[idx] = acc / len(seds)
np.savez(R / "cache/labeled_ss_controlled.npz", emb=emb, tucker=tuck, y=Y,
         files=np.array([k[0] for k in keys]), end_sec=np.array([k[1] for k in keys]), labels=np.array(LABELS))
print(f"ran Perch+Tucker on {len(keys)} windows in {(time.time()-t0)/60:.1f}min")

# ---- compare in-domain centroid vs Tucker (the production non-Aves signal) ----
def l2n(x): return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)
embn = l2n(emb); npos = Y.sum(0)
U = np.load(R / "cache/unlabeled_ss_strided.npz", allow_pickle=True)
uemb = l2n(U["embs"].astype(np.float32)); usc = U["scores"].astype(np.float32); ulab = [str(s) for s in U["labels"].tolist()]
sed_col = {s: j for j, s in enumerate(ulab)}
def AUC(sc, i):
    try: return roc_auc_score(Y[:, i], sc)
    except ValueError: return np.nan
def rp(a): a = np.asarray(a, float); return rankdata(a) / len(a)
aves = np.array([CLS[s] == "Aves" for s in LABELS])
ev = [i for i in range(234) if (not aves[i]) and npos[i] >= 10 and LABELS[i] in sed_col]
def indom(i, K=100):
    j = sed_col[LABELS[i]]; order = np.argsort(-usc[:, j])[:K]; return l2n(uemb[order].mean(0))
a_tuck = [AUC(tuck[:, i], i) for i in ev]; a_ind = [AUC(embn @ indom(i), i) for i in ev]
pe = [pearsonr(tuck[:, i], embn @ indom(i))[0] for i in ev]
print(f"\n=== non-Aves n_pos>=10: {len(ev)} species (CONTROLLED, aligned) ===")
print(f"Tucker SED (production non-Aves signal) AUC = {np.nanmean(a_tuck):.4f}")
print(f"in-domain centroid AUC                      = {np.nanmean(a_ind):.4f}")
print(f"pearson(Tucker, in-domain)                  = {np.nanmean(pe):.3f}  (low => decorrelated => can add)")
for w in (0.2, 0.3, 0.5):
    bl = [AUC((1 - w) * rp(tuck[:, i]) + w * rp(embn @ indom(i)), i) for i in ev]
    imp = sum(1 for k, i in enumerate(ev) if bl[k] > a_tuck[k])
    print(f"  blend Tucker+indom w={w}: AUC={np.nanmean(bl):.4f} (Δ vs Tucker {np.nanmean(bl)-np.nanmean(a_tuck):+.4f}, improved {imp}/{len(ev)})")
print("\nDECISION: if blend > Tucker AND pearson low => in-domain adds genuine signal over production Tucker => real LB lever")
