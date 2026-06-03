"""Validate SED window-offset TTA (genuinely-untried per audit L3 + blind-techniques #1) on the
controlled labeled-SS substrate. TTA = run Tucker 5-fold on the 5s clip at offsets {0, ±1.5s} and
average → captures boundary-straddling vocalizations (helps rare/blind species). NECESSARY-condition
check: (a) does it NOT degrade evaluable (Tucker 0.997)? (b) does it change predictions (rank-changing,
Pearson<1)? (c) per-window time cost (×3) for the 90-min CPU wall-time budget. Run: uv run python experiments/sed_tta_validate.py
"""
import numpy as np, pandas as pd, soundfile as sf, librosa, onnxruntime as ort, time, warnings
from pathlib import Path
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr
warnings.filterwarnings("ignore")
R = Path("/data/birdclef2026"); DATA = R / "data/birdclef-2026"; SSD = DATA / "train_soundscapes"
SR = 32000; WIN = 5 * SR
C = np.load(R / "cache/labeled_ss_controlled.npz", allow_pickle=True)
Y = C["y"]; LAB = [str(s) for s in C["labels"].tolist()]; tuck0 = C["tucker"]   # offset-0 (already computed)
files = [str(f) for f in C["files"].tolist()]; esec = C["end_sec"].astype(int)
cls = pd.read_csv(DATA / "taxonomy.csv").set_index(lambda i: None) if False else pd.read_csv(DATA / "taxonomy.csv")
CLS = cls.set_index(cls.primary_label.astype(str)).class_name.to_dict()
npos = Y.sum(0)

prov = ["CUDAExecutionProvider", "CPUExecutionProvider"]
seds = [ort.InferenceSession(p, providers=prov) for p in sorted((R / "model-weights/tucker_sed").glob("sed_fold*.onnx"))]
def mel_of(x):
    s = librosa.feature.melspectrogram(y=x, sr=SR, n_fft=2048, hop_length=512, n_mels=256, fmin=20, fmax=16000, power=2.0)
    s = librosa.power_to_db(s, top_db=80); s = (s - s.mean()) / (s.std() + 1e-6)
    return np.pad(s, ((0, 0), (0, max(0, 313 - s.shape[1]))))[:, :313].astype(np.float32)
sig = lambda z: 1 / (1 + np.exp(-z))
def sed_score(seg):
    m = mel_of(seg)[None, None]; acc = np.zeros(234, np.float32)
    for s in seds:
        c, f = s.run(None, {"mel": m})[:2]; acc += 0.5 * sig(c[0]) + 0.5 * sig(f.max(1)[0] if f.ndim == 3 else f[0])
    return acc / len(seds)

OFFS = [-1.5, 1.5]   # extra views (offset 0 already in tuck0); TTA = mean(tuck0, +views)
tta = tuck0.astype(np.float32).copy(); nview = np.ones((len(files), 1), np.float32)
cur_fn, cur_w = None, None; t0 = time.time(); nseg = 0
for idx in range(len(files)):
    fn, es = files[idx], int(esec[idx])
    if fn != cur_fn:
        w, _ = sf.read(str(SSD / fn), dtype="float32", always_2d=False); cur_w = w.mean(1) if w.ndim > 1 else w; cur_fn = fn
    for off in OFFS:
        st = int((es - 5 + off) * SR)
        if st < 0: continue
        seg = cur_w[st:st + WIN]
        if len(seg) < WIN: seg = np.pad(seg, (0, WIN - len(seg)))
        tta[idx] += sed_score(seg[:WIN].astype(np.float32)); nview[idx, 0] += 1; nseg += 1
tta /= nview
sec_per_view = (time.time() - t0) / max(nseg, 1)
print(f"computed {nseg} extra SED views, {sec_per_view*1000:.0f}ms/view (GPU); CPU ~{sec_per_view*1000*4:.0f}ms est")

def AUC(score, i):
    try: return roc_auc_score(Y[:, i], score)
    except ValueError: return np.nan
aves = np.array([CLS[s] == "Aves" for s in LAB])
ev = [i for i in range(234) if npos[i] >= 10]
evna = [i for i in ev if not aves[i]]
a0 = np.nanmean([AUC(tuck0[:, i], i) for i in ev]); at = np.nanmean([AUC(tta[:, i], i) for i in ev])
a0n = np.nanmean([AUC(tuck0[:, i], i) for i in evna]); atn = np.nanmean([AUC(tta[:, i], i) for i in evna])
pe = np.nanmean([pearsonr(tuck0[:, i], tta[:, i])[0] for i in ev])
print(f"\n=== SED window-TTA vs offset-0 (controlled, {len(ev)} evaluable / {len(evna)} non-Aves) ===")
print(f"all evaluable:  offset0={a0:.4f}  TTA={at:.4f}  Δ={at-a0:+.4f}")
print(f"non-Aves eval:  offset0={a0n:.4f}  TTA={atn:.4f}  Δ={atn-a0n:+.4f}")
print(f"pearson(offset0, TTA) over evaluable = {pe:.4f}  (<1 => rank-changing)")
print("NECESSARY CONDITION: TTA must NOT degrade evaluable (Δ≥~0). If Δ≥0 AND pearson<0.999 => safe rank-changing lever to submit blind.")
