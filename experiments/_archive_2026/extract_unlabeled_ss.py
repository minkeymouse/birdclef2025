"""Extract Perch emb + Tucker SED soft-scores for UNLABELED 2026 soundscapes (strided 6 win/file).
Excludes the 66 labeled files (no leakage into the site-safe eval). Reusable product for:
(a) in-domain non-Aves centroid (emcent refinement), (b) distillation pseudo-labels.
GPU (CUDA ONNX). Checkpoints every 500 files. Output: cache/unlabeled_ss_strided.npz
Run (background): nohup setsid uv run python -u experiments/extract_unlabeled_ss.py > LOG 2>&1 </dev/null & disown
"""
import numpy as np, pandas as pd, soundfile as sf, librosa, onnxruntime as ort
from pathlib import Path
import time, glob, sys

R = Path("/data/birdclef2026"); DATA = R / "data/birdclef-2026"; SSDIR = DATA / "train_soundscapes"
OUT = R / "cache/unlabeled_ss_strided.npz"
SR = 32000; WIN = 5 * SR; END_SECS = [5, 15, 25, 35, 45, 55]   # 6 strided 5s windows
N_FFT, HOP, N_MELS, FMIN, FMAX, TOP_DB = 2048, 512, 256, 20, 16000, 80   # Tucker verbatim

prov = ["CUDAExecutionProvider", "CPUExecutionProvider"]
perch = ort.InferenceSession(str(R / "model-weights/perch_v2_onnx/perch_v2.onnx"), providers=prov)
p_in = perch.get_inputs()[0].name
sed_paths = sorted(glob.glob(str(R / "model-weights/tucker_sed/sed_fold*.onnx")))
seds = [ort.InferenceSession(p, providers=prov) for p in sed_paths]
print(f"[init] perch in={p_in} prov={perch.get_providers()[0]} | sed folds={len(seds)}", flush=True)
LABELS = pd.read_csv(DATA / "sample_submission.csv").columns[1:].tolist()      # SED/output order (234)

labeled = set(pd.read_csv(DATA / "train_soundscapes_labels.csv")["filename"].astype(str))
files = sorted(p.name for p in SSDIR.glob("*.ogg") if p.name not in labeled)
print(f"[init] unlabeled SS files: {len(files)} (excluded {len(labeled)} labeled)", flush=True)

def mel_of(x):
    s = librosa.feature.melspectrogram(y=x, sr=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS,
                                       fmin=FMIN, fmax=FMAX, power=2.0)
    s = librosa.power_to_db(s, top_db=TOP_DB)
    s = (s - s.mean()) / (s.std() + 1e-6)
    if s.shape[1] < 313: s = np.pad(s, ((0, 0), (0, 313 - s.shape[1])))
    return s[:, :313].astype(np.float32)

def sigmoid(z): return 1.0 / (1.0 + np.exp(-z))

embs, scores, fidx, esec = [], [], [], []
t0 = time.time(); done = 0
for fi, fn in enumerate(files):
    try:
        wav, _ = sf.read(str(SSDIR / fn), dtype="float32", always_2d=False)
        if wav.ndim > 1: wav = wav.mean(axis=1)
    except Exception as e:
        print(f"[skip] {fn}: {e}", flush=True); continue
    wins = []
    for end in END_SECS:
        seg = wav[max(0, (end - 5) * SR):end * SR]
        if len(seg) < WIN: seg = np.pad(seg, (0, WIN - len(seg)))
        wins.append(seg[:WIN].astype(np.float32))
    W = np.stack(wins)                                              # (6,160000)
    # Perch emb (loop windows; perch onnx is single-sample)
    e6 = np.stack([perch.run(None, {p_in: w.reshape(1, -1)})[0][0] for w in W]).astype(np.float32)
    # Tucker SED 5-fold mean of 0.5*sig(clip)+0.5*sig(framemax)
    mels = np.stack([mel_of(w) for w in W])[:, None, :, :]         # (6,1,256,313)
    acc = np.zeros((6, 234), np.float32)
    for s in seds:
        clip, frame = s.run(None, {"mel": mels})[:2]
        fmax = frame.max(axis=1) if frame.ndim == 3 else frame
        acc += 0.5 * sigmoid(clip) + 0.5 * sigmoid(fmax)
    acc /= len(seds)
    embs.append(e6); scores.append(acc.astype(np.float32))
    fidx.extend([fi] * 6); esec.extend(END_SECS)
    done += 1
    if done % 500 == 0:
        el = time.time() - t0; eta = el / done * (len(files) - done)
        print(f"[{done}/{len(files)}] {el/60:.1f}min elapsed, ETA {eta/60:.1f}min", flush=True)
        np.savez(OUT, embs=np.concatenate(embs), scores=np.concatenate(scores),
                 file_idx=np.array(fidx), end_sec=np.array(esec),
                 files=np.array(files), labels=np.array(LABELS), partial=done)
np.savez(OUT, embs=np.concatenate(embs), scores=np.concatenate(scores),
         file_idx=np.array(fidx), end_sec=np.array(esec),
         files=np.array(files), labels=np.array(LABELS), partial=done)
print(f"[done] {done} files, {len(fidx)} windows, {(time.time()-t0)/60:.1f}min -> {OUT}", flush=True)
