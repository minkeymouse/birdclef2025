"""Extract Perch embeddings for data/external clips (the EXTRA Xeno-canto/iNat data, 1614 clips
by species — esp. all 35 Amphibia). Focal-style: first 5s window, same as perch_embed_extract.py.
Output: cache/external_perch_emb.npz (emb, species, files). Used to AUGMENT non-Aves centroids
beyond train.csv. GPU. Run after the unlabeled-SS extraction frees the GPU.
Run: uv run python experiments/emcent_external_aug.py
"""
import numpy as np, soundfile as sf, librosa, onnxruntime as ort
from pathlib import Path
import time, glob

R = Path("/data/birdclef2026"); EXT = R / "data/external"
SR = 32000; WIN = 5 * SR
prov = ["CUDAExecutionProvider", "CPUExecutionProvider"]
perch = ort.InferenceSession(str(R / "model-weights/perch_v2_onnx/perch_v2.onnx"), providers=prov)
pin = perch.get_inputs()[0].name
print(f"[init] perch prov={perch.get_providers()[0]}", flush=True)

clips = []
for d in sorted(EXT.iterdir()):
    if not d.is_dir() or d.name == "_logs":
        continue
    for f in sorted(list(d.glob("*.ogg")) + list(d.glob("*.mp3")) + list(d.glob("*.wav"))):
        clips.append((d.name, f))
print(f"[init] external clips: {len(clips)} across {len(set(c[0] for c in clips))} species", flush=True)

def load5s(fp):
    try:
        wav, sr = sf.read(str(fp), dtype="float32", always_2d=False)
        if wav.ndim > 1: wav = wav.mean(axis=1)
        if sr != SR: wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
    except Exception:
        wav, _ = librosa.load(str(fp), sr=SR, mono=True)
    if len(wav) < WIN: wav = np.pad(wav, (0, WIN - len(wav)))
    return wav[:WIN].astype(np.float32)

emb = np.zeros((len(clips), 1536), np.float32); sp = []; fn = []; ok = np.zeros(len(clips), bool)
t0 = time.time()
for i, (s, fp) in enumerate(clips):
    sp.append(s); fn.append(fp.name)
    try:
        emb[i] = perch.run(None, {pin: load5s(fp).reshape(1, -1)})[0][0]; ok[i] = True
    except Exception as e:
        print(f"[skip] {fp}: {e}", flush=True)
    if (i + 1) % 400 == 0:
        print(f"[{i+1}/{len(clips)}] {(time.time()-t0)/60:.1f}min", flush=True)
out = R / "cache/external_perch_emb.npz"
np.savez(out, emb=emb, species=np.array(sp), files=np.array(fn), ok=ok)
print(f"[done] {ok.sum()}/{len(clips)} ok, {(time.time()-t0)/60:.1f}min -> {out}", flush=True)
