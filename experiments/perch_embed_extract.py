"""Extract Perch embeddings for all train_audio clips."""
import onnxruntime as ort
import numpy as np
import soundfile as sf
from pathlib import Path
import pandas as pd
import time

sess = ort.InferenceSession(
    '/data/birdclef2026/model-weights/perch_v2_onnx/perch_v2.onnx',
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
print(f"Providers: {sess.get_providers()}")

AUDIO_DIR = Path("/data/birdclef2026/data/birdclef-2026/train_audio")
SR = 32000
WIN = 160000

train_df = pd.read_csv("/data/birdclef2026/data/birdclef-2026/train.csv")
files = train_df['filename'].tolist()
species = train_df['primary_label'].astype(str).tolist()
N = len(files)

embeddings = np.zeros((N, 1536), dtype=np.float32)
logits_234 = np.zeros((N, 14795), dtype=np.float32)
valid = np.zeros(N, dtype=bool)

t0 = time.time()
for i, fn in enumerate(files):
    fp = AUDIO_DIR / fn
    if not fp.exists():
        continue
    try:
        a, sr = sf.read(fp)
        if len(a.shape) > 1:
            a = a.mean(axis=1)
        if len(a) < WIN:
            a = np.pad(a, (0, WIN - len(a)))
        a = a[:WIN].astype(np.float32)
        out = sess.run(None, {'inputs': a.reshape(1, -1)})
        embeddings[i] = out[0][0]    # embedding (1536,)
        logits_234[i] = out[3][0]    # label (14795,)
        valid[i] = True
    except Exception as e:
        print(f"Error {fn}: {e}")

    if (i + 1) % 1000 == 0:
        elapsed = time.time() - t0
        eta = elapsed / (i + 1) * (N - i - 1)
        print(f"{i+1}/{N} ({elapsed:.0f}s, ETA {eta/60:.0f}min)", flush=True)

out_path = '/data/birdclef2026/cache/train_audio_perch_embeddings.npz'
np.savez_compressed(
    out_path,
    embeddings=embeddings[valid],
    logits=logits_234[valid],
    filenames=np.array(files)[valid],
    species=np.array(species)[valid],
)
print(f"Saved {valid.sum()} embeddings to {out_path}")
print(f"Total time: {(time.time()-t0)/60:.1f} min")
