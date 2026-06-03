"""NS lever step 1: pseudo-label the ~10.6k UNLABELED 2026 soundscapes (multi-site, same region
as test) with the Tucker SED 5-fold teacher. Produces soundscape-aware, site-unbiased soft targets
for distilling BirdSet-ConvNeXt (external-pretrained) -> avoids the focal domain gap AND site-22 trap.
Output: cache/ns_soundscape_softlabels.npz (filenames, end_sec, soft[N,234]).
"""
import numpy as np, pandas as pd, soundfile as sf, librosa, onnxruntime as ort
from pathlib import Path
import time, glob

ROOT = Path("/data/birdclef2026"); DATA = ROOT/"data/birdclef-2026"
SS_DIR = DATA/"train_soundscapes"
LABELED = set(pd.read_csv(DATA/"train_soundscapes_labels.csv")["filename"].astype(str))
PRIMARY = pd.read_csv(DATA/"taxonomy.csv")["primary_label"].astype(str).tolist()
NC = len(PRIMARY)
SR=32000; WIN=5*SR; NW=12
# Tucker SED mel config (matches phase4 cell 11 audio_to_mel)
N_FFT=2048; HOP=512; N_MELS=256; FMIN=20; FMAX=16000; TOP_DB=80

folds = sorted(glob.glob(str(ROOT/"model-weights/tucker_sed/sed_fold*.onnx")))
so = ort.SessionOptions(); so.intra_op_num_threads=4
sess = [ort.InferenceSession(f, sess_options=so, providers=["CUDAExecutionProvider","CPUExecutionProvider"]) for f in folds]
print(f"teacher folds: {len(sess)} | provider: {sess[0].get_providers()[0]}", flush=True)

def sig(x): return (1.0/(1.0+np.exp(-np.clip(x,-50,50)))).astype(np.float32)
def audio_to_mel(chunks):
    out=[]
    for x in chunks:
        s=librosa.feature.melspectrogram(y=x,sr=SR,n_fft=N_FFT,hop_length=HOP,n_mels=N_MELS,fmin=FMIN,fmax=FMAX,power=2.0)
        s=librosa.power_to_db(s,top_db=TOP_DB); s=(s-s.mean())/(s.std()+1e-6); out.append(s)
    return np.stack(out)[:,None].astype(np.float32)

# unlabeled pool (exclude the 66 labeled files to keep eval clean)
files=[p for p in sorted(SS_DIR.glob("*.ogg")) if p.name not in LABELED]
print(f"unlabeled soundscapes to pseudo-label: {len(files)}", flush=True)

fns=[]; ends=[]; soft=[]
t0=time.time()
for i,fp in enumerate(files):
    try:
        y,sr0=sf.read(fp,dtype="float32"); y=y.mean(1) if y.ndim>1 else y
        if sr0!=SR: y=librosa.resample(y,orig_sr=sr0,target_sr=SR)
        n=60*SR; y=np.pad(y,(0,max(0,n-len(y))))[:n]
        chunks=y.reshape(NW,WIN)
        mel=audio_to_mel(chunks)
        psum=np.zeros((NW,NC),dtype=np.float32)
        for s in sess:
            o=s.run(None,{s.get_inputs()[0].name:mel})
            psum+=0.5*sig(o[0])+0.5*sig(o[1].max(axis=1))
        p=psum/len(sess)
        stem=fp.stem
        for w in range(NW):
            fns.append(stem); ends.append((w+1)*5); soft.append(p[w])
    except Exception as e:
        print(f"err {fp.name}: {e}", flush=True)
    if (i+1)%500==0:
        el=time.time()-t0; eta=el/(i+1)*(len(files)-i-1)
        print(f"{i+1}/{len(files)} ({el:.0f}s, ETA {eta/60:.0f}min)", flush=True)

soft=np.stack(soft).astype(np.float16)
out=ROOT/"cache/ns_soundscape_softlabels.npz"
np.savez_compressed(out, filenames=np.array(fns), end_sec=np.array(ends,dtype=np.int16), soft=soft, labels=np.array(PRIMARY))
print(f"saved {len(soft)} windows -> {out} ({time.time()-t0:.0f}s)", flush=True)
# quick sanity: teacher confidence distribution
print(f"soft: mean max-per-window={soft.max(axis=1).mean():.3f}, frac windows with max>0.5={ (soft.max(axis=1)>0.5).mean():.3f}", flush=True)
