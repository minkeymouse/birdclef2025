"""Build HETEROGENEOUS-teacher pseudo for the distillation smoke test: run reconstructed exp59
(ConvNeXt, Pearson 0.70 vs Tucker — genuinely diverse) on the 63,552 unlabeled-SS windows, ensemble
with the cached Tucker 5-fold scores (0.65 Tucker + 0.35 exp59), filter + soft-trim, save in
PseudoDataset format. This is the diversity the prior B0-correlated pseudo (v11/v12/v62) lacked.
GPU ~40min. Run (bg): nohup setsid uv run python -u experiments/build_pseudo_hetero.py > LOG 2>&1 </dev/null & disown
"""
import numpy as np, soundfile as sf, torch, time, warnings
from pathlib import Path
import sys; sys.path.insert(0, "experiments")
from old_sed import load_old
warnings.filterwarnings("ignore")
R = Path("/data/birdclef2026"); SSD = R / "data/birdclef-2026/train_soundscapes"
SR = 32000; WIN = 5 * SR; W_TUCKER = 0.65; W_EXP59 = 0.35; TRIM = 0.1; KEEP = 40000

U = np.load(R / "cache/unlabeled_ss_strided.npz", allow_pickle=True)
tuck = U["scores"].astype(np.float32)            # (63552,234) Tucker 5-fold mean
fidx = U["file_idx"].astype(np.int32); esec = U["end_sec"].astype(np.int32); files = U["files"]
N = len(fidx); print(f"[init] {N} windows, {len(files)} files", flush=True)

dev = "cuda" if torch.cuda.is_available() else "cpu"
m = load_old("exp59_convnext_sed.pt", "convnext_tiny.fb_in22k_ft_in1k").to(dev)
ex = np.zeros((N, 234), np.float32)
# group rows by file for single audio load
from collections import defaultdict
byfile = defaultdict(list)
for r in range(N): byfile[int(fidx[r])].append(r)
t0 = time.time(); done = 0
with torch.no_grad():
    for fi, rows in byfile.items():
        try:
            w, _ = sf.read(str(SSD / str(files[fi])), dtype="float32", always_2d=False)
            w = w.mean(1) if w.ndim > 1 else w
        except Exception as e:
            print(f"[skip] {files[fi]}: {e}", flush=True); continue
        segs = []
        for r in rows:
            es = int(esec[r]); seg = w[max(0, (es - 5) * SR):es * SR]
            segs.append(np.pad(seg, (0, max(0, WIN - len(seg))))[:WIN].astype(np.float32))
        x = torch.tensor(np.stack(segs), dtype=torch.float32, device=dev)
        clip, _ = m(x)                              # (n,234) clip prob
        for j, r in enumerate(rows): ex[r] = clip[j].cpu().numpy()
        done += 1
        if done % 1000 == 0:
            el = time.time() - t0; print(f"[{done}/{len(byfile)}] {el/60:.1f}min ETA {el/done*(len(byfile)-done)/60:.1f}min", flush=True)

ens = W_TUCKER * tuck + W_EXP59 * ex
mx = ens.max(1)
thr = float(np.percentile(mx, 100 * (1 - KEEP / N)))
keep = mx >= thr
probs = ens[keep].copy(); probs[probs < TRIM] = 0.0
out = R / "experiments/sed/pseudo_hetero_teacher.npz"
np.savez(out, files=files, kept_file_idx=fidx[keep], kept_end_secs=esec[keep],
         kept_probs=probs.astype(np.float32), confidence_weights=probs.sum(1).astype(np.float32))
# quick decorrelation report on the pseudo windows
from scipy.stats import pearsonr
pe = np.nanmean([pearsonr(tuck[keep][:, c], ex[keep][:, c])[0] for c in range(234)
                 if tuck[keep][:, c].std() > 1e-6 and ex[keep][:, c].std() > 1e-6])
print(f"[done] exp59 inference {(time.time()-t0)/60:.1f}min | THR={thr:.3f} kept {keep.sum()}/{N} | "
      f"mean per-class Pearson(Tucker,exp59) on kept = {pe:.3f} | -> {out}", flush=True)
