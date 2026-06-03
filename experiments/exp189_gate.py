"""Gate exp189 (Tucker recipe + external non-Aves data) on the controlled labeled-SS substrate.
KEY question: does the external non-Aves data make exp189 CLOSE/EXCEED Tucker on evaluable non-Aves
(exp187 was only 0.976 vs Tucker 0.998)? If exp189 holds Aves AND lifts/holds non-Aves, the external
data helped -> blend is positive-EV (esp. for the blind non-Aves). N-fold ensemble of available folds.
Run: uv run python experiments/exp189_gate.py
"""
import numpy as np, pandas as pd, soundfile as sf, torch, glob, warnings
from pathlib import Path
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata, pearsonr
import sys; sys.path.insert(0, ".")
from experiments.sed.model import DistilledSED
from experiments.sed.config import exp189_tucker_external_nonaves
warnings.filterwarnings("ignore")
R = Path("/data/birdclef2026"); SSD = R / "data/birdclef-2026/train_soundscapes"; SR = 32000; WIN = 5 * SR
cfg = exp189_tucker_external_nonaves()
cks = sorted(glob.glob(str(R / "experiments/_data_pipelines/exp189_outputs/**/fold*/best_ckpt.pt"), recursive=True))
# optional: restrict to specific folds (avoid including a still-training partial fold) e.g. "0,1,2"
if len(sys.argv) > 1:
    _keep = set(sys.argv[1].split(","))
    cks = [c for c in cks if c.split("/")[-2].replace("fold", "") in _keep]
print(f"ensemble folds: {len(cks)} -> {[c.split('/')[-2] for c in cks]}")
if not cks: print("no checkpoints yet"); sys.exit(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"
models = []
for c in cks:
    m = DistilledSED(cfg, 234); d = torch.load(c, map_location="cpu", weights_only=False)
    m.load_state_dict(d.get("model_state_dict", d.get("state_dict", d)), strict=False); models.append(m.eval().to(dev))
C = np.load(R / "cache/labeled_ss_controlled.npz", allow_pickle=True)
Y = C["y"]; tuck = C["tucker"]; files = [str(f) for f in C["files"].tolist()]; esec = C["end_sec"].astype(int)
LAB = [str(s) for s in C["labels"].tolist()]; npos = Y.sum(0)
cls = pd.read_csv(R / "data/birdclef-2026/taxonomy.csv"); CLS = cls.set_index(cls.primary_label.astype(str)).class_name.to_dict()
aves = np.array([CLS[s] == "Aves" for s in LAB]); sig = lambda z: 1 / (1 + np.exp(-z))
ens = np.zeros((len(files), 234), np.float32); cur = None
with torch.no_grad():
    for i, (fn, es) in enumerate(zip(files, esec)):
        if fn != cur:
            w, _ = sf.read(str(SSD / fn), dtype="float32", always_2d=False); w = w.mean(1) if w.ndim > 1 else w; cur = fn
        seg = w[max(0, (es - 5) * SR):es * SR]; seg = np.pad(seg, (0, max(0, WIN - len(seg))))[:WIN]
        x = torch.tensor(seg, dtype=torch.float32, device=dev).unsqueeze(0)
        acc = np.zeros(234, np.float32)
        for m in models:
            o = m(x); fr = o[1]; fm = fr.max(dim=2).values if fr.ndim == 3 else fr
            acc += 0.5 * sig(o[0][0].cpu().numpy()) + 0.5 * sig(fm[0].cpu().numpy())
        ens[i] = acc / len(models)
def A(s, i):
    try: return roc_auc_score(Y[:, i], s)
    except ValueError: return np.nan
def rp(a): a = np.asarray(a, float); return rankdata(a) / len(a)
for grp, mask in [("all-eval", np.ones(234, bool)), ("non-Aves", ~aves), ("Aves", aves)]:
    ev = [i for i in range(234) if npos[i] >= 10 and mask[i]]
    if not ev: continue
    at = np.nanmean([A(tuck[:, i], i) for i in ev]); ae = np.nanmean([A(ens[:, i], i) for i in ev])
    pe = np.nanmean([pearsonr(tuck[:, i], ens[:, i])[0] for i in ev])
    bl = {w: np.nanmean([A((1 - w) * rp(tuck[:, i]) + w * rp(ens[:, i]), i) for i in ev]) for w in (0.10, 0.20, 0.30)}
    print(f"{grp} ({len(ev)}sp): Tucker={at:.4f} exp189={ae:.4f} pearson={pe:.3f} | " +
          " ".join(f"bl{w}={bl[w]:.4f}(Δ{bl[w]-at:+.4f})" for w in (0.10, 0.20, 0.30)))
print("Decision: if exp189 non-Aves >= Tucker (external data helped) AND Aves not-dragging -> blend non-Aves-safe.")
print("  (exp187 ref: non-Aves exp187=0.976 < Tucker 0.998 -> dragged. exp189 should be CLOSER if data helps.)")
