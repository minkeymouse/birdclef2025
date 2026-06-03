"""GATE check for the exp187 distillation smoke test (run AFTER fold-0 trains). Loads the fold-0
DistilledSED student, runs it on the 739 labeled-SS windows, and decides whether it's worth an LB slot:
  GATE PASS  iff  Pearson(student, Tucker) < 0.93 (genuinely decorrelated)  AND
                  rank-blend(Tucker, student) AUC on evaluable ≥ Tucker (does NOT drag).
If it drags evaluable (like exp59 did), it will drag LB → FAIL → do not submit.
Run: uv run python experiments/gate_exp187.py
"""
import numpy as np, pandas as pd, soundfile as sf, torch, glob, warnings
from pathlib import Path
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata, pearsonr
import sys; sys.path.insert(0, ".")
from experiments.sed.model import DistilledSED
from experiments.sed.config import exp187_hetero_teacher_pseudo
warnings.filterwarnings("ignore")
R = Path("/data/birdclef2026"); SSD = R / "data/birdclef-2026/train_soundscapes"; SR = 32000; WIN = 5 * SR

cks = sorted(glob.glob(str(R / "experiments/_data_pipelines/exp187_outputs/**/*fold0*.pt"), recursive=True)) \
    + sorted(glob.glob(str(R / "experiments/_data_pipelines/exp187_outputs/**/best*.pt"), recursive=True))
if not cks:
    print("NO exp187 fold-0 checkpoint found yet — train must finish first."); sys.exit(0)
ckpt = cks[0]; print(f"checkpoint: {ckpt}")
cfg = exp187_hetero_teacher_pseudo()
d = torch.load(ckpt, map_location="cpu", weights_only=False)
sd = d.get("model_state_dict", d.get("state_dict", d))
val_ss = d.get("val_ss", d.get("non_s22_macro_auc", d.get("best_metric", "?")))
print(f"recorded training val (non_s22) = {val_ss}")
dev = "cuda" if torch.cuda.is_available() else "cpu"
m = DistilledSED(cfg, 234); m.load_state_dict(sd, strict=False); m.eval().to(dev)

C = np.load(R / "cache/labeled_ss_controlled.npz", allow_pickle=True)
Y = C["y"]; tuck = C["tucker"]; files = [str(f) for f in C["files"].tolist()]; esec = C["end_sec"].astype(int)
LAB = [str(s) for s in C["labels"].tolist()]; npos = Y.sum(0)
cls = pd.read_csv(R / "data/birdclef-2026/taxonomy.csv"); CLS = cls.set_index(cls.primary_label.astype(str)).class_name.to_dict()
aves = np.array([CLS[s] == "Aves" for s in LAB])
sig = lambda z: 1 / (1 + np.exp(-z))
stu = np.zeros((len(files), 234), np.float32); cur = None
with torch.no_grad():
    for i, (fn, es) in enumerate(zip(files, esec)):
        if fn != cur:
            w, _ = sf.read(str(SSD / fn), dtype="float32", always_2d=False); w = w.mean(1) if w.ndim > 1 else w; cur = fn
        seg = w[max(0, (es - 5) * SR):es * SR]; seg = np.pad(seg, (0, max(0, WIN - len(seg))))[:WIN]
        out = m(torch.tensor(seg, dtype=torch.float32, device=dev).unsqueeze(0))
        clip, frame = out[0], out[1]
        fm = frame.max(dim=2).values if frame.ndim == 3 else frame
        stu[i] = (0.5 * sig(clip[0].cpu().numpy()) + 0.5 * sig(fm[0].cpu().numpy()))
def A(s, i):
    try: return roc_auc_score(Y[:, i], s)
    except ValueError: return np.nan
def rp(a): a = np.asarray(a, float); return rankdata(a) / len(a)
for grp, mask in [("all-eval", np.ones(234, bool)), ("non-Aves", ~aves)]:
    ev = [i for i in range(234) if npos[i] >= 10 and mask[i]]
    at = np.nanmean([A(tuck[:, i], i) for i in ev]); asu = np.nanmean([A(stu[:, i], i) for i in ev])
    pe = np.nanmean([pearsonr(tuck[:, i], stu[:, i])[0] for i in ev])
    bl = {w: np.nanmean([A((1 - w) * rp(tuck[:, i]) + w * rp(stu[:, i]), i) for i in ev]) for w in (0.2, 0.3)}
    print(f"{grp} ({len(ev)}sp): Tucker={at:.4f} student={asu:.4f} pearson={pe:.3f} | blend.2={bl[0.2]:.4f}(Δ{bl[0.2]-at:+.4f}) blend.3={bl[0.3]:.4f}(Δ{bl[0.3]-at:+.4f})")
    if grp == "all-eval":
        gate = (pe < 0.93) and (bl[0.2] >= at - 0.0005)
        print(f"  >>> GATE {'PASS' if gate else 'FAIL'} (pearson<0.93: {pe<0.93}; blend not-dragging: {bl[0.2]>=at-0.0005})")
print("PASS → export 5-fold + integrate + submit (select-best). FAIL → abort, protect anchor.")
