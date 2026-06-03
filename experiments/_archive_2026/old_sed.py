"""Reconstruct the OLD bird-SED class (mel→bn0→timm backbone→AttBlockV2 head) to load the diverse
teachers exp50 (HGNet) / exp59 (ConvNeXt) for heterogeneous-teacher distillation. Validated by
matching the recorded val_SS (exp59=0.859, exp50=0.838) on the labeled-SS controlled set.
Run: uv run python experiments/old_sed.py   (validates exp59 + exp50)
"""
import numpy as np, pandas as pd, soundfile as sf, torch, torch.nn as nn, torchaudio, timm, warnings
from pathlib import Path
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")
R = Path("/data/birdclef2026"); DATA = R / "data/birdclef-2026"; SSD = DATA / "train_soundscapes"
SR = 32000; WIN = 5 * SR

class MelWrap(nn.Module):
    def __init__(self, n_fft=2048, hop=512, n_mels=128, fmin=20, fmax=16000):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(SR, n_fft=n_fft, hop_length=hop, n_mels=n_mels,
                                                         f_min=fmin, f_max=fmax, power=2.0)
        self.db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)
    def forward(self, x):
        return self.db(self.mel(x))   # (B, n_mels, T)

class AttBlockV2(nn.Module):
    def __init__(self, inf, ncls):
        super().__init__()
        self.att = nn.Conv1d(inf, ncls, 1); self.cla = nn.Conv1d(inf, ncls, 1)
    def forward(self, x):  # (B,C,T)
        na = torch.softmax(torch.tanh(self.att(x)), dim=-1)
        cla = self.cla(x)
        clip = torch.sum(na * torch.sigmoid(cla), dim=2)   # clip prob
        return clip, cla

class OldSED(nn.Module):
    def __init__(self, backbone, n_mels=128, ncls=234, hop=512, fmin=20, fmax=16000):
        super().__init__()
        self.mel = MelWrap(hop=hop, n_mels=n_mels, fmin=fmin, fmax=fmax)
        self.bn0 = nn.BatchNorm2d(n_mels)
        self.backbone = timm.create_model(backbone, pretrained=False, in_chans=1, num_classes=0, global_pool="")
        self.head = AttBlockV2(self.backbone.num_features, ncls)
    def forward(self, x):
        x = self.mel(x).unsqueeze(1)          # (B,1,n_mels,T)
        x = x.transpose(1, 2)                  # (B,n_mels,1,T)
        x = self.bn0(x).transpose(1, 2)        # (B,1,n_mels,T)
        x = self.backbone.forward_features(x)  # (B,C,H,W)
        x = x.mean(dim=2)                      # pool freq -> (B,C,T')
        return self.head(x)                    # clip(B,234), frame(B,234,T')

def load_old(name, backbone, **kw):
    d = torch.load(R / f"model-weights/{name}", map_location="cpu", weights_only=False)
    m = OldSED(backbone, **kw); miss = m.load_state_dict(d["state_dict"], strict=False)
    print(f"  {name}: missing={len(miss.missing_keys)} unexpected={len(miss.unexpected_keys)} (recorded val_SS={d.get('val_SS'):.4f})")
    return m.eval()

if __name__ == "__main__":
    # build labeled-SS windows + y (same as controlled eval)
    LAB = pd.read_csv(DATA / "sample_submission.csv").columns[1:].tolist(); L2I = {s: i for i, s in enumerate(LAB)}
    def t2s(t): p = t.split(":"); return int(p[0]) * 3600 + int(p[1]) * 60 + int(p[2])
    df = pd.read_csv(DATA / "train_soundscapes_labels.csv"); df["es"] = df["end"].apply(t2s)
    win = {}
    for _, r in df.iterrows():
        y = win.setdefault((r["filename"], int(r["es"])), np.zeros(234, np.float32))
        for t in str(r["primary_label"]).split(";"):
            if t.strip() in L2I: y[L2I[t.strip()]] = 1.0
    keys = sorted(win); Y = np.stack([win[k] for k in keys]); npos = Y.sum(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for name, bb in [("exp59_convnext_sed.pt", "convnext_tiny.fb_in22k_ft_in1k"),
                     ("exp50_hgnet_sed.pt", "hgnetv2_b0.ssld_stage2_ft_in1k")]:
        m = load_old(name, bb).to(dev)
        preds = np.zeros((len(keys), 234), np.float32); cur = None
        with torch.no_grad():
            for i, (fn, es) in enumerate(keys):
                if fn != cur:
                    w, _ = sf.read(str(SSD / fn), dtype="float32", always_2d=False); w = w.mean(1) if w.ndim > 1 else w; cur = fn
                seg = w[max(0, (es - 5) * SR):es * SR]; seg = np.pad(seg, (0, max(0, WIN - len(seg))))[:WIN]
                clip, _ = m(torch.tensor(seg, dtype=torch.float32, device=dev).unsqueeze(0))
                preds[i] = clip[0].cpu().numpy()
        ev = [i for i in range(234) if npos[i] >= 10]
        auc = np.nanmean([roc_auc_score(Y[:, i], preds[:, i]) for i in ev if len(set(Y[:, i])) > 1])
        print(f"  -> {name} labeled-SS macro-AUC (n_pos>=10, {len(ev)}sp) = {auc:.4f}  [target ≈ recorded val_SS]")
