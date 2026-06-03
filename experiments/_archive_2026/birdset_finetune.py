"""Multi-day lever: fine-tune BirdSet-pretrained ConvNeXt (external Xeno-Canto-Large pretrain,
9736 eBird classes) on BirdCLEF-2026 234 species, for a DECORRELATED comparable SED to
recipe-pair-blend with Tucker. Honest risk: focal->soundscape domain gap + self-trained-weaker.
Fail-fast: per-epoch macro-AUC on labeled soundscapes; only worth integrating if comparable to Tucker.

Run: nohup setsid uv run python -u experiments/birdset_finetune.py > LOG 2>&1 < /dev/null & disown
"""
import os, time, math, json
import numpy as np, pandas as pd
import torch, torch.nn as nn
import torchaudio
import soundfile as sf
from pathlib import Path
from sklearn.metrics import roc_auc_score

ROOT = Path("/data/birdclef2026"); DATA = ROOT / "data/birdclef-2026"
DEV = "cuda"
SR = 32000; WIN = 5 * SR; IMG = 224
N_MELS = 224; N_FFT = 2048; HOP = 716  # ~224 frames over 5s (160000/716≈223)
EPOCHS = 6; BS = 48; LR = 1.5e-4
OUT = ROOT / "model-weights/birdset_ft"; OUT.mkdir(exist_ok=True)
torch.backends.cudnn.benchmark = True

tax = pd.read_csv(DATA / "taxonomy.csv")
LABELS = tax["primary_label"].astype(str).tolist()
L2I = {l: i for i, l in enumerate(LABELS)}; NC = len(LABELS)
cls_name = tax.set_index(tax["primary_label"].astype(str))["class_name"].to_dict()

# ---- mel ----
melspec = torchaudio.transforms.MelSpectrogram(
    sample_rate=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS, f_min=20, f_max=16000, power=2.0).to(DEV)
a2db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80).to(DEV)
_resize = torchaudio.transforms.Resize if False else None
import torch.nn.functional as F
def wav_to_img(wav):  # wav: (B, WIN) on DEV -> (B,1,224,224) normalized
    m = a2db(melspec(wav))                      # (B, n_mels, T)
    m = F.interpolate(m.unsqueeze(1), size=(IMG, IMG), mode="bilinear", align_corners=False)
    mn = m.amin(dim=(2,3), keepdim=True); mx = m.amax(dim=(2,3), keepdim=True)
    return (m - mn) / (mx - mn + 1e-6)

# ---- model: load BirdSet ConvNeXt, replace head 9736->234 ----
from transformers import ConvNextForImageClassification
print("loading BirdSet-ConvNeXt...", flush=True)
model = ConvNextForImageClassification.from_pretrained(
    "/tmp/birdset", num_labels=NC, ignore_mismatched_sizes=True)
model.to(DEV)
print("loaded; param M:", sum(p.numel() for p in model.parameters())/1e6, flush=True)

# ---- train data: train.csv focal clips ----
train = pd.read_csv(DATA / "train.csv")
train = train[train["primary_label"].astype(str).isin(L2I)].reset_index(drop=True)
sec_col = "secondary_labels" if "secondary_labels" in train.columns else None
AUDIO = DATA / "train_audio"

def load_clip(fn, rand=True):
    fp = AUDIO / fn
    try:
        info = sf.info(fp); n = info.frames
        if n <= WIN: a, _ = sf.read(fp, dtype="float32"); a = np.pad(a, (0, WIN-len(a)))
        else:
            st = np.random.randint(0, n-WIN) if rand else 0
            a, _ = sf.read(fp, start=st, frames=WIN, dtype="float32")
        if a.ndim > 1: a = a.mean(1)
        return a.astype(np.float32)
    except Exception:
        return np.zeros(WIN, np.float32)

class DS(torch.utils.data.Dataset):
    def __init__(self, df, rand=True): self.df=df.reset_index(drop=True); self.rand=rand
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        r=self.df.iloc[i]; y=np.zeros(NC,np.float32); y[L2I[str(r["primary_label"])]]=1.0
        if sec_col and isinstance(r[sec_col],str) and r[sec_col] not in ("","[]"):
            for s in r[sec_col].replace("[","").replace("]","").replace("'","").split(","):
                s=s.strip()
                if s in L2I: y[L2I[s]]=0.3
        return torch.from_numpy(load_clip(str(r["filename"]), self.rand)), torch.from_numpy(y)

dl = torch.utils.data.DataLoader(DS(train), batch_size=BS, shuffle=True, num_workers=8, drop_last=True, pin_memory=True)
print(f"train clips: {len(train)} | batches/epoch: {len(dl)}", flush=True)

# ---- validation: labeled soundscapes (multilabel ';') ----
vlab = pd.read_csv(DATA / "train_soundscapes_labels.csv")
vlab["splist"] = vlab["primary_label"].astype(str).str.split(";")
VSS = DATA / "train_soundscapes"
def build_val():
    Y=[]; W=[]
    files = sorted(set(vlab["filename"]))
    for fn in files:
        fp = VSS / fn
        if not fp.exists(): continue
        a,_=sf.read(fp,dtype="float32"); a=a.mean(1) if a.ndim>1 else a
        sub=vlab[vlab["filename"]==fn]
        for _,row in sub.iterrows():
            end=int(str(row["end"]).split(":")[-1]) if ":" in str(row["end"]) else int(row["end"])
            # end like 00:00:05 -> seconds
            t=str(row["end"]).split(":"); sec=int(t[0])*3600+int(t[1])*60+int(t[2]) if len(t)==3 else int(row["end"])
            s0=(sec-5)*SR
            seg=a[s0:s0+WIN]
            if len(seg)<WIN: seg=np.pad(seg,(0,WIN-len(seg)))
            y=np.zeros(NC,np.float32)
            for sp in row["splist"]:
                sp=sp.strip()
                if sp in L2I: y[L2I[sp]]=1.0
            W.append(seg.astype(np.float32)); Y.append(y)
    return np.stack(W), np.stack(Y)
Vx, Vy = build_val()
print(f"val windows: {len(Vx)}", flush=True)

@torch.no_grad()
def validate():
    model.eval(); preds=[]
    for i in range(0,len(Vx),64):
        wav=torch.from_numpy(Vx[i:i+64]).to(DEV)
        logit=model(wav_to_img(wav)).logits
        preds.append(torch.sigmoid(logit).cpu().numpy())
    P=np.concatenate(preds)
    aucs=[]
    for c in range(NC):
        if Vy[:,c].sum()>0 and Vy[:,c].sum()<len(Vy):
            aucs.append(roc_auc_score(Vy[:,c],P[:,c]))
    return float(np.mean(aucs)), len(aucs)

opt=torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
sched=torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=EPOCHS*len(dl), pct_start=0.1)
scaler=torch.cuda.amp.GradScaler()
lossf=nn.BCEWithLogitsLoss()
best=0.0
print("=== fail-fast: 1-batch forward check ===", flush=True)
wb,yb=next(iter(dl));
_img=wav_to_img(wb.to(DEV))
with torch.cuda.amp.autocast():
    out=model(_img).logits
print("forward OK, logits:", tuple(out.shape), "img range", float(_img.min()), float(_img.max()), flush=True)

for ep in range(EPOCHS):
    model.train(); t0=time.time(); run=0
    for bi,(wav,y) in enumerate(dl):
        wav=wav.to(DEV,non_blocking=True); y=y.to(DEV,non_blocking=True)
        opt.zero_grad()
        img=wav_to_img(wav)  # fp32 mel/dB OUTSIDE autocast (fp16 mel/dB underflows -> NaN, per exp169 commit)
        with torch.cuda.amp.autocast():
            logit=model(img).logits; loss=lossf(logit,y)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
        run+=loss.item()
        if bi%200==0: print(f"ep{ep} b{bi}/{len(dl)} loss {run/(bi+1):.4f} {time.time()-t0:.0f}s", flush=True)
    va,nev=validate()
    print(f"=== EP{ep} val_macro_auc={va:.4f} on {nev} eval species (best {best:.4f}) ===", flush=True)
    if va>best:
        best=va; torch.save(model.state_dict(), OUT/"birdset_ft_best.pt")
        print(f"saved best {best:.4f}", flush=True)
    # fail-fast: if after ep1 val is hopeless, stop
    if ep==1 and best<0.80:
        print(f"FAIL-FAST: val {best:.4f}<0.80 after 2 epochs -> domain gap too severe, stopping", flush=True); break
print(f"DONE best_val_macro_auc={best:.4f}", flush=True)
