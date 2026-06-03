"""
LoRA adapter + DPO training on Perch embeddings (train_audio + soundscapes).
Produces a lightweight adapter that modifies Perch embeddings for better
rare/unmapped species detection. Output: adapter weights for pipeline integration.

Usage: uv run python experiments/lora_dpo_train.py
"""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
import time, warnings
warnings.filterwarnings('ignore')


class LoRAAdapter(nn.Module):
    def __init__(self, d_in=1536, rank=64, alpha=0.3):
        super().__init__()
        self.down = nn.Linear(d_in, rank, bias=False)
        self.up = nn.Linear(rank, d_in, bias=False)
        self.alpha = alpha
        nn.init.kaiming_normal_(self.down.weight)
        nn.init.zeros_(self.up.weight)
    def forward(self, x):
        return x + self.alpha * self.up(F.gelu(self.down(x)))


class AdaptedHead(nn.Module):
    def __init__(self, d=1536, rank=64, n_cls=234):
        super().__init__()
        self.adapter = LoRAAdapter(d, rank)
        self.head = nn.Sequential(
            nn.Linear(d, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(512, n_cls))
    def forward(self, x):
        a = self.adapter(x)
        return self.head(a), a


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    tax = pd.read_csv("/data/birdclef2026/data/birdclef-2026/taxonomy.csv")
    species_list = sorted(tax['primary_label'].astype(str).tolist())
    sp_to_idx = {s:i for i,s in enumerate(species_list)}
    train_df = pd.read_csv("/data/birdclef2026/data/birdclef-2026/train.csv")
    tc = train_df.groupby(train_df['primary_label'].astype(str)).size().to_dict()

    # Load exp169 cached Perch embeddings (combined train_audio + soundscape)
    emb_all = np.load('/data/birdclef2026/experiments/_data_pipelines/exp169_outputs/perch_emb.npy')
    meta = np.load('/data/birdclef2026/experiments/_data_pipelines/exp169_outputs/meta.npz', allow_pickle=True)
    is_ss = meta['is_ss']
    primary_idx = meta['primary_idx']
    ss_label_str = meta['ss_label_str']

    # Split into train_audio (is_ss==0) and soundscape (is_ss==1)
    ta_mask = is_ss == 0
    ss_mask = is_ss == 1
    emb_ta = emb_all[ta_mask]
    emb_ss = emb_all[ss_mask]

    # Build Y_ta from primary_idx (single label per clip)
    pi_ta = primary_idx[ta_mask]
    Y_ta = np.zeros((len(emb_ta), 234), dtype=np.float32)
    for i, sp_i in enumerate(pi_ta):
        if 0 <= sp_i < 234:
            Y_ta[i, sp_i] = 1.0
    print(f"  Train audio: {len(emb_ta)} clips, {(Y_ta.sum(axis=0)>0).sum()} species present")

    # Build Y_ss from ss_label_str (multi-label, semicolon-separated)
    ss_labels = ss_label_str[ss_mask]
    Y_ss = np.zeros((len(emb_ss), 234), dtype=np.float32)
    for i, lab in enumerate(ss_labels):
        if lab is None: continue
        for sp in str(lab).split(';'):
            sp = sp.strip()
            if sp in sp_to_idx:
                Y_ss[i, sp_to_idx[sp]] = 1.0
    print(f"  Soundscapes: {len(emb_ss)} windows, {(Y_ss.sum(axis=0)>0).sum()} species present, "
          f"total positives={int(Y_ss.sum())}")

    # Species weights
    sw = torch.ones(234, device=device)
    for j in range(234):
        n = tc.get(species_list[j], 0)
        if n == 0: sw[j] = 5.0
        elif n < 20: sw[j] = 3.0
        elif n < 100: sw[j] = 2.0

    X_train = torch.tensor(emb_ta, dtype=torch.float32).to(device)
    Y_train = torch.tensor(Y_ta, dtype=torch.float32).to(device)
    X_val = torch.tensor(emb_ss, dtype=torch.float32).to(device)

    model = AdaptedHead(1536, rank=64, n_cls=234).to(device)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    # Phase 1: SFT
    print("\n=== Phase 1: SFT (50 epochs) ===")
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 50)
    N_ta = len(emb_ta)

    for ep in range(50):
        model.train()
        perm = torch.randperm(N_ta, device=device)
        total_loss = 0
        for i in range(0, N_ta, 256):
            idx = perm[i:i+256]
            lo, _ = model(X_train[idx])
            loss = F.binary_cross_entropy_with_logits(lo, Y_train[idx], pos_weight=sw)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); total_loss += loss.item()
        sched.step()

        if (ep+1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                vp = torch.sigmoid(model(X_val)[0]).cpu().numpy()
            aucs = [roc_auc_score(Y_ss[:,j], vp[:,j]) for j in range(234)
                    if 3<=int(Y_ss[:,j].sum())<len(Y_ss)-3]
            print(f"  Ep{ep+1}: loss={total_loss/(N_ta//256+1):.4f} val_macro={np.mean(aucs):.4f}")

    # Phase 2: DPO
    print("\n=== Phase 2: DPO (30 epochs) ===")
    opt2 = torch.optim.Adam(model.parameters(), lr=5e-5)
    best_val, best_state = 0, {k:v.clone() for k,v in model.state_dict().items()}

    for ep in range(30):
        model.train()
        perm = torch.randperm(N_ta, device=device)
        total_loss = 0

        for batch_start in range(0, N_ta, 512):
            idx = perm[batch_start:batch_start+512]
            lo, _ = model(X_train[idx])
            Y_batch = Y_ta[idx.cpu().numpy()]

            dpo_l = 0; np_ = 0
            for j in range(234):
                pi = np.where(Y_batch[:,j]==1)[0]
                ni = np.where(Y_batch[:,j]==0)[0]
                if len(pi)<1 or len(ni)<1: continue
                ns = min(100, len(pi)*len(ni))
                sp, sn = np.random.choice(pi,ns), np.random.choice(ni,ns)
                dpo_l += -F.logsigmoid(lo[sp,j]-lo[sn,j]).mean() * sw[j].item()
                np_ += 1

            if np_ > 0:
                bce = F.binary_cross_entropy_with_logits(lo, torch.tensor(Y_batch,device=device))
                loss = dpo_l/np_ + 0.05*bce
                opt2.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt2.step(); total_loss += loss.item()

        if (ep+1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                vp = torch.sigmoid(model(X_val)[0]).cpu().numpy()
            aucs = [roc_auc_score(Y_ss[:,j], vp[:,j]) for j in range(234)
                    if 3<=int(Y_ss[:,j].sum())<len(Y_ss)-3]
            macro = np.mean(aucs)

            cls_a = {}
            for cls in ['Aves','Amphibia','Insecta']:
                ca = [roc_auc_score(Y_ss[:,j], vp[:,j]) for j in range(234)
                      if tax[tax['primary_label'].astype(str)==species_list[j]]['class_name'].values[0]==cls
                      and 3<=int(Y_ss[:,j].sum())<len(Y_ss)-3]
                if ca: cls_a[cls] = np.mean(ca)

            print(f"  Ep{ep+1}: val={macro:.4f} A={cls_a.get('Aves',0):.3f} Am={cls_a.get('Amphibia',0):.3f} In={cls_a.get('Insecta',0):.3f}")
            if macro > best_val:
                best_val = macro
                best_state = {k:v.clone() for k,v in model.state_dict().items()}

    model.load_state_dict(best_state)
    out_path = '/data/birdclef2026/model-weights/lora_dpo_adapter.pt'
    torch.save({
        'state_dict': best_state,
        'species_list': species_list,
        'rank': 64,
        'val_macro': best_val,
    }, out_path)
    print(f"\nSaved to {out_path} (val_macro={best_val:.4f})")


if __name__ == '__main__':
    main()
