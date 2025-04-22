#!/usr/bin/env python3
"""
diffwave.py – minority‑class data synthesis
=========================================
Fine‑tunes a *class‑conditional* **DiffWave** model on the under‑represented
species in **BirdCLEF 2025** and then generates extra 10‑second waveforms that
are *immediately discoverable* by `process.py` because we patch `train.csv`
(option **A** in the design note).

CLI usage
---------
```bash
# 1) fine‑tune on all auto‑detected minority species
python diffwave.py train --epochs 40

# 2) only generate new clips (after training)
python diffwave.py generate --checkpoint models/diffwave/diffwave_minor_classes.pth

# 3) restrict to a whitelist & add just 3 extra epochs of fine‑tuning
python diffwave.py train --species plctan1,turvul --epochs 3
```

After `generate` completes:
1. New `.ogg` files live under `train_audio/<species>/synthetic_##.ogg`.
2. `processed/synthetic_manifest.csv` lists every synthetic file.
3. **train.csv is patched in‑place** with rows like:
   ```csv
   primary_label,filename,secondary_labels,rating
   turvul,synthetic_000.ogg,,
   ```
   so that a subsequent **python process.py** will treat the clips as bona‑fide
   labelled recordings.

The script is entirely self‑contained – you *don’t* have to touch `process.py`.
"""
from __future__ import annotations

import argparse
import math
import random
import textwrap
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import DataLoader, Dataset

# SpeechBrain DiffWave implementation
try:
    from speechbrain.lobes.models.diffwave import DiffWave
except Exception as exc:  # pragma: no cover – fail fast, clear msg
    raise ImportError("❌ diffwave.py requires `speechbrain` (pip install speechbrain)") from exc

from configure import CFG
from data_utils import load_audio

# ────────────────────────────────────────────────────────────────────────────
# Constants & hyper‑params
# ────────────────────────────────────────────────────────────────────────────
SAMPLE_RATE: int = 32_000
SEG_SECONDS: int = 10
SEG_SAMPLES: int = SAMPLE_RATE * SEG_SECONDS

# synthesis policy (piece‑wise)
THRESH_LOW = 20     # if <20 → make it 20 total
THRESH_HIGH = 50    # if 20≤n<50 → +5 synthetic
TARGET_LOW = 20
TARGET_MID = 5

DIFFWAVE_CFG: Dict[str, int] = {
    "residual_channels": 64,
    "dilation_cycle_length": 10,
    "layers": 30,
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ────────────────────────────────────────────────────────────────────────────

def compute_plan(train_csv: Path) -> Dict[str, int]:
    """Return mapping *species → n_extra* based on frequency in `train.csv`."""
    counts = pd.read_csv(train_csv)["primary_label"].value_counts()
    plan: Dict[str, int] = {}
    for sp, cnt in counts.items():
        if cnt < THRESH_LOW:
            plan[sp] = TARGET_LOW - cnt
        elif cnt < THRESH_HIGH:
            plan[sp] = TARGET_MID
        else:
            plan[sp] = 0
    return plan

# ────────────────────────────────────────────────────────────────────────────
# Dataset feeding raw waveforms to DiffWave
# ────────────────────────────────────────────────────────────────────────────
class MinorityDataset(Dataset):
    def __init__(self, species: List[str]):
        self.paths: List[Tuple[Path, str]] = []
        for sp in species:
            self.paths.extend([(p, sp) for p in (CFG.TRAIN_AUDIO_DIR / sp).glob("*.ogg")])
        self.s2i = {sp: i for i, sp in enumerate(species)}

    def __len__(self) -> int:  # noqa: D401
        return len(self.paths)

    def __getitem__(self, idx: int):
        fp, sp = self.paths[idx]
        wav = load_audio(fp)  # mono np.ndarray, float32, sr already 32 kHz
        wav = torch.from_numpy(wav).unsqueeze(0)  # [1, N]
        # pad / random‑crop to 10 s exactly
        if wav.size(1) < SEG_SAMPLES:
            rep = math.ceil(SEG_SAMPLES / wav.size(1))
            wav = wav.repeat(1, rep)[:, :SEG_SAMPLES]
        else:
            off = random.randint(0, wav.size(1) - SEG_SAMPLES)
            wav = wav[:, off:off + SEG_SAMPLES]
        return wav, self.s2i[sp]

# ────────────────────────────────────────────────────────────────────────────
# Class‑conditional DiffWave wrapper
# ────────────────────────────────────────────────────────────────────────────
class DiffWaveCond(torch.nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        self.diff = DiffWave(**DIFFWAVE_CFG)
        self.embed = torch.nn.Embedding(n_classes, DIFFWAVE_CFG["residual_channels"])

    def forward(self, audio: torch.Tensor, cls: torch.Tensor):
        cond = self.embed(cls)
        return self.diff(audio, cond)

    @torch.no_grad()
    def sample(self, cls_id: int, n: int = 1) -> torch.Tensor:  # (n,1,T)
        self.eval()
        noise = torch.randn(n, 1, SEG_SAMPLES, device=DEVICE)
        cond = self.embed(torch.full((n,), cls_id, device=DEVICE, dtype=torch.long))
        return self.diff.inference(noise, cond)

# ────────────────────────────────────────────────────────────────────────────
# Training & generation routines
# ────────────────────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    plan = compute_plan(CFG.TRAIN_CSV)
    species = args.species or [sp for sp, k in plan.items() if k > 0]
    if not species:
        print("Nothing to train – no minority species found.")
        return

    ds = MinorityDataset(species)
    loader = DataLoader(ds, batch_size=8, shuffle=True, num_workers=4, drop_last=True)

    model = DiffWaveCond(len(species)).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)

    for ep in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for wav, cls in loader:
            wav, cls = wav.to(DEVICE), cls.to(DEVICE)
            loss = model(wav, cls)
            opt.zero_grad(); loss.backward(); opt.step()
            running += loss.item() * wav.size(0)
        print(f"epoch {ep}/{args.epochs}  loss={running/len(loader.dataset):.5f}")

    ckpt_dir = CFG.DIFFWAVE_MODEL_DIR; ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "diffwave_minor_classes.pth"
    torch.save({"model": model.state_dict(), "species": species}, ckpt_path)
    print(f"✔ saved checkpoint → {ckpt_path}")


def patch_train_csv(rows: List[Tuple[str, str]]) -> None:
    """Append *rows* [(species, filename), …] to **train.csv** if missing."""
    train_df = pd.read_csv(CFG.TRAIN_CSV)
    existing = set(zip(train_df["primary_label"], train_df["filename"]))
    new_entries = [(sp, fn) for sp, fn in rows if (sp, fn) not in existing]
    if not new_entries:
        return

    # minimal schema – match existing columns, fill NA
    extra = pd.DataFrame(new_entries, columns=["primary_label", "filename"])
    for col in train_df.columns:
        if col not in extra.columns:
            extra[col] = np.nan
    train_df = pd.concat([train_df, extra[train_df.columns]], ignore_index=True)
    train_df.to_csv(CFG.TRAIN_CSV, index=False)
    print(f"📎 Patched train.csv with {len(new_entries)} synthetic rows.")


def generate(args: argparse.Namespace) -> None:
    ckpt = torch.load(args.checkpoint, map_location=DEVICE)
    model = DiffWaveCond(len(ckpt["species"]))
    model.load_state_dict(ckpt["model"]); model.to(DEVICE)
    sp2idx = {sp: i for i, sp in enumerate(ckpt["species"])}

    plan = compute_plan(CFG.TRAIN_CSV)
    manifest_rows, train_rows = [], []

    for sp, n_extra in plan.items():
        if n_extra == 0 or (args.species and sp not in args.species):
            continue
        cls_id = sp2idx.get(sp)
        if cls_id is None:
            print(f"[WARN] {sp} not in checkpoint, skipping.")
            continue

        tgt_dir = CFG.TRAIN_AUDIO_DIR / sp; tgt_dir.mkdir(parents=True, exist_ok=True)
        for k in range(n_extra):
            wav = model.sample(cls_id)[0].cpu().numpy()
            fn = f"synthetic_{k:03d}.ogg"; fp = tgt_dir / fn
            sf.write(fp, wav, SAMPLE_RATE, subtype="OGG")
            manifest_rows.append({"filepath": str(fp), "primary_label": sp, "synthetic": True})
            train_rows.append((sp, fn))
        print(f"generated {n_extra} for {sp}")

    # write manifest
    if manifest_rows:
        CFG.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        man_fp = CFG.PROCESSED_DIR / "synthetic_manifest.csv"
        pd.DataFrame(manifest_rows).to_csv(man_fp, index=False)
        print(f"✔ synthetic_manifest.csv rows={len(manifest_rows)}")

    # patch train.csv so process.py notices the new clips
    patch_train_csv(train_rows)

# ────────────────────────────────────────────────────────────────────────────
# CLI parsing
# ────────────────────────────────────────────────────────────────────────────

def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DiffWave minority‑class synthesiser")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="fine‑tune DiffWave")
    t.add_argument("--epochs", type=int, default=50)
    t.add_argument("--species", type=lambda s: s.split(","), help="CSV list of species")

    g = sub.add_parser("generate", help="generate synthetic audio")
    g.add_argument("--checkpoint", type=Path, required=True)
    g.add_argument("--species", type=lambda s: s.split(","), help="limit to these species")

    return p.parse_args()

# ────────────────────────────────────────────────────────────────────────────
# Main entry
# ────────────────────────────────────────────────────────────────────────────

def main() -> None:  # noqa: D401
    args = parse()
    if args.cmd == "train":
        train(args)
    else:
        generate(args)


if __name__ == "__main__":
    main()
