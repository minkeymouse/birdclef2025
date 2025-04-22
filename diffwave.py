#!/usr/bin/env python3
"""
diffwave.py – minority-class data synthesis
=========================================
Fine-tunes a *unconditional* **DiffWave** model on the under-represented
species in **BirdCLEF 2025**, then generates extra 10-second waveforms that
are immediately discoverable by `process.py` because we patch `train.csv`.

CLI usage
---------
```bash
# 1) fine-tune (unconditional) on all auto-detected minority species
python diffwave.py train --epochs 40

# 2) only generate new clips (after training)
python diffwave.py generate --checkpoint models/diffwave/diffwave_uncond.pth

# 3) restrict to a whitelist & add just 3 extra epochs
python diffwave.py train --species plctan1,turvul --epochs 3
```
"""
from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

# SpeechBrain DiffWave implementation
try:
    from speechbrain.lobes.models.DiffWave import DiffWave
except ImportError:
    raise ImportError("❌ diffwave.py requires `speechbrain` (pip install speechbrain)")

from configure import CFG
from data_utils import load_audio

# ────────────────────────────────────────────────────────────────────────────
# Constants & hyper-params
# ────────────────────────────────────────────────────────────────────────────
SAMPLE_RATE = 32_000
SEG_SECONDS = 5
SEG_SAMPLES = SAMPLE_RATE * SEG_SECONDS

THRESH_LOW = 20
THRESH_HIGH = 50
TARGET_LOW = 20
TARGET_MID = 5

DIFFWAVE_CFG = {
    "input_channels": 80,
    "residual_layers": 20,
    "residual_channels": 32,
    "dilation_cycle_length": 10,
    "total_steps": 50,
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ────────────────────────────────────────────────────────────────────────────

def compute_plan(train_csv: Path) -> Dict[str, int]:
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

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        fp, _ = self.paths[idx]
        wav = load_audio(fp)  # mono np.ndarray, float32, sr already 32 kHz
        wav = torch.from_numpy(wav).unsqueeze(0)
        if wav.size(1) < SEG_SAMPLES:
            rep = math.ceil(SEG_SAMPLES / wav.size(1))
            wav = wav.repeat(1, rep)[:, :SEG_SAMPLES]
        else:
            off = random.randint(0, wav.size(1) - SEG_SAMPLES)
            wav = wav[:, off : off + SEG_SAMPLES]
        return wav

# ────────────────────────────────────────────────────────────────────────────
# Unconditional DiffWave wrapper
# ────────────────────────────────────────────────────────────────────────────
class DiffWaveUncond(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # unconditional=True ensures no spectrogram is required
        self.diff = DiffWave(**DIFFWAVE_CFG, unconditional=True)

    def forward(self, audio: torch.Tensor):
        # sample random diffusion steps
        bs = audio.size(0)
        steps = torch.randint(0, self.diff.total_steps, (bs,), device=audio.device)
        # DiffWave expects diffusion embedding internally
        # It will embed steps via self.diff.diffusion_embedding
        return self.diff(audio, steps)

    @torch.no_grad()
    def sample(self, n: int = 1) -> torch.Tensor:
        self.eval()
        noise = torch.randn(n, 1, SEG_SAMPLES, device=DEVICE)
        # inference: unconditional=True, scale=SEG_SAMPLES
        return self.diff.inference(
            unconditional=True,
            scale=SEG_SAMPLES,
            condition=None,
            fast_sampling=True,
            device=DEVICE,
        )

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
    loader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=4, drop_last=True)

    model = DiffWaveUncond().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)

    for ep in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for wav in loader:
            wav = wav.to(DEVICE)
            pred = model(wav)
            # target is the waveform itself? use L2 loss for approximation
            loss = F.mse_loss(pred, wav)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.detach().item() * wav.size(0)
        print(f"epoch {ep}/{args.epochs}  loss={running/len(loader.dataset):.5f}")

    ckpt_dir = CFG.DIFFWAVE_MODEL_DIR; ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "diffwave_uncond.pth"
    torch.save(model.state_dict(), ckpt_path)
    print(f"✔ saved checkpoint → {ckpt_path}")


def patch_train_csv(rows: List[Tuple[str, str]]) -> None:
    train_df = pd.read_csv(CFG.TRAIN_CSV)
    existing = set(zip(train_df["primary_label"], train_df["filename"]))
    new_entries = [(sp, fn) for sp, fn in rows if (sp, fn) not in existing]
    if not new_entries:
        return
    extra = pd.DataFrame(new_entries, columns=["primary_label", "filename"] )
    for col in train_df.columns:
        if col not in extra.columns:
            extra[col] = np.nan
    train_df = pd.concat([train_df, extra[train_df.columns]], ignore_index=True)
    train_df.to_csv(CFG.TRAIN_CSV, index=False)
    print(f"📎 Patched train.csv with {len(new_entries)} synthetic rows.")


def generate(args: argparse.Namespace) -> None:
    model = DiffWaveUncond().to(DEVICE)
    ckpt = torch.load(args.checkpoint, map_location=DEVICE)
    model.load_state_dict(ckpt)

    plan = compute_plan(CFG.TRAIN_CSV)
    manifest_rows, train_rows = [], []

    for sp, n_extra in plan.items():
        if n_extra == 0 or (args.species and sp not in args.species):
            continue
        tgt_dir = CFG.TRAIN_AUDIO_DIR / sp; tgt_dir.mkdir(parents=True, exist_ok=True)
        for k in range(n_extra):
            wav = model.sample(n=1)[0].cpu().numpy()
            fn = f"synthetic_{k:03d}.ogg"; fp = tgt_dir / fn
            sf.write(fp, wav, SAMPLE_RATE, subtype="OGG")
            manifest_rows.append({"filepath": str(fp), "primary_label": sp, "synthetic": True})
            train_rows.append((sp, fn))
        print(f"generated {n_extra} for {sp}")

    if manifest_rows:
        CFG.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        man_fp = CFG.PROCESSED_DIR / "synthetic_manifest.csv"
        pd.DataFrame(manifest_rows).to_csv(man_fp, index=False)
        print(f"✔ synthetic_manifest.csv rows={len(manifest_rows)}")

    patch_train_csv(train_rows)


def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DiffWave minority-class synthesiser")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="fine-tune DiffWave")
    t.add_argument("--epochs", type=int, default=50)
    t.add_argument("--species", type=lambda s: s.split(","), help="CSV list of species")

    g = sub.add_parser("generate", help="generate synthetic audio")
    g.add_argument("--checkpoint", type=Path, required=True)
    g.add_argument("--species", type=lambda s: s.split(","), help="limit to these species")

    return p.parse_args()


def main() -> None:
    args = parse()
    if args.cmd == "train":
        train(args)
    else:
        generate(args)


if __name__ == "__main__":
    main()
