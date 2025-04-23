#!/usr/bin/env python3
"""
diffwave.py – minority-class data synthesis
=========================================
Fine-tunes a *conditional* DiffWave model on under-represented
species in BirdCLEF 2025, then generates extra 5-second waveforms
that process.py can incorporate into train.csv.

Usage:
  python diffwave.py train [--epochs E] [--species S1,S2] [--pretrained PATH]
  python diffwave.py generate [--checkpoint PATH] [--species S1,S2]
  python diffwave.py remove
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
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchaudio.transforms as T
from torch.utils.checkpoint import checkpoint
import bitsandbytes as bnb

# SpeechBrain DIFFWAVE
try:
    from speechbrain.lobes.models.DiffWave import DiffWave, DiffWaveDiffusion
    from speechbrain.nnet.diffusion import GaussianNoise
    from speechbrain.inference import DiffWaveVocoder
except ImportError:
    raise ImportError("diffwave.py requires speechbrain: pip install speechbrain")

from configure import CFG
from data_utils import load_audio

# ───── Config & Hyperparameters ─────
SAMPLE_RATE = 32_000
SEG_SECONDS = 5
SEG_SAMPLES = SAMPLE_RATE * SEG_SECONDS

THRESH_LOW, THRESH_HIGH = 20, 50
TARGET_LOW, TARGET_MID = 20, 5

DIFFWAVE_CFG = {
    "input_channels":        80,
    "residual_layers":       30,
    "residual_channels":     64,
    "dilation_cycle_length": 10,
    "total_steps":           50,
    "unconditional":         False,
}
BETA_START, BETA_END = 1e-4, 0.02
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Mel extractor: output shape [n_mels, time]
MEL_EXTRACTOR = T.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=1024,
    hop_length=256,
    n_mels=DIFFWAVE_CFG["input_channels"],
)

# ───── Helpers ─────

def compute_plan(csv_path: Path) -> Dict[str, int]:
    df = pd.read_csv(csv_path)
    counts = df["primary_label"].value_counts()
    plan: Dict[str,int] = {}
    for sp, cnt in counts.items():
        if cnt < THRESH_LOW:
            plan[sp] = TARGET_LOW - cnt
        elif cnt < THRESH_HIGH:
            plan[sp] = TARGET_MID
        else:
            plan[sp] = 0
    return plan

# ───── Dataset ─────
class MinorityDataset(Dataset):
    """Return waveform and mel (no extra channel dim) for 5s segments."""
    def __init__(self, species: List[str]):
        self.files: List[Tuple[Path,str]] = []
        for sp in species:
            folder = CFG.TRAIN_AUDIO_DIR / sp
            self.files += [(p, sp) for p in folder.glob("*.ogg")]

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        fp, _ = self.files[idx]
        wav = load_audio(fp)                 # np.ndarray[T]
        wav = torch.from_numpy(wav).unsqueeze(0)  # [1, T]
        # pad or random crop to SEG_SAMPLES
        if wav.size(1) < SEG_SAMPLES:
            reps = math.ceil(SEG_SAMPLES / wav.size(1))
            wav = wav.repeat(1, reps)[:, :SEG_SAMPLES]
        else:
            off = random.randint(0, wav.size(1) - SEG_SAMPLES)
            wav = wav[:, off:off+SEG_SAMPLES]
        # mel: [n_mels, mel_len]
        # mel-spectrogram: [n_mels, mel_len]
        mel = MEL_EXTRACTOR(wav).squeeze(0)
        # Crop mel frames so mel_len * hop_length == SEG_SAMPLES
        hop = MEL_EXTRACTOR.hop_length
        target_frames = SEG_SAMPLES // hop
        if mel.size(-1) > target_frames:
            mel = mel[..., :target_frames]
        return wav, mel

# ───── Model Wrapper ─────
class DiffWaveCond(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = DiffWave(**DIFFWAVE_CFG)
        self.diffusion = DiffWaveDiffusion(
            model=self.model,
            timesteps=DIFFWAVE_CFG["total_steps"],
            beta_start=BETA_START,
            beta_end=BETA_END,
            noise=GaussianNoise,
            show_progress=False,
        )
        betas = torch.linspace(BETA_START, BETA_END, DIFFWAVE_CFG["total_steps"], device=DEVICE)
        alphas = 1 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        self.sqrt_ab  = torch.sqrt(alpha_bar)
        self.sqrt_omb = torch.sqrt(1 - alpha_bar)

    def forward(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor,
        spectrogram: torch.Tensor,
    ) -> torch.Tensor:
        # x0: [B,1,T], noise: [B,1,T], spectrogram: [B,80,mel_len]
        xt = x0 * self.sqrt_ab[t].view(-1,1,1) + noise * self.sqrt_omb[t].view(-1,1,1)
        # checkpoint the core denoiser to save activation memory
        def denoise(x, tt, spec):
            return self.model(x, tt, spec)
        return checkpoint(denoise, xt, t, spectrogram)

    @torch.no_grad()
    def sample(self, n: int = 1) -> torch.Tensor:
        self.eval()
        wav = self.diffusion.inference(
            unconditional=False,
            fast_sampling_noise_schedule="linear",
            condition=None,
            device=DEVICE,
        )  # [n, T]
        return wav.unsqueeze(1)  # [n,1,T]

# ───── CSV Patch & Remove ─────

def patch_train_csv(rows: List[Tuple[str,str]]) -> None:
    df = pd.read_csv(CFG.TRAIN_CSV)
    exist = set(zip(df["primary_label"], df["filename"]))
    new = [r for r in rows if r not in exist]
    if not new:
        return
    extra = pd.DataFrame(new, columns=["primary_label","filename"])
    for c in df.columns:
        if c not in extra.columns:
            extra[c] = np.nan
    out = pd.concat([df, extra[df.columns]], ignore_index=True)
    out.to_csv(CFG.TRAIN_CSV, index=False)
    print(f"Patched train.csv with {len(new)} synthetic entries.")

def remove_synthetic() -> None:
    for spdir in CFG.TRAIN_AUDIO_DIR.iterdir():
        if spdir.is_dir():
            for f in spdir.glob("synthetic_*.ogg"):
                f.unlink(missing_ok=True)
    print("Removed all synthetic files.")

# ───── Training ─────

def train(args: argparse.Namespace) -> None:
    plan = compute_plan(CFG.TRAIN_CSV)
    spec_list = args.species or [s for s,k in plan.items() if k>0]
    if not spec_list:
        print("No under-represented species to train.")
        return

    ds = MinorityDataset(spec_list)
    loader = DataLoader(ds,
                        batch_size=CFG.DIFF_BATCH_SIZE,
                        shuffle=True,
                        num_workers=CFG.DIFF_NUM_WORKERS,
                        drop_last=True)

    model = DiffWaveCond().to(DEVICE)
    model = torch.compile(model, backend="inductor", mode="reduce-overhead")
    opt = bnb.optim.AdamW8bit(model.parameters(), lr=CFG.DIFF_LR * 0.1, eps=1e-6)
    if args.pretrained:
        sd = torch.load(args.pretrained, map_location=DEVICE)
        model.model.load_state_dict(sd, strict=False)
        print(f"Loaded pretrained weights from {args.pretrained}")
    else:
        vb = DiffWaveVocoder.from_hparams(
            source="speechbrain/tts-diffwave-ljspeech",
            savedir="pretrained_models/diffwave-ljspeech",
            run_opts={"device":DEVICE.type})
        sd = vb.mods["diffwave"].state_dict()
        model.model.load_state_dict(sd, strict=False)
        print("Loaded LJSpeech pretrained weights.")

    Tsteps = DIFFWAVE_CFG["total_steps"]
    for ep in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for wav, mel in loader:
            wav = wav.to(DEVICE)
            mel = mel.to(DEVICE)
            bs    = wav.size(0)
            noise = torch.randn_like(wav)
            t     = torch.randint(0, Tsteps, (bs,), device=DEVICE)

            # Plain FP32 training step
            pred = model(wav, t, noise, spectrogram=mel)
            loss = F.mse_loss(pred, noise)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            running += loss.item() * bs
        avg_loss = running / len(ds)
        print(f"Epoch {ep}/{args.epochs}  Loss {avg_loss:.6f}")

    outp = CFG.DIFFWAVE_MODEL_DIR / "diffwave_cond.pth"
    torch.save(model.state_dict(), outp)
    print(f"Saved checkpoint to {outp}")

# ───── Generation ─────

def generate(args: argparse.Namespace) -> None:
    model = DiffWaveCond().to(DEVICE)
    sd = torch.load(args.checkpoint, map_location=DEVICE)
    model.load_state_dict(sd)

    plan = compute_plan(CFG.TRAIN_CSV)
    manifest, rows = [], []
    for sp, n in plan.items():
        if n>0 and (not args.species or sp in args.species):
            outdir = CFG.TRAIN_AUDIO_DIR / sp
            outdir.mkdir(parents=True, exist_ok=True)
            for i in range(n):
                wav = model.sample(1)[0].squeeze().cpu().numpy()
                fn = f"synthetic_{i:03d}.ogg"
                fp = outdir / fn
                sf.write(fp, wav, SAMPLE_RATE, format="OGG", subtype="VORBIS")
                manifest.append({"filepath":str(fp),"primary_label":sp,"synthetic":True})
                rows.append((sp, fn))
            print(f"Generated {n} for {sp}")
    if manifest:
        CFG.PROCESSED_DIR.mkdir(exist_ok=True)
        pd.DataFrame(manifest).to_csv(CFG.PROCESSED_DIR/"synthetic_manifest.csv",index=False)
        patch_train_csv(rows)

# ───── CLI ─────

def parse_args():
    p = argparse.ArgumentParser(description="DiffWave minority-class synthesizer")
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--epochs", type=int, default=50)
    t.add_argument("--species", type=lambda s: s.split(","))
    t.add_argument("--pretrained", type=Path)
    g = sub.add_parser("generate")
    g.add_argument("--checkpoint", type=Path,
                   default=CFG.DIFFWAVE_MODEL_DIR/"diffwave_cond.pth")
    g.add_argument("--species", type=lambda s: s.split(","))
    sub.add_parser("remove")
    return p.parse_args()

def main():
    args = parse_args()
    if args.cmd=="train": train(args)
    elif args.cmd=="generate": generate(args)
    else: remove_synthetic()

if __name__=="__main__":
    main()
