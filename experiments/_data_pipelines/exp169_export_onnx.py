#!/usr/bin/env python3
"""Export exp169 5-fold ckpts to ONNX matching Tucker interface.

Tucker SED ONNX interface (from notebooks/birdclef-2026-mattia-fork/lib/tucker_sed.py):
  Input  : mel spectrogram (B, 1, n_mels=256, T) float32, externally preprocessed
           (per-spec z-score, n_fft=2048, hop=512, fmin=20, fmax=16000).
  Output : (B, 234) clip-level probabilities (sigmoid applied internally).

We strip the audio->mel + spec_aug + distill_head from the training model
so the export matches the inference interface 1-to-1.

Output: experiments/_data_pipelines/exp169_outputs/onnx/sed_fold{0..4}.onnx
"""
from __future__ import annotations
import argparse, os
from pathlib import Path
import torch, torch.nn as nn

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp169_distilled_sed import (  # type: ignore
    DistilledSED, GeMFreqPool, N_MELS, BACKBONE,
)

ROOT = Path("/data/birdclef2026")
OUT = ROOT / "experiments" / "_data_pipelines" / "exp169_outputs"
EXPORT_DIR = OUT / "onnx"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)


class InferenceWrapper(nn.Module):
    """Takes pre-computed mel (B, 1, M, T) -> (B, n_cls) clip probs."""

    def __init__(self, full: DistilledSED):
        super().__init__()
        self.backbone = full.backbone
        self.gem_freq = full.gem_freq
        self.bottleneck = full.bottleneck
        self.att = full.att
        self.cla = full.cla

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        h = self.backbone(mel)                     # (B, C, m', t')
        # In inference there's no distillation head; SED branch only.
        h_cls = self.gem_freq(h)                   # (B, C, T)
        h_cls = h_cls.transpose(1, 2)
        h_cls = self.bottleneck(h_cls)
        h_cls = h_cls.transpose(1, 2)              # (B, 512, T)
        a = torch.tanh(self.att(h_cls))
        norm_att = torch.softmax(a, dim=-1)
        framewise_logits = self.cla(h_cls)
        clip_logits = (norm_att * framewise_logits).sum(dim=2)
        return torch.sigmoid(clip_logits)


def export_one(ckpt_path: Path, out_path: Path, n_cls: int = 234, t_dim: int = 313):
    full = DistilledSED(n_cls=n_cls)
    state = torch.load(ckpt_path, map_location="cpu")
    full.load_state_dict(state["state_dict"])
    full.eval()
    wrap = InferenceWrapper(full).eval()

    dummy = torch.zeros(1, 1, N_MELS, t_dim)
    torch.onnx.export(
        wrap, dummy, str(out_path),
        input_names=["mel"], output_names=["probs"],
        dynamic_axes={"mel": {0: "batch", 3: "time"}, "probs": {0: "batch"}},
        opset_version=17, do_constant_folding=True,
    )
    sz = out_path.stat().st_size / 1024 / 1024
    print(f"  exported {out_path.name}  ({sz:.1f} MB)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = parser.parse_args()
    for f in args.folds:
        ckpt = OUT / f"fold{f}" / "best_ckpt.pt"
        if not ckpt.exists():
            print(f"skip fold{f}: no ckpt at {ckpt}")
            continue
        export_one(ckpt, EXPORT_DIR / f"sed_fold{f}.onnx")
    print("done.")


if __name__ == "__main__":
    main()
