"""sed/export.py — single ONNX export replacing 8 duplicate scripts.

Usage:
  uv run python -m experiments.sed.export --config exp175 --output-dir ...
"""
from __future__ import annotations
import argparse
from pathlib import Path

import torch
import torch.nn as nn

from .config import SedConfig
from .model import DistilledSED


class InferenceWrapper(nn.Module):
    """Strips audio→mel + spec_aug + distill_head from training model.
    Input: pre-computed mel (B, 1, n_mels, T) — Tucker SED ONNX interface.
    Output: (clip_logits (B, 234), framewise (B, T, 234)) — both raw logits.
    """

    def __init__(self, full: DistilledSED):
        super().__init__()
        self.backbone = full.backbone
        self.gem_freq = full.gem_freq
        self.bottleneck = full.bottleneck
        self.att = full.att
        self.cla = full.cla

    def forward(self, mel: torch.Tensor):
        h = self.backbone(mel)
        h_cls = self.gem_freq(h)
        h_cls = h_cls.transpose(1, 2)
        h_cls = self.bottleneck(h_cls)
        h_cls = h_cls.transpose(1, 2)
        att = torch.softmax(torch.tanh(self.att(h_cls)), dim=-1)
        framewise = self.cla(h_cls)
        clip_logits = (att * framewise).sum(dim=2)
        return clip_logits, framewise.transpose(1, 2)


def export_one(cfg: SedConfig, ckpt_path: Path, out_path: Path) -> None:
    full = DistilledSED(cfg, n_cls=234)
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    full.load_state_dict(sd["state_dict"], strict=False)
    full.eval()
    wrap = InferenceWrapper(full).eval()
    n_frames = (cfg.SR * cfg.TRAIN_DURATION_S) // cfg.HOP_LENGTH + 1
    dummy = torch.zeros(1, 1, cfg.N_MELS, n_frames)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrap, (dummy,), str(out_path),
        input_names=["mel"], output_names=["clip_logits", "framewise"],
        dynamic_axes={
            "mel": {0: "B", 3: "T"},
            "clip_logits": {0: "B"},
            "framewise": {0: "B", 1: "T"},
        },
        opset_version=17, dynamo=False,
    )
    print(f"  exported {out_path.name} ({out_path.stat().st_size/1024/1024:.1f} MB)", flush=True)


def export_all_folds(cfg: SedConfig) -> Path:
    out_dir = cfg.resolved_output_dir() / "onnx"
    for f in range(cfg.N_FOLDS):
        ckpt = cfg.resolved_output_dir() / f"fold{f}" / "best_ckpt.pt"
        if not ckpt.exists():
            print(f"  missing {ckpt}, skip")
            continue
        export_one(cfg, ckpt, out_dir / f"sed_fold{f}.onnx")
    print(f"done. ONNX dir: {out_dir}", flush=True)
    return out_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="exp175")
    args = parser.parse_args()
    from . import config as cfg_mod
    cfg = getattr(cfg_mod, f"{args.config}_tucker_actually" if args.config == "exp175"
                  else f"{args.config}")()
    export_all_folds(cfg)


if __name__ == "__main__":
    main()
