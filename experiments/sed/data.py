"""Data pipeline — single CachedDataset + PseudoDataset replacing 16 duplicates.

Behavior driven by SedConfig:
  - mixup variants (focal-focal, focal-soundscape) toggled by USE_FOCAL_*_MIXUP
  - rare-class upsampling when MIN_SAMPLE > 0
  - pseudo dataset only when SedConfig.pseudo is set
  - sampler choice (WeightedRandomSampler vs MixSamp-style two-stream)

The fold split logic is pulled out into helpers so configs can choose:
  - EVAL_SS_N_FILES (legacy fixed-N holdout from exp169)
  - per-fold filename group split (Tucker-style)
"""
from __future__ import annotations
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader, ConcatDataset, WeightedRandomSampler

from .config import SedConfig

ROOT = Path("/data/birdclef2026")
DATA = ROOT / "data" / "birdclef-2026"

SECONDARY_WEIGHT = 0.3


# ─── Audio I/O ─────────────────────────────────────────────────────────────

def load_audio(path: Path, target_sr: int) -> np.ndarray | None:
    try:
        wav, sr0 = sf.read(str(path), dtype="float32", always_2d=False)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr0 != target_sr:
            wav = torchaudio.functional.resample(
                torch.from_numpy(wav), sr0, target_sr
            ).numpy()
        return wav.astype(np.float32)
    except Exception:
        return None


def parse_secondary(x: str) -> list[str]:
    if pd.isna(x) or x in ("[]", ""):
        return []
    try:
        return [s.strip("'\" ") for s in str(x).strip("[]").split(",") if s.strip("'\" ")]
    except Exception:
        return []


# ─── Dataset ───────────────────────────────────────────────────────────────


class CachedDataset(Dataset):
    """Wraps the existing exp169v2 anchor cache (per-file pre-extracted Perch
    embeddings at K random anchors), with optional MixUp partners.

    Returns (wav, perch_emb, label_vec, primary_idx, source_kind).
    source_kind: 0=TA, 1=labeled_SS, 2=pseudo.
    """

    def __init__(
        self,
        cfg: SedConfig,
        indices: np.ndarray,
        l2i: dict,
        train: bool,
        secondary_map: dict,
        ss_labels_map: dict,
        primary_idx_full: np.ndarray,
        is_ss_full: np.ndarray,
        files: np.ndarray,
        n_anchors: np.ndarray,
        anchor_off: np.ndarray,
        anchor_emb: np.ndarray,
        bg_pool: Optional[np.ndarray] = None,
    ):
        self.cfg = cfg
        self.indices = indices
        self.l2i = l2i
        self.n_cls = len(l2i)
        self.train = train
        self.secondary_map = secondary_map
        self.ss_labels_map = ss_labels_map
        self.primary_idx_full = primary_idx_full
        self.is_ss_full = is_ss_full
        self.files = files
        self.n_anchors = n_anchors
        self.anchor_off = anchor_off
        self.anchor_emb = anchor_emb
        self.bg_pool = bg_pool

        is_ss_at = np.array([is_ss_full[gi] for gi in indices])
        self.ta_pool = np.where(is_ss_at == 0)[0]
        self.ss_pool = np.where(is_ss_at == 1)[0]

    def __len__(self):
        return len(self.indices)

    def _build_label(self, gi: int) -> np.ndarray:
        y = np.zeros(self.n_cls, dtype=np.float32)
        if self.is_ss_full[gi] == 1:
            for lbl in self.ss_labels_map.get(gi, []):
                if lbl in self.l2i:
                    y[self.l2i[lbl]] = 1.0
        else:
            y[int(self.primary_idx_full[gi])] = 1.0
            for sl in self.secondary_map.get(gi, []):
                if sl in self.l2i:
                    y[self.l2i[sl]] = SECONDARY_WEIGHT
        return y

    def _load_clip(self, gi: int, anchor_id: int) -> np.ndarray:
        is_ss = self.is_ss_full[gi]
        sub_dir = "train_soundscapes" if is_ss else "train_audio"
        wav = load_audio(DATA / sub_dir / self.files[gi], self.cfg.SR)
        win_samples = self.cfg.SR * self.cfg.TRAIN_DURATION_S
        if wav is None or len(wav) == 0:
            return np.zeros(win_samples, dtype=np.float32)
        if len(wav) < win_samples:
            reps = win_samples // len(wav) + 1
            return np.tile(wav, reps)[:win_samples].astype(np.float32)
        off = int(self.anchor_off[gi, anchor_id])
        off = max(0, min(len(wav) - win_samples, off))
        return wav[off:off + win_samples].astype(np.float32)

    def _load_partner(self, partner_pos: int):
        gi = int(self.indices[partner_pos])
        n_anch = max(1, int(self.n_anchors[gi]))
        anchor_id = random.randint(0, n_anch - 1)
        wav = self._load_clip(gi, anchor_id).astype(np.float32)
        perch = self.anchor_emb[gi, anchor_id].astype(np.float32)
        y = self._build_label(gi)
        return wav, perch, y

    def __getitem__(self, idx):
        cfg = self.cfg
        gi = int(self.indices[idx])
        n_anch = max(1, int(self.n_anchors[gi]))
        anchor_id = random.randint(0, n_anch - 1) if self.train else 0
        wav = self._load_clip(gi, anchor_id).astype(np.float32)
        perch = self.anchor_emb[gi, anchor_id].astype(np.float32)
        y = self._build_label(gi)
        is_ss = int(self.is_ss_full[gi])
        primary_idx = int(self.primary_idx_full[gi]) if is_ss == 0 else -1
        source_kind = 0 if is_ss == 0 else 1

        if self.train:
            mixed = False
            # Focal-Focal MixUp
            if (cfg.USE_FOCAL_FOCAL_MIXUP and is_ss == 0
                    and len(self.ta_pool) > 1
                    and random.random() < cfg.MIXUP_PROB):
                p_pos = int(np.random.choice(self.ta_pool))
                p_wav, p_perch, p_y = self._load_partner(p_pos)
                lam = float(np.random.beta(cfg.MIXUP_ALPHA, cfg.MIXUP_ALPHA))
                wav = (lam * wav + (1.0 - lam) * p_wav).astype(np.float32)
                perch = lam * perch + (1.0 - lam) * p_perch
                y = np.maximum(y, p_y) if cfg.MIXUP_HARD else lam * y + (1.0 - lam) * p_y
                mixed = True
            # Focal-Soundscape MixUp
            if (not mixed and cfg.USE_FOCAL_SC_MIXUP and is_ss == 0
                    and len(self.ss_pool) > 0
                    and random.random() < cfg.MIXUP_PROB):
                p_pos = int(np.random.choice(self.ss_pool))
                p_wav, p_perch, p_y = self._load_partner(p_pos)
                lam = float(np.random.beta(cfg.MIXUP_ALPHA, cfg.MIXUP_ALPHA))
                wav = (lam * wav + (1.0 - lam) * p_wav).astype(np.float32)
                perch = lam * perch + (1.0 - lam) * p_perch
                y = np.maximum(y, p_y) if cfg.MIXUP_HARD else lam * y + (1.0 - lam) * p_y
            # Gain + noise
            if random.random() < cfg.AUG_PROB:
                db = random.uniform(*cfg.AUG_GAIN_DB_RANGE)
                wav = wav * (10.0 ** (db / 20.0))
            if random.random() < cfg.AUG_PROB:
                snr = random.uniform(*cfg.AUG_NOISE_SNR_DB_RANGE)
                sig_p = float(np.mean(wav ** 2)) + 1e-8
                noise_p = sig_p / (10.0 ** (snr / 10.0))
                wav = wav + np.random.randn(len(wav)).astype(np.float32) * math.sqrt(noise_p)

        return (
            torch.from_numpy(wav.astype(np.float32)),
            torch.from_numpy(perch),
            torch.from_numpy(y),
            int(primary_idx),
            int(source_kind),
        )


class PseudoDataset(Dataset):
    """Soft-pseudo dataset (kept_probs from filter.py output).
    source_kind = 2.
    No Perch teacher embedding (placeholder zeros). Loss code masks distill MSE.
    """

    def __init__(self, cfg: SedConfig, kept_files, kept_file_idx, kept_end_secs,
                 kept_probs, source_dir: Path, train: bool = True, n_cls: int = 234):
        self.cfg = cfg
        self.files = kept_files
        self.file_idx = kept_file_idx
        self.end_secs = kept_end_secs
        self.probs = kept_probs.astype(np.float32)
        self.source_dir = source_dir
        self.train = train
        self.n_cls = n_cls
        self._wav_cache: dict[str, np.ndarray] = {}

    def __len__(self):
        return len(self.probs)

    def _load_60s(self, fname: str) -> np.ndarray | None:
        if fname in self._wav_cache:
            return self._wav_cache[fname]
        wav = load_audio(self.source_dir / fname, self.cfg.SR)
        if wav is None:
            return None
        target = 60 * self.cfg.SR
        if len(wav) < target:
            wav = np.pad(wav, (0, target - len(wav)))
        elif len(wav) > target:
            wav = wav[:target]
        if len(self._wav_cache) >= 50:
            self._wav_cache.pop(next(iter(self._wav_cache)))
        self._wav_cache[fname] = wav
        return wav

    def __getitem__(self, idx):
        cfg = self.cfg
        fi = int(self.file_idx[idx])
        end_sec = int(self.end_secs[idx])
        fname = str(self.files[fi])
        wav = self._load_60s(fname)
        win_samples = cfg.SR * cfg.TRAIN_DURATION_S
        if wav is None:
            wav = np.zeros(win_samples, dtype=np.float32)
        else:
            start = max(0, (end_sec - cfg.TRAIN_DURATION_S) * cfg.SR)
            chunk = wav[start:start + win_samples]
            if len(chunk) < win_samples:
                chunk = np.pad(chunk, (0, win_samples - len(chunk)))
            wav = chunk.astype(np.float32)
        if self.train and random.random() < cfg.AUG_PROB:
            db = random.uniform(*cfg.AUG_GAIN_DB_RANGE)
            wav = wav * (10.0 ** (db / 20.0))
        if self.train and random.random() < cfg.AUG_PROB:
            snr = random.uniform(*cfg.AUG_NOISE_SNR_DB_RANGE)
            sig_p = float(np.mean(wav ** 2)) + 1e-8
            noise_p = sig_p / (10.0 ** (snr / 10.0))
            wav = wav + np.random.randn(len(wav)).astype(np.float32) * math.sqrt(noise_p)

        return (
            torch.from_numpy(wav),
            torch.zeros(cfg.PERCH_EMBED_DIM, dtype=torch.float32),
            torch.from_numpy(self.probs[idx]),
            -1,
            2,  # source_kind=pseudo
        )


# ─── Two-stream batch sampler (Sydorskyi 2025 ratio) ────────────────────────


class TwoStreamBatchSampler:
    """Per-batch deterministic mix: cached + pseudo, with configurable ratio.

    If pseudo_weights is provided (Boredom 2025 1st recipe), pseudo rows
    are sampled with probability proportional to pseudo_weights (typically
    sum of pseudo labels per row). Cached rows remain uniform.
    """

    def __init__(self, n_cached: int, n_pseudo: int, batch_size: int,
                 num_batches: int, pseudo_share: float = 0.40, seed: int = 42,
                 pseudo_weights=None):
        self.n_cached = n_cached
        self.n_pseudo = n_pseudo
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.pseudo_per_batch = max(0, int(round(batch_size * pseudo_share)))
        self.cached_per_batch = batch_size - self.pseudo_per_batch
        self.pseudo_offset = n_cached
        self.rng = np.random.default_rng(seed)
        if pseudo_weights is not None:
            assert pseudo_weights.shape == (n_pseudo,), (
                f"pseudo_weights shape {pseudo_weights.shape} != ({n_pseudo},)")
            w = pseudo_weights.astype(np.float64)
            w = np.maximum(w, 1e-6)
            self.pseudo_probs = w / w.sum()
        else:
            self.pseudo_probs = None

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        for _ in range(self.num_batches):
            cached = self.rng.integers(0, self.n_cached, size=self.cached_per_batch)
            if self.pseudo_probs is not None:
                pseudo = self.rng.choice(self.n_pseudo, size=self.pseudo_per_batch,
                                          replace=True, p=self.pseudo_probs)
            else:
                pseudo = self.rng.integers(0, self.n_pseudo, size=self.pseudo_per_batch)
            batch = list(cached.tolist()) + [self.pseudo_offset + int(i) for i in pseudo]
            self.rng.shuffle(batch)
            yield [int(x) for x in batch]


# ─── Fold split helpers ────────────────────────────────────────────────────


def build_label_lookups():
    """Build PRIMARY_LABELS, label→idx map, taxonomy class_name map."""
    sub = pd.read_csv(DATA / "sample_submission.csv")
    primary_labels = sub.columns[1:].tolist()
    l2i = {c: i for i, c in enumerate(primary_labels)}
    return primary_labels, l2i


def split_ss_legacy(is_ss: np.ndarray, files: np.ndarray, eval_n_files: int, seed: int):
    """Legacy fixed-N-file holdout (used by exp169-174). Same eval set across all folds."""
    ss_files = sorted({files[i] for i in np.where(is_ss == 1)[0]})
    rng = np.random.RandomState(seed)
    rng.shuffle(ss_files)
    eval_files = set(ss_files[:eval_n_files])
    train_idx = np.array([i for i in np.where(is_ss == 1)[0] if files[i] not in eval_files], dtype=np.int64)
    eval_idx = np.array([i for i in np.where(is_ss == 1)[0] if files[i] in eval_files], dtype=np.int64)
    return train_idx, eval_idx


def split_ss_per_fold(is_ss: np.ndarray, files: np.ndarray, n_folds: int, seed: int):
    """Per-fold filename split (Tucker spec). Returns list of (train_idx, eval_idx) per fold."""
    ss_files = sorted({files[i] for i in np.where(is_ss == 1)[0]})
    rng = np.random.RandomState(seed)
    rng.shuffle(ss_files)
    file_to_fold = {f: i % n_folds for i, f in enumerate(ss_files)}
    folds = []
    for k in range(n_folds):
        train_idx = np.array(
            [i for i in np.where(is_ss == 1)[0] if file_to_fold.get(files[i], -1) != k],
            dtype=np.int64,
        )
        eval_idx = np.array(
            [i for i in np.where(is_ss == 1)[0] if file_to_fold.get(files[i], -1) == k],
            dtype=np.int64,
        )
        folds.append((train_idx, eval_idx))
    return folds
