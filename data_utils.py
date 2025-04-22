#!/usr/bin/env python
"""data_utils.py – shared utility layer for the **BirdCLEF 2025** solution
====================================================================
This module centralises *all* common data‑handling functionality so that the
rest of the pipeline (``process.py``, ``efficientnet.py``, ``regnety.py``,
``diffwave.py``) can remain lean.  Key capabilities:

* **Audio I/O** with deterministic resampling (+ optional WebRTC VAD removal)
* **Mel‑spectrogram extraction** via either SpeechBrain (GPU) or Librosa (CPU)
* **SpecAugment** + **CutMix** implementations fully driven by ``configure.CFG``
* **Soft‑label aware `torch.utils.data.Dataset`** (`MelDataset`)
* **One‑chunk‑per‑file sampling** (`FileWiseSampler`) to match training recipe
* **LRU‑cached on‑disk mel loader** to save repeated numpy I/O
* **Noise metric** helper used by ``process.py`` for fold‑0 split

The API surface is intentionally minimal – *import and call what you need*.
"""
from __future__ import annotations

import json
import os
import random
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple, Union

import cv2
import librosa
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler

from configure import CFG

__all__ = [
    "seed_everything",
    "compute_noise_metric",
    "load_audio",
    "trim_silence",
    "load_vad",
    "remove_speech",
    "compute_mel",
    "segment_audio",
    "spec_augment",
    "cutmix",
    "FileWiseSampler",
    "MelDataset",
]

# ────────────────────────────────────────────────────────────────────────
# Optional back‑ends – fail gracefully if missing
# ────────────────────────────────────────────────────────────────────────
try:
    # SpeechBrain provides a highly‑optimised GPU mel routine – use if present
    from speechbrain.lobes.models.HifiGAN import mel_spectrogram as sb_mel_spectrogram  # type: ignore

    _HAS_SB = True
except Exception:
    _HAS_SB = False

try:
    import webrtcvad  # type: ignore

    _HAS_VAD = True
except ImportError:
    _HAS_VAD = False

# ────────────────────────────────────────────────────────────────────────
# Reproducibility helpers
# ────────────────────────────────────────────────────────────────────────


def seed_everything(seed: int = 42) -> None:
    """Seed *every* RNG we know about for deterministic runs."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ────────────────────────────────────────────────────────────────────────
# Augmentations – SpecAugment & CutMix
# ────────────────────────────────────────────────────────────────────────


def _specaug_cfg() -> Tuple[int, int, int]:
    """Return the triplet *(freq_mask, time_mask, num_masks)* from ``CFG``."""
    fm = getattr(CFG, "SPEC_AUG_FREQ_MASK_PARAM", 10)
    tm = getattr(CFG, "SPEC_AUG_TIME_MASK_PARAM", 50)
    nm = getattr(CFG, "SPEC_AUG_NUM_MASKS", 2)
    return fm, tm, nm


def spec_augment(
    mel: np.ndarray,
    *,
    freq_mask_param: int | None = None,
    time_mask_param: int | None = None,
    num_masks: int | None = None,
) -> np.ndarray:
    """Apply **SpecAugment** (frequency & time masking) *in‑place* copy."""
    fm_d, tm_d, nm_d = _specaug_cfg()
    fmp = freq_mask_param or fm_d
    tmp = time_mask_param or tm_d
    nmk = num_masks or nm_d

    H, W = mel.shape
    out = mel.copy()
    for _ in range(nmk):
        # Frequency mask
        if fmp > 0:
            fh = np.random.randint(0, fmp + 1)
            f0 = np.random.randint(0, max(1, H - fh)) if fh else 0
            out[f0 : f0 + fh, :] = 0.0
        # Time mask
        if tmp > 0:
            th = np.random.randint(0, tmp + 1)
            t0 = np.random.randint(0, max(1, W - th)) if th else 0
            out[:, t0 : t0 + th] = 0.0
    return out


def cutmix(
    m1: np.ndarray,
    l1: torch.Tensor,
    m2: np.ndarray,
    l2: torch.Tensor,
) -> Tuple[np.ndarray, torch.Tensor]:
    """Horizontal **CutMix** for mels; label = length‑weighted interpolation."""
    W = m1.shape[1]
    if W < 2:
        return m1, l1  # avoid degenerate cases
    cut = np.random.randint(1, W)  # ensure at least 1 px on each side
    mixed = np.concatenate([m1[:, :cut], m2[:, cut:]], axis=1)
    alpha = cut / W
    label = l1 * alpha + l2 * (1.0 - alpha)
    return mixed, label


# ────────────────────────────────────────────────────────────────────────
# Audio helpers (load / trim / VAD)
# ────────────────────────────────────────────────────────────────────────


def load_audio(
    fp: Path | str,
    sample_rate: int | None = None,
    *,
    return_sr: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, int]]:
    """Load **mono** audio as 32‑bit float @ ``sample_rate`` (default CFG).

    Parameters
    ----------
    fp : Path | str
        File path to load. Any format supported by *librosa*.
    sample_rate : int | None, optional
        Desired sampling rate. ``None`` ⇒ use ``CFG.SAMPLE_RATE``.
    return_sr : bool, default = False
        If *True*, also return the effective sample‑rate alongside the audio.
    """
    sr = sample_rate or CFG.SAMPLE_RATE
    y, _ = librosa.load(str(fp), sr=sr, mono=True)
    y = y.astype(np.float32)
    if return_sr:
        return y, sr  # type: ignore[return-value]
    return y  # type: ignore[return-value]


def trim_silence(y: np.ndarray) -> np.ndarray:
    """Energy‑based leading / trailing trim."""
    y_trim, _ = librosa.effects.trim(y, top_db=CFG.TRIM_TOP_DB)
    return y_trim


# ── noise score used for fold‑0 selection ───────────────────────────────

def compute_noise_metric(y: np.ndarray) -> float:
    """Composite "*noise*" metric – smaller ⇒ cleaner recording."""
    return float(y.std() + y.var() + np.sqrt((y ** 2).mean()) + (y ** 2).sum())


# ── VAD helpers ─────────────────────────────────────────────────────────


def load_vad():
    """Return a *WebRTC VAD* instance & helper TS‑function or ``(None, None)``."""
    if not _HAS_VAD:
        return None, None

    vad = webrtcvad.Vad(3)

    def _iter_frames(wav: np.ndarray, sr: int, frame_ms: int = 30):
        flen = int(sr * frame_ms / 1000)
        for i in range(0, len(wav) - flen, flen):
            yield wav[i : i + flen]

    def _get_ts(wav: np.ndarray, sr: int):
        voiced: List[Tuple[int, int]] = []
        flen = int(sr * 0.03)  # 30 ms
        in_voiced = False
        start = 0
        for idx, frame in enumerate(_iter_frames(wav, sr)):
            speech = vad.is_speech((frame * 32768).astype("int16").tobytes(), sr)
            if speech and not in_voiced:
                start = idx * flen
                in_voiced = True
            elif not speech and in_voiced:
                voiced.append((start, idx * flen))
                in_voiced = False
        if in_voiced:
            voiced.append((start, len(wav)))
        return voiced

    return vad, _get_ts


def remove_speech(y: np.ndarray, vad_model, get_ts):
    """Zero‑out VAD‑detected speech regions (if VAD available)."""
    if vad_model is None:
        return y
    mask = np.ones_like(y, dtype=bool)
    for s, e in get_ts(y, CFG.SAMPLE_RATE):
        mask[s:e] = False
    return y[mask]


# ────────────────────────────────────────────────────────────────────────
# Mel‑spectrogram extraction
# ────────────────────────────────────────────────────────────────────────


def compute_mel(y: np.ndarray, *, to_db: bool = True) -> np.ndarray:
    """Return *128 × T* mel spectrogram in linear or dB scale."""

    if _HAS_SB:
        # SpeechBrain path – uses GPU if available, ~4‑6× faster than Librosa
        audio = torch.from_numpy(y).unsqueeze(0)
        mel = sb_mel_spectrogram(
            sample_rate=CFG.SAMPLE_RATE,
            hop_length=CFG.HOP_LENGTH,
            win_length=CFG.N_FFT,
            n_fft=CFG.N_FFT,
            n_mels=CFG.N_MELS,
            f_min=CFG.FMIN,
            f_max=CFG.FMAX,
            power=CFG.POWER,
            normalized=False,
            norm="slaney",
            mel_scale="slaney",
            compression=True,
            audio=audio,
        ).squeeze(0).cpu().numpy()
    else:
        mel = librosa.feature.melspectrogram(
            y=y,
            sr=CFG.SAMPLE_RATE,
            n_fft=CFG.N_FFT,
            hop_length=CFG.HOP_LENGTH,
            n_mels=CFG.N_MELS,
            fmin=CFG.FMIN,
            fmax=CFG.FMAX,
            power=CFG.POWER,
        )

    if to_db:
        mel = librosa.power_to_db(mel, ref=np.max)
    return mel.astype(np.float32)


# ────────────────────────────────────────────────────────────────────────
# Segmentation helper – yields fixed‑length chunks (wrap‑pad)
# ────────────────────────────────────────────────────────────────────────


def segment_audio(
    y: np.ndarray,
    *,
    chunk_sec: int = CFG.TRAIN_CHUNK_SEC,
    hop_sec: int = CFG.TRAIN_CHUNK_HOP_SEC,
    sr: int = CFG.SAMPLE_RATE,
):
    """Generator yielding ``(start_sec, chunk)`` padded to *chunk_sec* length."""
    chunk_len = int(chunk_sec * sr)
    hop_len = int(hop_sec * sr)
    n = len(y)
    for start in range(0, n, hop_len):
        chunk = y[start : start + chunk_len]
        if len(chunk) < chunk_len:
            chunk = np.pad(chunk, (0, chunk_len - len(chunk)), mode="wrap")
        yield start / sr, chunk


# ────────────────────────────────────────────────────────────────────────
# Mel on‑disk cache – speeds up CutMix double‑load
# ────────────────────────────────────────────────────────────────────────

_CACHE_SIZE = getattr(CFG, "MEL_CACHE_SIZE", 0)


@lru_cache(maxsize=_CACHE_SIZE if _CACHE_SIZE > 0 else None)
def _load_mel_cached(full_path: str) -> np.ndarray:  # pragma: no cover
    mel = np.load(full_path, allow_pickle=False).astype(np.float32)
    # *Min‑max norm* to [0,1] stabilises CNN training regardless of dB scale
    return (mel - mel.min()) / (mel.max() - mel.min() + 1e-6)


# ────────────────────────────────────────────────────────────────────────
# Dataset + Sampler for training
# ────────────────────────────────────────────────────────────────────────

class FileWiseSampler(Sampler[int]):
    """Yield *one* random chunk per source file each epoch (as in 2024 recipe)."""

    def __init__(self, df: pd.DataFrame, filepath_col: str = "filepath"):
        self.groups: Dict[str, Sequence[int]] = df.groupby(filepath_col).indices
        self.files: List[str] = list(self.groups.keys())

    def __iter__(self) -> Iterable[int]:
        random.shuffle(self.files)
        for fp in self.files:
            yield random.choice(self.groups[fp])

    def __len__(self) -> int:  # noqa: D401 – lint false‑positive
        return len(self.files)


class MelDataset(Dataset):
    """Thin wrapper around pre‑computed mel paths & JSON/NPY label vectors."""

    def __init__(
        self,
        df: pd.DataFrame,
        species2idx: Dict[str, int],
        *,
        augment: bool = False,
    ):
        self.df = df.reset_index(drop=True)
        self.s2i = species2idx
        self.augment = augment
        self.use_soft = bool(getattr(CFG, "USE_SOFT_LABELS", False))

    # ‑‑‑ internal helpers ‑──────────────────────────────────────────────
    def _load_norm(self, rel: str) -> np.ndarray:
        full = str(CFG.PROCESSED_DIR / rel)
        if _CACHE_SIZE > 0:
            mel = _load_mel_cached(full)
        else:
            mel = np.load(full, allow_pickle=False).astype(np.float32)
            mel = (mel - mel.min()) / (mel.max() - mel.min() + 1e-6)
        mel = cv2.resize(mel, CFG.TARGET_SHAPE, interpolation=cv2.INTER_LINEAR)
        return mel

    def _json_to_vec(self, js: str) -> torch.Tensor:
        vec = np.zeros(len(self.s2i), dtype=np.float32)
        for sp, w in json.loads(js).items():
            idx = self.s2i.get(sp)
            if idx is not None:
                vec[idx] = w
        return torch.from_numpy(vec)

    # ‑‑‑ Dataset interface ‑────────────────────────────────────────────
    def __len__(self) -> int:  # noqa: D401
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        mel = self._load_norm(row["mel_path"])

        if self.augment:
            mel = spec_augment(mel)

        # ── label handling ────────────────────────────────────────────
        if self.use_soft:
            if row["label_path"].endswith(".npy"):
                vec = np.load(CFG.PROCESSED_DIR / row["label_path"], allow_pickle=False)
                label_vec = torch.from_numpy(vec.astype(np.float32))
            else:
                label_vec = self._json_to_vec(row["label_json"])
        else:
            sp = row.get("primary_label") or row.get("label")
            label_vec = torch.tensor(self.s2i[sp], dtype=torch.long)

        mel_tensor = torch.tensor(mel).unsqueeze(0)  # [1,H,W]

        # ── CutMix ‑ only during *training* ───────────────────────────
        if (
            self.augment
            and random.random() < getattr(CFG, "CUTMIX_PROB", 0.0)
            and len(self.df) > 1
        ):
            j = random.randint(0, len(self.df) - 1)
            if j == idx:
                j = (j + 1) % len(self.df)
            row2 = self.df.iloc[j]
            mel2 = self._load_norm(row2["mel_path"])
            if self.augment:
                mel2 = spec_augment(mel2)

            if self.use_soft:
                if row2["label_path"].endswith(".npy"):
                    vec2 = np.load(CFG.PROCESSED_DIR / row2["label_path"], allow_pickle=False)
                    label_vec2 = torch.from_numpy(vec2.astype(np.float32))
                else:
                    label_vec2 = self._json_to_vec(row2["label_json"])
            else:
                sp2 = row2.get("primary_label") or row2.get("label")
                label_vec2 = torch.tensor(self.s2i[sp2], dtype=torch.long)

            mel, label_vec = cutmix(mel, label_vec, mel2, label_vec2)
            mel_tensor = torch.tensor(mel).unsqueeze(0)

        weight = float(row.get("weight", 1.0))
        return mel_tensor, label_vec, weight
