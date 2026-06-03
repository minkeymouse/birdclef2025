from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


def load_audio(path: str | Path, sr: int) -> np.ndarray:
    wav, file_sr = sf.read(str(path), dtype="float32", always_2d=False)
    if getattr(wav, "ndim", 1) == 2:
        wav = wav.mean(axis=1)
    if file_sr != sr:
        wav = librosa.resample(wav, orig_sr=file_sr, target_sr=sr)
    return np.asarray(wav, dtype=np.float32)


def load_audio_segment(path: str | Path, start_sec: float, end_sec: float, sr: int) -> np.ndarray:
    wav = load_audio(path, sr)
    return slice_audio_segment(wav, start_sec, end_sec, sr)


def slice_audio_segment(wav: np.ndarray, start_sec: float, end_sec: float, sr: int) -> np.ndarray:
    start = int(round(start_sec * sr))
    end = int(round(end_sec * sr))
    target_len = int(round((end_sec - start_sec) * sr))

    pad_left = max(0, -start)
    pad_right = max(0, end - len(wav))
    start = max(start, 0)
    end = min(end, len(wav))

    seg = wav[start:end]
    if pad_left or pad_right:
        seg = np.pad(seg, (pad_left, pad_right))
    if len(seg) < target_len:
        seg = np.pad(seg, (0, target_len - len(seg)))
    elif len(seg) > target_len:
        seg = seg[:target_len]
    return seg.astype(np.float32, copy=False)
