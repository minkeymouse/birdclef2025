import librosa
import numpy as np
from scipy.ndimage import zoom


def _resize_2d(x: np.ndarray, out_freq: int, out_time: int) -> np.ndarray:
    zf = out_freq / x.shape[0]
    zt = out_time / x.shape[1]
    return zoom(x, (zf, zt), order=1).astype(np.float32)


def _normalize_channel(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    mean = float(x.mean())
    std = float(x.std())
    if std < 1e-6:
        std = 1.0
    return (x - mean) / std


def make_feature(wav: np.ndarray, cfg: dict) -> np.ndarray:
    sr = int(cfg["audio"]["sample_rate"])
    fcfg = cfg["feature"]
    mel = librosa.feature.melspectrogram(
        y=wav,
        sr=sr,
        n_fft=int(fcfg["n_fft"]),
        hop_length=int(fcfg["hop_length"]),
        n_mels=int(fcfg["n_mels"]),
        fmin=float(fcfg["fmin"]),
        fmax=float(fcfg["fmax"]),
        power=2.0,
    ).astype(np.float32)

    channels = []
    if bool(fcfg.get("use_logmel", True)):
        logmel = np.log1p(mel)
        channels.append(logmel)
    if bool(fcfg.get("use_pcen", True)):
        pcen = librosa.pcen(
            mel * (2**31),
            sr=sr,
            hop_length=int(fcfg["hop_length"]),
            gain=float(fcfg.get("pcen_gain", 0.98)),
            bias=float(fcfg.get("pcen_bias", 2.0)),
            power=float(fcfg.get("pcen_power", 0.5)),
            time_constant=float(fcfg.get("pcen_time_constant", 0.4)),
            eps=1e-6,
        ).astype(np.float32)
        channels.append(pcen)
    if not channels:
        raise ValueError("At least one feature channel must be enabled.")

    out = []
    for ch in channels:
        ch = _resize_2d(ch, int(fcfg["image_freq"]), int(fcfg["image_time"]))
        if fcfg.get("normalize", "per_channel") == "per_channel":
            ch = _normalize_channel(ch)
        out.append(ch)
    return np.stack(out, axis=0).astype(np.float32)

