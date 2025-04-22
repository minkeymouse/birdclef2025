# configure.py – central configuration for the **BirdCLEF 2025** pipeline
# =============================================================================
# One authoritative source of truth for every tunable used by the codebase.
# Edit paths & hyper‑parameters here, re‑run the affected stage, and *never*
# hunt for magic numbers scattered across modules again. ❤
# =============================================================================

from __future__ import annotations

from pathlib import Path
from typing import List


class CFG:  # pylint: disable=too-few-public-methods
    """Global, immutable configuration namespace."""

    # ────────────────────────────────────────────────────────────────────────
    # Reproducibility & runtime
    # ────────────────────────────────────────────────────────────────────────
    SEED: int = 42              # master RNG seed
    DEVICE: str = "cuda"         # "cuda" | "gpu" | "cpu"

    # Helper: stable class ordering (optional)
    CLASSES: List[str] = []

    # ────────────────────────────────────────────────────────────────────────
    # Filesystem layout  – adjust to your environment / Kaggle dataset mount
    # ────────────────────────────────────────────────────────────────────────
    DATA_ROOT: Path = Path("/data/birdclef")

    TRAIN_AUDIO_DIR: Path = DATA_ROOT / "train_audio"
    TRAIN_SOUNDSCAPE_DIR: Path = DATA_ROOT / "train_soundscapes"
    TRAIN_CSV: Path = DATA_ROOT / "train.csv"
    TAXONOMY_CSV: Path = DATA_ROOT / "taxonomy.csv"
    TEST_DIR: Path = DATA_ROOT / "test_soundscapes"
    SAMPLE_SUBMISSION: Path = DATA_ROOT / "sample_submission.csv"

    # Pre‑processing outputs
    PROCESSED_DIR: Path = DATA_ROOT / "processed"  # mels/ labels/ *_metadata.csv

    # Model checkpoints
    MODELS_DIR: Path = DATA_ROOT / "models"
    EFF_MODEL_DIR: Path = MODELS_DIR / "efficientnet"
    REG_MODEL_DIR: Path = MODELS_DIR / "regnety"
    DIFFWAVE_MODEL_DIR: Path = MODELS_DIR / "diffwave"
    BENCHMARK_MODEL: Path = MODELS_DIR / "benchmark"  # ext. classifier or folder

    # Inference artifact
    SUBMISSION_OUT: Path = DATA_ROOT / "submission.csv"

    # ────────────────────────────────────────────────────────────────────────
    # Audio / spectrogram params – **keep consistent across modules**
    # ────────────────────────────────────────────────────────────────────────
    SAMPLE_RATE: int = 32_000
    N_FFT: int = 1024
    HOP_LENGTH: int = 500
    N_MELS: int = 128
    FMIN: int = 40
    FMAX: int = 15_000
    POWER: float = 2.0

    TARGET_SHAPE: tuple[int, int] = (256, 256)  # resize for CNN

    # ────────────────────────────────────────────────────────────────────────
    # Chunking strategy
    # ────────────────────────────────────────────────────────────────────────
    TRAIN_CHUNK_SEC: int = 10
    TRAIN_CHUNK_HOP_SEC: int = 5
    SC_SEG_SEC: int = 5  # evaluation granularity

    FOLD0_RATIO: float = 0.80  # % of cleanest clips used for training

    # ────────────────────────────────────────────────────────────────────────
    # Pre‑processing heuristics
    # ────────────────────────────────────────────────────────────────────────
    TRIM_TOP_DB: int = 20
    RMS_THRESHOLD: float = 0.01
    MIN_RATING: int = 0
    PSEUDO_THRESHOLD: float = 0.50

    USE_SOFT_LABELS: bool = True
    LABEL_WEIGHT_PRIMARY: float = 0.95
    LABEL_WEIGHT_BENCH: float = 0.05

    RARE_COUNT_THRESHOLD: int = 20
    RARE_WEIGHT: float = 2.0
    PSEUDO_WEIGHT: float = 0.5

    MEL_CACHE_SIZE: int = 2048

    # ────────────────────────────────────────────────────────────────────────
    # Data‑augmentation
    # ────────────────────────────────────────────────────────────────────────
    SPEC_AUG_FREQ_MASK_PARAM: int = 10
    SPEC_AUG_TIME_MASK_PARAM: int = 50
    SPEC_AUG_NUM_MASKS: int = 2
    CUTMIX_PROB: float = 0.5

    # ────────────────────────────────────────────────────────────────────────
    # EfficientNet‑B0
    # ────────────────────────────────────────────────────────────────────────
    EFF_NUM_MODELS: int = 2
    EFF_BATCH_SIZE: int = 32
    EFF_EPOCHS: int = 10
    EFF_LR: float = 2e-3
    EFF_WEIGHT_DECAY: float = 1e-4
    EFF_NUM_WORKERS: int = 4

    # ────────────────────────────────────────────────────────────────────────
    # RegNetY‑0.8GF
    # ────────────────────────────────────────────────────────────────────────
    REG_NUM_MODELS: int = 2
    REG_BATCH_SIZE: int = 32
    REG_EPOCHS: int = 10
    REG_LR: float = 2e-3
    REG_WEIGHT_DECAY: float = 1e-4
    REG_NUM_WORKERS: int = 4

    # ────────────────────────────────────────────────────────────────────────
    # DiffWave minority‑class synthesis
    # ────────────────────────────────────────────────────────────────────────
    DIFF_BATCH_SIZE: int = 16
    DIFF_EPOCHS: int = 100
    DIFF_LR: float = 1e-4
    DIFF_NUM_WORKERS: int = 4
    DIFF_RARE_THRESHOLD: int = 10  # if real recordings < 10 ⇒ target for synth

    # ────────────────────────────────────────────────────────────────────────
    # Inference / ensemble parameters
    # ────────────────────────────────────────────────────────────────────────
    INF_BATCH_SIZE: int = 16
    INF_NUM_WORKERS: int = 4
    INF_SMOOTH_NEIGHBORS: int = 2  # ±2 × 5 s segments

    # ────────────────────────────────────────────────────────────────────────
    # CPU optimisation (OpenVINO / ONNX)
    # ────────────────────────────────────────────────────────────────────────
    USE_OPENVINO: bool = True
    OV_NUM_THREADS: int | None = None

    # ────────────────────────────────────────────────────────────────────────
    # Convenience
    # ────────────────────────────────────────────────────────────────────────
    @classmethod
    def use_cuda(cls) -> bool:  # tiny helper
        return str(cls.DEVICE).lower() in {"cuda", "gpu"}


# EOF
