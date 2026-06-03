"""Canonical Tucker SED spec — single source of truth.

Source: tuckerarrants/bc2026-distilled-sed (Tucker's published training notebook,
pulled to /tmp/recon_kernels/_tucker_sed/bc2026-distilled-sed.ipynb on 2026-05-07).

Every hyperparam below is *verbatim* from that notebook's S1 (Imports + Config)
and S3 (Model). If you derive from this spec for an experiment, override only
the variables you intentionally change. silent_drift_check.py compares any
SedConfig vs this canonical reference and flags deviations.

Why this file exists: from exp169 (commit 80783bd, 2026-04-30) through exp174b,
all 8 SED training scripts carried `drop_rate=0.1` in addition to
`drop_path_rate=0.1`. Tucker's canonical recipe uses `drop_path_rate=0.1` only.
We did not detect this for ~5 weeks because there was no canonical spec to diff
against. This file fixes that.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Tuple


@dataclass(frozen=True)
class TuckerSpec:
    """Tucker bc2026-distilled-sed published recipe — frozen reference."""

    # ─── Audio / mel ─────────────────────────────────────────────
    SR: int = 32000
    TRAIN_DURATION_S: int = 5
    VAL_DURATION_S: int = 5
    N_FFT: int = 2048
    HOP_LENGTH: int = 512
    N_MELS: int = 256
    FMIN: int = 20
    FMAX: int = 16000
    AMPLITUDE_TO_DB_TOP_DB: int = 80

    # ─── Backbone ────────────────────────────────────────────────
    BACKBONE_NAME: str = "tf_efficientnet_b0.ns_jft_in1k"
    BACKBONE_PRETRAINED: bool = True
    BACKBONE_IN_CHANS: int = 1
    BACKBONE_NUM_CLASSES: int = 0
    BACKBONE_GLOBAL_POOL: str = ""
    BACKBONE_DROP_PATH_RATE: float = 0.1
    # NB: drop_rate is NOT set by Tucker; timm default = 0.0
    BACKBONE_DROP_RATE: float | None = None  # explicit None = "do not pass arg"

    # ─── SED head architecture ────────────────────────────────────
    HIDDEN_DIM: int = 512
    GEM_FREQ_P_INIT: float = 3.0
    BOTTLENECK_DROPOUT_1: float = 0.25  # before Linear
    BOTTLENECK_DROPOUT_2: float = 0.5   # after ReLU
    ATT_CLA_BIAS: bool = True
    ATT_CLA_INIT: str = "xavier_uniform"  # NOT timm default
    ATT_CLA_BIAS_INIT: float = 0.0

    # Stop gradient: SED head trained on h.detach() (backbone trains via distill only)
    STOP_GRAD_TO_BACKBONE: bool = True

    # ─── Distillation ─────────────────────────────────────────────
    USE_PERCH_DISTILL: bool = True
    PERCH_EMBED_DIM: int = 1536
    DISTILL_HEAD_GAP: bool = True   # AdaptiveAvgPool2d(1) → Flatten → Linear
    ALPHA_DISTILL: float = 1.0      # weight for MSE distill loss

    # ─── Loss ─────────────────────────────────────────────────────
    BCE_CLIP_WEIGHT: float = 0.5
    BCE_FRAME_MAX_WEIGHT: float = 0.5

    # ─── Optimizer / schedule ─────────────────────────────────────
    OPTIMIZER: str = "AdamW"
    LR: float = 5e-4
    MIN_LR: float = 1e-6  # Tucker's eta_min for cosine
    WD: float = 1e-4
    WARMUP_EPOCHS: int = 2
    WARMUP_START_FACTOR: float = 1 / 25  # LinearLR start_factor
    SCHEDULER: str = "warmup_cosine"
    GRAD_CLIP_NORM: float = 1.0

    # ─── Training ─────────────────────────────────────────────────
    EPOCHS: int = 25
    BATCH: int = 64
    USE_AMP: bool = True

    # ─── Augmentation ─────────────────────────────────────────────
    AUG_PROB: float = 0.5
    AUG_GAIN_DB_RANGE: Tuple[float, float] = (-6.0, 6.0)
    AUG_NOISE_SNR_DB_RANGE: Tuple[float, float] = (10.0, 30.0)
    SPEC_FREQ_MASK_PARAM: int = 10
    SPEC_TIME_MASK_PARAM: int = 10
    SPEC_NUM_FREQ_MASKS: int = 1
    SPEC_NUM_TIME_MASKS: int = 2

    # ─── MixUp ────────────────────────────────────────────────────
    USE_FOCAL_FOCAL_MIXUP: bool = True
    USE_FOCAL_SC_MIXUP: bool = True
    MIXUP_PROB: float = 0.5
    MIXUP_ALPHA: float = 0.4
    MIXUP_HARD: bool = True  # label union via np.maximum

    # ─── Sampling ─────────────────────────────────────────────────
    SAMPLER: str = "MixSamp"  # deterministic per-batch source mixing
    SOURCE_SHARE_FOCAL: float = 0.9
    SOURCE_SHARE_SC: float = 0.1
    MIN_SAMPLE: int = 20  # rare-species upsampling

    # ─── Validation ───────────────────────────────────────────────
    N_FOLDS: int = 5
    FOCAL_FOLD_BY: str = "primary_label_stratified"
    SC_FOLD_BY: str = "filename_per_fold"  # NOT fixed-N-file holdout
    CHECKPOINT_CRITERION: str = "non_s22_macro_auc"  # masks S22 site

    # ─── Inference ────────────────────────────────────────────────
    INFER_BLEND: str = "0.5*sigmoid(clip)+0.5*sigmoid(frame_max)"


# Canonical instance
TUCKER = TuckerSpec()


def to_dict(spec: TuckerSpec | None = None) -> dict:
    """Serialize spec to plain dict (for diff/yaml)."""
    s = spec or TUCKER
    return {f.name: getattr(s, f.name) for f in s.__dataclass_fields__.values()}


# What Tucker's published recipe does NOT define / leaves to default:
# - random seed (uses 42 in S1)
# - val_files split for soundscapes (per-fold via filename groups)
# - exact validation eval batch size
# - exact ONNX export wrapper details (re-mapped to Conv1d for tracing in S6)
