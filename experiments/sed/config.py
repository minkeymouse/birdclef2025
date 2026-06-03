"""SedConfig — dataclass for one SED training run.

Override pattern: `cfg = SedConfig.from_tucker(drop_rate=0.1)` produces a config
that diffs from Tucker spec by exactly one field. silent_drift_check.py uses
this dataclass to surface ALL deviations, intentional or accidental.

A run = one SedConfig + one fold seed. Output is a frozen ckpt + trained val
metrics + the exact config used. Reproducible by replaying the same SedConfig.
"""
from __future__ import annotations
from dataclasses import dataclass, field, fields, asdict, replace
from pathlib import Path
from typing import Optional, Tuple

from .tucker_spec import TUCKER, TuckerSpec


@dataclass
class PseudoConfig:
    """Pseudo-label dataset config (None means no pseudo)."""
    pseudo_npz: Optional[str] = None  # path to filter.py output (kept_probs etc)
    pseudo_share_per_batch: float = 0.40  # Sydorskyi 2025 ratio
    label_threshold: float = 0.5         # treat soft prob > thr as positive in MixUp
    keep_as: str = "soft"                # 'soft' or 'hard'


@dataclass
class FineTuneConfig:
    """Fine-tune from existing ckpts (rather than from-scratch)."""
    base_ckpt_dir: Optional[str] = None  # path containing fold{0..4}/best_ckpt.pt
    bottleneck_remap: bool = False       # legacy bottleneck.0 → bottleneck.1 remap
    finetune_lr: float = 1e-5            # override LR for fine-tune
    finetune_epochs: int = 5             # override EPOCHS for fine-tune


@dataclass
class SedConfig:
    """One SED training run. Defaults = Tucker canonical spec."""

    name: str = "tucker_baseline"
    seed: int = 42
    output_dir: str = "experiments/runs/{name}"
    ANCHOR_CACHE_DIR: Optional[str] = None  # None = exp169v2 default; set to use an extended anchor cache

    # ── Audio / mel ──────────────────────────────────────────────
    SR: int = TUCKER.SR
    TRAIN_DURATION_S: int = TUCKER.TRAIN_DURATION_S
    VAL_DURATION_S: int = TUCKER.VAL_DURATION_S
    N_FFT: int = TUCKER.N_FFT
    HOP_LENGTH: int = TUCKER.HOP_LENGTH
    N_MELS: int = TUCKER.N_MELS
    FMIN: int = TUCKER.FMIN
    FMAX: int = TUCKER.FMAX
    AMPLITUDE_TO_DB_TOP_DB: int = TUCKER.AMPLITUDE_TO_DB_TOP_DB

    # ── Backbone ─────────────────────────────────────────────────
    BACKBONE_NAME: str = TUCKER.BACKBONE_NAME
    BACKBONE_PRETRAINED: bool = True
    BACKBONE_DROP_PATH_RATE: float = 0.1
    BACKBONE_DROP_RATE: Optional[float] = None  # None = don't pass arg (Tucker default)

    # ── SED head ─────────────────────────────────────────────────
    HIDDEN_DIM: int = 512
    GEM_FREQ_P_INIT: float = 3.0
    BOTTLENECK_DROPOUT_1: float = 0.25
    BOTTLENECK_DROPOUT_2: float = 0.5
    ATT_CLA_BIAS: bool = True
    ATT_CLA_INIT: str = "xavier_uniform"
    STOP_GRAD_TO_BACKBONE: bool = True

    # ── Distillation ─────────────────────────────────────────────
    USE_PERCH_DISTILL: bool = True
    PERCH_EMBED_DIM: int = 1536
    ALPHA_DISTILL: float = 1.0

    # ── Loss ─────────────────────────────────────────────────────
    BCE_CLIP_WEIGHT: float = 0.5
    BCE_FRAME_MAX_WEIGHT: float = 0.5
    LOSS_FAMILY: str = "bce"  # 'bce' (Tucker) or 'softauc' (Babych 2025 1st place)
    SOFTAUC_TEMPERATURE: float = 1.0  # temperature for pairwise log-loss

    # ── Optimizer / schedule ─────────────────────────────────────
    LR: float = TUCKER.LR
    MIN_LR: float = TUCKER.MIN_LR
    WD: float = TUCKER.WD
    WARMUP_EPOCHS: int = TUCKER.WARMUP_EPOCHS
    WARMUP_START_FACTOR: float = TUCKER.WARMUP_START_FACTOR
    GRAD_CLIP_NORM: float = TUCKER.GRAD_CLIP_NORM

    # ── Training ─────────────────────────────────────────────────
    EPOCHS: int = TUCKER.EPOCHS
    BATCH: int = TUCKER.BATCH
    USE_AMP: bool = True

    # ── Augmentation ─────────────────────────────────────────────
    AUG_PROB: float = TUCKER.AUG_PROB
    AUG_GAIN_DB_RANGE: Tuple[float, float] = TUCKER.AUG_GAIN_DB_RANGE
    AUG_NOISE_SNR_DB_RANGE: Tuple[float, float] = TUCKER.AUG_NOISE_SNR_DB_RANGE
    SPEC_FREQ_MASK_PARAM: int = TUCKER.SPEC_FREQ_MASK_PARAM
    SPEC_TIME_MASK_PARAM: int = TUCKER.SPEC_TIME_MASK_PARAM
    SPEC_NUM_FREQ_MASKS: int = TUCKER.SPEC_NUM_FREQ_MASKS
    SPEC_NUM_TIME_MASKS: int = TUCKER.SPEC_NUM_TIME_MASKS

    # ── MixUp ────────────────────────────────────────────────────
    USE_FOCAL_FOCAL_MIXUP: bool = TUCKER.USE_FOCAL_FOCAL_MIXUP
    USE_FOCAL_SC_MIXUP: bool = TUCKER.USE_FOCAL_SC_MIXUP
    MIXUP_PROB: float = TUCKER.MIXUP_PROB
    MIXUP_ALPHA: float = TUCKER.MIXUP_ALPHA
    MIXUP_HARD: bool = TUCKER.MIXUP_HARD

    # ── Sampling ─────────────────────────────────────────────────
    SOURCE_SHARE_FOCAL: float = TUCKER.SOURCE_SHARE_FOCAL
    SOURCE_SHARE_SC: float = TUCKER.SOURCE_SHARE_SC
    MIN_SAMPLE: int = TUCKER.MIN_SAMPLE

    # ── Validation ───────────────────────────────────────────────
    N_FOLDS: int = TUCKER.N_FOLDS
    EVAL_SS_N_FILES: Optional[int] = None  # legacy: fixed-N-file holdout. None = per-fold split.
    CHECKPOINT_CRITERION: str = TUCKER.CHECKPOINT_CRITERION  # 'non_s22_macro_auc' / 'val_ss' / 'val_ta'

    # ── Pseudo / Fine-tune extensions ────────────────────────────
    pseudo: Optional[PseudoConfig] = None
    finetune: Optional[FineTuneConfig] = None

    # ── Per-class pos_weight boost (Sprint 2026-05-17 v11 direction) ──
    # Applies BCE pos_weight upweighting for actual-positive samples in
    # rare/ultra-rare classes. Targets F9 unified mechanism (ours
    # under-confident on ultra-rare positives).
    pos_weight_ultra_rare: float = 1.0   # n_train_audio < 5 classes
    pos_weight_rare: float = 1.0          # 5 ≤ n_train_audio < 20 classes

    # ── Noisy Student v3 knobs (Boredom 2025 1st recipe) ─────────
    # MIXUP_PROB = 1.0 applies mixup to every batch — Sydorskyi 2025 2nd
    # also used full mixup. Default Tucker = 0.5 (every other batch).
    # use_pseudo_weighted_sampler: confidence-weighted sampling of pseudo
    # rows by sum of pseudo labels (Boredom recipe).
    use_pseudo_weighted_sampler: bool = False

    # ── Run metadata ─────────────────────────────────────────────
    notes: str = ""

    # ── Helpers ──────────────────────────────────────────────────
    @classmethod
    def from_tucker(cls, **overrides) -> "SedConfig":
        """Start from Tucker spec, apply overrides, return new config."""
        return cls(**overrides) if not overrides else replace(cls(), **overrides)

    def diff_from_tucker(self) -> dict:
        """Return dict of fields where this config differs from canonical Tucker."""
        diff = {}
        tucker_d = {f.name: getattr(TUCKER, f.name) for f in fields(TUCKER)}
        my_d = asdict(self)
        for k, tv in tucker_d.items():
            if k in my_d and my_d[k] != tv:
                diff[k] = {"tucker": tv, "ours": my_d[k]}
        return diff

    def resolved_output_dir(self) -> Path:
        """Output dir with {name} and {seed} placeholders.

        For backward-compat, if output_dir doesn't contain {seed}, return as-is.
        Otherwise interpolate with self.seed.
        """
        s = self.output_dir
        if "{seed}" in s:
            return Path(s.format(name=self.name, seed=self.seed))
        return Path(s.format(name=self.name))


# ── Pre-canned configs ─────────────────────────────────────────────

def tucker_canonical() -> SedConfig:
    """Tucker recipe verbatim, no extras."""
    return SedConfig(name="tucker_canonical")


def exp169_legacy() -> SedConfig:
    """Replicates exp169v2 — what we actually trained, with the silent-drift bugs.
    Useful for retroactively documenting what our 0.938 anchor really is."""
    return SedConfig(
        name="exp169_legacy",
        BACKBONE_DROP_RATE=0.1,           # ← silent drift since exp169
        ATT_CLA_INIT="default",           # ← no xavier, used timm default
        EPOCHS=20,                         # exp169v2 used 20, not Tucker's 25
        EVAL_SS_N_FILES=11,                # ← fixed 11-file SS holdout (not per-fold)
        CHECKPOINT_CRITERION="val_ta",     # ← exp169v2 saved best val_TA, not val_SS
        WARMUP_EPOCHS=1,                    # exp169v2 used 1 ep warmup
        USE_FOCAL_FOCAL_MIXUP=False,       # exp169v2: no waveform mixup (just BG aug)
        USE_FOCAL_SC_MIXUP=False,
        MIN_SAMPLE=0,                       # exp169v2: no rare upsampling
        notes="What our 0.938 LB anchor (exp169v2) actually was — silent drifts documented",
    )


def exp171_misnamed_tucker() -> SedConfig:
    """exp171 was named 'Tucker-exact' but had silent drifts. Documents them."""
    return SedConfig(
        name="exp171_misnamed_tucker",
        BACKBONE_DROP_RATE=0.1,           # silent drift
        ATT_CLA_INIT="default",           # silent drift
        EVAL_SS_N_FILES=11,                # silent drift
        CHECKPOINT_CRITERION="val_ss",     # used val_SS, not non_s22_macro_auc
        WARMUP_EPOCHS=2,                   # tucker-correct
        EPOCHS=25,                          # tucker-correct
        USE_FOCAL_FOCAL_MIXUP=True,        # tucker-correct
        USE_FOCAL_SC_MIXUP=True,           # tucker-correct
        MIN_SAMPLE=20,                      # tucker-correct
        notes="Named 'Tucker-exact' but 4 deviations remain. LB 0.937.",
    )


def exp175_tucker_actually() -> SedConfig:
    """exp175 = exp171 minus drop_rate + xavier init. Closest we have to Tucker.

    Output dir uses {seed} placeholder so multi-seed runs don't collide:
      seed=42 → experiments/_data_pipelines/exp175_outputs/seed42/
      seed=43 → .../seed43/
    """
    return SedConfig(
        name="exp175_tucker_actually",
        output_dir="experiments/_data_pipelines/exp175_outputs/seed{seed}",
        BACKBONE_DROP_RATE=None,            # Tucker correct
        ATT_CLA_INIT="xavier_uniform",      # Tucker correct
        EVAL_SS_N_FILES=11,                 # ← still legacy here
        CHECKPOINT_CRITERION="val_ss",      # ← still legacy here
        WARMUP_EPOCHS=2,
        EPOCHS=25,
        USE_FOCAL_FOCAL_MIXUP=True,
        USE_FOCAL_SC_MIXUP=True,
        MIN_SAMPLE=20,
        notes="exp175 fixes drop_rate + xavier. 2 deviations remain (eval_ss, criterion).",
    )


def exp177_b1_backbone() -> SedConfig:
    """exp177 = exp175 with B1 backbone instead of B0. Multi-arch ensemble.

    tf_efficientnet_b1.ns_jft_in1k has more capacity (~6.5M vs 4M params).
    Same recipe (drift fix + xavier init) as exp175.
    """
    return SedConfig(
        name="exp177_b1_backbone",
        output_dir="experiments/_data_pipelines/exp177_outputs/seed{seed}",
        BACKBONE_NAME="tf_efficientnet_b1.ns_jft_in1k",
        BACKBONE_DROP_RATE=None,
        ATT_CLA_INIT="xavier_uniform",
        EVAL_SS_N_FILES=11,
        CHECKPOINT_CRITERION="val_ss",
        WARMUP_EPOCHS=2,
        EPOCHS=25,
        USE_FOCAL_FOCAL_MIXUP=True,
        USE_FOCAL_SC_MIXUP=True,
        MIN_SAMPLE=20,
        notes="exp177 = exp175 + B1 backbone. Multi-arch ensemble member.",
    )


def exp178_softauc() -> SedConfig:
    """exp178 = exp175 + SoftAUCLoss instead of BCE.

    Babych 2025 1st place writeup ("Multi-Iterative Noisy Student") used SoftAUCLoss
    for direct AUC optimization. Same backbone (B0) and recipe as exp175 otherwise.
    Single change: BCE → softauc.
    """
    return SedConfig(
        name="exp178_softauc",
        output_dir="experiments/_data_pipelines/exp178_outputs/seed{seed}",
        BACKBONE_DROP_RATE=None,
        ATT_CLA_INIT="xavier_uniform",
        EVAL_SS_N_FILES=11,
        CHECKPOINT_CRITERION="val_ss",
        WARMUP_EPOCHS=2,
        EPOCHS=25,
        USE_FOCAL_FOCAL_MIXUP=True,
        USE_FOCAL_SC_MIXUP=True,
        MIN_SAMPLE=20,
        LOSS_FAMILY="softauc",
        SOFTAUC_TEMPERATURE=1.0,
        notes="exp178 = exp175 + softauc loss. Babych 2025 1st place direct-AUC optimization.",
    )


def exp176_per_fold_ss() -> SedConfig:
    """exp176 = exp175 + per-fold SS split (Tucker-correct).
    Eliminates EVAL_SS_N_FILES=11 drift. Each fold uses 4/5 of SS files for train,
    1/5 for eval. Over 5 folds, every SS file is in eval exactly once.
    Hypothesis: rare-class supervision improves because no SS file is permanently held out.
    """
    return SedConfig(
        name="exp176_per_fold_ss",
        output_dir="experiments/_data_pipelines/exp176_outputs",
        BACKBONE_DROP_RATE=None,
        ATT_CLA_INIT="xavier_uniform",
        EVAL_SS_N_FILES=None,               # ← per-fold split (Tucker correct)
        CHECKPOINT_CRITERION="val_ss",
        WARMUP_EPOCHS=2,
        EPOCHS=25,
        USE_FOCAL_FOCAL_MIXUP=True,
        USE_FOCAL_SC_MIXUP=True,
        MIN_SAMPLE=20,
        notes="exp176 fixes EVAL_SS_N_FILES drift in addition. 1 deviation (CHECKPOINT_CRITERION).",
    )


def exp182_pos_boost() -> SedConfig:
    """exp182 = exp175 + per-class pos_weight boost on rare/ultra-rare.

    Sprint 2026-05-17 v11 direction: F9 confirmed ours under-confident on
    ultra-rare positives. v10 (filter design) was wrong direction (suppress);
    exp182 tests opposite (boost actual-positive supervision).

    pos_weight semantics in BCE: positive samples are weighted by pos_weight,
    so loss gradient on positive ultra-rare classes is amplified ×3.0.
    Negative samples unchanged → baseline noise unaffected.

    Hypothesis: macro AUC stays ~v53 baseline (no degradation on common
    classes) AND unlabeled SS ultra-rare p99 ratio approaches 1.0× (Tucker
    level) because the model learns to spike on actual ultra-rare positives.

    Anti-pattern: if macro AUC drops > -0.005, this is v10-style failure
    (over-aggressive direction). If macro AUC stays but ultra-rare p99 ratio
    INCREASES (worse than v53 1.83×), boost induces baseline elevation
    (symmetric inverse of v10). Either failure mode triggers v12 redesign.
    """
    return SedConfig(
        name="exp182_pos_boost",
        output_dir="experiments/_data_pipelines/exp182_outputs",
        BACKBONE_DROP_RATE=None,
        ATT_CLA_INIT="xavier_uniform",
        EVAL_SS_N_FILES=11,
        CHECKPOINT_CRITERION="val_ss",
        WARMUP_EPOCHS=2,
        EPOCHS=25,
        USE_FOCAL_FOCAL_MIXUP=True,
        USE_FOCAL_SC_MIXUP=True,
        MIN_SAMPLE=20,
        pos_weight_ultra_rare=3.0,
        pos_weight_rare=2.0,
        notes="exp182 = exp175 + pos_weight boost. Sprint 2026-05-17 v11 direction.",
    )


def exp180_pseudo_v10() -> SedConfig:
    """exp180 = exp176 + pseudo dataset v10 (axis-2-aware filter, 2026-05-17 Phase F).

    v10 = refined pseudo labels (147k windows from 71k unique (file, end) tuples):
    - Tier A common (n_train ≥ 20): 121k rows, keep ~87% of v7
    - Tier B rare (5 ≤ n_train < 20): 20k rows, keep ~73% of v7
    - Tier C ultra-rare (n_train < 5): 5k rows, keep ~7% of v7 (axis-2 noise filter)

    pseudo_share_per_batch = 0.30 (lower than Sydorskyi 0.40 to bias toward labeled data;
    rare-class pseudo signal in v10 already targets ultra-rare, so we don't need overweight).

    Hypothesis: training a new SED variant on v10 supervision should make the resulting
    model less prone to ultra-rare baseline elevation in broader OOD (axis-2 reduced at
    weights level rather than inference patch).

    Validation:
      - Run silent_drift_check.py first
      - Compare disagreement breakdown (ours_v10 vs Tucker) — should reduce
        n_train<5 picks gap (Tucker 81 vs ours 34 → narrower)
      - Compare unlabeled SS ours/tucker p99 ratio — should drop from 2.2× toward 1.0×
      - LB submit only if local axis-2 metrics improve AND silent_drift_check PASS
    """
    return SedConfig(
        name="exp180_pseudo_v10",
        output_dir="experiments/_data_pipelines/exp180_outputs",
        BACKBONE_DROP_RATE=None,
        ATT_CLA_INIT="xavier_uniform",
        EVAL_SS_N_FILES=None,               # per-fold (Tucker correct, exp176 inheritance)
        CHECKPOINT_CRITERION="val_ss",
        WARMUP_EPOCHS=2,
        EPOCHS=25,
        USE_FOCAL_FOCAL_MIXUP=True,
        USE_FOCAL_SC_MIXUP=True,
        MIN_SAMPLE=20,
        pseudo=PseudoConfig(
            pseudo_npz="experiments/sed/pseudo_v10.npz",
            pseudo_share_per_batch=0.30,
        ),
        notes="exp180 = exp176 + v10 pseudo (axis-2-aware filter). Sprint 2026-05-17 Phase F.",
    )


def exp184_noisy_student_balanced() -> SedConfig:
    """exp184 = exp183 + F6 fix (per-class balanced pseudo + labeled SS exclusion).

    v62 post-mortem (2026-05-18) identified: exp183 over-predicts pseudo-rich
    classes (65380 30k chunks → 10 FP) and under-predicts pseudo-poor classes
    (516975 4 chunks → 8 missed). Class distribution in pseudo = class bias in student.

    F6 fix in pseudo_v12_balanced.npz:
      (1) per-class cap = 1500 chunks (drops ~30k from 65380, ~25k from 22973, etc.)
      (2) Exclude 66 labeled SS files from pseudo gen (was 1.6% direct leak)
      Expected: ~15-20k chunks total (vs 49k v11)

    All other hyperparams identical to exp183 (same Noisy Student noise injection,
    40% pseudo share, drop_path 0.2). Isolates per-class balance effect.

    Hypothesis: if v62 regression was driven by class imbalance, exp184 should
    recover toward v53 baseline. If exp184 still regresses, the issue is
    fundamental to pseudo-from-own-teacher (channel limit) — need foundation
    swap or multi-arch teacher next.
    """
    return SedConfig(
        name="exp184_noisy_student_balanced",
        output_dir="experiments/_data_pipelines/exp184_outputs",
        BACKBONE_DROP_RATE=None,
        BACKBONE_DROP_PATH_RATE=0.2,
        ATT_CLA_INIT="xavier_uniform",
        EVAL_SS_N_FILES=None,
        CHECKPOINT_CRITERION="val_ss",
        WARMUP_EPOCHS=2,
        EPOCHS=25,
        USE_FOCAL_FOCAL_MIXUP=True,
        USE_FOCAL_SC_MIXUP=True,
        MIN_SAMPLE=20,
        pseudo=PseudoConfig(
            pseudo_npz="experiments/sed/pseudo_v12_balanced.npz",
            pseudo_share_per_batch=0.40,
        ),
        notes="exp184 = exp183 + F6 (per-class cap 1500 + labeled SS exclusion). "
              "Diagnostic-driven fix for v62 -0.007 regression caused by pseudo "
              "class imbalance bias in student.",
    )


def exp185_noisy_student_full() -> SedConfig:
    """exp185 = full Boredom 2025 1st Noisy Student recipe.

    Additions vs exp183/exp184:
      - pseudo with power transform alpha=0.7 (Boredom's reshape on probs)
      - MIXUP_PROB = 1.0 (Sydorskyi 2025 2nd exact ratio, every batch)
      - use_pseudo_weighted_sampler = True (confidence-weighted)
      - Keep drop_path 0.2, pseudo_share 0.40 from exp183.

    Targets the v62 -0.007 regression by adding the noise-injection
    components that exp183/184 missed.

    Validation:
      - same gates as exp183 (labeled AUC ≥ 0.997, p99 ratio ≤ 1.5×)
      - smoke test fold 0 first, then 5-fold if smoke OK
    """
    return SedConfig(
        name="exp185_noisy_student_full",
        output_dir="experiments/_data_pipelines/exp185_outputs",
        BACKBONE_DROP_RATE=None,
        BACKBONE_DROP_PATH_RATE=0.2,        # Noisy Student noise
        ATT_CLA_INIT="xavier_uniform",
        EVAL_SS_N_FILES=None,               # per-fold (Tucker correct)
        CHECKPOINT_CRITERION="val_ss",
        WARMUP_EPOCHS=2,
        EPOCHS=25,
        USE_FOCAL_FOCAL_MIXUP=True,
        USE_FOCAL_SC_MIXUP=True,
        MIXUP_PROB=1.0,                     # Boredom/Sydorskyi: every batch mixed
        MIN_SAMPLE=20,
        pseudo=PseudoConfig(
            pseudo_npz="experiments/sed/pseudo_noisy_student_iter1_p0.70.npz",
            pseudo_share_per_batch=0.40,    # Sydorskyi exact
        ),
        use_pseudo_weighted_sampler=True,
        notes="exp185 = full Noisy Student recipe: power transform + mixup 1.0 + "
              "weighted sampler. Diagnostic fix for exp183/184 partial-recipe "
              "regression (v62 = -0.007).",
    )


def exp183_noisy_student_iter1() -> SedConfig:
    """exp183 = exp176 + Noisy Student iteration 1 (Sydorskyi 2025 2nd place recipe).

    Differs from exp180 (which used v10 binarized pseudo, failed -0.025 AUC):
      - pseudo_npz = pseudo_v11_noisy.npz (REAL soft probs from 10-ckpt ensemble teacher,
        NOT binarized — v10 had all probs = 1.0, that was the bug)
      - pseudo_share_per_batch = 0.40 (Sydorskyi exact ratio, vs exp180's 0.30)
      - BACKBONE_DROP_PATH_RATE = 0.2 (Noisy Student asymmetric noise: clean teacher,
        noisy student. Tucker base = 0.1; bumped to 0.2 for student-side regularization)

    Selection logic baked into pseudo_v11 generation (2nd place exact):
      - Chunk kept iff max(ensemble_prob) ≥ 0.5
      - Per retained chunk, classes with prob < 0.1 zeroed out
      - Remaining classes kept AS SOFT (mean ensemble sigmoid)

    Why this should work where v3/v7/v10/v11 didn't (sprint F11 failure mode):
      - Soft labels prevent student from learning hard misclassification
      - Ensemble teacher (10 ckpts) > single-model teacher (v33 in v3/v7)
      - Noise asymmetry (drop_path higher) breaks circular distillation per Xie 2019

    Validation gates (Phase C):
      - labeled SS macro AUC ≥ 0.997 (v53 baseline 0.998; allow 1 σ slack)
      - unlabeled SS p99 ratio vs Tucker ≤ 1.5× (v53 baseline 1.83×; aim ≤1.5×)
      - Pearson with v53 in [0.93, 0.97] (different enough to add info, not orthogonal)
    """
    return SedConfig(
        name="exp183_noisy_student_iter1",
        output_dir="experiments/_data_pipelines/exp183_outputs",
        BACKBONE_DROP_RATE=None,
        BACKBONE_DROP_PATH_RATE=0.2,        # Noisy Student: increase student noise
        ATT_CLA_INIT="xavier_uniform",
        EVAL_SS_N_FILES=None,               # per-fold (Tucker correct)
        CHECKPOINT_CRITERION="val_ss",
        WARMUP_EPOCHS=2,
        EPOCHS=25,
        USE_FOCAL_FOCAL_MIXUP=True,
        USE_FOCAL_SC_MIXUP=True,
        MIN_SAMPLE=20,
        pseudo=PseudoConfig(
            pseudo_npz="experiments/sed/pseudo_v11_noisy.npz",
            pseudo_share_per_batch=0.40,    # Sydorskyi exact (vs exp180's 0.30)
        ),
        notes="exp183 = Noisy Student iter 1. Sydorskyi 2nd place recipe with REAL soft labels. "
              "First proper Noisy Student attempt — prior pseudo v3/v7/v10/v11 all used "
              "binarized labels which broke the soft signal.",
    )


def exp186_deep_b1_softauc_pseudo() -> SedConfig:
    """exp186 = DEEP run combining proper-recipe axes that were only ever tried SEPARATELY:
      - B1 backbone (decorrelate from Tucker B0; exp177 proven)
      - SoftAUCLoss (direct macro-AUC; exp178 proven wired in model.py)
      - Tucker-5fold teacher pseudo with TUNED threshold + soft labels (vs v11's blind 0.5)
      - drop_path 0.2 (Noisy Student student-side noise)
      - CHECKPOINT_CRITERION non_s22_macro_auc (Tucker-correct; masks the site-22 bias that
        prior exp175-185 'val_ss' criterion baked in)
    Stage A: teacher = Tucker (validates machinery). Stage B will swap pseudo_npz for a
    diverse-teacher ensemble (reconstruct exp50 HGNet / exp59 ConvNeXt / exp84b) to break the
    teacher-correlation ceiling that capped v11/v12 (Pearson 0.97-0.99).
    """
    return SedConfig(
        name="exp186_deep_b1_softauc_pseudo",
        output_dir="experiments/_data_pipelines/exp186_outputs/seed{seed}",
        BACKBONE_NAME="tf_efficientnet_b1.ns_jft_in1k",
        BACKBONE_DROP_RATE=None,
        BACKBONE_DROP_PATH_RATE=0.2,
        ATT_CLA_INIT="xavier_uniform",
        EVAL_SS_N_FILES=None,                       # per-fold (Tucker-correct)
        CHECKPOINT_CRITERION="non_s22_macro_auc",   # Tucker-correct (masks site-22)
        WARMUP_EPOCHS=2,
        EPOCHS=25,
        USE_FOCAL_FOCAL_MIXUP=True,
        USE_FOCAL_SC_MIXUP=True,
        MIN_SAMPLE=20,
        LOSS_FAMILY="softauc",
        pseudo=PseudoConfig(
            pseudo_npz="experiments/sed/pseudo_deep_tucker.npz",
            pseudo_share_per_batch=0.40,
        ),
        notes="exp186 = B1 + softauc + Tucker-teacher pseudo(tuned thr) + drop_path0.2 + "
              "non_s22 criterion. Combines proper-recipe axes never combined. Stage A.",
    )


def exp187_hetero_teacher_pseudo() -> SedConfig:
    """exp187 = exp186 but with HETEROGENEOUS-teacher pseudo (Tucker 0.65 + exp59 ConvNeXt 0.35,
    Pearson 0.70 = genuinely decorrelated, unlike the B0-correlated v11/v12/v62 teachers). The one
    distillation variant whose teacher diversity is real. Gated smoke test: train fold 0, accept only
    if val_SS(non_s22)≥0.930 AND Pearson(student,Tucker)<0.93. Blind-submit downside = 1 slot (select-best)."""
    return SedConfig(
        name="exp187_hetero_teacher_pseudo",
        output_dir="experiments/_data_pipelines/exp187_outputs/seed{seed}",
        BACKBONE_NAME="tf_efficientnet_b1.ns_jft_in1k",
        BACKBONE_DROP_RATE=None,
        BACKBONE_DROP_PATH_RATE=0.2,
        ATT_CLA_INIT="xavier_uniform",
        EVAL_SS_N_FILES=None,
        CHECKPOINT_CRITERION="non_s22_macro_auc",
        WARMUP_EPOCHS=2,
        EPOCHS=25,
        USE_FOCAL_FOCAL_MIXUP=True,
        USE_FOCAL_SC_MIXUP=True,
        MIXUP_PROB=1.0,
        MIN_SAMPLE=20,
        LOSS_FAMILY="softauc",
        pseudo=PseudoConfig(
            pseudo_npz="experiments/sed/pseudo_hetero_teacher.npz",
            pseudo_share_per_batch=0.40,
        ),
        use_pseudo_weighted_sampler=True,
        notes="exp187: B1 + SoftAUC + heterogeneous teacher pseudo (Tucker 0.65 + exp59 ConvNeXt 0.35). "
              "Breaks the B0-correlated ceiling of v11/v12/v62. Gated smoke test.",
    )


def exp188_convnext_hetero_pseudo() -> SedConfig:
    """exp188 = exp187 but ConvNeXt-tiny student (2nd architecture for a multi-arch distilled ensemble,
    Babych-2025-style). Same hetero pseudo + SoftAUC + noisy-student. Ensemble with exp187 (B1) → more
    diverse SED → may drag less / generalize better to the blind-184 species. Idle-GPU bounded test (fold0 first)."""
    return SedConfig(
        name="exp188_convnext_hetero_pseudo",
        output_dir="experiments/_data_pipelines/exp188_outputs/seed{seed}",
        BACKBONE_NAME="convnext_tiny.fb_in22k_ft_in1k",
        BACKBONE_DROP_RATE=None,
        BACKBONE_DROP_PATH_RATE=0.2,
        ATT_CLA_INIT="xavier_uniform",
        EVAL_SS_N_FILES=None,
        CHECKPOINT_CRITERION="non_s22_macro_auc",
        WARMUP_EPOCHS=2,
        EPOCHS=25,
        USE_FOCAL_FOCAL_MIXUP=True,
        USE_FOCAL_SC_MIXUP=True,
        MIXUP_PROB=1.0,
        MIN_SAMPLE=20,
        LOSS_FAMILY="softauc",
        pseudo=PseudoConfig(
            pseudo_npz="experiments/sed/pseudo_hetero_teacher.npz",
            pseudo_share_per_batch=0.40,
        ),
        use_pseudo_weighted_sampler=True,
        notes="exp188 = exp187 + ConvNeXt-tiny (2nd arch for multi-arch distilled ensemble).",
    )


def exp189_tucker_external_nonaves() -> SedConfig:
    """exp189 = canonical Tucker recipe (exp175) + EXTERNAL non-Aves focal data folded into the anchor
    cache (experiments/_data_pipelines/exp189_outputs/anchors.npz = exp169v2 + ~555 external non-Aves
    clips from data/external). The BirdCLEF-2025 winners' KEY ingredient (external non-Aves supervision)
    that this stack never used. This STRENGTHENS the SED with data — it does NOT dilute Tucker with a
    weaker model, the structural difference from the refuted exp187. BCE keeps it calibrated/blendable
    like Tucker (avoids exp187's SoftAUC logit-scale blend distortion). Site-safe checkpoint selection."""
    return SedConfig(
        name="exp189_tucker_external_nonaves",
        output_dir="experiments/_data_pipelines/exp189_outputs/seed{seed}",
        ANCHOR_CACHE_DIR="experiments/_data_pipelines/exp189_outputs",
        BACKBONE_DROP_RATE=None,            # Tucker correct
        ATT_CLA_INIT="xavier_uniform",      # Tucker correct
        EVAL_SS_N_FILES=None,
        CHECKPOINT_CRITERION="non_s22_macro_auc",   # site-safe (multi-site test, not site-22)
        WARMUP_EPOCHS=2,
        EPOCHS=25,
        USE_FOCAL_FOCAL_MIXUP=True,
        USE_FOCAL_SC_MIXUP=True,
        MIN_SAMPLE=20,
        LOSS_FAMILY="bce",                  # calibrated like Tucker → clean to blend
        notes="exp189 = Tucker recipe + external non-Aves data (winners' ingredient, never used in this stack).",
    )
