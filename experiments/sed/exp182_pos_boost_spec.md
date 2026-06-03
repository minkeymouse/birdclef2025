# exp182 — Ultra-rare positive boost SED training (v11 successor to v10)

**Status**: design only, NOT implemented. Triggered by Phase G v10 local validation result
showing filter-design = wrong direction.

## Hypothesis (post-sprint F9 confirmation)

Sprint Phase G confirmed F9 (unified mechanism): ours under-confident on ultra-rare
positives. v10 filter design (Tier C 92% dropped) crushed model confidence on
ultra-rare uniformly → macro AUC -0.025 catastrophic regression.

**True fix direction**: keep ultra-rare supervision intact AND apply higher
weight on actual-positive ultra-rare samples → model learns "ultra-rare positives
are real signal worth fighting for."

## Required code changes

### config.py — pos_weight option
```python
@dataclass
class SedConfig:
    ...
    # NEW: per-class positive sample weight for BCE (boost ultra-rare)
    pos_weight_ultra_rare: float = 1.0   # multiplier for n_train < 5 classes
    pos_weight_rare: float = 1.0          # multiplier for n_train < 20 classes
```

### model.py — hybrid_loss with pos_weight
```python
def hybrid_loss(..., cfg: SedConfig, ...):
    ...
    # Build per-class pos_weight tensor if requested
    pw_ur = getattr(cfg, "pos_weight_ultra_rare", 1.0)
    pw_r = getattr(cfg, "pos_weight_rare", 1.0)
    if pw_ur != 1.0 or pw_r != 1.0:
        # Need to load n_per_sp once (cached at module level)
        pos_weight = torch.ones(y.shape[1], device=clip_logits.device)
        pos_weight[n_train_arr < 5] = pw_ur
        pos_weight[(n_train_arr >= 5) & (n_train_arr < 20)] = pw_r
        bce_clip = F.binary_cross_entropy_with_logits(clip_logits, y, pos_weight=pos_weight)
        bce_frame = F.binary_cross_entropy_with_logits(frame_max, y, pos_weight=pos_weight)
    else:
        bce_clip = F.binary_cross_entropy_with_logits(clip_logits, y)
        bce_frame = F.binary_cross_entropy_with_logits(frame_max, y)
    ...
```

### Config factory
```python
def exp182_pos_boost() -> SedConfig:
    """exp182 = exp175 + pos_weight 3× on ultra-rare, 2× on rare. v11 direction.
    
    Hypothesis: BCE pos_weight upweighting actual-positive samples in
    ultra-rare classes during training makes the model produce sharper
    confidence on true positives, without affecting negative-row baseline.
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
```

## Validation plan (avoid v10 failure mode)

1. **Drift check**: must pass `silent_drift_check.py`.
2. **Single fold first**: fold 0 seed 42, ~1h on RTX 5090.
3. **Local validation** (use the SAME script `exp180_v10_local_validation.py`):
   - Macro AUC on labeled SS: target ≥ 0.997 (NOT below v53 0.998 baseline)
   - Pearson vs Tucker: target ≥ 0.96 (vs v10's 0.876 catastrophe)
   - Top-1 agreement: target ≥ 0.75 (vs v10's 0.447)
   - Unlabeled SS ultra-rare p99: target close to Tucker 0.067, NOT below 0.04 (over-suppression sign)

4. **Falsifiable bar**:
   - If macro AUC ≥ v53 -0.001 AND ultra-rare p99 ratio close to 1.0× → LB candidate
   - If macro AUC drops more than -0.005 from v53 → reject (over-boost in wrong direction)

## Expected scenarios

### A — Both metrics improve toward Tucker
Pos_weight successfully boosts ultra-rare confidence without crushing common-class
performance. → train 5 folds, deploy as new SED variant. Potential v53 → 0.945+.

### B — Macro AUC stable, ultra-rare p99 ratio unchanged
Pos_weight effect washed out by Mixup / SpecAug / random batch composition.
Try higher weight (5×, 10×) or focal-loss variant.

### C — Macro AUC degrades, similar to v10
Pos_weight created overconfidence on ultra-rare even at negative rows.
Falsifies "boost is right direction" hypothesis. Then F9 may need further revision.

### D — Macro AUC stable, but ultra-rare p99 ratio INCREASES (worse than v53)
Pos_weight made ultra-rare baseline elevation worse on broader OOD.
This is the symmetric inverse of v10 failure. Suggests the right fix is at the
data-distribution level, not loss-weight level.

## Compute budget

- Single fold: ~1h. v53 LB-ready: 5h.
- v59 LB result (Phase G post-sprint) informs whether to run exp182.

## Stack alternatives

- exp182 (pos_weight 3×) + exp180 (v10 pseudo at lower share, e.g., 0.20) =
  exp183. Combines boost direction with filtered pseudo. Risk: complexity.
- exp182 alone is the cleanest test of the boost-direction hypothesis.

## NOT this design

- ❌ Focal loss alone (exp178 SoftAUC variant already tested 2026-05-15,
  Pearson 0.668 OK but solo AUC drop)
- ❌ Class-balanced sampler (raises ultra-rare rows in batch, but doesn't
  address loss-weight; tested as MIN_SAMPLE which only handles n < 20)
- ❌ Two-stage training (warmstart on Tucker spec, then fine-tune with
  pos_weight) — too many moving parts; isolate one variable first.
