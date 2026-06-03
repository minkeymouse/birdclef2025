# exp181 — 2024-oversample SED training spec (2026-05-17 sprint output)

**Status**: design only, NOT implemented. Sprint Phase E discovery F8/F9 motivates this.

## Hypothesis

Sprint Phase E unification (F9): ours SED under-trained on 2024 recording-year
patterns. labeled SS year distribution: 2021=576, 2022=504, **2024=120**, 2025=278.
2024 has 60% less data than 2022. Tucker (same recipe, different seed) happens to
fit 2024 better by random init luck.

**Fix**: oversample 2024 SS rows during training by factor 5× → 600 effective rows.
Match 2022's representation.

## Required code changes

### config.py — new SedConfig option
```python
@dataclass
class SedConfig:
    ...
    # NEW: year-based oversample (None = uniform sampling)
    ss_year_weights: Optional[dict] = None  # e.g., {2024: 5.0}
```

### train.py — extend MIN_SAMPLE block
Around line 235 (after MIN_SAMPLE TA upsample):

```python
# Year-based SS oversample (exp181)
if cfg.ss_year_weights:
    import re
    def year_of(fn):
        m = re.match(r"BC2026_\w+_\d+_S\d+_(\d{4})", str(fn))
        return int(m.group(1)) if m else None

    ss_in_train = train_idx[is_ss[train_idx] == 1]
    extra_ss = []
    for ss_row in ss_in_train:
        fname = files[ss_row]  # need to verify column mapping
        year = year_of(fname)
        weight = cfg.ss_year_weights.get(year, 1.0)
        if weight > 1:
            # Add (weight - 1) extra copies
            extra_ss.extend([ss_row] * int(weight - 1))
    if extra_ss:
        train_idx = np.concatenate([train_idx, np.array(extra_ss)])
        n_extra = len(extra_ss)
        print(f"  Year oversample: +{n_extra} SS rows (2024 weight={cfg.ss_year_weights.get(2024, 1)})", flush=True)
```

### New config factory
```python
def exp181_year_oversample() -> SedConfig:
    """exp181 = exp176 + 2024 SS row oversample (×5)."""
    return SedConfig(
        name="exp181_year_oversample",
        output_dir="experiments/_data_pipelines/exp181_outputs",
        BACKBONE_DROP_RATE=None,
        ATT_CLA_INIT="xavier_uniform",
        EVAL_SS_N_FILES=None,
        CHECKPOINT_CRITERION="val_ss",
        WARMUP_EPOCHS=2,
        EPOCHS=25,
        USE_FOCAL_FOCAL_MIXUP=True,
        USE_FOCAL_SC_MIXUP=True,
        MIN_SAMPLE=20,
        ss_year_weights={2024: 5.0},
        notes="exp181 = exp176 + 2024 oversample. Sprint 2026-05-17 Phase E F8/F9.",
    )
```

## Validation plan

1. **Drift check first**: must pass `silent_drift_check.py`.
2. **Single fold first** (fold 0 seed 42, ~1h on RTX 5090).
3. **Local validation**:
   - Macro AUC on labeled SS year-stratified:
     - 2024 macro AUC (should improve from current Tucker-vs-ours level)
     - 2021/2022/2025 macro AUC (should NOT degrade)
   - Tucker-vs-ours_exp181 disagreement breakdown:
     - In 2024 disagree rows: Both-correct count should increase
     - "Tucker only correct" count should decrease (currently 9 of 30)
4. **If local validation passes** (axis-7 metric improves, others stable):
   - Train 5 folds (sequential, ~5h)
   - Deploy as new SED ensemble variant
   - LB test with v53 base + exp181 fold filter

## Expected outcome scenarios

### A — Local 2024 macro AUC improves significantly
- Confirms axis-7 = training under-representation explanation
- LB submission worth doing (1 slot)
- Direction: stack with v59 OR replace v59 (depending on v59 LB result)

### B — Local 2024 macro AUC unchanged
- 2024 oversample doesn't help → maybe axis-7 is acoustic-features not data-quantity
- Investigate: are 2024 acoustic patterns truly different from 2021/2022/2025?
  - Run spectral analysis on 2024 vs other years' SS audio
- Alternative: per-year-specific augmentation (e.g., 2024-recording-specific noise injection)

### C — Local 2021/2022/2025 macro AUC degrades
- 2024 oversample over-fits to 2024 at the expense of other years
- Reduce weight from ×5 to ×3 or ×2

## Compute budget

- Fold 0 training: ~1h (Tucker recipe 25 epochs)
- Total 5 folds: ~5h
- + ONNX export + Kaggle dataset upload + notebook push: ~30min
- Total to LB-ready: ~5.5h

## Risk

- 2024 over-fit (Scenario C above). Mitigated by weight tuning.
- Diminishing returns (Tucker's advantage may be 60-70% from seed luck not data quantity).
  Then 2024 oversample → +0.001 LB at best.
- Doesn't fix axis-2 (ultra-rare baseline elevation in broader OOD) which is a
  separate manifestation of the same root cause.

## Combinations

- exp180 (v10 pseudo) + exp181 (year oversample) = exp182. Probably saturates
  improvement; first test each in isolation.
- exp175 (Tucker spec) + 5 seeds with 2024 oversample = multi-seed ensemble.
  Direction similar to v45 axis decomposition Slot 2/3 (already shown 0 effect).
