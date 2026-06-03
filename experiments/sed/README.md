# sed/ — Refactored SED training + LB orchestration package

**Goals**:
1. Kill silent drift. Single source of truth for Tucker spec. One config dataclass. One model class. One trainer.
2. Autonomous overnight LB orchestration (push, poll, submit, update registry).

## Why this exists

Before refactor (5 weeks of fork-based development):
- 27 scripts in `_data_pipelines/`, ~5,700 LOC, ~90% duplication
- `MelExtractor`, `DistilledSED`, `hybrid_loss` duplicated 10-21× each
- `drop_rate=0.1` silent drift carried from exp169 → all later SED scripts
- "exp171_tucker_exact" name was a lie — 5 deviations from Tucker
- 6 different notebook patch scripts, stateful, ordering-sensitive

After refactor (1,748 LOC total, ~70% reduction):
- `tucker_spec.py` — frozen canonical reference (62 fields)
- `config.py` — `SedConfig` dataclass + auto-diff against Tucker
- One model, one data pipeline, one trainer
- `silent_drift_check.py` — CI-style drift detector

## Files

```
sed/
├── _common.py                 # SHARED: paths, slugs, anchor commit, Kaggle helpers, macro_auc
├── tucker_spec.py             # canonical Tucker recipe (frozen, 62 fields)
├── config.py                  # SedConfig + pre-canned configs (tucker_canonical, exp175_*, exp176_*, exp177_*)
├── silent_drift_check.py      # CI-style drift detector (scans archived scripts + configs)
├── notebook_state.py          # idempotent notebook patcher (replaces 6 patch scripts)
├── model.py                   # DistilledSED with config-driven init
├── data.py                    # CachedDataset, PseudoDataset, TwoStreamBatchSampler
├── train.py                   # train_one_fold(SedConfig, fold) → ckpt; CLI w/ --config / --seed
├── export.py                  # ONNX export wrapper
├── deploy.py                  # generic deploy (export → upload → notebook patch → push)
├── deploy_exp175.py           # exp175-specific deploy (legacy)
├── deploy_exp176.py           # exp176-specific deploy (legacy)
├── deploy_ensemble.py         # multi-seed + multi-arch ensemble deploy (current)
│
│ === Orchestration (current) ===
├── auto_deploy_ensemble.py    # Wait for ensemble chain → deploy → submit
├── one_shot_submit.py         # Generic submit with --version + --message
├── lb_poller.py               # Dump submissions CSV every 10min
├── lb_processor.py            # Auto-fill lb_registry when new LB scores arrive
├── memory_update.py           # Write project memory entry once results arrive
├── paper_update.py            # Patch paper section once results arrive
└── orch_status.py             # Single-shot orchestration status snapshot
```

(Q-test scripts moved to `_archive_2026_pre_sed_refactor/_post_qtest_2026_05_08/` after Q1/Q2/Q3 LB results recorded in registry.)

## Quick reference — running things

### Single-fold training
```bash
uv run python -m experiments.sed.train --config exp175_tucker_actually --fold 0
uv run python -m experiments.sed.train --config exp175_tucker_actually --seed 43  # all folds, alt seed
```

### Drift check (run before any new SED training)
```bash
uv run python -m experiments.sed.silent_drift_check
```

### Background autonomous loop
```bash
# 1. Auto-deploy when chain completes
nohup setsid bash -c 'KAGGLE_API_TOKEN=... uv run python -u -m experiments.sed.auto_deploy_ensemble' &

# 2. LB poller (dump submissions every 10min)
nohup setsid bash -c 'KAGGLE_API_TOKEN=... uv run python -u -m experiments.sed.lb_poller' &

# 3. LB processor (auto-update lb_registry when scores arrive)
nohup setsid bash -c 'uv run python -u -m experiments.sed.lb_processor' &
```

## Naming convention

- Experiments named by **mechanism** not sequential number:
  - `tucker_canonical` / `tucker_exact`
  - `exp176_per_fold_ss` (Tucker-correct EVAL split)
  - `exp177_b1_backbone` (multi-arch ensemble member)
  - `pseudo_iter1_strict` / `pseudo_iter1_relaxed` (future)
- Each run output dir: `experiments/_data_pipelines/<name>_outputs/seed{seed}/fold{0..4}/`
  (seed-specific subdirectory prevents multi-seed runs from colliding)
- LB submissions tagged with config name + seed in lb_registry.yaml.

## Multi-seed / multi-arch ensemble

To run multi-seed ensemble for a config:

```bash
# Train 3 seeds sequentially
for seed in 42 43 44; do
    uv run python -m experiments.sed.train --config exp175_tucker_actually --seed $seed
done

# Train alternative architecture
uv run python -m experiments.sed.train --config exp177_b1_backbone --seed 42

# Deploy combined ensemble (auto-discovers all completed seeds + configs)
uv run python -m experiments.sed.deploy_ensemble
```

`deploy_ensemble.py` collects all available `(config, seed, fold)` triples, exports them as `sed_fold00.onnx` through `sed_fold{N-1}.onnx` into a single dataset, uploads, and pushes the notebook with appropriate dataset slug.

## Tucker recipe match (verified 2026-05-08)

Pulled Tucker's published training notebook from Kaggle and confirmed:
- All architecture, hyperparam, augmentation, optimizer, scheduler, loss settings MATCH
- Recipe-level Tucker replication is COMPLETE
- The 0.003 LB gap (ours 0.937 vs Tucker 0.941) is NOT in any spec hyperparam — it's random seed luck + numerical micro-detail in training run.
- See `memory/project_tucker_recipe_match.md` for full audit table.

The path to close the gap is multi-seed + multi-arch ensemble averaging out random seed variance.
