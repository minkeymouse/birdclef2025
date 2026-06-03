# Experiments

> **Final 2026 result: LB 0.950** (EoS8). Production kernel `notebooks/birdclef-2026-eos8-verbatim/`.
> **Key finding:** the wall is local→LB *transfer* under cross-region (site-22) covariate shift, NOT model
> quality — a SED that beats the baseline on local eval (exp189) is LB-flat. See
> `paper/covariate_shift_findings_2026_06_03.md`.

## Layout (core, reproducible)

```
experiments/
├── sed/                         SED training package (Tucker-spec). config.py = recipes incl.
│                                exp189_tucker_external_nonaves (winners' external-non-Aves data lever);
│                                train.py (ANCHOR_CACHE_DIR override), model.py, export.py, deploy_ensemble.py,
│                                tucker_spec.py (frozen 62-field recipe), silent_drift_check.py (run before any run)
├── sidecar_src/                 PCEN/ConvNeXt sidecar (PYTHONPATH=experiments; module sidecar_src.*)
├── _eval_harness/               eval_soundscapes.py — competition macro-AUC on labeled SS
├── _data_pipelines/             anchor caches + trained outputs:
│                                  exp169v2_random_anchor_cache.py (base cache: train_audio + labeled SS)
│                                  exp189_build_external_cache.py  (base + 552 external non-Aves clips)
│                                  exp169v2_outputs/ exp175_outputs/ exp189_outputs/ ...
├── eval_labeled_ss_controlled.py  THE correct local baseline (Perch+Tucker, 739 aligned windows) →
│                                   cache/labeled_ss_controlled.npz. Validate candidates vs THIS, never ProtoSSM.
├── exp189_gate.py               fold-filtered ensemble gate vs Tucker on the controlled substrate
├── exp189_patch_notebook.py     deploy: blend an external-data SED into the eos8 SED stream (full|non-Aves modes)
├── dial_patch.py                generic single-constant notebook patcher (single-variable LB tests)
├── perch_embed_extract.py       Perch v2 ONNX embedding extraction
├── local_validator.py           pre-submit sanity (Δ macro-AUC vs anchor; catches NaN/shape/all-zeros)
├── lb_registry.yaml             EVERY LB submission (hypothesis + outcome). Single source of truth.
├── _scratch_logs/               gitignored working run logs (NOT in git — deletable)
├── _records_2026/               session logs preserved IN GIT (SESSION_SUMMARY, decision log) — survives delete
└── _archive_2026/               2026 one-off lever scripts (emcent/ptlam/ptax2/hg/r1/rpair patches, distill
                                 pseudo builders, poll utils, probes) — kept for "what we tried" reference
```

## Reproduce the headline experiment (exp189 = external-data SED)
```bash
# 1. build the external-augmented anchor cache (Perch on train_audio + labeled SS + 552 external non-Aves clips)
uv run python experiments/_data_pipelines/exp189_build_external_cache.py
# 2. train (5 folds; ~1 h/fold on a 5090). Tucker recipe + external data, site-safe checkpoint.
uv run python -m experiments.sed.silent_drift_check
uv run python -m experiments.sed.train --config exp189_tucker_external_nonaves --fold 0   # repeat --fold 1..4
# 3. gate vs production Tucker on the controlled substrate (folds 0,1,2)
uv run python experiments/eval_labeled_ss_controlled.py     # builds cache/labeled_ss_controlled.npz (once)
uv run python experiments/exp189_gate.py 0,1,2
# 4. export + deploy (blend into eos8 SED stream)
uv run python -m experiments.sed.export --config exp189_tucker_external_nonaves
uv run python experiments/exp189_patch_notebook.py full 0.40    # -> notebooks/birdclef-2026-eos8-exp189-full
```
Result: exp189 beat Tucker on every evaluable group (all-eval 0.9979>0.9972) but LB-flat (0.950) — the
transfer wall. Full record + the 5 levers tested 2026-06-03 in `lb_registry.yaml`.

## Long-running jobs
```bash
LOG=experiments/_scratch_logs/<name>_$(date +%Y%m%d_%H%M%S).log
nohup setsid uv run python -u <script> > "$LOG" 2>&1 < /dev/null & disown
```
