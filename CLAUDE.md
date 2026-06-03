# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Solution code for the **Kaggle BirdCLEF+ 2026** competition (identify Pantanal-wetland species from 5-second windows of passive-monitoring audio; 234 species across Aves/Insecta/Amphibia/Mammalia/Reptilia). Metric: **macro-averaged ROC-AUC that skips classes with no true positives**. Full brief in `OVERVIEW.md`; prior-year winning writeup in `WINNING_SOLUTION_2025.md`.

## Repository architecture

**Two worlds, bridged by Kaggle datasets.** Models are *trained locally* (RTX 5090) but *scored on Kaggle* inside a CPU-only, no-internet notebook kernel. The scored `submission.csv` is produced **by the notebook on Kaggle**, not by any script here. Local `submission*.csv` files are downloaded copies / individual streams kept for analysis.

The bridge is always the same loop:
```
train locally → export ONNX/weights → upload as a private Kaggle dataset
  → reference it in notebooks/<dir>/kernel-metadata.json:dataset_sources
  → kaggle kernels push → kaggle competitions submit → log in lb_registry.yaml
```

**The scored inference recipe lives in the notebook cells, not in repo Python.** The repo's Python trains the *components* the notebook consumes. The blend itself (Perch ProtoSSM 60% + Tucker SED 40% → rank blend → taxonomy smoothing → sidecar/BirdNET corrections) is implemented across the notebook's ~26 cells, forked from the public "EoS8" notebook.

Key directories (non-obvious roles only):

- **`notebooks/birdclef-2026-eos8-*/`** — the Kaggle inference kernels = the actual submission artifacts. Each dir is one kernel: `notebook.ipynb` + `kernel-metadata.json` (declares `dataset_sources` / `model_sources` / `kernel_sources`). Inference runs CPU-only, internet disabled. Kept top-level (post-2026 reorg): **`eos8-verbatim`** = production reference (LB 0.950); **`eos8-phase4`** = enriched experiment base (Karnakbayev Two-Pass SSM — per-class blend + Perch probes + priors; 0.950) that every lever forked from; **`eos8-exp189-full`** = the final-day external-data SED kernel (LB-flat but decisive — see FINAL section); `mattia-fork` = older 0.949 reference. **All other single-variable `eos8-<lever>` dirs (`r1`, `hgblend`, `rpair`, `um00`, `lam50`, `ps55/65`, `tax25`, `yz00/10`, `exp187`, `ptlam`, `ptax2`, …) are in `notebooks/_archive_2026/`** (gitignored; each isolates one knob, results in `lb_registry.yaml`).
- **`experiments/sed/`** — Tucker-SED local training package. `tucker_spec.py` = frozen 62-field canonical recipe; `config.py`'s `SedConfig` auto-diffs against it (incl. `exp189_tucker_external_nonaves`); run `silent_drift_check.py` before any new SED run. Flow: `train.py` → `export.py` (ONNX) → deploy via the top-level `exp189_patch_notebook.py` against `eos8-phase4`. NOTE: the legacy orchestration scripts (`deploy_ensemble.py`, `deploy.py`, `_common.py`, `lb_poller`/`lb_processor`, `sed/_archive/`) target the superseded **mattia-fork / ensemble_v3** lineage (LB 0.938, not production) — kept for reference, not the current path. See `experiments/sed/README.md`.
- **`experiments/sidecar_src/`** — separate PCEN/ConvNeXt "sidecar" training pipeline (RunPod-oriented; invoked as module `sidecar_src.*` with `PYTHONPATH=experiments`). Produces a **class-wise masked rank correction** gated by an OOF-gate CSV, blended into the anchor — not a standalone model. Configs in `sidecar_src/configs/*.yaml` (DPO + BCE variants). See `experiments/sidecar_src/README.md`.
- **`experiments/lb_registry.yaml`** — single source of truth for every LB submission (hypothesis before, lessons after); the `anchor:` entry is the never-fall-below baseline. **Tracked in git.**
- **`experiments/_eval_harness/eval_soundscapes.py`** — local macro-AUC on labeled `train_soundscapes` (competition metric).
- **`experiments/local_validator.py`** — pre-submit gate: Δ macro-AUC of a candidate vs anchor. Catches bugs (NaN/shape/all-zeros) but is **not** an LB proxy — labeled SS is anti-correlated with LB at the margin (see Hard rules).
- **`experiments/_data_pipelines/`** — per-experiment trained outputs (exp169/175/176/187/188/189/ensemble_v3; gitignored) + the anchor-cache builders `exp169v2_random_anchor_cache.py` (base) and `exp189_build_external_cache.py` (base + external non-Aves).
- **`experiments/eval_labeled_ss_controlled.py`** — THE correct controlled local baseline (Perch+Tucker on 739 aligned windows → `cache/labeled_ss_controlled.npz`); validate candidates vs this, never the ProtoSSM strawman. **`exp189_gate.py`** / **`exp189_patch_notebook.py`** — the external-data SED gate + deploy. **`dial_patch.py`** — generic single-constant notebook patcher. **`perch_embed_extract.py`** — Perch embedding extraction.
- **`experiments/_archive_2026/`** — all 2026 one-off lever/research scripts (the `*_patch*.py` lever generators, `poll_*.py`, `emcent_*`, `structural_analysis.py`, `lora_dpo_train.py`, `build_pseudo_*`, probes, …) — "what we tried", not the active pipeline. **`experiments/_records_2026/`** — preserved session logs (gitignored `_scratch_logs/` was copied here so it survives in git).
- **`model-weights/`** — local trained artifacts (ONNX/`.pt`/`.npz`); the source for Kaggle dataset uploads. Gitignored.
- **`data/birdclef-2026/`** — competition data (`train_audio/`, `train_soundscapes/`, `taxonomy.csv`, `sample_submission.csv`). `birdclef-2025/` = prior year; `external/` = extra downloaded audio. Gitignored.
- **`perch_v2/`** — frozen Google Perch v2 TF SavedModel (the foundation embedding model). Gitignored; ONNX copy in `model-weights/perch_v2_onnx/`.

**What's tracked vs not:** `.gitignore` excludes essentially all binaries, data, weights, logs, and even most `*.csv`/`*.json`/`*.yaml` — with explicit `!` exceptions for `experiments/lb_registry.yaml` and `notebooks/*/kernel-metadata.json`. So the git repo is Python source + READMEs + those two config types + this file. Trained models and the notebook's input datasets live on Kaggle / local disk, never in git.

## Common commands

**Environment:** `uv` (Python 3.13), resolved from `pyproject.toml` + `uv.lock`. Prefix everything with `uv run` (`uv run python …`, `uv run kaggle …`); no separate install step. **There is no unit-test suite or linter** — "validation" = `local_validator.py` + `_eval_harness/eval_soundscapes.py` + the LB.

SED training (run from repo root, module = `experiments.sed.*`):
```bash
uv run python -m experiments.sed.silent_drift_check                                  # ALWAYS before a new SED run
uv run python -m experiments.sed.train --config exp175_tucker_actually --fold 0      # single fold
uv run python -m experiments.sed.train --config exp175_tucker_actually --seed 43     # all folds, alt seed
uv run python -m experiments.sed.deploy_ensemble                                     # export ONNX → upload → patch+push kernel
```

Sidecar training (note `PYTHONPATH=experiments`, module = `sidecar_src.*`):
```bash
PYTHONPATH=experiments uv run python -m sidecar_src.training.train \
    --config sidecar_src/configs/dpo_pcen_convnext.yaml --fold 0
```

Eval / pre-submit validation:
```bash
uv run python experiments/_eval_harness/eval_soundscapes.py
uv run python experiments/local_validator.py --candidate /tmp/cand.csv --anchor /tmp/anchor.csv
```

Long-running jobs (RTX 5090 32GB; Perch ONNX ≈ 0.15s/clip GPU):
```bash
LOG=experiments/_scratch_logs/<name>_$(date +%Y%m%d_%H%M%S).log
nohup setsid uv run python -u <script> > "$LOG" 2>&1 < /dev/null & disown
# or: scripts/run_exp.sh <script.py> [args...]   # unbuffered, tees to _scratch_logs, writes .pid
```

Kaggle submission (auth: bearer token at `~/.kaggle/access_token`; username `ultimatumgame`; **5 submissions/day**):
```bash
uv run kaggle kernels push -p notebooks/<dir>
uv run kaggle kernels status ultimatumgame/<slug>
uv run kaggle competitions submit -c birdclef-2026 -f submission.csv \
    -k ultimatumgame/<slug> -v <VERSION> -m "msg"            # -f is REQUIRED — CLI silently no-ops without it
uv run kaggle competitions submissions -c birdclef-2026 --csv | head -5
```
Production kernel: `eos8-verbatim` (LB 0.950). Model-weights dataset: `ultimatumgame/birdclef2026-model-weights`. **Log every submission in `experiments/lb_registry.yaml`** (hypothesis before, lessons after).

---

## Current state (2026-06-03 FINAL — competition closed; final LB 0.950)

**Final score**: LB **0.950** (EoS8 verbatim fork; protected via Kaggle select-best). **Public top 0.963.**
Kaggle scoring noise SD ≥ 0.002. The −0.013 gap was **never closed locally** but is **not a ceiling** — it is
cross-region *transfer* headroom (see the FINAL section below). For BirdCLEF 2027, this repo is cleaned +
reproducible: core pipeline at `experiments/` top-level, every lever in `lb_registry.yaml`, what-we-tried in
`*/_archive_2026/`, the research thesis in `paper/` (read `paper/README.md` first).

### Working stance (read this before any LB / strategy analysis)

Null and regression results **narrow the search; they never close it.** A flat single-variable trial means *that one knob* isn't the lever — not that the problem is solved. This project has collapsed "N flat trials → we're at the ceiling" four separate times (2026-05-01 / 05-03 / 05-09, and the 06-01/02 session), and each time a new axis proved it wrong (Tucker SED swap +0.009; public 0.943 → 0.950). Do not repeat it. Concretely:

- **The scarce resource is LB feedback (5 submits/day), not compute.** One full SED training ≈ 3 h on the 5090; ~2 days remain → dozens of local training / experiment runs are available, plus ~10 LB slots. "We're out of chances" is false — the right question is always "which experiment do we run next."
- **Read `lb_registry.yaml` outcomes literally.** An entry calling a knob "flat / exhausted" means *that knob* is flat — it is a map of where signal *isn't*, which sharpens where to look. Never aggregate those into a search-wide verdict.
- **End every analysis with ≥3 untested axes and the next concrete run.** Forbidden framings: "true / real ceiling", "exhausted", "structurally bounded", "no headroom", "X is final". (See memory `feedback_no_ceiling_lock` — four documented relapses; this is the highest-priority behavioral rule on this project.)
- Tone: objective **and** relentlessly forward-seeking — neither triumphalist nor defeatist.

### 2026-06-03 FINAL — the wall is TRANSFER, not model quality (the key learning for 2027)

Final day, all 5 LB slots spent on genuinely-new levers (full record in `lb_registry.yaml`):

| lever | LB | what it was |
|---|---|---|
| exp187 | 0.938 | distilled student SED (weaker) blended in → dragged |
| ptlam | 0.950 | per-taxon prior λ (non-Aves) → flat |
| **exp189** W=0.40 | **0.950** | **Tucker recipe + external non-Aves data → FIRST own-SED to BEAT Tucker on local eval, yet LB-flat** |
| exp189 W=0.70 | 0.949 | dose follow-up → flat |
| ptax2 | 0.950 | drop genus tax-smoothing for Insecta sonotypes (site-invariant) → flat |

**The decisive result (exp189):** we did exactly the BirdCLEF-2025-winner ingredient (external non-Aves
data, done properly) and built a SED that **beats the production Tucker on every evaluable group**
(all-eval 0.9979 > 0.9972; gate `experiments/exp189_gate.py`). It moved the LB by **0.000**. A model that is
*strictly better on the validation distribution* produces zero test movement → **the bottleneck is local→LB
transfer under cross-region covariate shift (labeled SS is ~65% site-22 = 954/1478 windows; test is multi-site), NOT SED quality,
data volume, or any post-proc knob.** Across all interventions, local Δ of +0.0006 to +0.067 ALL gave ~0 LB
(decoupling table in `paper/covariate_shift_findings_2026_06_03.md`).

**This is a *located bottleneck*, not a ceiling.** The 0.963 top proves the signal exists. The 2027 lever must
target TRANSFER, not local quality:
1. **Multi-site supervision / validation** — the root fix (= the paper thesis). Naïve LOSO on the existing
   labels is unreliable (site-22 bias survives it); a genuinely site-diverse label source is needed.
2. **A cross-region-robust foundation embedding** with native non-Aves coverage (Perch 2.0 / BirdMAE class —
   **vet provenance before download**; 2026 dropped Perch 2.0 as an untrusted artifact).
3. **Final-level (not SED-stream) integration** of an external-data SED — untested (bypasses gate/Gaussian cascade).
4. **Domain-adversarial / site-invariant training objective** — the only model-side route at the actual wall.

(Reproduce exp189: see `experiments/README.md`. The older "trainable≠evaluable / dial-tuning flat" analysis
that led here is preserved below + in `paper/structural_analysis_2026_06_02.md`.)

## Historical analysis (pre-2026-06-03) — SUPERSEDED by the FINAL section above

> ⚠ Everything below is the pre-final working hypothesis, kept as the paper's evidence trail. exp189 refuted
> its core claim: a SED that BEATS Tucker on every evaluable group was LB-flat → the bottleneck is local→LB
> TRANSFER (covariate shift), **not** the SED / ProtoSSM non-Aves suppression. Read the FINAL section for the
> current explanation; treat the "bottleneck / headroom / open questions" framing below as historical.

### Why the public recipe converged to 0.950

All public notebooks use the same recipe and converge to ~0.950:
```
Audio → Perch v2 → 1536-d embedding → ProtoSSM (60%) ─┐
                 └→ 14,795 logits → 203/234 mapped     │ rank blend → taxonomy smooth → sidecars
Audio → Tucker SED (mel → B0 → 234 logits) (40%) ──────┘
```

### The bottleneck is NOT the SED

Tucker SED achieves AUC 0.97+ on all evaluable species including cold-start Insecta (verified 2026-05-27). The problem is **ProtoSSM suppressing non-Aves species**:

| Class | ProtoSSM mean score | Species | Perch mapped |
|---|---|---|---|
| Aves | +0.24 | 162 | 162/162 |
| Insecta | **-0.37** | 28 | **0/28** |
| Mammalia | **-0.66** | 8 | 6/8 |
| Amphibia | **-1.72** | 35 | 32/35 |
| Reptilia | -0.21 | 1 | 0/1 |

31 species unmapped by Perch → Perch logit = 0 → ProtoSSM has zero signal.
60/40 ProtoSSM/SED blend dilutes SED's good predictions for these species.

### Perch model is frozen

- TF SavedModel: `trainable_variables = 0`
- ONNX → PyTorch conversion: fails (shape inference error)
- 102M params (custom EfficientNet with prototypical network head, NOT standard B3)
- **Cannot fine-tune with current tools**

### What Perch embedding reveals

For unmapped species, Perch embedding still separates TP from TN (cosine similarity 0.33-0.59 between centroids). The information EXISTS in the embedding — the 14,795-class prototypical head just doesn't have entries for these species.

Perch head structure: `prototypes: (14795, 1536, 4)` — each species has 4 prototypes in 1536-d space. Unmapped species have no prototypes.

### PCEN sidecar analysis

OOF gate enables only 35/234 species (15%). Key: 12 cold-start Insecta with AUC 0.68-1.00.
`WEIGHT_CAP = 0.030` limits correction to 3%. (Tested 0.03→0.10 = LB 0.950 flat → the sidecar correction is
insufficient regardless of the cap; the "too conservative" hypothesis was refuted.)

### Local eval limitation

Only 50/234 species have ≥10 positives in soundscapes. 79% of LB depends on non-evaluable species. All local experiments are measured on this 21% subset.

## Experiments run (2026-05-27, with results)

| Experiment | Result | Conclusion |
|---|---|---|
| Recipe-pair vs Tucker SED | Pearson 0.50, AUC 0.58 vs 0.67 | Diverse but weaker → no ensemble gain |
| HGNet-B0 (public) vs Tucker | Pearson 0.35, AUC 0.69 vs 0.67 | Different and locally better, LB unverified |
| PCEN sidecar gate analysis | 12 Insecta AUC 0.68-1.00, W=0.03 | W_cap 0.03→0.10 = LB 0.950 flat → cap NOT the limiter |
| SED head RL fine-tune (soundscapes) | Already AUC 0.987 | SED saturated on evaluable data |
| SED head RL fine-tune (train_audio) | 0.967→0.961 (worse) | SED not the bottleneck |
| MLP on Perch embedding (3-fold CV) | Unmapped 0.50→0.73 | Perch has info, head doesn't use it |
| DPO on Perch embedding (3-fold CV) | Unmapped 0.50→0.96, Insecta 0.50→0.96 | Best result but overfitting risk |
| LoRA+DPO adapter (3-fold CV) | Macro 0.69→0.96, cos 0.995 | Adapter preserves embedding structure |
| Perch TF fine-tune | trainable_variables=0 | Blocked |
| Perch ONNX→PyTorch | Shape inference error | Blocked |
| EoS8 + increased PCEN weights (W_cap→0.10) | LB 0.950 | Flat — sidecar insufficient regardless of cap |

## What failed (don't retry)

- **Own-trained SED variants** (exp169-183): Pearson 0.97-0.99 with each other.
- **Pseudo-label iteration** (6 attempts): all failed.
- **NS sidecar** (2026-05-26): negative TP_shift.
- **Inference tweaks**: quantile cal, rare dampen, BirdNET attach — all noise.
- **Loss swaps**: DPO/focal/soft-AUC/contrastive — BCE is optimal.
- **Foundation alternatives**: CLAP, iVAE/iVDFM, AudioMAE — all failed.
- **Ensemble beyond recipe-pair**: multi-seed 15 = multi-arch 10 = 0.938.
- **RL on SED head**: already saturated at 0.97+.
- **Re-tuning these EoS8 inference dials** (2026-06-02): proto/SED blend, hour-prior λ, yukiZ weight, unmapped-ProtoSSM→0 — all flat 0.949–0.950 (within noise). (These dials, not the search.)
- **New SED streams on EoS8** (2026-06-02): HGNet replace 0.937 / blend 0.949; BirdSet-ConvNeXt external-pretrain val ≈ random (domain gap); exp175/176 recipe-pair on phase4 → 0.944.

## Research direction: RL post-training (novel, untried in bioacoustics)

**Core idea**: Perch embedding has information for unmapped species but the classification head doesn't use it. Post-training can fill this gap.

**DPO showed best results** (Insecta 0.50→0.96) by directly optimizing ranking. But trained on 708 soundscape windows only — needs train_audio scale for robustness.

**Blockers**:
1. Perch frozen → can only train heads/adapters on frozen embeddings (= what ProtoSSM already does)
2. Training on frozen embeddings is redundant with ProtoSSM unless we change the training objective (DPO/RL vs BCE)
3. Need Perch embeddings for all train_audio (~88 min GPU extraction) before full-scale DPO

**Open questions**:
- Does DPO's ranking-based training give genuinely different results from ProtoSSM's BCE training on the SAME embeddings?
- Can per-class blend weights (more SED for Insecta, more ProtoSSM for Aves) close the gap without new models?
- Can we reconstruct Perch in PyTorch by manually mapping ONNX weights?

## Hard rules

- **Kaggle noise ≥ 0.002**: deltas within ±0.002 are noise.
- **Local eval unreliable**: 50/234 species only, site-22-biased. Don't gate on local improvement (exp189
  beat Tucker locally yet was LB-flat — the defining lesson).
- **5 submissions/day** (BirdCLEF rule): budget carefully in future competitions.

## Paper

CLEF 2026 working note. Competition closed **2026-06-03**; paper due **2026-06-17**.
Thesis: covariate shift in cross-region bioacoustic monitoring. **Primary contribution: the transfer-wall
result** (exp189 strictly-better-locally yet LB-flat; local Δ +0.0006–0.067 → ~0 LB) — see
`paper/covariate_shift_findings_2026_06_03.md`. RL/DPO-on-embedding (Insecta 0.50→0.96 on a 708-window subset,
never LB-tested) is a supporting embedding-probe finding, not the main thesis.
