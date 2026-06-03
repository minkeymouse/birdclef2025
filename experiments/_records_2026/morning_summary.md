# Morning summary — 2026-05-08

## Overnight LB results (5/5 slots used May 7 UTC)

| v | Test | LB | Δ vs anchor 0.938 | Outcome |
|---|---|---|---|---|
| v22 | Ulyanov blend on exp174a + konbu 5% | 0.937 | −0.001 | submitted yesterday |
| v26 | Q1: drop all Insecta (28 cls) | 0.881 | **−0.057** | regression_strong |
| v27 | Q2: drop all Amphibia (35 cls) | 0.834 | **−0.104** | regression_strong |
| v28 | Q3: drop rare cls (n_ta<5, 42 cls) | 0.837 | **−0.101** | regression_strong |
| v30 | exp175 5-fold (silent drift fix) | 0.937 | −0.001 | matched anchor |

(v23/v24/v25 errored on Kaggle — apply_state bug; v29 errored — dataset upload race; both replaced by v26-v28/v30.)

## Key findings

### 1. Q-test: all 3 subsets are NET POSITIVE STRONG
- Implied LB-side AUC for these subsets: Insecta ~0.975, Amphibia ~1.0, rare ~1.0
- Our SED predictions on hidden test for Insecta/Amphibia/rare classes are HIGHLY ACCURATE
- Q3 vs Q1 +0.044 differential → rare non-Insecta (Amphibia + Mammalia) carries extra signal beyond Insecta

### 2. Silent drift hypothesis REJECTED
- exp175 (drop_rate fix + xavier init) = LB 0.937, indistinguishable from anchor exp169 (0.938)
- The 5-week silent drift was a real bug (real consistency improvement) but NOT the LB bottleneck
- The 0.003 LB gap to Tucker public weights remains unexplained by this fix

## Pseudo strategy decisions (high-confidence)

Q-test results give us a clear pseudo plan:
1. **Generate pseudo for Insecta/Amphibia/rare classes** — high threshold (≥0.7), oversample
2. **Multi-arch teacher ensemble** to reduce circular-distillation bias (B0 + B1 if available)
3. **Avoid pseudo on Aves majority** — don't add what's already saturated (likely already predicting well at LB level)

## Tomorrow's slot plan (5 fresh slots after UTC reset 09:00 KST)

Priority queue:

**Slot 1: exp176 5-fold (per-fold SS split, Tucker-correct EVAL)**
- Tests if per-fold SS split (vs fixed 11-file holdout) closes any LB gap
- Different mechanism from exp175 silent drift fix
- exp176 will finish ~09:30 KST (after reset)
- Direct A/B vs exp175 v30 (0.937)

**Slot 2: Pseudo iter informed by Q-results**
- Generate pseudo specifically for Insecta+Amphibia+rare classes
- Use exp175 5-fold + exp176 5-fold ensemble as teacher (multi-arch substitute)
- High threshold (max prob ≥ 0.7), strict per-class filter
- Train SED with extra pseudo for these 3 subsets

**Slot 3: Multi-seed ensemble of exp175 (if exp176 doesn't move LB)**
- exp175 with seed=43 OR seed=44
- Average 3 seeds × 5 folds = 15 ckpts
- Tests random-seed variance hypothesis (different lever from drift)

**Slot 4-5: TBD based on Slot 1-3 results**

## Background processes still running

- `exp176` training (fold 2 ep20+, will finish ~09:30 KST)
- `lb_poller` (10min CSV dump)
- `lb_processor` (5min auto-fill registry)

## Files in canonical state

- `experiments/sed/` — refactored 1,748 LOC + _common.py shared utilities
- `experiments/_archive_2026_pre_sed_refactor/` — 27 archived fork-based scripts
- `experiments/lb_registry.yaml` — v22-v30 entries with hypothesis/outcome/lessons
- `experiments/_scratch_logs/lb_results_summary.md` — auto-appended LB scores
- `paper/exp_current.tex` — Q-test section + numerical table updated
- `memory/project_q_test_results.md` — pseudo strategy decisions
- `memory/MEMORY.md` — updated index

## Lessons recorded

- `feedback_kaggle_submit_flag.md` — `kaggle competitions submit` MUST include `-f submission.csv`
- `project_q_test_program.md` + `project_q_test_results.md` — Q-test methodology + outcomes
- `project_2026_05_08_refactor.md` — sed/ package consolidation
