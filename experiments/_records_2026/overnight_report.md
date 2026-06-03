# Overnight LB push report (2026-05-08, ~01:30 → 09:00 KST)

## RESULTS (final, 05:43)

**Q-tests all returned strong NET POSITIVE — pseudo high-priority for all three subsets:**
**exp175 silent drift fix did NOT help — within noise.**

| v | Q | Subset (drop) | LB | Δ | Implied subset AUC |
|---|---|---|---|---|---|
| v26 | Q1 | Insecta (28 cls) | 0.881 | −0.057 | 0.975 |
| v27 | Q2 | Amphibia (35 cls) | 0.834 | −0.104 | ~1.000 |
| v28 | Q3 | rare n_ta<5 (42 cls) | 0.837 | −0.101 | ~1.000 |
| v30 | exp175 5-fold (silent drift fix) | **0.937** | **−0.001** | within noise |

**Silent drift hypothesis REJECTED**: drop_rate fix + xavier init didn't close the 0.003 LB gap to Tucker. The 5-week silent drift was real (real bug, real consistency improvement) but NOT the LB bottleneck. Other systematic differences are responsible.

**Key insight**: All three subsets contribute massively to LB. Pseudo for any/all is valuable. Q3-Q1 differential (+0.044) shows rare non-Insecta classes ALSO carry signal beyond Insecta.

**Practical pseudo plan (next step after exp175)**:
1. Generate pseudo specifically for these subsets (high threshold ≥0.7).
2. Multi-arch teacher ensemble (B0 + B1 if exp174_b1 finishes) for less circular distillation.
3. Train SED with extra pseudo for Insecta/Amphibia/rare classes.
4. Test 1-slot LB to verify direction transfers.



## Plan executed

User invoked autonomous overnight mode at 01:30 KST. Strategic context:
0.945 catch-up requires SED quality + blend transfer. Step c (own konbu
head + linear blend with own SED) was confirmed dead-end via local
labeled-SS test. Step a (multi-seed/per-fold-SS retrain) requires hours
of training. Pre-pseudo information gathering via subset diagnostics
gives more value per slot than naive submissions.

## 4-slot allocation

| Slot | Kaggle ver | Test | Hypothesis |
|---|---|---|---|
| 1 | v26 | Q1: Drop all Insecta (28 cls, 12% of macro) | Insecta net positive vs negative on LB |
| 2 | v27 | Q2: Drop all Amphibia (35 cls, 15%) | Amphibia net positive vs negative |
| 3 | v28 | Q3: Drop rare classes (n_train_audio<5, 42 cls, 18%) | Rare-supervision net positive vs negative |
| 4 | v29 | exp175 5-fold (silent drift fix) | drop_rate removal + xavier init closes 0.003 SED gap |

**Bug fix at 01:55**: First push (v23/v24/v25) errored on Kaggle because `NotebookState.apply_state` regex deleted the `make_sed_session` helper function (which was defined between `def find_sed_dir` and the patch block). Re-pushed as v26/v27/v28 with `notebook_state.py` patched to only replace the patch-marked block (`# ===== exp169 patch:` onward), preserving helpers.

Slots NOT consumed by failed Kaggle runs (failed = no submission). Today's slot count still 5/5 available.

## Why these tests

For Q1/Q2/Q3, replacing class columns with constant 0.5 → AUC=0.5 for
those classes. Macro AUC drops by `(true_AUC - 0.5) × subset_fraction`.
LB Δ tells us subset's contribution sign and magnitude.

Each maps to a pseudo strategy decision:
- Subset hurts LB when dropped → currently helping → pseudo for that subset valuable
- Subset helps LB when dropped → currently hurting → exclude from pseudo

For v26, exp175 is the first SED training with all silent drifts fixed
(drop_rate removed, xavier init applied). Tests if 5 weeks of plateau
was caused by fork-based copy-paste.

## Background processes (started 2026-05-08 01:30+)

```
PID range          Process
exp175_*           exp175 FOLDs 2-4 training (resume; fold0/1 already done)
auto_deploy_exp175 waits for fold4 → deploy + submit v29
lb_poller          dump submissions CSV every 10min
lb_processor       auto-update lb_registry every 5min
q_test_submit      round-robin try-submit v26/v27/v28 (re-pushed after bug fix)
```

When exp175 FOLD 4 completes, auto_deploy_exp175 runs:
1. `deploy_exp175.main()`: reset notebook to v5 anchor, swap exp169→exp175 SED, export 5 ONNX, upload bc2026-exp175-sed dataset, push notebook v26
2. `submit_when_ready(v26)`: poll until Kaggle re-run done, submit v26 with hypothesis-bearing message

When all submissions land, lb_processor auto-fills lb_registry pending entries (v26, v27, v28, v29).

## Files added during this session

- `experiments/sed/q_test_runner.py` — Q variant builder + push
- `experiments/sed/q_test_submit.py` — round-robin submit poller
- `experiments/sed/q_test_interpret.py` — read results + print pseudo strategy recommendations
- `experiments/sed/auto_deploy_exp175.py` — wait+deploy+submit chain
- `experiments/sed/lb_poller.py` — periodic submissions CSV dump
- `experiments/sed/lb_processor.py` — auto-update registry
- `experiments/sed/orch_status.py` — quick status snapshot
- `experiments/_audits_post_v26/exp_konbu_head.py` — Option A (M5/M6/M7 with pos_weight; baseline LogReg comparison included)
- `experiments/_audits_post_v26/exp_sed_divergence.py` — Option B (Tucker vs exp175 fold0 divergence)
- `experiments/_audits_post_v26/exp_blend_tune.py` — Step c (linear/rank-pct blend sweep)
- `experiments/_audits_post_v26/exp_blend_per_class.py` — selective per-class blend
- `experiments/_audits_post_v26/exp_exp175_5fold_eval.py` — exp175 partial vs exp169 5-fold (failed: GPU OOM, retry later)

## Local test results (already complete)

### Option A: konbu head from public data
| Variant | val_AUC on labeled SS |
|---|---|
| LogReg-OVR balanced (exp22 baseline) | 0.687 |
| MLP 512 + multilabel | 0.735 |
| **MLP 1024→512 + primary + pos_weight** | **0.7754** |
| 5-seed MLP ensemble | 0.718 |

→ Reproducible head from Perch features + train_audio. Saved at `model-weights/own_konbu_head_m7.pt`.

### Option B: Tucker fold0 vs exp175 fold0 (full labeled SS)
- Tucker fold0: macro AUC **0.9868**
- exp175 fold0: **0.9831**
- Δ +0.0037 (matches the persistent 0.003 LB gap to within noise)
- Pearson 0.928, Top-1 agreement 0.754
- Tucker advantage on rare classes (strher2 +0.235, 47158son05 +0.196, nacnig1 +0.111, 74113 +0.078)
- Mechanism: 47158sonXX classes have all SS positives in our fixed 11-file holdout → never seen in training. Tucker's per-fold SS split exposes them in 4/5 folds.

### Step c: own konbu head + own SED blend (DEAD END)
- own SED 5-fold: 0.9685
- own head: 0.7741
- Pearson(SED, head): 0.408 (decorrelated)
- All linear/rank-pct/per-class/oracle/row-uncertainty blends: REGRESSION
- head better in only 2 of 75 valid classes
- v22 LB (-0.001) confirmed at local

## Mechanism interpretation map

For each Q-test result (delta vs anchor 0.938):

```
Δ < -0.005 (subset hurts a lot):  subset is net-very-positive on LB.
                                    Pseudo for that subset is high-value.
Δ < -0.002 (subset hurts mildly):  subset is net-positive.
                                    Pseudo for that subset is positive-EV.
|Δ| ≤ 0.002 (within noise):         subset is roughly neutral.
                                    Pseudo unlikely to move LB by much.
Δ > +0.002 (drop helps mildly):    subset is net-negative on LB.
                                    Don't pseudo-augment that subset.
Δ > +0.005 (drop helps a lot):     subset is net-very-negative.
                                    Suppress or zero out that subset's predictions in production.
```

For the exp175 v26 submit:

```
LB ≥ 0.940:  silent drift fix matters. Multi-seed ensemble next.
LB ≈ 0.938:  drift fix doesn't transfer. Pivot to per-fold SS (exp176, queued).
LB ≤ 0.936:  silent drift fix made things worse. Investigate: random seed unlucky?
```

## What to do tomorrow (sequence)

1. Read `experiments/lb_registry.yaml` for v26/v27/v28/v29 outcomes.
2. Run `uv run python -m experiments.sed.q_test_interpret` for pseudo strategy recommendations.
3. Check exp176 progress (started ~04:30 KST after exp175 done; will complete ~09:30).
4. Based on Q-test pattern + exp175 LB + exp176 progress, decide tomorrow's 5-slot strategy.

If Q-test shows clear pseudo direction (e.g., Insecta is high-value + drift fix didn't help), tomorrow's plan can target Insecta-specific pseudo with exp176 SED as teacher.

If everything is in noise band, fall back to multi-seed ensemble as planned.
