# Autonomous session summary — 2026-06-02 → 06-03 (user asleep ~8h, woke briefly mid-session)

Honest, forward assessment. Anchor LB = 0.950 (never beaten, never broken). Deadline 06-03 23:59 UTC.
NOTE on time: local machine clock is KST (UTC+9); all LB/deadline times here are UTC.

## What I tested at the LB today (5/5 slots used)
| # | lever | LB | read |
|---|---|---|---|
| 1 | exp187 — distilled student SED (B1+SoftAUC, hetero-teacher pseudo) full-blend W=0.10 | **0.938** | REGRESSION — a weaker student dilutes Tucker |
| 2 | ptlam — per-taxon prior λ (non-Aves 0.15, Aves 0.65) | **0.950** | flat — non-Aves priors already near-uniform |
| 3 | exp189 — Tucker recipe + EXTERNAL non-Aves data, full-blend W=0.40 | **0.950** | flat |
| 4 | exp189 W=0.70 (dose) | **0.949** | flat (slight down at higher W) |
| 5 | ptax2 — drop genus tax-smoothing for 25 Insecta sonotypes (site-invariant) | **0.950** | flat |

## The one genuine achievement (your distillation/external-data push was RIGHT)
**exp189 is the first own-trained SED in this project to BEAT the public Tucker SED on local eval** — and it
did so by exactly the BirdCLEF-2025-winner ingredient you insisted on: the same Tucker recipe **+ 552 external
non-Aves focal clips** (Xeno-canto/iNat) folded into the training cache. Controlled gate (3-fold) vs Tucker:
all-eval 0.9979 vs 0.9972, non-Aves 0.9983 vs 0.9978, Aves 0.9966 vs 0.9953 — beats it on **every** group;
rank-blend net-positive on all taxa. This is the structural opposite of exp187 (which *diluted* Tucker with a
weaker model and dragged). External data, done properly, produced a measurably better model.

## The honest finding (NOT a ceiling — a located bottleneck)
A model that is **strictly better on the validation distribution moved the LB by 0.000**. exp189 W=0.40→0.950,
W=0.70→0.949. So the bottleneck is **not** SED quality, training-data volume, or any post-proc knob — it is
**local→LB transfer under cross-region covariate shift** (our labeled soundscapes are 68% site-22; the test is
multi-site). Every local-improving lever (ptlam, exp189, ptax2) is LB-flat; the one that dragged (exp187) was
weaker. The 0.950→0.963 gap is real (top teams prove the signal exists) but it lives in the **transfer**, not
in anything our site-22-biased validation can see us improve.

## Untested axes that remain (the lever exists; it must address TRANSFER, not local quality)
1. **Multi-site supervision** — the root fix: reduce the site-22 bias in what we validate/select on (we lack
   multi-site labels; this is the resource gap, and is exactly the CLEF-paper thesis).
2. **A different foundation embedding** with native non-Aves + cross-region robustness (Perch 2.0 dropped as an
   untrusted download; a properly-vetted alternative is open).
3. **Final-level integration** of exp189 (rank-blend AFTER all post-proc, like the phase4 B0 cell) rather than
   into the SED stream — untested; may pass the signal through differently.
4. **Per-taxon ENSEMBLE_W** (Insecta ProtoSSM weight 0.35→0.05) — the Proto/SED MIX per taxon, untested.
5. exp189 trained with an explicitly **site-invariant** objective/validation (the only thing that targets the
   actual wall).

## Final submission selection (at deadline)
Best public = 0.950, achieved by several submissions (anchor eos8-verbatim, ptlam, exp189-W0.40, ptax2 — all
tied). Pick any 2 of the 0.950s for the private LB; the 0.950 anchor is protected by select-best regardless.

## Reusable assets built this session
`experiments/_data_pipelines/exp189_build_external_cache.py` (external-data cache), `sed/config.py:exp189_*`
(+ `ANCHOR_CACHE_DIR` override in `train.py`), `exp189_gate.py` (fold-filtered ensemble gate), `exp189_patch_
notebook.py` (full/non-Aves modes), `ptlam_patch.py`, `ptax2_patch.py`, `bc2026-exp189-sed` Kaggle dataset.
Paper: `paper/covariate_shift_findings_2026_06_03.md` updated — exp189 is the **purest** demonstration of the
covariate-shift thesis (better-locally yet LB-flat isolates transfer from capacity).
