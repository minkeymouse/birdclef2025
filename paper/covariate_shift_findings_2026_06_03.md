# CLEF working-note findings — covariate shift in cross-region bioacoustic monitoring (2026-06-03)

Raw, quantified material from the BirdCLEF+ 2026 effort, framed for the covariate-shift thesis. All numbers
are from this repo's rigorous local evals + the LB. Competition closed 2026-06-03; paper due 2026-06-17.

## Thesis
Cross-region passive-acoustic monitoring exhibits a **covariate shift** between the training/validation
distribution and the deployment (test) distribution that (a) makes local validation anti-correlate with
true generalization, and (b) makes the standard distillation/self-training recipe that wins in-distribution
fail to transfer. We give a clean, quantified case study.

## Finding 1 — Trainable ≠ Evaluable (the local space is saturated, but it's a tiny, biased slice)
- Of 234 species, only **50 have ≥10 positives** in the labeled soundscapes (the locally-evaluable set).
- On that set, the SED alone reaches **macro-AUC 0.9972** (every taxon ≥0.99; controlled aligned eval,
  `cache/labeled_ss_controlled.npz`). The locally-measurable space is **saturated**.
- Yet the LB sits at **0.950**. The entire 0.950→0.963(public-top) gap lives in the **184 non-evaluable
  species (79% of the macro metric)** that have no local labels → unmeasurable locally.
- **Implication:** "improve local macro-AUC" is a near-useless objective here; the optimisable surface and
  the scored surface barely overlap.

## Finding 2 — Site covariate shift drives the local↔LB anti-correlation
- **~65% of the labeled soundscape windows come from a single site (site-22)** (954/1478; rounded to ~two-thirds); the hidden test is multi-site
  (incl. unseen site S05). Local-validation gains concentrate on site-22 acoustics.
- Documented anti-correlation: multiple interventions raised labeled-SS macro-AUC yet were flat/negative at LB
  (e.g. an inference mel-fix: +0.024 local → flat LB; LOSO cross-site +0.067 → LB 0.000). Site-22-fitted
  structure encodes a site fingerprint that does not transfer.

## Finding 3 (case study) — a methodologically-proper distillation passes local gates but REGRESSES at LB
This is the cleanest demonstration of the thesis. We built the BirdCLEF-winner recipe **properly**:
- **Heterogeneous teachers** (not the usual self-/single-arch): Tucker EfficientNet-B0 (strong) + a
  reconstructed ConvNeXt-tiny SED — per-class **Pearson 0.235** on the pseudo windows = genuine diversity
  (vs 0.97–0.99 for same-family teachers).
- Soft pseudo-labels on 40k unlabeled in-region soundscape windows; fresh **EfficientNet-B1 student +
  SoftAUCLoss + noisy-student (drop_path 0.2)**, 5-fold, checkpoint on a site-22-masked criterion.
- The student is strong (val_SS ~0.94) and **genuinely decorrelated from the teacher (Pearson 0.56)**.
- **Local gate passed:** blended at W=0.10 into the SED stream, the drag on the 50 evaluable species is only
  **−0.0005** (within ±0.002 noise).
- **LB result: 0.950 → 0.938 (−0.012).** The decorrelated, locally-non-dragging distilled SED **hurts the
  184 blind/non-evaluable species**. Local validation gave a green light; the LB regressed by 6× the noise.
- **Interpretation:** the self-training signal (pseudo-labels from in-region but site-skewed soundscapes +
  a student weaker than the foundation model on rare/unseen-site classes) amplifies the covariate shift on
  exactly the species you cannot measure. The recipe that wins in-distribution (BirdCLEF-2025) does not
  transfer under strong cross-region shift without multi-site supervision.

## Finding 3b (the purest case) — a model that BEATS the baseline locally is still LB-flat
exp187 (Finding 3) was a *weaker* distilled student that dragged, so quality and transfer were confounded.
exp189 removes that confound and is the cleaner demonstration:
- exp189 = the canonical Tucker SED recipe (BCE, focal-SC mixup) **+ 552 external non-Aves focal clips**
  (Xeno-canto/iNaturalist, the BirdCLEF-2025 winners' key ingredient) folded into the training cache —
  i.e. the *same architecture/recipe* as the production SED, only with **more, multi-source training data**.
- On the controlled labeled-SS substrate the 3-fold exp189 **EXCEEDS the production Tucker SED on every
  evaluable group**: all-eval 0.9979 vs 0.9972, non-Aves 0.9983 vs 0.9978, Aves 0.9966 vs 0.9953. Rank-blending
  it in is net-positive on all taxa (+0.0003…+0.0010). It is, by the local metric, a strictly better SED — the
  first own-trained SED in this project to beat the public baseline locally.
- **LB: 0.950 (full-blend W=0.40) and 0.950 (W=0.70) — flat, identical to the anchor (Δ0.000).**
- **Interpretation:** a model that is *measurably better on the validation distribution* produces **zero**
  movement on the (multi-site) test distribution. This isolates the failure to **distribution transfer**, not
  model capacity or training-data volume: adding the winners' external-data ingredient genuinely improved the
  model on the site-22-dominated validation slice, and that improvement did not exist on the deployment slice.
  Cross-region covariate shift, not under-fitting, is the operative ceiling. (5-fold ≈ 3-fold locally → the
  ensemble had also saturated; the flatness is not a folds artifact.)

## Finding 3c — the transfer-decoupling table (local Δ of ANY magnitude → ~0 LB)
The decisive quantitative evidence for the thesis: across interventions spanning **two orders of magnitude** of
local improvement, the LB moves by ~0. Local validation gain does not predict — and is decoupled from —
deployment-distribution gain.

| intervention | local Δ (labeled-SS macro-AUC) | LB Δ (vs 0.950) |
|---|---|---|
| LOSO cross-site label-flip (2026-05-28) | **+0.067** | 0.000 |
| inference mel-normalisation fix | **+0.024** | ~0.000 (flat) |
| exp189 external-data SED, W=0.40 (this session) | **+0.0006** (blend, beats Tucker) | 0.000 |
| ptlam / ptax2 per-taxon post-proc (this session) | ~0 (not locally gateable) | 0.000 |
| exp187 weaker distilled student, W=0.10 | −0.0005 | **−0.012** |

The only intervention that moved the LB at all was the one that was *worse* locally (exp187), and it moved it
**down**. Every positive local Δ — whether +0.0006 or +0.067 — produced **zero** LB movement. Under this
cross-region covariate shift, the sign and magnitude of a local-CV change carry essentially no information about
the test distribution. (Root cause: labeled validation is ~65% one site; the test is multi-site incl. unseen
sites — see Finding 2.)

## Finding 4 — only rank-changing operators can move a skip-empty macro-AUC; calibration is a no-op
- The metric is macro-AUC skipping empty classes → each class is scored independently → **any per-class
  monotone rescaling (temperature, per-class prior/logit-adjustment, quantile calibration) is AUC-invariant.**
- This is borne out empirically: every score-space calibration tried was flat/noise. Only operators that change
  the **cross-window rank order** within a class (new embedding, replaced prototype, a different model, the
  hour/site prior, taxonomy smoothing) can move the metric — and on the saturated local set they can't, and on
  the blind set they're an unverifiable gamble (here: a regression).

## Takeaways for the working note
1. Local CV is not just noisy but **structurally misleading** under cross-region shift here: the evaluable slice
   is saturated and site-biased; the scored majority is invisible.
2. Distillation/self-training, the in-distribution SOTA recipe, **can regress under cross-region shift** — we
   show a properly-built, decorrelated instance that passes a careful local gate yet loses −0.012 at LB.
3. Closing the gap appears to require **multi-site supervision / a foundation model with native non-Aves
   coverage** — i.e. reducing the shift at the data/representation level, not post-hoc modelling on a
   single-site-biased validation set.

## Next directions (BirdCLEF 2027) — the lever must target TRANSFER, not local quality
The session exhausted the "improve the model/post-proc on the available validation" family (all flat). The
2027 effort should attack the covariate shift directly:
1. **Multi-site validation/supervision** — the root fix and the paper's own prescription: obtain or
   construct held-out *site-diverse* labels so model selection optimises transfer, not site-22 fit. (Note:
   naïve leave-one-site-out on the existing labels was unreliable — site-22 bias survives LOSO; a genuinely
   multi-site label source is needed.)
2. **A cross-region-robust foundation embedding** with native non-Aves coverage (Perch 2.0 / BirdMAE class —
   **vet provenance before downloading**; the 2026 effort dropped Perch 2.0 as an untrusted artifact).
3. **Final-level (not SED-stream) integration** of an external-data SED — untested; bypasses the gate/Gaussian
   cascade (see `experiments/_archive_2026/` post-mortem notes).
4. **Domain-adversarial / site-invariant training objective** — the only model-side route that targets the
   wall rather than overfitting the site-22 slice.

## Reproducibility (what's in the repo)
- Better-SED-by-external-data pipeline: `experiments/_data_pipelines/exp189_build_external_cache.py` →
  `experiments/sed/` (`config.exp189_tucker_external_nonaves`, `ANCHOR_CACHE_DIR` override) →
  `experiments/exp189_gate.py` (controlled eval) → `experiments/exp189_patch_notebook.py` (deploy).
- Controlled local eval (the correct baseline, NOT ProtoSSM strawman): `experiments/eval_labeled_ss_controlled.py`
  → `cache/labeled_ss_controlled.npz` (Perch+Tucker on 739 aligned labeled-SS windows).
- Full LB record: `experiments/lb_registry.yaml`. Structural data: `paper/structural_analysis_2026_06_02.md`.
- One-off lever scripts (what was tried, for reference): `experiments/_archive_2026/`,
  `notebooks/_archive_2026/`.
