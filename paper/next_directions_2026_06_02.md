# Next directions — beyond the 0.950 local optimum (2026-06-02)

> ⚠ HISTORICAL candidate menu, written 2026-06-02 BEFORE the final-day LB tests. Several headline claims here
> were empirically REFUTED on 06-03 — flagged so a 2027 reader doesn't act on them:
> - "the non-Aves bottleneck (zero Perch prototypes) is directly fixable" / "external data closes the gap":
>   exp189 (external non-Aves data, beat Tucker on every evaluable group) was **LB-flat (0.950)**.
> - Tier 2a "multi-teacher distill → fresh SED = highest LB-proven upside": exp187 ran exactly this →
>   **LB 0.938 (regression)**.
> The bottleneck is local→LB **TRANSFER** (cross-region covariate shift), not coverage/recipe. The scout-agent
> IDs below are stale (not re-queryable). Canonical account: `covariate_shift_findings_2026_06_03.md`.

Deep-research synthesis from 4 parallel literature scouts (Hugging Face Papers + arXiv + venue
proceedings), June 2026. Goal: find levers OUTSIDE the local optimum (Perch v2 frozen + ProtoSSM
+ Tucker SED → LB 0.950; public top 0.963).

Scout agent IDs (re-queryable via SendMessage for deeper follow-up):
- Track A foundation models: `a6d0024d9167d90f1`
- Track B domain adaptation: `a3f0014ddd0e1ee85`
- Track C semi/self-supervised: `ac5faa45f37fa4b12`
- Track D few-shot/long-tail: `afa7051bfcdc46b6f`

---

## Cross-cutting insights (read first)

1. **ROC-AUC is rank-invariant to per-class monotone rescaling.** Re-weighting a species' scores
   by any constant/temperature/prior cannot change that class's AUC. This *mathematically explains*
   why six days of dial-tuning (proto/SED weight, hour-prior λ, per-taxon scaling, quantile cal)
   were flat. **The levers that move macro-AUC must change the per-window RANK ORDER** — i.e. inject
   genuinely new signal (new embedding, new/replaced prototype, new SED), not post-hoc rescaling.
   This retires the entire "rescale the existing scores" family and focuses effort correctly.

2. **The root bottleneck (31 non-Aves species with zero Perch prototype) is now directly fixable.**
   Two independent routes, both new:
   - Replace the frozen embedding model with **Perch 2.0**, which *has* non-Aves prototypes (Track A).
   - Keep frozen Perch v2 but **build the missing prototypes ourselves** from `train_audio` + the
     10,600 unlabeled soundscapes (TEEN / TIM / T3A / DC — Tracks B,C,D all converged here).
   The in-project fact that Perch embeddings already separate TP/TN for unmapped species
   (centroid cosine 0.33–0.59) is exactly the precondition these methods need.

3. **Label-free > label-tuned, because of the site-22 trap.** Every method below that wins does so
   WITHOUT tuning on the labeled soundscapes (which are 68% one site and anti-correlate with LB).
   Validation must avoid site-22: use held-out-taxon CV, the 12 PCEN-gated Insecta OOF (AUC
   0.68–1.00 from `train_audio` CV), or unsupervised criteria (MMD), NOT labeled-SS macro-AUC.

4. **There is a BirdCLEF-scale-proven escape.** BirdCLEF+ 2025 1st/2nd place moved 0.87→0.922 with
   heterogeneous multi-teacher distillation onto a fresh SED (SoftAUCLoss, 50/50 labeled/unlabeled).
   Our past pseudo-label failures were *self*-distillation / single-teacher — a different mechanism.

---

## Tiered action plan

Ordered by expected value ÷ cost. Tier 0 is free and carries no LB risk — start there.

### Tier 0 — today, no training slot, no LB risk

- **T0a. Perch 2.0 prototype-coverage probe.** Load `cgeorgiaw/Perch` (arXiv:2508.04665) via
  `perch-hoplite`; extract embeddings; check whether the 31 currently-unmapped BirdCLEF-2026
  non-Aves species now have non-zero prototypes / positive activations. This single check decides
  whether the headline model-swap (Tier 3) is viable. Attacks bottleneck #1 at the root.
  *Risk to verify:* CPU build / ONNX export may hit the same shape-inference error that blocked
  Perch v2; license = Apache-2.0 (OK). GPU+TF needed for the probe.

- **T0b. TEEN training-free prototype calibration** (NeurIPS 2023, arXiv:2312.05229) on frozen
  Perch v2. Compute per-class `train_audio` centroids for the 72 non-Aves species; nudge each toward
  acoustically-similar Aves base prototypes via softmax(cosine/τ). ~80 lines numpy, no training, no
  soundscape labels (site-22-safe), ships as a ~1.4 MB `.npz`, CPU-inferable. Blend as a non-Aves-only
  third stream (β≈0.15–0.25), bypassing the zero-logit ProtoSSM entries. Validate on the 12 PCEN-gated
  Insecta OOF. **Highest confidence-to-effort of everything found.**

### Tier 1 — cheap local (GPU hours, no full training)

- **T1a. α-BN domain alignment on Tucker SED** (arXiv:2110.04065). Recompute the EfficientNet-B0
  BatchNorm running stats over the 10,600 unlabeled soundscapes, α-blend with source stats, bake into
  ONNX. ~1 h, label-free, drop-in ONNX swap, directly targets the focal→soundscape feature shift.
  Select α unsupervised via MMD (no labeled-SS). Track B's top single-slot pick.
- **T1b. Unlabeled-soundscape pseudo-prototypes (T3A / TIM)** (NeurIPS 2021 arXiv via T3A; TIM
  bioacoustic DCASE +27% F, MDPI Bioengineering 2024). Needs the ~88 min Perch-embedding extraction on
  the 10.6k SS first. Build per-class pseudo-prototypes gated by Tucker SED score (>0.3 positive,
  <0.05 negative oracle); replace the zero-signal ProtoSSM entries for the 31 unmapped species.
  Maps perfectly onto ProtoSSM's `(N,1536,4)` prototype structure.

### Tier 2 — one training slot (~3 h) each

- **T2a. Multi-teacher cross-model distillation → fresh SED + SoftAUCLoss.** BirdCLEF+ 2025
  1st/2nd-place recipe (CEUR-WS Vol-4038 paper_256; github VSydorskyy). Ensemble Tucker 5-fold + a
  *public* model's logits on the 10.6k unlabeled SS → soft pseudo-labels → train a fresh EfficientNet-B0
  student on 50/50 labeled-focal/unlabeled-SS, supplementing non-Aves focal data to avoid Aves-collapse.
  **Highest LB-proven upside; also highest retry-risk** (must use heterogeneous teachers — single-teacher
  self-distillation is the documented failure; SoftAUCLoss "tried" before was on the SED head directly,
  not in this ensemble-distillation context — flag and verify).
- **T2b. RLDAM macro-AUC margin loss** (AAAI 2025 arXiv:2412.18231; theory ICML 2023 arXiv:2305.05248).
  Per-class adaptive margin `Δ_k = λ/|D_k+|^(1/4)` + rare-class reweighting; a *theoretically grounded*
  macro-AUC surrogate, distinct from the failed plain focal/soft-AUC. One Tucker retrain; judge on the
  bottom-quartile-by-clip-count species, not overall labeled-SS.

### Tier 3 — bigger swings / overnight / strongest CLEF-paper material

- **T3a. Perch 2.0 integration** (if T0a is positive): re-export ONNX, rebuild the ProtoSSM head with
  real non-Aves prototypes, swap the embedding stream. The single highest-EV move; restores signal for
  31 species at ~zero architectural cost. Gated entirely on the T0a probe + ONNX-export feasibility.
- **T3b. esp-aves2-sl-beats-bio** (arXiv:2508.11845, EarthSpeciesProject on HF) — SSL bioacoustic
  encoder with the best measured focal→soundscape generalization (0.01 vs 0.09 AUC drop on domain shift),
  fine-tunable, open weights (CC-BY-NC-SA — verify Kaggle terms). As a replacement or 3rd stream;
  CPU-inference budget (BEATs ~90M) is the gating risk → needs ONNX/quantization.
- **T3c. Masked-spectrogram continued pretraining on unlabeled SS** (MaskSpec arXiv:2204.12768;
  SONAR arXiv:2509.15703; BirdMAE in arXiv:2508.01277). Domain-adaptive pretraining of a fresh
  ConvNeXt/EffNet backbone on the 10.6k SS (class-agnostic → cannot amplify site fingerprint or collapse
  to Aves), then supervised head on focal data. Overnight (~6–12 h pretrain + 3 h finetune). **Best paper
  angle** (novel domain-adaptive pretraining for multi-taxa Pantanal soundscapes).

---

## Per-track top pick + "one 3-h slot" vote

| Track | #1 pick | Source | One-slot vote |
|---|---|---|---|
| A — foundation models | Perch 2.0 (drop-in 1536-d, has non-Aves) | arXiv:2508.04665 / HF `cgeorgiaw/Perch` | Perch 2.0 prototype-coverage probe |
| B — domain adaptation | T3A pseudo-prototypes (frozen, label-free) | NeurIPS 2021; α-BN arXiv:2110.04065 | α-BN BN-stat update on Tucker |
| C — semi/self-sup | Multi-teacher distillation → fresh SED | CEUR-WS Vol-4038 paper_256 | Multi-teacher distill + SoftAUCLoss |
| D — few-shot/long-tail | TEEN training-free calibration | NeurIPS 2023 arXiv:2312.05229 | TEEN (<5 min) + TIM (<1 h) prototype `.npz` |

**Convergent recommendation:** the cheapest high-EV moves (TEEN + frozen-embedding prototypes for
non-Aves, and α-BN) need NO LB slot to build and are site-22-safe — do them today. In parallel, the
Perch 2.0 probe decides the headline swap. Reserve the first real training slot for the BirdCLEF-2025-
proven multi-teacher distillation.

## CLEF paper angle
α-BN, T3A/NOTELA-LAME, SSL continued pretraining, and multi-teacher distillation are all
covariate-shift remedies → directly serve the cross-region covariate-shift thesis. The
"AUC-invariance ⇒ only rank-changing levers move macro-AUC" observation + the trainable≠evaluable /
site-22 structural diagnosis is itself a clean methodological contribution.

## Honesty notes vs the "already failed" list
- TEEN/TIM/T3A/DC ≠ the failed "MLP/DPO on Perch embedding": those were *supervised, label-tuned* (site-22 trap); these are training-free or unsupervised-transductive on *unlabeled* data.
- Multi-teacher distillation ≠ the failed "pseudo-label iteration (6×)": those were self-/single-teacher (circular); the proven recipe uses heterogeneous teachers + soft labels + AUC loss. Still the highest retry-risk item — verify the distinction holds before spending the slot.
- RLDAM ≠ the failed "soft-AUC/focal": per-class adaptive margin with a macro-AUC generalization bound, not confidence down-weighting. Carries retry-risk; gate on rare-tail per-class AUC.
- α-BN ≠ the failed "linear site/file-mean centering of Perch": different layer (SED BatchNorm), does not touch the Perch embedding or subtract species signal.

---

## Tier-0 EXECUTED (2026-06-02) — `experiments/teen_nonaves_probe.py`

Validation basis (site-22-safe, alignment verified): `cache/perch_arrays.npz` embs (708×1536) +
`cache/sidecar_dpo/y_soundscape.npy` (708×234) — confirmed aligned (macro-AUC of cached `scores`
vs y = 0.739, ≠ 0.5; class order == `oof_gate`; `y.sum(0)` == gate `n_pos` exactly).

- **TEEN (T0b) → NULL.** Cross-taxon prototype calibration (borrow from Aves base) *hurts* non-Aves:
  best config collapses to α=0.9 (own-centroid-only), Δ macro-AUC −0.003…−0.006 across n_pos≥5/10/20.
  The agent's cross-taxon-contamination risk was correct. **Mechanism ruled out.**
- **RAW Perch-v2 centroid stream → the real lever.** Mean `train_audio` embedding per class, cosine to
  SS embedding, gives non-Aves soundscape macro-AUC **0.85–0.88** (n_pos≥5/10/20), beating cached ref
  `scores` by **+0.03–0.04**. Rank-blending it into ref (non-Aves only) lifts evaluable-non-Aves
  macro-AUC **+0.030 (w=0.2) … +0.046 (w=0.6)**, 7–8/14 spp improve, flat across w (no knife-edge),
  Aves untouched. Centroids saved → `model-weights/teen_nonaves_prototypes.npz` (`raw_centroids`).
- **Limits:** only 44/72 non-Aves have `train_audio` (median 3 clips; 28 have NONE → no centroid, untouched);
  eval is on the site-biased labeled SS; local≠LB. But the centroid is built site-invariantly and the
  stream changes per-window rank order → it clears the falsifiable submit bar.

**→ Next move (needs an LB slot + Kaggle dataset/kernel push = user confirm):** patch `eos8-phase4`
into `eos8-emcent` — upload `teen_nonaves_prototypes.npz`, score cosine-to-centroid for the 44 non-Aves
species in-kernel (Perch embeddings already computed there), rank-blend at w≈0.3–0.4 for non-Aves only,
submit one slot. This is the BirdCLEF-2025-proven "embedding probe as extra stream" idea, but built
site-invariantly and gated to the taxa where the dominant ProtoSSM stream is weakest.

**Updated one-slot priority:** (1) the embedding-centroid non-Aves stream above (built + locally
positive, just needs the LB slot); (2) Perch 2.0 coverage probe (T0a, needs model download); (3)
multi-teacher distillation (T2a, first 3-h training slot).
