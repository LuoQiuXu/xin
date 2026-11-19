# CC-MIA Causal Attribution & Oracle Upper-Bound Protocol

Status: implementation protocol; do not implement Margin-aware Gate in this phase.

## Frozen clarification, 2026-11-19 (user instructions)

Each seed replays the parent's complete 30-epoch BASE_C3 trajectory. Record
Calibration-only source-to-target influence at every epoch and verify the
final BASE_C3 and paired C2 classifier states against the actual parent
checkpoints. Historical full-holdout rankings do not determine any mask.
Many/Medium/Few remain fixed from the original TRAIN counts (>100,
20..100, <20); they are never redefined using either development subset.

Pre-register the following practical-effect flags for each oracle relative
to BASE_C3: mean Many accuracy gain >=2.0 percentage points, mean fixed
Many margin gain >=0.10, mean Few accuracy change >=-2.0 percentage points,
and both primary effects positive together in at least two of three seeds.
Report the individual seed effects and whether each reaches these cutoffs.
These cutoffs do not express statistical significance and do not automate
Margin-aware Gate admission. The final structured decision must provide
supporting evidence, opposing evidence, limitations, and a recommendation
for human scientific review. No Gate design or implementation occurs here.

### Fixed implementation definitions, before Evaluation access

- Split seed: 20261119. Within each pre-existing class allocation, sort
  original TRAIN IDs, shuffle with NumPy SeedSequence([20261119, class_id]),
  then take 10 Calibration and 10 Evaluation. Persist row positions and raw
  TRAIN IDs. The existing prepared order declares 20-row blocks for classes
  1..99; validate Calibration labels immediately and Evaluation labels only
  after all-seed mask sealing. Reject any disagreement. No full holdout
  array, including an mmap, is loaded before sealing.
- Eligible pairs: all original non-Many sources (classes 35..99) to Many
  targets represented in Calibration (classes 1..34). Class 0 has no
  Calibration data and is excluded from ranking, while its classifier row
  remains trainable and its original gradients remain untouched. There
  are 65*34=2210 eligible pairs; K=221 by exact half-up rounding.
- Competitor q_y: for each represented Many class y, choose the non-Many
  row with the largest class-mean **initial Calibration** logit. Freeze this
  class map before Evaluation. Evaluation does not select competitors.
- The intervention removes a **single classifier row** contribution. For
  ranking, project each accumulated source direct contribution to its
  eligible target row only, using the parent's fixed-margin projection
  helper's target_logit_step output. This equals that pair's change to the
  fixed y-vs-q_y margin because q_y is non-Many and its row is not masked.
  Also record the parent's full source margin (including the non-Many
  competitor-row term) as a separate diagnostic. Do not rank a one-row
  intervention using a change to an unmodified competitor row.
- Temporal score: cumulative direct SGD-momentum parameter contribution
  along the original BASE_C3 trajectory at epoch 30, projected onto the
  ten Calibration examples of each target class and averaged. Per-batch
  direct gradients are FP32, as in the parent; the unchanged parent's FP64
  attribution accumulator propagates them through the frozen SGD
  recurrence. FP64 is bookkeeping, not model-training precision.
- Oracle Selection retains exactly K eligible pairs maximizing the sum
  of these row-local Calibration scores, suppressing the other eligible
  pairs. All ineligible direct terms and the clean path are unchanged.
  Oracle Suppression is an alias of TARGETED, with identical mask hash
  and one shared training result; it is not an additional independent arm.
- RANDOM_MATCHED uses source TRAIN bins <20, 20..49, 50..100 and exact target
  class counts, with RNG seed seed+710001. FREQUENCY_MATCHED uses the same
  source bins and target TRAIN bins 101..199, 200..349, >=350, with an
  independent RNG seed seed+820001. Match exact joint counts. Report
  continuous-count standardized mean differences and mask overlap;
  controls may naturally overlap with TARGETED.
- BASE_C3 uses the original autograd update exactly. Interventions use
  the actual clean C2-autograd gradient at their **own current state** plus
  the retained analytic source direct gradients. Validate unmasked
  reconstruction against original C3 autograd using the parent's unchanged
  atol=2e-5, rtol=2e-4. Report this finite-precision decomposition residual;
  never add an observed final reconstruction residual to force a pass.
- Use the validated parent's independent SGD parameter-writeback replay
  ledger for each actual optimizer. Preserve its bitwise parameter/buffer
  replay and operator-based rounding bound. Parent paired attribution
  reconstruction retains abs=2e-6, rel=5e-3.
- Calibration replay follows successful CPU tests and a real one-batch
  baseline parity check. It is a reference replay for mask estimation,
  not intervention formal training. Seal all three seeds' actual masks,
  rankings, split, competitor maps, source hashes, and scores together.
  Then run the production real-data Smoke for every actual arm and seed.
  Only a passing production Smoke admits the 3x30 paired formal runs.
- Class-stratified bootstrap: 2000 replicates, RNG seed 20261120,
  identical resampled examples for paired arms. Cross-seed mean intervals
  use the same sampled examples across seeds and are conditional on these
  three seeds, not a population inference over training randomness.
- No optional shuffled-sign arm or sensitivity sweep is added in this
  finite experiment. No evaluation-driven remasking, early stopping,
  hyperparameter selection, resume, or overwrite is allowed.

## Goal
Test whether the negative source-class -> Many-target fixed-margin interactions identified by CC-MIA are causally actionable, and estimate the attainable headroom under calibration-only oracle selection.

## Parent and immutable inputs
Parent: experiment/ccsam-cross-class-margin-influence-audit-20261117.
Reuse the sealed Clean Stage-I checkpoint, prepared CIFAR-100-LT IF=100 TRAIN split, original sampler plans, transforms, C2/C3 objectives, optimizer, and official TEST prohibition. Do not modify parent files or historical results. New outputs go only under this experiment directory.

## Critical split discipline
The 1,980-image TRAIN-derived holdout is development data, not an independent test set. Deterministically split it within each represented class into CALIBRATION and EVALUATION halves (10/10 per classes 1..99; class 0 remains absent). Persist exact indices and hashes. Calibration may determine source-target rankings and oracle masks. Evaluation labels/features may only be read after masks are frozen and hashed. No official CIFAR-100 TEST access.

All model selection and thresholding use CALIBRATION only. Report EVALUATION as a held-out development partition, never as independent-test generalization.

## Causal intervention
Run paired C2/C3 baseline and C3 interventions from identical Clean Stage-I classifier initialization, same per-seed sampler, augmentation realization, batch order, and optimizer settings.

The intervention unit is a source-class -> target-Many-class classifier-row gradient contribution, not a vague class-level sample count. For each batch, decompose the C3 direct perturbation gradient by source label and target classifier row using the validated CC-MIA analytic formula. Apply a frozen binary mask to these direct perturbation gradient components before they enter the C3 optimizer update. Do not mask clean CE gradients, class weights, C2, or the feature perturbation itself. Document that this is a controlled optimizer-gradient intervention, not a new end-to-end loss.

Arms:
- BASE_C3: original C3, no suppression.
- TARGETED: suppress the K most negative source->Many target pairs selected using CALIBRATION-only influence scores.
- RANDOM_MATCHED: suppress exactly K eligible pairs sampled deterministically, matching source frequency bins and target Many class counts of TARGETED.
- FREQUENCY_MATCHED: suppress K pairs sampled deterministically from the same source-frequency x target-frequency strata, without using influence signs.
Use the same K, update count, learning rate, momentum and weight decay. Freeze masks before evaluation.

The causal contrast of interest is TARGETED minus RANDOM_MATCHED (and FREQUENCY_MATCHED), with paired seeds. A positive result is not established by a single favorable seed.

## Oracle upper bounds
On CALIBRATION only, construct:
- ORACLE_SUPPRESSION: suppress the K most negative source->Many pairs (same feasible intervention family and same K as TARGETED).
- ORACLE_SELECTION: among eligible source->Many pairs, select the K most positive beneficial pairs to retain / equivalently choose the best K pair set under a precisely documented objective. This is an optimistic selection bound, not a deployable method.
Do not call any oracle result a theoretical global upper bound. It is a calibration-constrained empirical oracle bound for this intervention family.

If targeted and oracle masks are identical by construction, do not report them as independent methods; report their equivalence and focus on calibration-estimation uncertainty. Oracle Selection must be explicitly distinct from suppression and use a fixed K budget.

## Leakage and fairness controls
- Generate masks using calibration-only scores; serialize mask, ranking table, calibration indices hash, code hash, and random seeds before evaluation.
- Ensure no evaluation label, metric, or per-example margin enters mask construction.
- Random controls match pair count K and source/target frequency strata; report balance diagnostics.
- Include a negative-control mask with shuffled influence signs if computationally feasible.
- No hyperparameter sweep. K is fixed before evaluation, derived from the eligible pair count and predeclared as 10% of eligible pairs (round half up); also report sensitivity descriptively only if pre-registered.
- Keep source and target semantics explicit: source = perturbed input ground-truth class; target = classifier row receiving direct gradient.
- Check gradient additivity, masked-gradient reconstruction, optimizer state finiteness, and exact zeroing of masked pair components.

## Metrics
Primary on EVALUATION partition:
1. Many accuracy (percentage points).
2. Mean fixed Many-vs-non-Many margin using competitors selected from CALIBRATION only and frozen.
3. Overall / macro accuracy and Few accuracy (guard against sacrificing tail gains).
4. Per-class Many accuracy and margin.
Secondary: targeted/random paired seed deltas, bootstrap CI over evaluation examples (class-stratified), mask stability across seeds, calibration/evaluation score agreement.

Report all 3 seeds (123,124,125), mean, sample SD, paired differences, and class-stratified bootstrap 95% intervals. Do not claim population-level statistical significance from this small development holdout.

## Execution gates
1. Source audit: read parent train_ccmia.py, train_ccsam.py, ccmia_core.py and parent protocol; reuse rather than duplicate formulas.
2. CPU unit tests: pair masking, source/target indexing, additivity, deterministic split, matched random controls, no leakage.
3. Real-data smoke: one paired minibatch per arm/seed; verify original C3 trace parity before intervention and exact masked component accounting. No formal output before pass.
4. Formal: 3 seeds, fixed 30 epochs, no resume/overwrite/best-epoch selection.
5. Analysis: produce machine-readable JSON/CSV and human-readable report. Abort on any failed gate.

## Required outputs
- protocol and code hashes, prepared manifest hash, calibration/evaluation split manifest
- mask tables per seed and arm; pair-level calibration scores; matched-control balance table
- preflight/smoke reports and formal completion ledger
- per-seed epoch metrics, per-class evaluation metrics, paired deltas, bootstrap intervals
- gradient decomposition / masking reconstruction diagnostics
- final report with decision: causal evidence supported / not supported / inconclusive; oracle headroom substantial / small / invalid.

## Decision rules (predeclared)
Support for actionable negative interactions requires TARGETED to outperform BOTH matched random controls on Many accuracy and fixed margin in at least 2/3 seeds, with consistent paired mean direction and no material collapse in Few accuracy. Report uncertainty; do not equate this with definitive causality beyond the controlled intervention.

Proceed to Margin-aware Gate design only if:
(a) intervention is valid and targeted beats matched controls;
(b) Oracle Suppression has measurable headroom over BASE_C3;
(c) Oracle Selection shows a non-trivial attainable gap;
(d) effects are not driven by one class or one seed.
Otherwise stop or revise the hypothesis. No Gate implementation in this branch.
