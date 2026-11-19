# CC-MIA Causal Attribution & Oracle Upper-Bound Protocol

Status: implementation protocol; do not implement Margin-aware Gate in this phase.

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
