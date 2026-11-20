# TP-MAG: Tail-Preserving Margin-Aware Gate

Status: one finite method-construction experiment. No additional mechanism audit or hyperparameter search is authorized by this protocol.

## 1. Research hypothesis

The existing CC-MIA causal experiment found that suppressing selected non-Many source -> Many classifier-row direct perturbation-gradient components can improve Many accuracy and fixed Many margins, but incurs a Few-accuracy cost. The logit-geometry diagnostic further showed that the Many/Few trade-off is not explained by the tested one-dimensional log-prior adjustment family.

Hypothesis H1:

For a direct perturbation-gradient component from source class s to target Many classifier row y, the same row update has two distinct first-order consequences:

- It can improve or damage the target Many class's fixed margin against its frozen non-Many competitor.
- It can improve or damage the source class's margin against the aggregate Many-logit competitor set.

A continuous gate based on both consequences can retain useful Many-directed transfer while attenuating contributions that endanger the source class's tail margin.

This is a falsifiable hypothesis. The experiment is not allowed to add further modules if it fails.

## 2. Frozen setup

Repository: LuoQiuXu/xin

Parent branch: experiment/ccmia-logit-geometry-diagnostics-20261120

Method branch: experiment/tail-preserving-margin-aware-gate-20261120

Pinned parent commit: bb8536273effd8a478812509a5389c6c01c154ae

Parent experiment: H2T_repro_audit/ccsam_causal_oracle_20261119

Dataset: CIFAR-100-LT, IF=100

Backbone: frozen Clean Stage-I ResNet-32

Classifier feature: 64-dimensional Clean-compatible GAP

Seeds: 123, 124, 125

Training: 30 epochs, batch size 64, 5,100 optimizer updates per seed

Optimizer: SGD, lr=0.01, momentum=0.9, weight decay=2e-4, Nesterov=False

Precision: FP32; AMP disabled

Many: original TRAIN count > 100

Medium: 20 <= original TRAIN count <= 100

Few: original TRAIN count < 20

Calibration/Evaluation: the parent's sealed 990/990 TRAIN-derived development split. Each represented class has 10 examples per partition; class 0 is absent. Expected per-partition group counts are Many=340, Medium=350, Few=300.

The Clean Stage-I checkpoint, training images/labels, sampling plans, C0 batch traces, augmentation generator, and all parent provenance checks are reused. The frozen backbone and its BatchNorm state must remain bitwise unchanged.

## 3. Pairwise margin definitions

Let G[s,y] = (G_W[s,y], G_b[s,y]) be the direct perturbation-gradient component for source class s and target classifier row y, using the unchanged CC-MIA decomposition. Source means the ground-truth class of the perturbed training input. Target means the classifier row receiving the direct gradient.

Only the existing eligible family is gated:

- source s: original TRAIN count <= 100 and represented in Calibration;
- target y: original TRAIN count > 100 and represented in Calibration.

For this split, this is exactly 65 source classes x 34 target Many classes = 2,210 pairs. Many-source direct perturbation gradients must remain exactly zero. All ineligible source/target components receive gate 1 and remain unchanged.

### 3.1 Target Many benefit

For each represented Many class y, select q_y once using the initial Clean Stage-I classifier: q_y is the non-Many class with the largest mean initial Calibration logit over examples whose label is y. Freeze q_y.

Define the target margin:

m_head,y(x) = z_y(x) - z_q_y(x).

Let mu_y be the mean 64-D Calibration feature for class y.

The pair G[s,y] changes only target row y. Under a hypothetical SGD step theta_y' = theta_y - eta * G[s,y], its first-order average target-margin change is:

Delta m_head[s,y] = -eta * (G_W[s,y] dot mu_y + G_b[s,y]).

Define the unit-learning-rate benefit score:

b[s,y] = -(G_W[s,y] dot mu_y + G_b[s,y]).

Positive b means the pair update increases the target Many margin to first order. The fixed competitor row q_y is not changed by this pair.

### 3.2 Source tail-preservation risk

For a non-Many source class s, define its margin against the aggregate represented Many set M:

m_tail,s(x) = z_s(x) - logsumexp over k in M of z_k(x).

Use the initial Clean Stage-I classifier to compute:

p_y^M(x) = softmax({z_k(x): k in M})_y.

For each eligible pair (s,y), freeze the Calibration-only quantities:

v[s,y] = mean over x with label s of p_y^M(x) * x

a[s,y] = mean over x with label s of p_y^M(x).

The first-order tail-margin change under the same hypothetical SGD step is:

Delta m_tail[s,y] = eta * (G_W[s,y] dot v[s,y] + G_b[s,y] * a[s,y]).

Define the unit-learning-rate tail shift:

t[s,y] = G_W[s,y] dot v[s,y] + G_b[s,y] * a[s,y].

Positive t preserves or improves the source class's margin against the aggregate Many set. Negative t is a tail-margin risk.

This is an aggregate Many-competitor margin, not a guarantee for every non-Many-vs-non-Many boundary.

## 4. Gate definition

B[s,y] = max(0, b[s,y])

R[s,y] = max(0, -t[s,y])

g[s,y] = B[s,y] / (B[s,y] + R[s,y] + 1e-12).

The epsilon is a fixed numerical constant, not a tunable hyperparameter.

Properties:

- If an eligible pair's direct gradient is exactly zero, g = 1 as an identity operation; there is no contribution to suppress.
- For a nonzero eligible pair, if b <= 0, g = 0: do not retain a pair that has no positive first-order target-Many benefit.
- If b > 0 and t >= 0, g is approximately 1: retain a beneficial, tail-safe pair.
- If b > 0 and t < 0, 0 < g < 1: attenuate the pair according to the relative head benefit and tail risk.
- Every ineligible pair has g = 1.
- The gate is recalculated for every training minibatch from the current direct pair gradient and the frozen Calibration reference. Gate-distribution summaries are computed over active (nonzero) eligible pair gradients, while the total eligible-pair count is reported separately.

The actual optimizer gradient is:

G_TPMAG = G_clean_CE + sum over source s of g[s,:] * G_direct[s,:].

The clean CE gradient, feature perturbation delta, class weights, optimizer, and all non-eligible direct-gradient components are unchanged.

This is a first-order local gate. Momentum, weight decay and later state feedback can make the realized multi-step parameter trajectory differ from its one-step margin prediction; final Evaluation metrics remain decisive.

## 5. Experimental comparison

Only TP-MAG is newly trained. Do not rerun or modify parent arms.

Compare the new TP-MAG final checkpoint against the already completed parent results for each identical seed:

- C2
- BASE_C3
- TARGETED
- RANDOM_MATCHED
- FREQUENCY_MATCHED
- ORACLE_SELECTION

The parent run uses the same Clean Stage-I initialization, exact per-seed sampler, augmentation trace, frozen backbone, optimizer and 30-epoch schedule. The TP-MAG loader must verify every batch against the parent's C0 trace.

Primary metrics on Evaluation:

1. Many accuracy (percentage points).
2. Mean fixed Many-vs-non-Many margin, using the competitor map selected from Calibration and frozen.

Tail-preservation metrics:

3. Few accuracy (percentage points).
4. Overall and macro accuracy.
5. Per-class Many and Few accuracy.

Also report gate distribution, retained direct-gradient norm ratio, source/target pair counts, gradient reconstruction errors, optimizer steps and frozen-backbone checks.

Do not select an epoch or gate threshold using Evaluation. No Evaluation-driven tuning is allowed.

## 6. Explicit stop/go criteria

Reuse the practical thresholds declared before the parent Oracle experiment.

Primary head-efficacy requirement versus BASE_C3:

- mean Many-accuracy gain >= +2.0 percentage points;
- mean fixed Many-margin gain >= +0.10;
- both primary effects are positive together in at least 2 of 3 seeds.

Tail floor versus BASE_C3:

- mean Few-accuracy change >= -2.0 percentage points.

Tail-preserving trade-off requirement versus TARGETED:

- TP-MAG mean Many accuracy must be at least TARGETED mean Many accuracy;
- TP-MAG mean Few accuracy must be at least TARGETED mean Few accuracy;
- at least one of these two mean metrics must be strictly better.

The experiment is a GO only if all three requirements pass. Otherwise it is a NO-GO: report the failure and stop this Gate line. Do not adjust epsilon, add a threshold, change the calibration reference, tune a coefficient, add a loss, or run another mechanism variant.

A successful GO only admits external-dataset validation as the next scientific step; it does not by itself establish a publishable method or SOTA result.

## 7. Execution gates

1. CPU: syntax-check all files and pass all unit tests.
2. Preflight: verify branch ancestry, parent completion, prepared inputs, exact counts, source hashes and Calibration-only reference construction. Do not read Evaluation.
3. Production Smoke: one real batch and one optimizer update for each of the three seeds. Verify original C3 gradient reconstruction, gate bounds, unchanged ineligible components, finite FP32 gradients, real C0 trace identity, optimizer writeback and frozen backbone.
4. Formal: train TP-MAG for all three seeds, 30 epochs each. No AMP, resume, overwrite, early stopping or best-epoch selection.
5. Only after all three seed runs finish, read the sealed Evaluation partition and calculate final comparisons.

Any failed gate aborts the run. Do not silently repair or continue with altered settings.

## 8. Leakage, data and artifact restrictions

- Calibration only constructs the fixed reference and competitor map.
- Evaluation is not read before all three TP-MAG formal runs have completed.
- Evaluation is a TRAIN-derived development partition, not an independent test set. The historical Stage-I checkpoint was selected using the original 1,980-example holdout; this split does not undo that history.
- Official CIFAR-100 TEST must never be opened, loaded, evaluated or used for selection.
- Do not regenerate the Clean Stage-I checkpoint, prepared arrays, split, sampler plans or parent baseline results.
- Do not alter any parent experiment file.
- Do not commit checkpoints, NPY/NPZ caches or local logs.
- Output directories are exclusive. No resume or overwrite.
