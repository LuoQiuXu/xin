# CC-SAM-TPR: Tail Parameter Reallocation with a Low-Rank Residual

Status: experimental protocol; all proposed mechanisms require empirical validation.
Base commit: `4ab34ad00c86bcac60d765123f153b76ff097f47` (C4 branch).
Dataset/protocol: CIFAR-100-LT IF=100, ResNet-32 Clean Stage-I backbone, 64-D GAP, Stage-II seeds 123/124/125, 30 epochs, batch 64, 170 steps/epoch. Official CIFAR-100 TEST is prohibited.

## 1. Motivation from completed C0-C4 evidence

C1 retains a consistent Few gain over C2 but substantially reduces Many accuracy. C3 shows that simply zeroing direct Many perturbations does not restore Many. C4 improves Many relative to C1 but sacrifices part of the C1 Few gain; its mean Few gain over C2 is about +1.83 percentage points, below the prior +2 pp exploratory target. Thus the next step tests whether tail enhancement can be carried by a small, explicit tail-only parameter residual rather than by updating the shared classifier.

These observations motivate the hypothesis; they do not prove shared-parameter competition is the only cause.

## 2. Model

For each seed, use the corresponding completed C2 final-epoch classifier as the frozen general classifier on the same frozen Clean Stage-I 64-D GAP representation:

`z_g(f)=W_g f+b_g`.

Add one shared low-rank residual with class-specific coefficients:

`DeltaW = M_tail (B A)`,
where `A in R^(r x 64)`, `B in R^(100 x r)`, and `M_tail[c]=1` iff the sealed TRAIN count for class c is <=100 (Medium/Few), otherwise 0 (Many). Start with rank `r=8`. Initialize A with a deterministic Kaiming/orthogonal initialization and B exactly zero. Do not add a residual bias.

The student logits are `z_t(f)=W_g f+b_g+DeltaW f`. The C2 classifier is frozen; only A and B are trainable. The backbone and BN remain frozen. This is one backbone, one fixed general classifier, and one low-rank residual—not a multi-expert model.

Important: because the residual rows for Many are exactly zero, the Many class's own linear logit row is unchanged. However, Medium/Few logits still participate in the 100-way softmax and can affect Many predictions. Therefore a Many-input full-distribution preservation term remains necessary.

## 3. Objective

Retain the original, numerically validated C1 feature perturbation calculation and C1 weighted adversarial/clean objective, but compute its current logits and feature-gradient perturbation using the effective student classifier `W_g+DeltaW`. Detach the analytic perturbation exactly as in C1; do not change its normalization, class weights, alpha schedule, or loss reduction.

`L_C1(f,y; W_eff) = alpha * mean[w_y CE(W_eff(f+delta),y)] + (1-alpha)*mean[CE(W_eff f,y)]`.

On original, unperturbed features for Many input examples only, preserve the frozen C2 full 100-class distribution:

`L_HP = T^2 mean_{i: n[y_i]>100} KL(softmax(z_g(f_i)/T) || softmax(z_t(f_i)/T))`,
with `T=2`. If no Many examples occur in a batch, return differentiable zero. Use detached teacher logits.

### Dominant-subspace soft de-overlap

Do NOT impose full orthogonality between W_g and DeltaW: W_g may span the entire 64-D feature space, making full orthogonality degenerate. Before training, compute the top-k right singular vectors V_k of W_g (k=8, deterministic FP32 SVD), shared across the seed's run. Add the scale-normalized penalty:

`L_sub = ||DeltaW V_k||_F^2 / (||DeltaW||_F^2 + eps)`.

This discourages residual energy from duplicating the dominant classifier directions without demanding impossible full-space orthogonality. Log the top-8 singular energy fraction and residual projection ratio. If SVD is numerically unstable or shapes fail, stop; do not silently skip this term.

### Confusion-aware tail-row separation

Construct a symmetric confusion graph using the frozen C2 classifier's predictions on all 10,847 prepared Stage-II TRAIN examples, using the clean unaugmented view and frozen Clean Stage-I GAP features (never validation or official TEST). Include only pairs where both classes are tail-eligible (count <=100), and select the top 5 unordered off-diagonal confusion pairs by symmetric normalized confusion; deterministic tie-break by class IDs. For each selected pair (c,c'), penalize squared cosine similarity between their residual classifier rows `DeltaW[c]` and `DeltaW[c']`, with eps-safe normalization. Use `L_div` as the mean over selected pairs. If fewer than one eligible pair exists, record that fact and set L_div to differentiable zero; do not fabricate pairs.

This is an explicit hypothesis, not a claim that all semantically similar classes should be orthogonal. Save the exact pair list, confusion scores, TRAIN labels/image hash, Clean Stage-I checkpoint hash, and C2 teacher checkpoint hash before training. The validation holdout is TRAIN-derived development data and must not be used for hyperparameter search.

Unified objective:

`L = L_C1 + 1.0*L_HP + 0.01*L_sub + 0.01*L_div`.

These are fixed first-run values; no sweep, adaptive coefficient, warmup, early stopping, or best-epoch selection. Log every component separately. If regularizer scale is unexpectedly dominant (e.g. either weighted regularizer > 10% of L_C1 at initialization or during the first 10 batches), stop and report rather than tuning after observing validation results.

## 4. Initialization and training

- C2 teacher/base: matching seed's completed C2 final_classifier checkpoint; verify checksum/status and load exact classifier state.
- Backbone: same frozen Clean Stage-I checkpoint, SHA256 `7ced9d07f71335357c08623573f038acd0dfdcdeadb55d95d57b332a14f83a88`.
- A: deterministic nonzero initialization; B: all zeros. Verify DeltaW is exactly zero at step 0 and effective logits exactly equal C2 logits.
- Train only A/B using SGD lr=0.01, momentum=0.9, weight_decay=0.0002, nesterov=False, matching C4 optimizer unless the actual sealed protocol code proves otherwise.
- Seeds 123/124/125; use each seed's sealed sampling plan and continuous augmentation RNG. Match C1 batch indices/labels, augmented-image hashes, and GAP-feature hashes for all 5,100 batches per seed.
- 30 epochs, 170 optimizer steps/epoch, fixed epoch 30 primary.
- Keep original prepared arrays, C0-C4 runs, source code and traces read-only. New outputs only under a new TPR experiment subdirectory.
- Never load, instantiate, decode, evaluate, or select using official CIFAR-100 TEST.

## 5. Required implementation tests

1. CPU unit tests for factor shapes, rank, mask semantics (>100 Many; <=100 residual eligible), zero-init identity, finite forward/backward, and effective classifier merge.
2. Verify Many residual rows are exactly zero and Many base weights/biases remain bitwise equal to C2.
3. Verify teacher and backbone have no gradients and remain bitwise unchanged after a disposable optimizer step.
4. Verify A and B both receive finite gradients after B has moved from zero; avoid a dead factorization.
5. Verify `L_HP` full 100-class KL on Many input rows and differentiable zero for no-Many batches.
6. Verify C1 analytic feature perturbation against the existing direct-autograd reference with original tolerances, now using W_eff.
7. Verify `L_sub` finite, scale normalized, and exactly zero for zero residual (with safe denominator); verify gradients finite after residual becomes nonzero.
8. Verify confusion graph is deterministic, tail-only, symmetric, top-5, and derived only from fixed TRAIN-derived validation holdout.
9. Verify paired C1 trace equality for the first two real batches in Smoke and all batches in Formal.
10. Verify no official TEST access and no modifications to protected C0-C4 files/results.

## 6. Execution gates

Preflight must verify C0-C4 integrity, all three C2 base checkpoints, Clean Stage-I checkpoint, prepared arrays, sampling plans, C1 traces, C4 artifacts, and TEST-free provenance. Missing/mismatched artifact => stop.

Smoke: run two real batches per seed, exact paired trace checks, all tests above, disposable optimizer update, and output a smoke report. No formal checkpoint may be created during Smoke.

Formal: only after all Smoke gates pass, train 3 seeds x 30 epochs. No auto-resume, overwrite, restart, or silent recovery. If any seed fails, preserve logs/partial output and stop.

## 7. Required analysis

Report fixed epoch 30 for C0/C1/C2/C3/C4/TPR:
- Overall, Macro, Many, MediumN, Few, Head, OfficialMedium, Tail.
- Per-seed and mean +/- sample SD (ddof=1), paired deltas versus C2, C1, and C4.
- Tail residual norm by class, rank/effective singular values, dominant-subspace projection ratio, selected confusion pairs and their cosine similarities.
- Many teacher-student KL and top-1 agreement.
- Component losses and gradient norms over training.
- Integrity checksums and explicit TEST-free declaration.

No statistical-significance, independent-backbone, cross-dataset, or paper-level novelty claims from these three Stage-II seeds and one TRAIN-derived holdout. Clearly state class 0 has no validation examples if this remains true.

## 8. Decision criteria

The experiment is promising only if it retains a meaningful portion of the C1 Few gain while keeping Many materially closer to C2 than C1 does, without an Overall collapse. Report exact deltas; do not select a best epoch or change thresholds after seeing results. Treat any failed metric as evidence, not as a reason for an unplanned tuning sweep.
