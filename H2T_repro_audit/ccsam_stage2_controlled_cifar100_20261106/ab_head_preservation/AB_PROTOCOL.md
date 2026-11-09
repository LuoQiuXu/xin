# C4 protocol — CC-SAM with Many-only head-preservation distillation

## 1. Scientific question

The sealed C0/C1/C2 Stage-II comparison on CIFAR-100-LT IF=100 showed that the original CC-SAM feature-gradient perturbation improves Few accuracy but substantially reduces Many accuracy. C3 then removed direct Many feature perturbations while leaving the original C1 objective and all classifier updates intact. C3 retained the Few benefit but did not recover Many accuracy.

C4 tests one constructive method rather than another mechanism audit:

> Can the full C1 tail-enhancement perturbation be retained while a frozen C2 classifier supplies a Many-input decision-distribution reference that limits head-class degradation?

C4 is an exploratory classifier-stage method. It is not a new backbone, a representation-learning method, or a full reproduction of the original two-stage CC-SAM paper.

## 2. Fixed method definition

### Module A — full C1 feature perturbation

Use the original, already numerically validated C1 implementation without changing its gradient calculation, normalization, class weights, alpha schedule, or loss:

    L_C1 = alpha * mean_i[w_yi * CE(W(f_i + delta_i) + b, y_i)]
           + (1-alpha) * mean_i[CE(W f_i + b, y_i)]

The original CC-SAM feature perturbation is applied to every class, including Many. Do not use C3 masking in C4.

### Module B — Many-only head-preservation distillation

For each Stage-II seed, load that same seed's completed C2 classifier checkpoint as a frozen teacher. C2 is the weighted-CE/no-feature-perturbation arm already completed in the sealed experiment.

The C4 student starts from the original Clean Stage-I classifier initialization, not from the C2 teacher checkpoint.

For each original, unperturbed 64-D GAP feature f_i, compute teacher and student logits. Distillation is applied only when the sealed TRAIN count of the sample's ground-truth class is strictly greater than 100:

    M_i = 1[count[y_i] > 100]

    p_T(i) = softmax(z_C2(f_i) / T)
    p_S(i) = softmax(z_C4(f_i) / T)

    L_HP = T^2 * mean_{i:M_i=1} KL(p_T(i) || p_S(i))

The KL compares the full 100-class output distributions. The input subset is Many-only; the target distribution is not restricted to Many output classes.

If a batch has no Many examples, L_HP is a differentiable zero for that batch.

### Unified objective

    L_C4 = L_C1 + lambda_HP * L_HP

Frozen first-run values:

- Temperature T = 2.0.
- Distillation coefficient lambda_HP = 1.0.
- Many threshold: original sealed TRAIN count strictly greater than 100.
- No hyperparameter search, warm-up schedule, adaptive coefficient, or extra module.

## 3. Teacher and student pairing

For each seed 123, 124, and 125:

- Student: new classifier initialized from the same sealed Clean Stage-I classifier checkpoint used by C0-C3.
- Teacher: corresponding seed's final epoch-30 C2 classifier checkpoint.
- Teacher weights and gradients: frozen; teacher is always in eval mode.
- Student: only trainable component.
- Backbone and BN: frozen and checked bitwise unchanged.
- Teacher checkpoint SHA, initial classifier SHA, and Clean Stage-I SHA are verified before Smoke and Formal.
- Teacher is evaluated on the same clean GAP features as the student distillation branch. It is not evaluated on perturbed features.

The three C2 teachers differ by their Stage-II seeds but share the same fixed Clean Stage-I representation. Do not describe them as independent-backbone teachers.

## 4. Frozen experimental protocol

- Repository: LuoQiuXu/xin
- C4 branch: experiment/ccsam-head-preservation-distill-20261109
- Base: completed C3 branch experiment/ccsam-c3-no-many-perturb-20261108
- Experiment root: H2T_repro_audit/ccsam_stage2_controlled_cifar100_20261106
- C4 implementation directory: ab_head_preservation
- Dataset: CIFAR-100-LT, IF=100, 10,847 official TRAIN images.
- Backbone: sealed Clean Stage-I ResNet-32, seed 42, best epoch 184.
- Clean Stage-I checkpoint SHA256: 7ced9d07f71335357c08623573f038acd0dfdcdeadb55d95d57b332a14f83a88.
- Feature: 64-D GAP using the existing clean_compatible_gap path.
- Stage II: 30 epochs, batch size 64, 170 steps per epoch.
- Seeds: 123, 124, 125.
- Optimizer: SGD, lr=0.01, momentum=0.9, weight_decay=0.0002, nesterov=False.
- Precision: FP32; no AMP, MixUp, gradient accumulation, early stopping, or scheduler changes.
- Data augmentation and sampling plans: exact existing sealed per-seed plans and continuous augmentation RNG.
- Primary epoch: fixed epoch 30.
- Validation: the existing 1,980-image holdout from official TRAIN, 20 images per class ID 1..99, class 0 absent.
- Official CIFAR-100 TEST: prohibited throughout preflight, Smoke, formal training, and analysis.

## 5. Comparisons and causal scope

C0-C3 are reused and must remain byte-identical. Only C4 is newly trained.

- C4 − C1: incremental effect of adding Many-only C2 teacher distillation to the full C1 perturbation.
- C4 − C2: whether C4 preserves a Few benefit over weighted CE without perturbation.
- C4 − C3: teacher-guided preservation versus removing direct Many perturbations.
- C4 − C0: total change relative to the balanced-CE classifier-only control.
- C1 − C2 and C3 − C2: historical reference contrasts.

C4 must be paired against the original C1 batch trace for every seed and all 5,100 batches. The sample indices/labels, augmented image hashes, and GAP feature hashes must match exactly.

## 6. Exploratory success gates

These are decision thresholds for the next research iteration, not statistical-significance tests:

1. C4 recovers at least half of the original C1-to-C2 Many accuracy gap.
2. C4 retains at least +2.0 pp Few accuracy over C2.
3. C4 Overall is no worse than C2 by more than 0.2 pp.
4. Many recovery is positive in at least two of three Stage-II seeds.
5. Few gain over C2 is positive in at least two of three Stage-II seeds.

Report all per-seed values, means, sample standard deviations, and class-wise diagnostics. Do not conceal a failed gate or select a best epoch.

## 7. Required preflight and protection

Preflight must verify:

- All original nine C0/C1/C2 runs against the independent published SHA256 index, including local-only checkpoints and batch traces.
- All archived C3 artifacts against c3_selective/ARTIFACT_SHA256.csv, including the local-only C3 checkpoints and traces.
- All three local C2 teacher checkpoints, metadata, status SHA, and shared initial classifier SHA.
- Clean Stage-I checkpoint, prepared arrays, transform source, and all three sealed sampling plans.
- Original C0-C3 results and source/protocol hashes.

If any teacher checkpoint, original trace, C3 local-only artifact, prepared array, or source input is missing or differs, stop. Do not retrain C0-C3, regenerate plans, reconstruct traces from metrics, or skip a checksum gate.

## 8. Smoke requirements

For each of the three seeds:

- Use the first two real batches from the sealed plan with the original continuous augmentation RNG.
- Require exact equality with the original C1 trace for indices/labels, augmented images, and GAP features.
- Verify C1 analytic feature gradients against the existing direct-autograd reference with unchanged tolerances.
- Verify C4 total loss equals C1 base loss plus lambda times Many-only KL.
- Verify the strict count threshold (>100), finite loss, finite student gradients, and nonzero Many-only KD on the second Smoke batch.
- Perform one disposable student classifier SGD update.
- Verify teacher has no gradients and is bitwise unchanged.
- Verify backbone and BN are bitwise unchanged.
- Verify the official TEST guard remains clean.

Smoke does not create formal checkpoints or train epochs.

## 9. Formal training budget

Only the new C4 arm:

- 3 seeds × 30 epochs = 90 new epochs.
- 170 optimizer steps per epoch.
- 15,300 optimizer steps total.
- Original C0-C3 directories are read-only.
- New outputs only under ab_head_preservation/results/seed_*/C4_CCSAM_HEAD_PRESERVE/.
- No automatic resume, overwrite, rerun, hyperparameter change, or best-epoch selection.

If a seed fails, preserve its directory and logs, stop, and report the failure. Do not delete or silently restart a partial run.

## 10. Analysis outputs

After all three C4 runs complete, analyze fixed epoch 30 and write:

- analysis/final_epoch_metrics.csv
- analysis/paired_deltas.csv
- analysis/per_class.csv
- analysis/summary.json
- analysis/AB_RESULTS.md

Report Overall, Macro, Many, MediumN, Few, and compatibility metrics Head/OfficialMedium/Tail. Include teacher Many accuracy, teacher-student Many KL, and teacher-student Many top-1 agreement.

Class 0 participates in Many distillation during training but has no validation images. Explicitly state that validation Many results do not evaluate class 0.

Three Stage-II seeds share one Clean Stage-I backbone and a development holdout. Do not claim independent-backbone generalization, statistical significance, official TEST performance, or paper-level novelty from this experiment alone.
