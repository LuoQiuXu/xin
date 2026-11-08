# CC-SAM Stage II controlled transfer on CIFAR-100-LT

This experiment transfers only the feature-gradient perturbation part of Stage II from the official CC-SAM implementation to the already trained, sealed Clean Stage I checkpoint in this repository. It is not a full reproduction of the original two-stage CC-SAM training pipeline.

Official paper: Class-Conditional Sharpness-Aware Minimization for Deep Long-Tailed Recognition, CVPR 2023.

Official code: https://github.com/zzpustc/CC-SAM

## Current validation and execution status (2026-11-07)

Gradient numerical alignment is complete. All 510 real epoch-0 batches across
seeds 123/124/125 passed the original analytic/autograd gate; the admitted
candidate's final deltas were byte-identical to the oracle. All 10 deterministic
duplicate-label stress cases passed the unchanged tolerances.

The single post-fix production Smoke passed C0/C1/C2, paired inputs and
initialization, disposable classifier updates, and bitwise-frozen backbone/BN.
Its original report is retained in
[validation_20261106_05/smoke_report.json](validation_20261106_05/smoke_report.json)
and has been copied byte-for-byte to [smoke_report.json](smoke_report.json), the
path read by the formal entry point. Smoke was not rerun during this copy.

Preparation also passed: the sealed 40.050505% TRAIN-holdout accuracy was
reproduced, and all 976,230 sampling positions across three seeds and 30 epochs
matched the official continuous sampler. The earlier C1 Smoke failure is
historical evidence preserved in validation_20261106_03; it is superseded by the
[gradient fix report](validation_20261106_05/C1_GRADIENT_NUMERICS_FIX_REPORT.md).

Formal Stage II was authorized and completed on 2026-11-07: all nine runs
(seeds 123/124/125 x C0/C1/C2) finished 30 epochs, with 45,900 actual optimizer
steps. Full batch traces and initialization matched across each seed's arms;
the formal TEST guard recorded zero access attempts. No Smoke was rerun.

See the [fixed-epoch results](CCSAM_STAGE2_RESULTS.md),
[structured summary](summary.json), and
[actual execution report](formal_launch_20261106_01/FORMAL_EXECUTION_REPORT.md).
The [prelaunch checklist](FINAL_PRELAUNCH_CHECK.md) preserves the readiness state
before this authorized execution.

## Why the perturbation is applied to 64-D GAP features

The official CC-SAM CIFAR ResNet32 implementation pools its final convolutional map before returning the feature variable used by the Stage-II feature-gradient operation. Its feature-gradient variable is therefore a 64-dimensional vector, not the full 64 x 8 x 8 convolutional map.

The Clean Stage I backbone in this repository returns a 64 x 8 x 8 map. The shared clean_compatible_gap helper uses F.avg_pool2d(feature_map, feature_map.size(3)) and the same view(batch, -1) as the sealed Clean classifier, preserving its floating-point reduction path and dtype. Preparation, Smoke and the formal feature-extraction path all use this helper; run_stage.py delegates to these entry points. Adaptive average pooling is mathematically equivalent here but can differ numerically in FP32, so it is not used for this Clean-compatible path.

The original classifier-path equivalence check remains active with unchanged rtol=1e-5 and atol=1e-6. The perturbation remains in the 64-D embedding space; the original Clean model/checkpoint and the CC-SAM perturbation objective are unchanged.

## Controlled arms

C0_BALANCED_CE:
- Frozen Clean Stage I backbone.
- Class-aware balanced sampling.
- Ordinary CE.
- Same Clean Stage I classifier initialization as C1 and C2.
- This is a matched cRT-style classifier-only control, not a claim of strict canonical cRT with a randomly reinitialized head.

C1_CCSAM_FEATURE_GRAD:
- Same backbone, classifier initialization, sample order, augmented images, optimizer, and training budget.
- Official CC-SAM Stage-II class weights and alpha schedule.
- Adds class-conditional gradient-ascent perturbations to pooled 64-D features.

C2_WEIGHTED_CE_NO_PERTURB:
- Same as C1, including class-aware sampling, class weights, alpha schedule, optimizer, and initialization.
- Sets the feature perturbation delta exactly to zero.
- C1 minus C2 is the primary estimate of the incremental contribution of feature perturbation.

C2 minus C0 estimates the contribution of class weighting and the alpha blend without perturbation. C1 minus C0 measures the combined Stage-II transfer effect.

## Faithful and efficient gradient calculation

The official implementation obtains a class-specific feature gradient with autograd for each class in the batch. Since the transferred head is a single linear layer, the derivative has a closed form:

d CE / d f = (softmax(logits) - one_hot(label)) @ W

The implementation uses log_softmax(logits).exp() to align with CrossEntropy's numerical probability path, applies class-mean scaling in logit space before the linear product, and normalizes the class gradient in the full zero-padded batch shape. The smoke test retains the direct autograd reference and unchanged rtol=1e-5, atol=1e-7 gate. The mathematical gradient and perturbation objective are unchanged; numerical equality is documented by the real-batch and duplicate-label checks above.

## Frozen protocol

Sampling validation compares every index against the pinned, unmodified official
CC-SAM ClassAwareSampler (official_reference/Samplers.py). The original local
producer, private RNG and continuous class/data cycles are unchanged. Epoch
boundaries can straddle class cycles, so each epoch's counts are recorded rather
than required to differ by at most one. The full 30-epoch sequence must match for
each of seeds 123, 124 and 125.

- CIFAR-100-LT, IF=100; 10,847 official TRAIN images.
- Clean Stage I checkpoint SHA256: 7ced9d07f71335357c08623573f038acd0dfdcdeadb55d95d57b332a14f83a88.
- Clean Stage I seed 42, best epoch 184.
- The same classifier initialization is loaded independently for every arm and seed.
- Stage II: 30 epochs, batch size 64, SGD, LR 0.01, momentum 0.9, weight decay 0.0002.
- Official CC-SAM CIFAR Stage-II values: class-weight divisor 2.0, adver_reg 1.0, alpha ramp over 30 epochs capped at 0.3.
- Stage-II seeds: 123, 124, 125.
- FP32; no AMP, MixUp, gradient accumulation, early stopping, or hyperparameter search.
- The backbone is kept in eval mode. Every backbone parameter and BN buffer is checked bitwise unchanged.
- Primary result is fixed epoch 30, not the best validation epoch.

The 1980-image validation partition is held out from official CIFAR-100 TRAIN, contains 20 images per class ID 1..99, and has no class-0 examples. It has been reused during prior development and is not an independent test set. All Stage-II seeds share one Clean Stage I backbone; they measure Stage-II stochastic variability conditional on that representation, not independent-backbone variability.

## Run sequence

Run from the repository root in PowerShell with
D:\app\minicondaaa\envs\xiaoju\python.exe.

For this completed checkout, steps 1 through 4 below are historical command
references. Preparation, Smoke, all nine formal runs, and fixed-epoch analysis
are complete. Do not rerun them over the existing artifacts. The original
passing Smoke report remains at the native formal-entry path.

1. Prepare the frozen TRAIN split and verify the Clean Stage I checkpoint:

   python H2T_repro_audit\ccsam_stage2_controlled_cifar100_20261106\prepare_ccsam.py

   Optional path overrides:

   python H2T_repro_audit\ccsam_stage2_controlled_cifar100_20261106\prepare_ccsam.py --checkpoint "D:\path\to\clean_stage1_best.pth" --data-root "D:\path\to\cifar100"

2. Run only the one-batch-per-arm smoke test:

   If a previous preparation stopped with partial arrays, the explicitly approved
   completion command is prepare_ccsam.py --complete-existing-preparation.
   It recomputes and verifies sealed data and features exactly, preserves every
   existing array byte, and exclusively creates missing plans and manifests.
   Completed or unexpected preparation artifacts are refused; nothing is deleted.

   python H2T_repro_audit\ccsam_stage2_controlled_cifar100_20261106\train_ccsam.py --smoke-only

   This performs disposable updates only. It verifies paired images/features, exact analytic-gradient equivalence, finite classifier gradients, classifier updates, and bitwise-frozen backbone/BN. It does not start formal training.

3. Formal training is a separate explicit command and should be run only after reviewing the code and smoke report:

   python H2T_repro_audit\ccsam_stage2_controlled_cifar100_20261106\train_ccsam.py --formal

4. Analyze completed runs:

   python H2T_repro_audit\ccsam_stage2_controlled_cifar100_20261106\analyze_ccsam.py

## Outputs

Preparation:
- prepared/prepared_manifest.json
- prepared/test_guard_prepare.json
- prepared/sampling_plan_seed123.npy, sampling_plan_seed124.npy, sampling_plan_seed125.npy
- prepared/train_images.npy, train_labels.npy, validation_labels.npy, validation_gap_features.npy

Smoke:
- smoke_report.json

Formal:
- results/seed_*/C0_BALANCED_CE/
- results/seed_*/C1_CCSAM_FEATURE_GRAD/
- results/seed_*/C2_WEIGHTED_CE_NO_PERTURB/
- Per-arm batch_trace.csv, metrics.csv, status.json, final_classifier.pth
- formal_completed.json and formal_test_guard.json

Analysis:
- final_epoch_metrics.csv
- paired_deltas.csv
- summary.json
- CCSAM_STAGE2_RESULTS.md

The scripts refuse to overwrite preparation, smoke, or existing formal run directories. They do not download data, retrain Stage I, instantiate CIFAR100(train=False), or access the official TEST pickle.
