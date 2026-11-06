# CC-SAM Stage II controlled transfer on CIFAR-100-LT

This experiment transfers only the feature-gradient perturbation part of Stage II from the official CC-SAM implementation to the already trained, sealed Clean Stage I checkpoint in this repository. It is not a full reproduction of the original two-stage CC-SAM training pipeline.

Official paper: Class-Conditional Sharpness-Aware Minimization for Deep Long-Tailed Recognition, CVPR 2023.

Official code: https://github.com/zzpustc/CC-SAM

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

The code computes this exactly in a vectorized way, then normalizes each class subset's gradient exactly as the official per-class autograd operation does. The smoke test compares the vectorized result against a direct autograd reference and fails if they differ beyond tolerance. This avoids hundreds of separate autograd calls per batch without changing the mathematical gradient.

## Frozen protocol

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

Run from the repository root in PowerShell.

1. Prepare the frozen TRAIN split and verify the Clean Stage I checkpoint:

   python H2T_repro_audit\ccsam_stage2_controlled_cifar100_20261106\prepare_ccsam.py

   Optional path overrides:

   python H2T_repro_audit\ccsam_stage2_controlled_cifar100_20261106\prepare_ccsam.py --checkpoint "D:\path\to\clean_stage1_best.pth" --data-root "D:\path\to\cifar100"

2. Run only the one-batch-per-arm smoke test:

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
