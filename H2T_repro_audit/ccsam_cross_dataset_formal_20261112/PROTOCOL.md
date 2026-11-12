# CC-SAM cross-dataset formal experiment protocol

Version: 1.0.0
Created: 2026-11-12
Base branch: experiment/cross-dataset-data-audit-20261111
Implementation branch: experiment/ccsam-cross-dataset-formal-20261112

## 1. Scientific objective

Test whether the already selected classifier-stage method transfers from CIFAR-100-LT to two external long-tailed benchmarks with different visual semantics and class cardinalities:

- ImageNet-LT: object recognition, 1,000 classes.
- Places-LT: scene recognition, 365 classes.

This is a controlled cross-dataset transfer experiment, not a new mechanism search. Do not add C3, TPR, a new gate, extra experts, multimodal components, or new loss terms.

## 2. Frozen experimental arms

| Cross-dataset arm | CIFAR source arm | Definition |
|---|---|---|
| D0 | C2_WEIGHTED_CE_NO_PERTURB | CC-SAM Stage-II weighted CE and alpha blend, with the feature perturbation exactly zero |
| D1 | C1_CCSAM_FEATURE_GRAD | Full class-conditional feature-gradient perturbation |
| D2 | C4_CCSAM_HEAD_PRESERVE | Full D1 perturbation plus Many-input-only distillation from the matching D0 teacher |

D0 is not ordinary unweighted CE. D0 and D1 both use the same class-aware sampler, class weights, alpha schedule, optimizer, number of steps, and initial classifier. D1-D0 therefore estimates the incremental contribution of feature perturbation.

D2 retains the complete D1 perturbation. It does not use the C3 Many-perturbation mask. Its teacher is the final D0 classifier from the same dataset and same seed. The D2 student starts from the same Stage-I classifier initialization as D0 and D1, not from the D0 teacher checkpoint.

No other arms are part of this experiment.

## 3. Data provenance and split rules

### ImageNet-LT

Local root: D:\datasets\ImageNet-LT

Expected audited inventory:

- Train: 115,846 images, 1,000 classes, 25 Parquet shards, class counts 5–1,280.
- Validation: 20,000 images, 1,000 classes, exactly 20 per class, 5 Parquet shards.
- Test: 50,000 images, 1,000 classes, exactly 50 per class, 15 Parquet shards.

Use the Parquet train split for all training and feature extraction. The Hugging Face validation split may be evaluated for diagnostics, but it must not select checkpoints, epochs, hyperparameters, or method variants. The official 50,000-image test split is sealed until all formal runs and the analysis protocol are frozen.

Read labels as the audited original 0-based class IDs. Do not reorder class IDs by frequency or remap them using a newly inferred order.

### Places-LT

Local root: D:\datasets\places-lt

Use the repository's official manifests:

- H2T\datasets\data_txt\Places_LT_train.txt
- H2T\datasets\data_txt\Places_LT_test.txt

Expected audited inventory:

- Train: 62,500 images, 365 classes, class counts 5–4,980.
- Test: 36,500 images, 365 classes, exactly 100 per class.

The extracted train directory is addressed only through the official train manifest. The extracted val directory is the official test split, not a validation split. Do not read it during training, checkpoint selection, Smoke, or feature preparation. The 7,300 unreferenced images in the extracted tree must be excluded. Do not use directory scanning as a substitute for the official TXT manifests.

Map class folders using the exact audited rule: derive the folder for each original label from the official train manifest by joining the nested class-folder components with a hyphen; map official test filenames into val/<derived-class-folder>/<filename>. Require one unique folder per label and one unique resolved path per manifest row. Do not use fallback matching.

### Split leakage rules

- No test image, test label, or test loader may be opened or instantiated during Stage-I training, feature extraction, Stage-II training, or Smoke.
- ImageNet-LT validation is diagnostic only and never used for model selection.
- Places-LT has no validation set. Do not treat its val directory as validation.
- Test evaluation is a separate final command after all 18 Stage-II runs are complete.
- The existing data audit is accepted. Do not re-download or repeat the full audit.

## 4. Common Stage-I representation protocol

Train one independent Stage-I model for each dataset and seed: 2 datasets × 3 seeds = 6 backbones.

- Seeds: 123, 124, 125.
- Backbone: torchvision ResNet-50, weights=None, trained from random initialization.
- Feature dimension: 2,048-D global-average-pooled feature.
- Classifier: one linear layer, 2,048 → 1,000 for ImageNet-LT or 2,048 → 365 for Places-LT.
- Input resolution: 224 × 224.
- Train transform: RandomResizedCrop(224), RandomHorizontalFlip, ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0), ToTensor, ImageNet mean/std normalization.
- Stage-I sampling: ordinary instance sampling with epoch shuffle; no class-aware sampler.
- Objective: CE + MixUp, MixUp alpha = 1.0.
- Epochs: 90 fixed epochs.
- Batch size: 16.
- Precision: CUDA AMP enabled when CUDA is available; no gradient accumulation.
- Optimizer: SGD, learning rate 0.01, momentum 0.9, weight decay 0.0002, Nesterov false.
- Schedule: 5-epoch linear warm-up followed by cosine decay to zero over the remaining fixed budget.
- No pretrained weights, no early stopping, no best-epoch selection. The epoch-90 checkpoint is the only Stage-I checkpoint used by downstream arms.
- ImageNet-LT validation may be logged at the end for diagnostics, but cannot replace the epoch-90 checkpoint. Places-LT has no validation evaluation during this phase.

Why this common protocol: the same architecture, input size, initialization policy, optimizer, and training budget make the cross-dataset comparison easier to interpret. In particular, not loading full ImageNet-1K pretrained weights avoids potential image overlap with the ImageNet-LT benchmark. This is a controlled method-transfer setup; do not describe it as a strict reproduction of the official H2T ImageNet-LT or Places-LT training recipes.

## 5. Feature extraction and Stage-II protocol

After each Stage-I run, freeze the entire ResNet-50 and all BN buffers. Extract one deterministic 2,048-D feature per official training image using Resize(256), CenterCrop(224), ToTensor, and ImageNet normalization. Store features in float32 memory-mapped arrays in the original manifest row order. The same feature cache is shared by D0, D1, and D2 for that dataset and seed.

Stage-II trains only a linear classifier. The frozen backbone remains in eval mode and is not included in the optimizer. No image augmentation or feature-cache regeneration is allowed during Stage II.

- Epochs: 30 fixed epochs.
- Batch size: 64 (the last batch may be smaller).
- Optimizer: SGD, LR 0.01, momentum 0.9, weight decay 0.0002, Nesterov false.
- Precision: FP32.
- MixUp: disabled.
- Gradient accumulation: disabled.
- Sampling: the pinned official CC-SAM ClassAwareSampler from H2T_repro_audit/ccsam_stage2_controlled_cifar100_20261106/official_reference/Samplers.py, SHA256 c84dfcb46c747a2ed1131df533184b2f9bf1724bc855a3d774830e795f60e174. Generate and save one 30-epoch sampling plan per dataset and seed. Each epoch has exactly N_train sampled indices. The identical plan is consumed by D0, D1, and D2 for a given dataset/seed. Do not reimplement or silently alter the official sampler.
- Initial classifier: copy the epoch-90 Stage-I classifier. Its state hash must match across D0, D1, and D2 within a dataset/seed.

### Class weights and alpha

For class c with official training count n_c:

w_c = log(max_j(n_j) / n_c + 0.01) / 2.0

For zero-based Stage-II epoch e:

alpha_e = min((e + 1) / 30, 0.3)

### D0: C2 weighted CE, no perturbation

For clean feature f_i and classifier logits z_i:

L_D0 = alpha * mean_i[w_yi * CE(z_i, y_i)] + (1-alpha) * mean_i[CE(z_i, y_i)]

The feature perturbation delta is exactly zero.

### D1: C1 full CC-SAM feature-gradient perturbation

For each class represented in a batch, compute the gradient of the mean clean CE for that class with respect to the batch feature tensor. Normalize the full zero-padded class-gradient tensor exactly as the existing CIFAR implementation does. For each sample of class c:

delta_i = 1.0 * w_c * g_i / ||g_c||_2

where g_c is the class-conditional gradient tensor and the norm includes the full batch-shaped zero-padded tensor.

Then:

L_D1 = alpha * mean_i[w_yi * CE(W(f_i + delta_i) + b, y_i)] + (1-alpha) * mean_i[CE(W f_i + b, y_i)]

Use the vectorized linear-classifier gradient only if it passes an autograd reference test on real and synthetic batches at rtol=1e-5, atol=1e-7. Stage-II remains FP32.

### D2: C4 full perturbation + Many-only head-preservation distillation

D2 uses the full D1 loss and adds a Many-input-only KL term from the matching D0 teacher.

Many is defined from official TRAIN counts strictly greater than 100. Medium is 20–100 inclusive. Few is below 20.

For each clean feature f_i:

- Teacher: matching same-dataset/same-seed D0 final epoch-30 classifier, frozen, eval mode, no gradients.
- Student: D2 classifier on clean f_i.
- Apply KL only when n_yi > 100.
- The KL output distribution spans all classes, not only Many output classes.

L_HP = T^2 * mean_{i: n_yi>100} KL(softmax(z_D0(f_i)/T) || softmax(z_D2(f_i)/T))

T = 2.0, lambda_HP = 1.0.

L_D2 = L_D1 + lambda_HP * L_HP

If a batch contains no Many inputs, L_HP is a differentiable zero. Do not distill on perturbed features. Do not alter the teacher, temperature, coefficient, Many threshold, or objective.

## 6. Pairing and execution order

For every dataset and seed:

1. Train the independent Stage-I ResNet-50 from scratch.
2. Extract and seal the deterministic train feature cache.
3. Generate and seal the official class-aware sampling plan.
4. Train D0 from the Stage-I classifier initialization.
5. Train D1 from the same Stage-I classifier initialization and same sampling plan.
6. Train D2 from the same Stage-I classifier initialization and same sampling plan, using the completed D0 checkpoint as its frozen teacher.

D0/D1/D2 must share the same Stage-I checkpoint, feature cache, class counts, initial classifier state, sampling plan, batch boundaries, and 30-epoch budget. D2 must verify the matching D0 teacher checkpoint SHA before Smoke and Formal.

Seeds are independent end-to-end: each dataset/seed pair has its own Stage-I backbone. Do not reuse one Stage-I backbone across seeds or datasets.

Total formal workload: 6 Stage-I runs + 18 Stage-II runs. Do not silently skip a seed or arm.

## 7. Evaluation and analysis

All primary results are fixed final checkpoints: Stage I epoch 90 and Stage II epoch 30. No best-epoch selection.

After all formal runs are complete, run one separate final evaluation over the official test splits. Report:

- Overall top-1 accuracy.
- Many, Medium, Few macro accuracy (mean of per-class accuracies; bins from official TRAIN counts).
- Per-class accuracy and test support.
- Optional top-5 accuracy as a secondary metric.
- Mean and sample standard deviation across the three independent end-to-end seeds.
- Paired per-seed deltas: D1-D0, D2-D0, and D2-D1.
- Per-seed results, not only aggregate means.

Expected train-count bins:

| Dataset | Many (>100) | Medium (20–100) | Few (<20) |
|---|---:|---:|---:|
| ImageNet-LT | 385 | 479 | 136 |
| Places-LT | 131 | 163 | 71 |

Do not claim statistical significance from three seeds. Do not use the test set for model selection or hyperparameter tuning. Interpret overall and group-wise changes together; a Few gain with a Many collapse is not an unqualified improvement.

## 8. Resource settings

Target hardware is a Windows laptop with an RTX 5070 Laptop GPU and 8 GB VRAM.

- Stage I: batch 16, AMP, 224-pixel input, no gradient accumulation.
- Stage II: cached 2,048-D float32 features, batch 64, FP32.
- DataLoader workers should default to 4 and be configurable down to 0 for Windows debugging.
- Do not silently lower image resolution, change the backbone, change epochs, or change the batch size after an OOM. Stop and report the exact failure and peak memory. Any protocol change must be explicit and recorded before continuing.

## 9. Required output layout

All new files and outputs must remain under:

H2T_repro_audit/ccsam_cross_dataset_formal_20261112/

Suggested layout:

- cross_dataset_core.py
- cross_dataset_data.py
- cross_dataset_models.py
- prepare_experiment.py
- train_stage1.py
- extract_features.py
- train_stage2.py
- evaluate_final.py
- analyze_results.py
- run_cross_dataset.py
- experiment_protocol.json
- preflight/
- stage1/<dataset>/seed_<seed>/
- features/<dataset>/seed_<seed>/
- stage2/<dataset>/seed_<seed>/<D0|D1|D2>/
- final_evaluation/
- analysis/

Every run must save config, seed, data manifest fingerprint, source hashes, checkpoint SHA256, epoch metrics, status JSON, and logs. Refuse to overwrite completed or partial run directories. No auto-resume and no silent restart; preserve partial artifacts for diagnosis.
