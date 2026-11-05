# Structured Feature Perturbation for CIFAR-100-LT

## Research question

Can a frozen long-tailed representation be improved by generating tail-class features from real augmentation-induced variation directions, and does projecting those directions toward the nearest competing decision boundary add value beyond random perturbations?

This is an exploratory mechanism test, not a claim of novelty or a claim of improved official-test accuracy.

## Locked two-stage setup

- Dataset: CIFAR-100-LT, exponential imbalance factor 100, 10,847 training images, 100 classes, class counts 500 down to 5.
- Stage I: the existing Clean ResNet-32 checkpoint, seed 42, epoch 184.
- Stage-I checkpoint SHA256: 7ced9d07f71335357c08623573f038acd0dfdcdeadb55d95d57b332a14f83a88.
- Stage II: frozen backbone and frozen BN; train only the 64-to-100 linear classifier.
- Stage-II seeds: 42, 43, 44. These are three Stage-II stochastic runs sharing one Stage-I checkpoint, not three independent end-to-end seeds.
- Stage-II schedule: 10 epochs, batch size 128, 85 updates per epoch, SGD, momentum 0.9, weight decay 5e-4, original cosine schedule with Stage-II lr factor 0.2, plain CE.
- Each arm starts from the exact same Stage-I classifier weights and a fresh optimizer.
- All five arms use the same receiver/donor indices and the same frozen augmented feature stream within each seed.

The implementation reads the existing Clean Stage-I experiment as a read-only dependency. It does not edit H2T, PI-H2T, the old Clean Stage-I directory, or historical result files.

## Five controlled arms

1. crt: cRT-style classifier-only baseline. Use balanced receiver features and the original Stage-I classifier initialization; no fusion or generated perturbation.
2. h2t_pooled: replace 16 of 64 pooled channels with the instance-distribution donor feature, keeping the receiver label. The same 16-channel mask is shared across each batch, matching the original H2T channel-mask semantics.
3. gaussian: add an isotropic Gaussian direction to tail features.
4. subspace_random: sample a random direction from the class-conditional augmentation-induced feature subspace.
5. subspace_boundary: project the frozen Stage-I nearest-competitor boundary normal into that class subspace and move the feature toward the boundary without crossing it.

### Why pooled H2T is a valid controlled comparator

The original H2T operation replaces whole feature-map channels using a channel mask that is constant over spatial positions. Global average pooling is linear, so:

GAP((1-M) F_r + M F_d) = (1-M) GAP(F_r) + M GAP(F_d).

The unit test verifies this identity. The h2t_pooled arm is therefore exactly equivalent to channel replacement followed by GAP for this operation. It is a frozen-BN, classifier-only controlled comparator, not a claim of a full official H2T reproduction.

## Structured augmentation subspace

For each training image, the instance-distribution donor stream supplies one independently augmented view in each of the first two epochs. Let the two frozen pooled features be z_i^a and z_i^b. The paired variation is:

d_i = z_i^a - z_i^b.

For class c, compute the covariance of centered d_i vectors, then shrink it toward an equal-class-weighted global within-class covariance:

C_c^shrunk = n_c/(n_c+tau) C_c + tau/(n_c+tau) C_global,

with fixed tau=20. The top eight eigenvectors define U_c. This shrinkage is intended to reduce unstable estimates for small classes. All estimates use training data only.

A structured random direction is sampled in span(U_c), with coefficients weighted by the square roots of the retained eigenvalues, then normalized to unit length.

## Boundary-directed perturbation and norm matching

For a tail feature z with label y, the frozen Stage-I classifier selects the strongest non-target competitor j. Define:

g = w_y - w_j and m = s_y(z) - s_j(z).

Project the pairwise boundary normal into the class augmentation subspace:

p = U_y U_y^T g.

Eligible samples must belong to the official H2T Tail IDs 71–99, have a positive frozen Stage-I margin, and have projection alignment ||p||/||g|| >= 0.05. Class 70 has 19 training examples but belongs to the official Medium index group, so it is not perturbed. The frequency-based Few metric (count < 20) includes class 70; both Few and official Tail metrics are reported. Their perturbation is:

delta_E = -r p/||p||,

where:

r = min(0.5 * m/||p||, 0.25 * ||z||).

Thus the construction moves at most halfway to the selected pairwise boundary and is capped at 25% of the feature norm. It must not cross that boundary. Ineligible samples receive zero perturbation.

For every sample, gaussian and subspace_random receive the exact same per-example radius r as subspace_boundary. This isolates perturbation direction from perturbation magnitude. All generated perturbations are applied only to official Tail receivers (class IDs 71–99); official Head (0–35) and Medium (36–70) receivers remain unchanged.

## Data and leakage safeguards

- Training features are extracted from the locked 10,847-image LT training split using the original CIFAR train transform.
- Donor indices are a permutation of all LT training examples each epoch, preserving the instance-distribution donor stream.
- Receiver classes are sampled uniformly, then a receiver example is sampled within that class.
- All arms share the same per-seed receiver/donor schedules, augmented features, labels, and H2T masks.
- The existing 1,980-image holdout comes from CIFAR-100's official TRAIN partition. It lacks class 0 and has been used in earlier development. It is used only for a fixed epoch-10 diagnostic; no best-checkpoint selection is performed.
- The official CIFAR-100 TEST file and CIFAR100(train=False) are blocked by a runtime guard. No official-test metric is produced in this round.
- Stage-II seeds share one Stage-I checkpoint. Report their mean, standard deviation, and paired per-seed deltas descriptively; do not claim independent Stage-I replication or statistical significance.

## Primary and diagnostic metrics

Primary: Overall, Many, Medium, Few, Head, OfficialMedium, Tail, Macro at the fixed tenth Stage-II epoch.

Diagnostics:
- Tail-to-Head error rate (true classes 71–99 predicted as 0–35) and Head-to-Tail error rate (true classes 0–35 predicted as 71–99). Frequency-based Few metrics also include class 70.
- Mean true-vs-nearest-competitor margin, Few-class mean margin, and Few negative-margin fraction.
- Boundary eligibility and clipping rates.
- Per-example perturbation norm matching error.
- Subspace projection residual for subspace_random.
- Confusion matrix and per-class validation accuracy.

## Decision rules

- subspace_boundary should beat both gaussian and subspace_random on Few/Tail while avoiding a material Many/Head regression before it is worth a more expensive follow-up.
- If subspace_random matches or beats subspace_boundary, the boundary direction is not adding demonstrated value.
- If Gaussian matches the structured arms, the learned subspace has not shown incremental value.
- If all three generated arms fail against cRT and H2T-pooled, stop this candidate rather than adding more gates or modules.
- A positive result is only a reason to run an independent-dataset and untouched-test confirmation after the method and hyperparameters are frozen.

## Commands

Run from the repository root in the existing PyTorch environment:

~~~powershell
$env:PYTHONIOENCODING='utf-8'
$env:PYTHONDONTWRITEBYTECODE='1'
& 'D:\app\anaconda\envs\pytorch\python.exe' -B -u -m unittest discover -s H2T_repro_audit/structured_feature_perturbation_20261105 -p "test_*.py"
& 'D:\app\anaconda\envs\pytorch\python.exe' -B -u H2T_repro_audit/structured_feature_perturbation_20261105/run_experiment.py prepare
& 'D:\app\anaconda\envs\pytorch\python.exe' -B -u H2T_repro_audit/structured_feature_perturbation_20261105/run_experiment.py smoke
& 'D:\app\anaconda\envs\pytorch\python.exe' -B -u H2T_repro_audit/structured_feature_perturbation_20261105/run_experiment.py train
& 'D:\app\anaconda\envs\pytorch\python.exe' -B -u H2T_repro_audit/structured_feature_perturbation_20261105/run_experiment.py report
~~~

The script refuses to overwrite an existing preparation, smoke run, or formal run. For a fresh rerun, create a new experiment branch/directory rather than deleting or overwriting historical results.

Generated feature caches and per-run checkpoints/logs are written under ignored feature_cache/ and runs/ folders. The small tracked deliverables are prepare_manifest.json, smoke_status.json, formal_complete.json, summary.csv, summary.json, RESULTS.md, and test_guard.json.
