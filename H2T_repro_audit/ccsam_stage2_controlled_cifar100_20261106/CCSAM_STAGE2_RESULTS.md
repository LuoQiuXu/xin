# CC-SAM Stage II transfer: CIFAR-100-LT controlled results

Primary comparison uses fixed epoch 30. Official CIFAR-100 TEST was not loaded or evaluated.

## Final-epoch metrics (%)

| Seed | Arm | Overall | Head | Official Medium | Tail | Many | Medium-n | Few |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 123 | C0_BALANCED_CE | 47.071 | 64.000 | 49.143 | 24.138 | 65.000 | 49.143 | 24.333 |
| 123 | C1_CCSAM_FEATURE_GRAD | 46.313 | 57.286 | 49.000 | 29.828 | 58.088 | 48.714 | 30.167 |
| 123 | C2_WEIGHTED_CE_NO_PERTURB | 47.020 | 62.286 | 49.714 | 25.345 | 63.235 | 49.714 | 25.500 |
| 124 | C0_BALANCED_CE | 46.768 | 63.429 | 48.857 | 24.138 | 64.412 | 48.857 | 24.333 |
| 124 | C1_CCSAM_FEATURE_GRAD | 46.364 | 57.714 | 49.000 | 29.483 | 58.529 | 48.714 | 29.833 |
| 124 | C2_WEIGHTED_CE_NO_PERTURB | 46.869 | 61.857 | 49.429 | 25.690 | 62.794 | 49.429 | 25.833 |
| 125 | C0_BALANCED_CE | 46.768 | 63.857 | 48.000 | 24.655 | 65.000 | 47.857 | 24.833 |
| 125 | C1_CCSAM_FEATURE_GRAD | 46.364 | 57.857 | 48.571 | 29.828 | 58.676 | 48.286 | 30.167 |
| 125 | C2_WEIGHTED_CE_NO_PERTURB | 46.818 | 62.714 | 48.000 | 26.207 | 63.676 | 48.000 | 26.333 |

## Paired deltas (percentage points)

| Comparison | Metric | Mean Δ | Sample SD | Positive seeds | Per-seed Δ |
|---|---|---:|---:|---:|---|
| C1_minus_C0_full_transfer | Overall | -0.522 | 0.204 | 0/3 | -0.758, -0.404, -0.404 |
| C1_minus_C0_full_transfer | Macro | -0.522 | 0.204 | 0/3 | -0.758, -0.404, -0.404 |
| C1_minus_C0_full_transfer | Head | -6.143 | 0.515 | 0/3 | -6.714, -5.714, -6.000 |
| C1_minus_C0_full_transfer | OfficialMedium | +0.190 | 0.360 | 2/3 | -0.143, +0.143, +0.571 |
| C1_minus_C0_full_transfer | Tail | +5.402 | 0.263 | 3/3 | +5.690, +5.345, +5.172 |
| C1_minus_C0_full_transfer | Many | -6.373 | 0.516 | 0/3 | -6.912, -5.882, -6.324 |
| C1_minus_C0_full_transfer | MediumN | -0.048 | 0.436 | 1/3 | -0.429, -0.143, +0.429 |
| C1_minus_C0_full_transfer | Few | +5.556 | 0.255 | 3/3 | +5.833, +5.500, +5.333 |
| C1_minus_C2_feature_perturbation | Overall | -0.556 | 0.134 | 0/3 | -0.707, -0.505, -0.455 |
| C1_minus_C2_feature_perturbation | Macro | -0.556 | 0.134 | 0/3 | -0.707, -0.505, -0.455 |
| C1_minus_C2_feature_perturbation | Head | -4.667 | 0.459 | 0/3 | -5.000, -4.143, -4.857 |
| C1_minus_C2_feature_perturbation | OfficialMedium | -0.190 | 0.675 | 1/3 | -0.714, -0.429, +0.571 |
| C1_minus_C2_feature_perturbation | Tail | +3.966 | 0.456 | 3/3 | +4.483, +3.793, +3.621 |
| C1_minus_C2_feature_perturbation | Many | -4.804 | 0.473 | 0/3 | -5.147, -4.265, -5.000 |
| C1_minus_C2_feature_perturbation | MediumN | -0.476 | 0.675 | 1/3 | -1.000, -0.714, +0.286 |
| C1_minus_C2_feature_perturbation | Few | +4.167 | 0.441 | 3/3 | +4.667, +4.000, +3.833 |
| C2_minus_C0_weighting_and_curriculum | Overall | +0.034 | 0.077 | 2/3 | -0.051, +0.101, +0.051 |
| C2_minus_C0_weighting_and_curriculum | Macro | +0.034 | 0.077 | 2/3 | -0.051, +0.101, +0.051 |
| C2_minus_C0_weighting_and_curriculum | Head | -1.476 | 0.297 | 0/3 | -1.714, -1.571, -1.143 |
| C2_minus_C0_weighting_and_curriculum | OfficialMedium | +0.381 | 0.330 | 2/3 | +0.571, +0.571, +0.000 |
| C2_minus_C0_weighting_and_curriculum | Tail | +1.437 | 0.199 | 3/3 | +1.207, +1.552, +1.552 |
| C2_minus_C0_weighting_and_curriculum | Many | -1.569 | 0.225 | 0/3 | -1.765, -1.618, -1.324 |
| C2_minus_C0_weighting_and_curriculum | MediumN | +0.429 | 0.247 | 3/3 | +0.571, +0.571, +0.143 |
| C2_minus_C0_weighting_and_curriculum | Few | +1.389 | 0.192 | 3/3 | +1.167, +1.500, +1.500 |

## Interpretation guardrails

- C1 minus C2 is the primary feature-perturbation comparison. The sampler, class weights, alpha schedule, classifier initialization, optimizer, training budget, and augmented inputs are paired.
- C2 minus C0 estimates the contribution of class weighting and the alpha blend without feature perturbation.
- C1 minus C0 measures the combined transferred Stage-II objective.
- Do not claim statistical significance from three Stage-II seeds.
- All three seeds share one Clean Stage-I backbone; they are not three independent representation seeds.
- The validation partition is drawn from official TRAIN, contains no class-0 examples, and is not an independent test set.
- This experiment transfers the feature-gradient part of CC-SAM Stage II; it is not a full reproduction of original CC-SAM.

See summary.json and paired_deltas.csv for machine-readable results.
