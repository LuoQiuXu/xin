# C4 — CC-SAM with Many-only head-preservation distillation

Primary results are fixed Stage-II epoch 30 on the sealed TRAIN-derived holdout. Official CIFAR-100 TEST was never accessed.

## Per-seed final metrics (%)

| Seed | Arm | Overall | Many | MediumN | Few |
|---:|---|---:|---:|---:|---:|
| 123 | C0_BALANCED_CE | 47.071 | 65.000 | 49.143 | 24.333 |
| 123 | C1_CCSAM_FEATURE_GRAD | 46.313 | 58.088 | 48.714 | 30.167 |
| 123 | C2_WEIGHTED_CE_NO_PERTURB | 47.020 | 63.235 | 49.714 | 25.500 |
| 123 | C3_NO_MANY_PERTURB | 46.162 | 57.500 | 48.857 | 30.167 |
| 123 | C4_CCSAM_HEAD_PRESERVE | 46.768 | 61.471 | 49.000 | 27.500 |
| 124 | C0_BALANCED_CE | 46.768 | 64.412 | 48.857 | 24.333 |
| 124 | C1_CCSAM_FEATURE_GRAD | 46.364 | 58.529 | 48.714 | 29.833 |
| 124 | C2_WEIGHTED_CE_NO_PERTURB | 46.869 | 62.794 | 49.429 | 25.833 |
| 124 | C3_NO_MANY_PERTURB | 46.364 | 58.235 | 48.571 | 30.333 |
| 124 | C4_CCSAM_HEAD_PRESERVE | 47.020 | 61.765 | 48.714 | 28.333 |
| 125 | C0_BALANCED_CE | 46.768 | 65.000 | 47.857 | 24.833 |
| 125 | C1_CCSAM_FEATURE_GRAD | 46.364 | 58.676 | 48.286 | 30.167 |
| 125 | C2_WEIGHTED_CE_NO_PERTURB | 46.818 | 63.676 | 48.000 | 26.333 |
| 125 | C3_NO_MANY_PERTURB | 46.364 | 57.941 | 48.429 | 30.833 |
| 125 | C4_CCSAM_HEAD_PRESERVE | 46.768 | 61.471 | 49.143 | 27.333 |

## Mean ± sample SD (%)

| Arm | Overall | Many | MediumN | Few |
|---|---:|---:|---:|---:|
| C0_BALANCED_CE | 46.869 ± 0.175 | 64.804 ± 0.340 | 48.619 ± 0.675 | 24.500 ± 0.289 |
| C1_CCSAM_FEATURE_GRAD | 46.347 ± 0.029 | 58.431 ± 0.306 | 48.571 ± 0.247 | 30.056 ± 0.192 |
| C2_WEIGHTED_CE_NO_PERTURB | 46.902 ± 0.105 | 63.235 ± 0.441 | 49.048 ± 0.918 | 25.889 ± 0.419 |
| C3_NO_MANY_PERTURB | 46.296 ± 0.117 | 57.892 ± 0.370 | 48.619 ± 0.218 | 30.444 ± 0.347 |
| C4_CCSAM_HEAD_PRESERVE | 46.852 ± 0.146 | 61.569 ± 0.170 | 48.952 ± 0.218 | 27.722 ± 0.536 |

## C4 paired deltas (percentage points)

| Comparison | Overall | Many | MediumN | Few |
|---|---:|---:|---:|---:|
| C4_minus_C1 | +0.505 | +3.137 | +0.381 | -2.333 |
| C4_minus_C2 | -0.051 | -1.667 | -0.095 | +1.833 |
| C4_minus_C3 | +0.556 | +3.676 | +0.333 | -2.722 |
| C4_minus_C0 | -0.017 | -3.235 | +0.333 | +3.222 |

## Predeclared exploratory gates

- PASS: many_recovers_at_least_half_of_C1_to_C2_gap
- NOT MET: few_gain_vs_C2_at_least_2pp
- PASS: overall_within_0_2pp_of_C2
- PASS: many_recovery_positive_in_at_least_2_seeds
- PASS: few_gain_positive_in_at_least_2_seeds

- Mean Many recovery relative to the C1–C2 gap: 0.6530612244897954
- Few gain retained relative to the C1–C2 gain: 0.43999999999999984
- Overall delta versus C2: -0.051 pp

## Interpretation guardrails

- C4−C1 measures the added effect of Many-only C2 teacher distillation while retaining the full C1 perturbation.
- C4−C2 measures whether C4 preserves a Few benefit over the no-perturbation weighted-CE teacher.
- C4−C3 compares teacher-guided preservation against direct Many-perturbation removal.
- KL is applied only to Many inputs, but it compares the full 100-class output distribution.
- Class 0 is excluded from validation; Many validation metrics cover only observed Many classes.
- Three Stage-II seeds share one frozen Stage-I representation; no significance claim is justified.
- No official TEST data, labels, or test loader are used.

Machine-readable details: summary.json, paired_deltas.csv, per_class.csv, final_epoch_metrics.csv.
