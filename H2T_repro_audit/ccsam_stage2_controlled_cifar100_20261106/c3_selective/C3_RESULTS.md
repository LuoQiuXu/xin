# C3 — Exclude Many-class feature perturbation, fixed Stage II

C3: delta=0 exactly when original TRAIN class count >100; all remaining delta and weighted/clean CE are unchanged.
Original C0/C1/C2 are reused, not rerun. Only C3 for seeds 123,124,125 was trained.
All primary values use sealed TRAIN-holdout, epoch 30; official TEST was never accessed.

## Per-seed final metrics (%)

| Seed | Arm | Overall | Many | MediumN | Few |
|---:|---|---:|---:|---:|---:|
| 123 | C0_BALANCED_CE | 47.071 | 65.000 | 49.143 | 24.333 |
| 123 | C1_CCSAM_FEATURE_GRAD | 46.313 | 58.088 | 48.714 | 30.167 |
| 123 | C2_WEIGHTED_CE_NO_PERTURB | 47.020 | 63.235 | 49.714 | 25.500 |
| 123 | C3_NO_MANY_PERTURB | 46.162 | 57.500 | 48.857 | 30.167 |
| 124 | C0_BALANCED_CE | 46.768 | 64.412 | 48.857 | 24.333 |
| 124 | C1_CCSAM_FEATURE_GRAD | 46.364 | 58.529 | 48.714 | 29.833 |
| 124 | C2_WEIGHTED_CE_NO_PERTURB | 46.869 | 62.794 | 49.429 | 25.833 |
| 124 | C3_NO_MANY_PERTURB | 46.364 | 58.235 | 48.571 | 30.333 |
| 125 | C0_BALANCED_CE | 46.768 | 65.000 | 47.857 | 24.833 |
| 125 | C1_CCSAM_FEATURE_GRAD | 46.364 | 58.676 | 48.286 | 30.167 |
| 125 | C2_WEIGHTED_CE_NO_PERTURB | 46.818 | 63.676 | 48.000 | 26.333 |
| 125 | C3_NO_MANY_PERTURB | 46.364 | 57.941 | 48.429 | 30.833 |

## Paired changes (percentage points; 3 Stage-II seeds)

| Contrast | Overall | Many | MediumN | Few |
|---|---:|---:|---:|---:|
| C3_minus_C1_remove_Many_perturbation | -0.051 | -0.539 | +0.048 | +0.389 |
| C3_minus_C2_nonMany_perturbation | -0.606 | -5.343 | -0.429 | +4.556 |
| C3_minus_C0_total_difference | -0.572 | -6.912 | +0.000 | +5.944 |
| C1_minus_C2_original_effect | -0.556 | -4.804 | -0.476 | +4.167 |

## Interpretation limits

- C3-C1 tests the direct contribution of Many feature perturbations in the presence of shared classifier updates.
- C3-C2 measures the remaining effect when ONLY Medium/Few feature perturbations are active.
- Even with no direct Many perturbation, the shared Softmax classifier can move Many decision boundaries.
- Class 0 is absent from validation, so 'Many' describes observed Many classes only.
- Three Stage-II seeds share a single fixed Stage-I representation; no statistical significance claim.
- Validation derives from official TRAIN and was previously used for development; not independent TEST.

Files: c3_summary.json, c3_final_epoch_metrics.csv, c3_paired_deltas.csv, c3_per_class.csv.
