# Epoch-160 source-to-observed-group attribution

- Samples: 10847
- Observed-Few raw-margin pooled AUC: 0.676740
- Weighted source-wise AUC reconstruction: 0.676740
- Observed-Few noisy samples: 1342
- Many-to-Few noisy samples: 1097 (81.74% of Few noise)
- Class-adaptive Few q90 threshold (score-only): 1.1284409609
- Reference Few samples above q90: 154

## Observed-Few source breakdown

| Source group | Total in observed Few | Clean | Noisy | Share of Few noise |
|---|---:|---:|---:|---:|
| Many | 1097 | 0 | 1097 | 81.74% |
| Medium | 213 | 0 | 213 | 15.87% |
| Few | 226 | 194 | 32 | 2.38% |

## Pairwise separation: Clean Few vs noisy source-to-Few

| Noisy source | n negatives | Raw-margin AUC | Raw-margin AP | LC+prior AUC | LC+prior AP |
|---|---:|---:|---:|---:|---:|
| Many->Few | 1097 | 0.6802 | 0.2944 | 0.6949 | 0.3217 |
| Medium->Few | 213 | 0.6623 | 0.6659 | 0.6211 | 0.6415 |
| Few->Few | 32 | 0.6551 | 0.9275 | 0.6524 | 0.9235 |

## Pooled AUC decomposition check

| Score | Pooled Few AUC | Weighted source AUC | Absolute difference |
|---|---:|---:|---:|
| raw_margin | 0.676740 | 0.676740 | 0.00e+00 |
| lc_plus_log_prior_margin | 0.682152 | 0.682152 | 0.00e+00 |
| inverse_prior_margin | 0.645475 | 0.645475 | 0.00e+00 |
| class_normalized_raw_margin | 0.682990 | 0.682990 | 0.00e+00 |

## Logit / entropy availability

NOT AVAILABLE: the audited sample_scores.csv writer stores position/index, observed/clean labels, clean/noisy status, observed-label group and four margin scores only. It contains no logits, predicted class, softmax probabilities or entropy. No model loading or re-inference was performed.

## Interpretation guardrails

- Source group is derived from clean_label only for this offline evaluation.
- Observed group is derived from observed_label and the fixed clean IF=100 class-count vector.
- The gate reference uses observed_label and margin scores only; clean_label/is_clean are not used.
- Pairwise AUC compares Clean Few positives against one source-group's noisy Few negatives.
- The weighted pairwise AUC must reconstruct the pooled Few AUC; otherwise the script stops.
- No model is loaded, no GPU is initialized, and no inference or training is run.
- A scalar margin cannot recover the predicted class, maximum head-class logit, or prediction entropy.

## Outputs

- observed_group_composition.csv
- noisy_source_to_observed_counts.csv
- noisy_source_to_observed_row_percent.csv
- few_source_breakdown.csv
- few_margin_distribution_summary.csv
- few_pairwise_source_metrics.csv
- few_auc_source_decomposition.csv
- gate_reference.json (score-only; usable by the subsequent training gate)
