# NTC-BCL v2 Stage-1 estimator audit — asym04_coarse_cycle

- Estimator sealed before truth access: yes
- Posterior semantics: balanced_posterior (primary), long-tailed observed-prior proxy (sensitivity)
- Transition reference: theoretical_protocol_matrix
- Estimated balanced EM converged: False
- Balanced EM iterations / final delta: 50 / 4.594e-04

## Transition and prior estimation

| Estimator | T MAE | T Frobenius | T diagonal MAE | Prior MAE | Prior KL | Estimated noise | Realized noise |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| balanced_posterior | 0.004970 | 2.958008 | 0.071711 | 0.001026 | 0.034205 | 0.3610 | 0.4037 |
| long_tailed_posterior_sensitivity | 0.005281 | 3.076110 | 0.088094 | 0.001487 | 0.022872 | 0.3408 | 0.4037 |

## Probability accuracy

| q variant | Overall | Many | Medium | Few |
| --- | ---: | ---: | ---: | ---: |
| raw_balanced_q | 61.685% | 64.257% | 55.471% | 22.295% |
| prior_calibrated_q | 69.180% | 75.612% | 47.381% | 5.902% |
| em_posterior_gamma | 88.439% | 91.908% | 79.336% | 39.344% |
| observed_prior_long_tailed_q | 67.751% | 73.266% | 50.000% | 8.197% |
| long_tailed_posterior_gamma | 86.964% | 89.483% | 79.860% | 54.098% |

## Soft-BCL denominator diagnostics

| q variant | support floor | positive mass / anchor | negative inverse-denominator mass / anchor | denominator min / median / max |
| --- | ---: | ---: | ---: | ---: |
| raw_balanced_q | 0.00 | 17.48176 | 0.00430 | 1.2628 / 24.188 / 117.72 |
| raw_balanced_q | 0.50 | 17.48176 | 0.00314 | 2.8814 / 24.248 / 117.72 |
| prior_calibrated_q | 0.00 | 62.38842 | 0.00647 | 1.0023 / 84.697 / 186.99 |
| prior_calibrated_q | 0.50 | 62.38842 | 0.00330 | 2.8687 / 84.769 / 186.99 |
| em_posterior_gamma | 0.00 | 213.10632 | 13.79265 | 1.0004 / 244.36 / 286.3 |
| em_posterior_gamma | 0.50 | 213.10632 | 5.08134 | 2.639 / 244.37 / 286.3 |

## Limitations

- This is a fixed-teacher estimator audit, not an end-to-end v2 training result.
- The epoch-200 v1 noisy-BCL teacher is known to have weak Few accuracy; do not interpret poor estimates as an isolated estimator failure without the q diagnostics.
- Transition rows for five-example classes are weakly identified and are shrunk toward the high-support empirical template.
- No clean labels, corruption masks, or official TEST data were used during estimation.
