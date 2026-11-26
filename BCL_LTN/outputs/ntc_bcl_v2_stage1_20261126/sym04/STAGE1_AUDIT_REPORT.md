# NTC-BCL v2 Stage-1 estimator audit — sym04

- Estimator sealed before truth access: yes
- Posterior semantics: balanced_posterior (primary), long-tailed observed-prior proxy (sensitivity)
- Transition reference: theoretical_protocol_matrix
- Estimated balanced EM converged: False
- Balanced EM iterations / final delta: 50 / 1.614e-04

## Transition and prior estimation

| Estimator | T MAE | T Frobenius | T diagonal MAE | Prior MAE | Prior KL | Estimated noise | Realized noise |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| balanced_posterior | 0.007356 | 3.513900 | 0.347675 | 0.003615 | 0.110487 | 0.0540 | 0.4037 |
| long_tailed_posterior_sensitivity | 0.007382 | 3.537414 | 0.350421 | 0.003661 | 0.113095 | 0.0503 | 0.4037 |

## Probability accuracy

| q variant | Overall | Many | Medium | Few |
| --- | ---: | ---: | ---: | ---: |
| raw_balanced_q | 61.685% | 64.257% | 55.471% | 22.295% |
| prior_calibrated_q | 68.268% | 73.889% | 49.476% | 11.475% |
| em_posterior_gamma | 62.063% | 61.990% | 62.224% | 63.279% |
| observed_prior_long_tailed_q | 68.332% | 73.810% | 50.116% | 12.459% |
| long_tailed_posterior_gamma | 61.971% | 61.877% | 62.165% | 63.607% |

## Soft-BCL denominator diagnostics

| q variant | support floor | positive mass / anchor | negative inverse-denominator mass / anchor | denominator min / median / max |
| --- | ---: | ---: | ---: | ---: |
| raw_balanced_q | 0.00 | 17.48176 | 0.00430 | 1.2628 / 24.188 / 117.72 |
| raw_balanced_q | 0.50 | 17.48176 | 0.00314 | 2.8814 / 24.248 / 117.72 |
| prior_calibrated_q | 0.00 | 41.80195 | 0.00273 | 1.48 / 57.728 / 164.56 |
| prior_calibrated_q | 0.50 | 41.80195 | 0.00235 | 2.8526 / 57.807 / 164.56 |
| em_posterior_gamma | 0.00 | 112.69685 | 0.27289 | 1.0024 / 107.55 / 202.03 |
| em_posterior_gamma | 0.50 | 112.69685 | 0.14403 | 2.0384 / 107.6 / 202.03 |

## Limitations

- This is a fixed-teacher estimator audit, not an end-to-end v2 training result.
- The epoch-200 v1 noisy-BCL teacher is known to have weak Few accuracy; do not interpret poor estimates as an isolated estimator failure without the q diagnostics.
- Transition rows for five-example classes are weakly identified and are shrunk toward the high-support empirical template.
- No clean labels, corruption masks, or official TEST data were used during estimation.
