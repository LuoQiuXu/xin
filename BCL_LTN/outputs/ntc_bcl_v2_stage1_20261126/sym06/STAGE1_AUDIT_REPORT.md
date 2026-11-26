# NTC-BCL v2 Stage-1 estimator audit — sym06

- Estimator sealed before truth access: yes
- Posterior semantics: balanced_posterior (primary), long-tailed observed-prior proxy (sensitivity)
- Transition reference: theoretical_protocol_matrix
- Estimated balanced EM converged: False
- Balanced EM iterations / final delta: 50 / 2.848e-03

## Transition and prior estimation

| Estimator | T MAE | T Frobenius | T diagonal MAE | Prior MAE | Prior KL | Estimated noise | Realized noise |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| balanced_posterior | 0.006888 | 1.824252 | 0.130679 | 0.004217 | 0.181263 | 0.5369 | 0.6096 |
| long_tailed_posterior_sensitivity | 0.007873 | 1.897461 | 0.127203 | 0.004793 | 0.179913 | 0.5093 | 0.6096 |

## Probability accuracy

| q variant | Overall | Many | Medium | Few |
| --- | ---: | ---: | ---: | ---: |
| raw_balanced_q | 61.685% | 64.257% | 55.471% | 22.295% |
| prior_calibrated_q | 66.774% | 72.133% | 48.545% | 14.426% |
| em_posterior_gamma | 62.506% | 64.257% | 58.789% | 32.787% |
| observed_prior_long_tailed_q | 67.318% | 71.951% | 52.270% | 18.033% |
| long_tailed_posterior_gamma | 60.662% | 61.525% | 59.488% | 42.295% |

## Soft-BCL denominator diagnostics

| q variant | support floor | positive mass / anchor | negative inverse-denominator mass / anchor | denominator min / median / max |
| --- | ---: | ---: | ---: | ---: |
| raw_balanced_q | 0.00 | 17.48176 | 0.00430 | 1.2628 / 24.188 / 117.72 |
| raw_balanced_q | 0.50 | 17.48176 | 0.00314 | 2.8814 / 24.248 / 117.72 |
| prior_calibrated_q | 0.00 | 39.15437 | 0.00670 | 1.0043 / 53.995 / 160.63 |
| prior_calibrated_q | 0.50 | 39.15437 | 0.00370 | 2.8821 / 54.06 / 160.63 |
| em_posterior_gamma | 0.00 | 83.04713 | 0.03395 | 1.0024 / 70.565 / 189.63 |
| em_posterior_gamma | 0.50 | 83.04713 | 0.01776 | 2.5406 / 70.643 / 189.63 |

## Limitations

- This is a fixed-teacher estimator audit, not an end-to-end v2 training result.
- The epoch-200 v1 noisy-BCL teacher is known to have weak Few accuracy; do not interpret poor estimates as an isolated estimator failure without the q diagnostics.
- Transition rows for five-example classes are weakly identified and are shrunk toward the high-support empirical template.
- No clean labels, corruption masks, or official TEST data were used during estimation.
