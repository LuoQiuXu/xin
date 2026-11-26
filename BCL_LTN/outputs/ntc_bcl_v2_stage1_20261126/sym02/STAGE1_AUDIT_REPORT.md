# NTC-BCL v2 Stage-1 estimator audit — sym02

- Estimator sealed before truth access: yes
- Posterior semantics: balanced_posterior (primary), long-tailed observed-prior proxy (sensitivity)
- Transition reference: theoretical_protocol_matrix
- Estimated balanced EM converged: False
- Balanced EM iterations / final delta: 50 / 4.045e-03

## Transition and prior estimation

| Estimator | T MAE | T Frobenius | T diagonal MAE | Prior MAE | Prior KL | Estimated noise | Realized noise |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| balanced_posterior | 0.004175 | 1.792254 | 0.117758 | 0.001888 | 0.047042 | 0.1568 | 0.2028 |
| long_tailed_posterior_sensitivity | 0.004350 | 1.634099 | 0.110316 | 0.001778 | 0.033124 | 0.1417 | 0.2028 |

## Probability accuracy

| q variant | Overall | Many | Medium | Few |
| --- | ---: | ---: | ---: | ---: |
| raw_balanced_q | 61.685% | 64.257% | 55.471% | 22.295% |
| prior_calibrated_q | 68.756% | 74.875% | 48.079% | 8.197% |
| em_posterior_gamma | 82.622% | 83.556% | 82.014% | 59.016% |
| observed_prior_long_tailed_q | 69.005% | 75.125% | 48.254% | 8.852% |
| long_tailed_posterior_gamma | 82.852% | 83.284% | 82.014% | 75.082% |

## Soft-BCL denominator diagnostics

| q variant | support floor | positive mass / anchor | negative inverse-denominator mass / anchor | denominator min / median / max |
| --- | ---: | ---: | ---: | ---: |
| raw_balanced_q | 0.00 | 17.48176 | 0.00430 | 1.2628 / 24.188 / 117.72 |
| raw_balanced_q | 0.50 | 17.48176 | 0.00314 | 2.8814 / 24.248 / 117.72 |
| prior_calibrated_q | 0.00 | 52.70698 | 0.00380 | 1.0068 / 72.343 / 177.32 |
| prior_calibrated_q | 0.50 | 52.70698 | 0.00253 | 2.8871 / 72.417 / 177.32 |
| em_posterior_gamma | 0.00 | 187.89361 | 0.15204 | 0.98738 / 214.81 / 263.87 |
| em_posterior_gamma | 0.50 | 187.89361 | 0.06705 | 2.1037 / 214.82 / 263.87 |

## Limitations

- This is a fixed-teacher estimator audit, not an end-to-end v2 training result.
- The epoch-200 v1 noisy-BCL teacher is known to have weak Few accuracy; do not interpret poor estimates as an isolated estimator failure without the q diagnostics.
- Transition rows for five-example classes are weakly identified and are shrunk toward the high-support empirical template.
- No clean labels, corruption masks, or official TEST data were used during estimation.
