# NTC-BCL v2 Stage-1 estimator audit — cifar100n_real

- Estimator sealed before truth access: yes
- Posterior semantics: balanced_posterior (primary), long-tailed observed-prior proxy (sensitivity)
- Transition reference: empirical_finite_sample_matrix
- Estimated balanced EM converged: False
- Balanced EM iterations / final delta: 50 / 3.765e-03

## Transition and prior estimation

| Estimator | T MAE | T Frobenius | T diagonal MAE | Prior MAE | Prior KL | Estimated noise | Realized noise |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| balanced_posterior | 0.005565 | 2.581843 | 0.129512 | 0.001275 | 0.064788 | 0.3177 | 0.3597 |
| long_tailed_posterior_sensitivity | 0.005802 | 2.429105 | 0.126282 | 0.002055 | 0.051021 | 0.2870 | 0.3597 |

## Probability accuracy

| q variant | Overall | Many | Medium | Few |
| --- | ---: | ---: | ---: | ---: |
| raw_balanced_q | 61.685% | 64.257% | 55.471% | 22.295% |
| prior_calibrated_q | 69.005% | 75.601% | 46.449% | 5.246% |
| em_posterior_gamma | 79.414% | 83.908% | 65.774% | 26.230% |
| observed_prior_long_tailed_q | 68.111% | 73.663% | 50.000% | 9.508% |
| long_tailed_posterior_gamma | 77.551% | 80.655% | 67.695% | 43.279% |

## Soft-BCL denominator diagnostics

| q variant | support floor | positive mass / anchor | negative inverse-denominator mass / anchor | denominator min / median / max |
| --- | ---: | ---: | ---: | ---: |
| raw_balanced_q | 0.00 | 17.48176 | 0.00430 | 1.2628 / 24.188 / 117.72 |
| raw_balanced_q | 0.50 | 17.48176 | 0.00314 | 2.8814 / 24.248 / 117.72 |
| prior_calibrated_q | 0.00 | 62.12697 | 0.00776 | 1.0022 / 84.51 / 186.86 |
| prior_calibrated_q | 0.50 | 62.12697 | 0.00370 | 2.8631 / 84.592 / 186.86 |
| em_posterior_gamma | 0.00 | 174.99141 | 0.13190 | 0.99283 / 202.57 / 259.66 |
| em_posterior_gamma | 0.50 | 174.99141 | 0.05832 | 2.1809 / 202.61 / 259.66 |

## Limitations

- This is a fixed-teacher estimator audit, not an end-to-end v2 training result.
- The epoch-200 v1 noisy-BCL teacher is known to have weak Few accuracy; do not interpret poor estimates as an isolated estimator failure without the q diagnostics.
- Transition rows for five-example classes are weakly identified and are shrunk toward the high-support empirical template.
- No clean labels, corruption masks, or official TEST data were used during estimation.
