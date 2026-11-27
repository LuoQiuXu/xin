# NTC-BCL v2 Report — cifar100n_real

- Epochs completed: `200` (warmup `20`, semantic ramp `20`, EM interval `5`)
- Unknown transition & clean prior: no true noise rate or symmetry used during training
- Prototype input mode: `direction` | support floor ratio: `0.5` | EMA decay: `0.99`

## Official TEST Metrics

| Evaluator | Overall | Many | Medium | Few |
| --- | ---: | ---: | ---: | ---: |
| student | 42.02% | 53.40% | 43.83% | 26.63% |
| ema_teacher | 41.82% | 52.94% | 43.80% | 26.53% |

## TRAIN Clean Accuracy & Memorization

| Epoch | Student Clean (O/Ma/Me/Fe) | Student Mem% | EMA Clean (O/Ma/Me/Fe) | EMA Mem% |
| ---: | --- | ---: | --- | ---: |
| 40 | 41.22 / 43.83 / 29.52 / 22.56 | 10.46% | 62.20 / 64.64 / 52.20 / 39.02 | 16.58% |
| 100 | 61.15 / 63.49 / 51.81 / 37.20 | 17.58% | 73.55 / 75.06 / 68.52 / 51.83 | 23.14% |
| 160 | 61.92 / 62.75 / 59.73 / 46.34 | 18.89% | 77.31 / 78.78 / 73.02 / 52.44 | 25.50% |
| 180 | 82.85 / 84.13 / 79.08 / 61.59 | 31.65% | 82.39 / 83.48 / 79.28 / 63.41 | 31.62% |
| 200 | 83.71 / 84.79 / 80.45 / 66.46 | 33.50% | 83.56 / 84.61 / 80.45 / 65.85 | 33.50% |

## Online Anchored EM Evolution (Offline Truth Audit)

- Realized dataset noise: `0.3597`
- Final blended estimated noise (Epoch 195): `0.2708` (raw `0.2654`)
- Final blended `T` MAE: `0.006627` | `T` diag MAE: `0.1701` | Prior KL: `0.12818` | Min prior: `9.30e-05`

## Prototype & Contrastive Health (Final Epoch)

- Classifier `||W_c||` min / median / max: `0.2018` / `2.4085` / `3.0704` (small weights <0.15: `0`)
- Prototype off-diag cosine (dir input) mean / max: `0.0436` / `0.8224`
- Prototype off-diag cosine (raw input) mean / max: `0.0393` / `0.8199`
- Soft-BCL q entropy mean: `1.7071` | min denominator weight: `1.1920` | anchor positive mass mean: `3.8210`
