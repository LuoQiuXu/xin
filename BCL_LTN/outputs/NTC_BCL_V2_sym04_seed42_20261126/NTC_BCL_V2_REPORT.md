# NTC-BCL v2 Report — sym04

- Epochs completed: `200` (warmup `20`, semantic ramp `20`, EM interval `5`)
- Unknown transition & clean prior: no true noise rate or symmetry used during training
- Prototype input mode: `direction` | support floor ratio: `0.5` | EMA decay: `0.99`

## Official TEST Metrics

| Evaluator | Overall | Many | Medium | Few |
| --- | ---: | ---: | ---: | ---: |
| student | 38.94% | 55.46% | 43.74% | 14.07% |
| ema_teacher | 39.03% | 55.49% | 44.11% | 13.90% |

## TRAIN Clean Accuracy & Memorization

| Epoch | Student Clean (O/Ma/Me/Fe) | Student Mem% | EMA Clean (O/Ma/Me/Fe) | EMA Mem% |
| ---: | --- | ---: | --- | ---: |
| 40 | 42.32 / 45.68 / 30.67 / 13.92 | 1.19% | 49.52 / 53.05 / 37.90 / 16.49 | 1.67% |
| 100 | 53.68 / 56.33 / 46.00 / 23.20 | 2.35% | 61.61 / 64.20 / 54.58 / 29.38 | 2.85% |
| 160 | 58.67 / 61.14 / 51.78 / 28.87 | 2.72% | 68.92 / 71.36 / 63.07 / 34.54 | 4.09% |
| 180 | 76.99 / 78.75 / 74.73 / 41.75 | 6.51% | 78.49 / 80.10 / 76.66 / 44.85 | 5.80% |
| 200 | 79.28 / 81.00 / 76.76 / 46.39 | 6.69% | 79.44 / 81.15 / 76.86 / 46.91 | 6.58% |

## Online Anchored EM Evolution (Offline Truth Audit)

- Realized dataset noise: `0.4037`
- Final blended estimated noise (Epoch 195): `0.4271` (raw `0.4209`)
- Final blended `T` MAE: `0.007662` | `T` diag MAE: `0.1539` | Prior KL: `0.06326` | Min prior: `9.21e-05`

## Prototype & Contrastive Health (Final Epoch)

- Classifier `||W_c||` min / median / max: `0.1142` / `2.2564` / `2.9157` (small weights <0.15: `7`)
- Prototype off-diag cosine (dir input) mean / max: `0.0461` / `0.9991`
- Prototype off-diag cosine (raw input) mean / max: `0.0505` / `0.9998`
- Soft-BCL q entropy mean: `1.9622` | min denominator weight: `1.3790` | anchor positive mass mean: `3.6448`
