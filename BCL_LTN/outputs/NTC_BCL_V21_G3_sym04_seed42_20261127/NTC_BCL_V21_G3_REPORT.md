# NTC-BCL v2.1 G3 Report — sym04

- Epochs completed: `200` (warmup `20`, semantic ramp `20`, EM interval `5`)
- Unknown transition & clean prior: no true noise rate or symmetry used during training
- Prototype input mode: `direction` | support floor ratio: `0.5` | EMA decay: `0.99`
- Module B: PiW(raw soft support) + constrained KL + Hellinger prior calibration (lambda_h=1.0, pi_min=3.0/N) + transition blend=0.35
- No DEATG, no temporal probability smoothing, and no epoch-based transition freeze; fit statistics are diagnostics only.

## Zero-Intervention Observed-Label Fit Diagnostics

| Epoch | F_top20 | F_top50 | F_all_macro |
| ---: | ---: | ---: | ---: |
| 20 | 0.160660 | 0.149033 | 0.114203 |
| 25 | 0.205411 | 0.197674 | 0.142005 |
| 30 | 0.247593 | 0.225145 | 0.165882 |
| 35 | 0.263124 | 0.244550 | 0.177120 |
| 40 | 0.279073 | 0.269271 | 0.198524 |
| 45 | 0.295072 | 0.290323 | 0.211412 |
| 50 | 0.322194 | 0.303814 | 0.225973 |
| 55 | 0.321517 | 0.299337 | 0.221395 |
| 60 | 0.323876 | 0.312483 | 0.235979 |
| 65 | 0.335753 | 0.332814 | 0.243645 |
| 70 | 0.333981 | 0.348312 | 0.262711 |
| 75 | 0.329671 | 0.320346 | 0.238100 |
| 80 | 0.346860 | 0.349315 | 0.268272 |
| 85 | 0.361941 | 0.361253 | 0.274494 |
| 90 | 0.330854 | 0.347363 | 0.264443 |
| 95 | 0.356718 | 0.365746 | 0.277922 |
| 100 | 0.355441 | 0.370234 | 0.280600 |
| 105 | 0.352574 | 0.377532 | 0.291428 |
| 110 | 0.351899 | 0.379561 | 0.290853 |
| 115 | 0.346289 | 0.376936 | 0.293822 |
| 120 | 0.357872 | 0.384764 | 0.301991 |
| 125 | 0.361552 | 0.380963 | 0.297389 |
| 130 | 0.367083 | 0.381925 | 0.292278 |
| 135 | 0.361486 | 0.378895 | 0.299322 |
| 140 | 0.361596 | 0.385116 | 0.299132 |
| 145 | 0.368401 | 0.403942 | 0.308310 |
| 150 | 0.370075 | 0.394135 | 0.301975 |
| 155 | 0.358895 | 0.393638 | 0.307082 |
| 160 | 0.370714 | 0.405628 | 0.316310 |
| 165 | 0.381800 | 0.425674 | 0.334018 |
| 170 | 0.389562 | 0.438593 | 0.345816 |
| 175 | 0.395792 | 0.444672 | 0.355945 |
| 180 | 0.393178 | 0.448151 | 0.362356 |
| 185 | 0.406849 | 0.457497 | 0.367301 |
| 190 | 0.404068 | 0.457038 | 0.370757 |
| 195 | 0.404971 | 0.458010 | 0.370978 |

## Official TEST Metrics

| Evaluator | Overall | Many | Medium | Few |
| --- | ---: | ---: | ---: | ---: |
| student | 39.72% | 57.40% | 45.11% | 12.80% |
| ema_teacher | 39.54% | 57.34% | 44.83% | 12.60% |

## TRAIN Clean Accuracy & Memorization

| Epoch | Student Clean (O/Ma/Me/Fe) | Student Mem% | EMA Clean (O/Ma/Me/Fe) | EMA Mem% |
| ---: | --- | ---: | --- | ---: |
| 40 | 41.09 / 44.66 / 29.41 / 7.22 | 1.74% | 49.57 / 53.48 / 36.74 / 12.37 | 1.51% |
| 100 | 47.12 / 49.57 / 39.92 / 19.59 | 2.15% | 64.13 / 66.35 / 59.50 / 28.87 | 2.63% |
| 160 | 60.14 / 63.74 / 47.06 / 32.99 | 3.08% | 69.57 / 71.09 / 68.08 / 36.60 | 3.70% |
| 180 | 77.60 / 79.11 / 76.57 / 42.27 | 5.57% | 78.62 / 80.49 / 75.60 / 44.33 | 5.59% |
| 200 | 80.43 / 82.38 / 76.95 / 46.39 | 6.17% | 80.50 / 82.41 / 77.15 / 46.91 | 6.17% |

## Online Anchored EM Evolution (Offline Truth Audit)

- Realized dataset noise: `0.4037`
- Estimated noise raw / calibrated / blended (Epoch 195): `0.4104` / `0.4097` / `0.4159`
- Final blended `T` MAE: `0.006533` | `T` diag MAE: `0.0972`
- Prior MAE (raw / calibrated / blended): `0.00173` / `0.00167` / `0.00163`
- Prior KL (raw / calibrated / blended): `0.03947` / `0.03348` / `0.03244` | Min blended prior: `2.77e-04`

## Prototype & Contrastive Health (Final Epoch)

- Classifier `||W_c||` min / median / max: `0.1136` / `2.2581` / `3.0389` (small weights <0.15: `5`)
- Prototype off-diag cosine (dir input) mean / max: `0.0411` / `0.9991`
- Prototype off-diag cosine (raw input) mean / max: `0.0444` / `0.9999`
- Soft-BCL q entropy mean: `2.0268` | min denominator weight: `1.2805` | anchor positive mass mean: `3.6056`
