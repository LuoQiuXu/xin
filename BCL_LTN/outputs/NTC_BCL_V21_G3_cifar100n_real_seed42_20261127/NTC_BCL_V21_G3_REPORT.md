# NTC-BCL v2.1 G3 Report — cifar100n_real

- Epochs completed: `200` (warmup `20`, semantic ramp `20`, EM interval `5`)
- Unknown transition & clean prior: no true noise rate or symmetry used during training
- Prototype input mode: `direction` | support floor ratio: `0.5` | EMA decay: `0.99`
- Module B: PiW(raw soft support) + constrained KL + Hellinger prior calibration (lambda_h=1.0, pi_min=3.0/N) + transition blend=0.35
- No DEATG, no temporal probability smoothing, and no epoch-based transition freeze; fit statistics are diagnostics only.

## Zero-Intervention Observed-Label Fit Diagnostics

| Epoch | F_top20 | F_top50 | F_all_macro |
| ---: | ---: | ---: | ---: |
| 20 | 0.313196 | 0.283884 | 0.215957 |
| 25 | 0.380455 | 0.345615 | 0.251254 |
| 30 | 0.410994 | 0.382145 | 0.285111 |
| 35 | 0.440421 | 0.431537 | 0.327622 |
| 40 | 0.481847 | 0.468214 | 0.351196 |
| 45 | 0.484875 | 0.491372 | 0.381440 |
| 50 | 0.499964 | 0.492339 | 0.381786 |
| 55 | 0.505366 | 0.492351 | 0.367920 |
| 60 | 0.493009 | 0.510867 | 0.396139 |
| 65 | 0.510913 | 0.519303 | 0.407332 |
| 70 | 0.524346 | 0.534520 | 0.425640 |
| 75 | 0.518815 | 0.521041 | 0.413593 |
| 80 | 0.505937 | 0.526257 | 0.421166 |
| 85 | 0.535662 | 0.539436 | 0.426734 |
| 90 | 0.538772 | 0.542508 | 0.424319 |
| 95 | 0.532377 | 0.547830 | 0.434962 |
| 100 | 0.554873 | 0.549514 | 0.437532 |
| 105 | 0.541910 | 0.542974 | 0.443790 |
| 110 | 0.571140 | 0.576976 | 0.456718 |
| 115 | 0.555269 | 0.544164 | 0.430458 |
| 120 | 0.561083 | 0.567442 | 0.454233 |
| 125 | 0.540831 | 0.554065 | 0.451657 |
| 130 | 0.544144 | 0.570197 | 0.462400 |
| 135 | 0.516364 | 0.516939 | 0.424853 |
| 140 | 0.548191 | 0.567128 | 0.463103 |
| 145 | 0.584803 | 0.587723 | 0.470912 |
| 150 | 0.568000 | 0.581745 | 0.472388 |
| 155 | 0.581668 | 0.585994 | 0.465249 |
| 160 | 0.567982 | 0.592612 | 0.479741 |
| 165 | 0.596249 | 0.611563 | 0.502128 |
| 170 | 0.617862 | 0.629692 | 0.521747 |
| 175 | 0.629409 | 0.647196 | 0.531640 |
| 180 | 0.635101 | 0.655103 | 0.545046 |
| 185 | 0.638678 | 0.664133 | 0.556940 |
| 190 | 0.638662 | 0.665372 | 0.558494 |
| 195 | 0.645414 | 0.670896 | 0.560089 |

## Official TEST Metrics

| Evaluator | Overall | Many | Medium | Few |
| --- | ---: | ---: | ---: | ---: |
| student | 42.50% | 55.63% | 44.69% | 24.63% |
| ema_teacher | 42.46% | 55.37% | 44.83% | 24.63% |

## TRAIN Clean Accuracy & Memorization

| Epoch | Student Clean (O/Ma/Me/Fe) | Student Mem% | EMA Clean (O/Ma/Me/Fe) | EMA Mem% |
| ---: | --- | ---: | --- | ---: |
| 40 | 43.25 / 46.16 / 30.21 / 22.56 | 10.02% | 61.04 / 63.48 / 51.22 / 36.59 | 16.07% |
| 100 | 60.66 / 62.63 / 52.30 / 43.90 | 18.25% | 71.97 / 73.78 / 65.59 / 48.17 | 20.27% |
| 160 | 65.66 / 67.02 / 61.68 / 42.68 | 19.48% | 79.12 / 81.21 / 71.65 / 52.44 | 24.19% |
| 180 | 83.28 / 84.99 / 77.03 / 62.20 | 30.29% | 83.04 / 84.58 / 76.93 / 67.07 | 29.81% |
| 200 | 85.07 / 86.61 / 78.89 / 69.51 | 32.06% | 85.01 / 86.54 / 78.69 / 70.73 | 32.06% |

## Online Anchored EM Evolution (Offline Truth Audit)

- Realized dataset noise: `0.3597`
- Estimated noise raw / calibrated / blended (Epoch 195): `0.2591` / `0.2597` / `0.2654`
- Final blended `T` MAE: `0.006227` | `T` diag MAE: `0.1665`
- Prior MAE (raw / calibrated / blended): `0.00300` / `0.00301` / `0.00298`
- Prior KL (raw / calibrated / blended): `0.11139` / `0.11311` / `0.10909` | Min blended prior: `4.26e-04`

## Prototype & Contrastive Health (Final Epoch)

- Classifier `||W_c||` min / median / max: `1.2043` / `2.4049` / `3.0713` (small weights <0.15: `0`)
- Prototype off-diag cosine (dir input) mean / max: `0.0499` / `0.8134`
- Prototype off-diag cosine (raw input) mean / max: `0.0457` / `0.8146`
- Soft-BCL q entropy mean: `1.7456` | min denominator weight: `1.2946` | anchor positive mass mean: `3.8966`
