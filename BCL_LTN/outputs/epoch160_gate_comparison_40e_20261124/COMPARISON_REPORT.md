# BCL-LTN epoch-160 continuation comparison

The three FP32 arms restored the same epoch-160 model, SGD momentum, and RNG. They used batch 256, all 10,847 TRAIN samples per epoch, and identical sampler and first-batch augmentation hashes.

Arm A matches the archived epoch-200 model state SHA256 425cdbfe6b296ccbc7de11fca696e523bb6055c5760f2c86877de7057a502a2b.
The fixed observed-Few q90 was 1.128440960907.
Clean labels and the corruption mask were decoded for evaluation after each frozen TRAIN diagnostic pass.

## Balanced TEST after all TRAIN diagnostics

| Arm | Overall | Many | Medium | Few | Overall delta vs A (pp) | Few delta vs A (pp) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BCL_ORIGINAL | 32.22% | 50.43% | 35.66% | 6.97% | +0.00 | +0.00 |
| BCL_GLOBAL_RAW_MARGIN | 32.75% | 50.77% | 36.77% | 7.03% | +0.53 | +0.07 |
| BCL_FEW_CLASS_ADAPTIVE_Q90 | 32.37% | 50.49% | 36.06% | 6.93% | +0.15 | -0.03 |

## TRAIN checkpoints

Accuracy and memorization use original clean source-group membership. Gate rows use observed-label groups.

| Arm | Epoch | Clean accuracy Overall / Many / Medium / Few | Wrong-label memorization Overall / Many / Medium / Few | Eval Few acceptance (all / clean / noisy) |
| --- | ---: | --- | --- | --- |
| BCL_ORIGINAL | 180 | 75.03% / 77.95% / 69.24% / 27.32% | 9.27% / 9.23% / 9.54% / 9.01% | 100.00% / 100.00% / 100.00% |
| BCL_ORIGINAL | 200 | 77.57% / 80.50% / 71.46% / 30.93% | 11.33% / 11.54% / 10.28% / 10.81% | 100.00% / 100.00% / 100.00% |
| BCL_GLOBAL_RAW_MARGIN | 180 | 73.92% / 76.97% / 67.50% / 25.77% | 8.65% / 8.64% / 8.81% / 8.11% | 11.13% / 25.77% / 9.02% |
| BCL_GLOBAL_RAW_MARGIN | 200 | 76.47% / 79.38% / 70.49% / 29.90% | 10.32% / 10.32% / 10.43% / 9.91% | 13.67% / 29.90% / 11.33% |
| BCL_FEW_CLASS_ADAPTIVE_Q90 | 180 | 74.80% / 77.70% / 69.24% / 26.29% | 9.20% / 9.14% / 9.40% / 9.91% | 25.91% / 40.72% / 23.77% |
| BCL_FEW_CLASS_ADAPTIVE_Q90 | 200 | 77.27% / 80.18% / 71.46% / 29.90% | 11.19% / 11.37% / 10.43% / 9.91% | 28.71% / 49.48% / 25.71% |

## Observed-Few logit signals

Many-to-Few is the positive class for AUC. All signals came from the same required TRAIN passes at epochs 180 and 200.
The epoch-160 source CSV supplied no logits or entropy.

| Arm | Epoch | Signal | Clean Few n / mean / median / IQR | Many-to-Few n / mean / median / IQR | AUC |
| --- | ---: | --- | --- | --- | ---: |
| BCL_ORIGINAL | 180 | max_head_logit | 194 / 3.6481 / 3.5477 / 1.5670 | 1097 / 4.4047 / 4.1908 / 2.1354 | 0.6408 |
| BCL_ORIGINAL | 180 | many_probability_mass | 194 / 0.2182 / 0.2040 / 0.1691 | 1097 / 0.3462 / 0.2978 / 0.2435 | 0.7022 |
| BCL_ORIGINAL | 180 | maximum_probability | 194 / 0.2679 / 0.2021 / 0.2051 | 1097 / 0.2791 / 0.2151 / 0.2365 | 0.5171 |
| BCL_ORIGINAL | 180 | entropy | 194 / 3.0813 / 3.3529 / 0.8509 | 1097 / 3.0719 / 3.2831 / 0.9368 | 0.4846 |
| BCL_ORIGINAL | 180 | negative entropy | | | 0.5154 |
| BCL_ORIGINAL | 200 | max_head_logit | 194 / 3.6527 / 3.4637 / 1.5100 | 1097 / 4.4892 / 4.2963 / 2.2218 | 0.6515 |
| BCL_ORIGINAL | 200 | many_probability_mass | 194 / 0.1999 / 0.1846 / 0.1555 | 1097 / 0.3333 / 0.2828 / 0.2581 | 0.7080 |
| BCL_ORIGINAL | 200 | maximum_probability | 194 / 0.2813 / 0.2070 / 0.2500 | 1097 / 0.2867 / 0.2208 / 0.2486 | 0.5241 |
| BCL_ORIGINAL | 200 | entropy | 194 / 3.0083 / 3.3137 / 1.1307 | 1097 / 3.0128 / 3.2309 / 0.9743 | 0.4771 |
| BCL_ORIGINAL | 200 | negative entropy | | | 0.5229 |
| BCL_GLOBAL_RAW_MARGIN | 180 | max_head_logit | 194 / 3.6116 / 3.4811 / 1.4894 | 1097 / 4.3595 / 4.1969 / 2.0755 | 0.6423 |
| BCL_GLOBAL_RAW_MARGIN | 180 | many_probability_mass | 194 / 0.2197 / 0.2153 / 0.1658 | 1097 / 0.3444 / 0.3016 / 0.2370 | 0.6989 |
| BCL_GLOBAL_RAW_MARGIN | 180 | maximum_probability | 194 / 0.2580 / 0.1974 / 0.2085 | 1097 / 0.2732 / 0.2105 / 0.2316 | 0.5234 |
| BCL_GLOBAL_RAW_MARGIN | 180 | entropy | 194 / 3.1222 / 3.3695 / 0.9104 | 1097 / 3.0992 / 3.2977 / 0.8962 | 0.4798 |
| BCL_GLOBAL_RAW_MARGIN | 180 | negative entropy | | | 0.5202 |
| BCL_GLOBAL_RAW_MARGIN | 200 | max_head_logit | 194 / 3.6559 / 3.5609 / 1.4609 | 1097 / 4.4985 / 4.3439 / 2.2175 | 0.6543 |
| BCL_GLOBAL_RAW_MARGIN | 200 | many_probability_mass | 194 / 0.2039 / 0.1970 / 0.1726 | 1097 / 0.3386 / 0.2880 / 0.2621 | 0.7049 |
| BCL_GLOBAL_RAW_MARGIN | 200 | maximum_probability | 194 / 0.2773 / 0.2013 / 0.2148 | 1097 / 0.2880 / 0.2245 / 0.2553 | 0.5273 |
| BCL_GLOBAL_RAW_MARGIN | 200 | entropy | 194 / 3.0191 / 3.3130 / 1.0969 | 1097 / 3.0116 / 3.2143 / 0.9712 | 0.4722 |
| BCL_GLOBAL_RAW_MARGIN | 200 | negative entropy | | | 0.5278 |
| BCL_FEW_CLASS_ADAPTIVE_Q90 | 180 | max_head_logit | 194 / 3.6637 / 3.5643 / 1.5910 | 1097 / 4.4119 / 4.2361 / 2.1341 | 0.6396 |
| BCL_FEW_CLASS_ADAPTIVE_Q90 | 180 | many_probability_mass | 194 / 0.2187 / 0.2072 / 0.1643 | 1097 / 0.3474 / 0.3013 / 0.2468 | 0.7023 |
| BCL_FEW_CLASS_ADAPTIVE_Q90 | 180 | maximum_probability | 194 / 0.2655 / 0.2074 / 0.2036 | 1097 / 0.2798 / 0.2155 / 0.2378 | 0.5198 |
| BCL_FEW_CLASS_ADAPTIVE_Q90 | 180 | entropy | 194 / 3.0860 / 3.3313 / 0.9456 | 1097 / 3.0696 / 3.2784 / 0.9374 | 0.4838 |
| BCL_FEW_CLASS_ADAPTIVE_Q90 | 180 | negative entropy | | | 0.5162 |
| BCL_FEW_CLASS_ADAPTIVE_Q90 | 200 | max_head_logit | 194 / 3.6705 / 3.5115 / 1.5412 | 1097 / 4.5135 / 4.3238 / 2.1995 | 0.6518 |
| BCL_FEW_CLASS_ADAPTIVE_Q90 | 200 | many_probability_mass | 194 / 0.2010 / 0.1848 / 0.1601 | 1097 / 0.3356 / 0.2848 / 0.2604 | 0.7073 |
| BCL_FEW_CLASS_ADAPTIVE_Q90 | 200 | maximum_probability | 194 / 0.2814 / 0.2074 / 0.2481 | 1097 / 0.2890 / 0.2229 / 0.2543 | 0.5276 |
| BCL_FEW_CLASS_ADAPTIVE_Q90 | 200 | entropy | 194 / 3.0023 / 3.3110 / 1.1790 | 1097 / 3.0025 / 3.2154 / 0.9778 | 0.4757 |
| BCL_FEW_CLASS_ADAPTIVE_Q90 | 200 | negative entropy | | | 0.5243 |

## Runtime and protocol evidence

| Arm | Epochs | Actual SGD updates | Wall seconds | Peak allocated GiB | Peak reserved GiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| BCL_ORIGINAL | 40 | 1720 | 667.8 | 1.878 | 2.242 |
| BCL_GLOBAL_RAW_MARGIN | 40 | 1720 | 808.3 | 1.881 | 2.242 |
| BCL_FEW_CLASS_ADAPTIVE_Q90 | 40 | 1720 | 801.0 | 1.881 | 2.242 |

Official TEST opens until all TRAIN work completed 0. Official TEST file opens during final evaluation 2.
The environment, input and checkpoint hashes are in the JSON receipts. Per-batch relation counts are in each arm's gate_diagnostics.csv.
The reported differences describe this fixed Seed 42 synthetic symmetric-noise continuation.
