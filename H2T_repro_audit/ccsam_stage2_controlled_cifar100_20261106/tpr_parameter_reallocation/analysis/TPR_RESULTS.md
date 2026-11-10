# CC-SAM-TPR Results

- Analyzed: 2026-11-10T21:08:29+08:00
- Primary epoch: 30
- Seeds: 123, 124, 125
- Base: frozen matching-seed C2 classifier
- Rank: 8; MORE peak: 200.0
- Official CIFAR-100 TEST used: no

## Paired results

| Comparison | Overall | Many | MediumN | Few |
|---|---:|---:|---:|---:|
| TPR_minus_C2 | -0.034 ± 0.029 pp (seeds -0.051, +0.000, -0.051) | -0.098 ± 0.085 pp (seeds -0.147, +0.000, -0.147) | -0.190 ± 0.218 pp (seeds -0.143, -0.429, +0.000) | +0.222 ± 0.255 pp (seeds +0.167, +0.500, +0.000) |
| TPR_minus_C4 | +0.017 ± 0.177 pp (seeds +0.202, -0.152, +0.000) | +1.569 ± 0.516 pp (seeds +1.618, +1.029, +2.059) | -0.095 ± 0.918 pp (seeds +0.571, +0.286, -1.143) | -1.611 ± 0.536 pp (seeds -1.833, -2.000, -1.000) |
| TPR_minus_C1 | +0.522 ± 0.127 pp (seeds +0.657, +0.505, +0.404) | +4.706 ± 0.389 pp (seeds +5.000, +4.265, +4.853) | +0.286 ± 0.571 pp (seeds +0.857, +0.286, -0.286) | -3.944 ± 0.509 pp (seeds -4.500, -3.500, -3.833) |
| TPR_minus_C3 | +0.572 ± 0.210 pp (seeds +0.808, +0.505, +0.404) | +5.245 ± 0.594 pp (seeds +5.588, +4.559, +5.588) | +0.238 ± 0.595 pp (seeds +0.714, +0.429, -0.429) | -4.333 ± 0.289 pp (seeds -4.500, -4.000, -4.500) |
| TPR_minus_C0 | +0.000 ± 0.101 pp (seeds -0.101, +0.101, +0.000) | -1.667 ± 0.225 pp (seeds -1.912, -1.618, -1.471) | +0.238 ± 0.165 pp (seeds +0.429, +0.143, +0.143) | +1.611 ± 0.347 pp (seeds +1.333, +2.000, +1.500) |

## Exploratory gates

- PASS — many_within_0_2pp_of_C2_mean
- NOT MET — few_gain_vs_C2_at_least_2pp_mean
- PASS — overall_within_0_2pp_of_C2_mean
- PASS — many_not_worse_than_C2_by_more_than_0_2pp_in_at_least_2_seeds
- PASS — few_gain_positive_vs_C2_in_at_least_2_seeds
- NOT MET — few_within_0_5pp_of_C4_mean

## Interpretation constraints

- The validation set is a 1,980-image holdout from official TRAIN; class 0 has no validation examples.
- The three seeds share one frozen Clean Stage-I backbone and one development holdout.
- This is a classifier-stage MORE-inspired adaptation, not a full MORE reproduction or a statistical-significance claim.
- Read summary.json, final_epoch_metrics.csv, paired_deltas.csv, and per_class.csv together; do not select a favorable metric in isolation.
