# TP-MAG Final Development Report

Decision: **NO_GO**

All metrics below are on the sealed TRAIN-derived Evaluation partition. This is not an independent test set.
Official CIFAR-100 TEST access: false.

## Paired differences versus BASE_C3

| Seed | Overall delta (pp) | Many accuracy delta (pp) | Fixed Many margin delta | Few accuracy delta (pp) |
|---:|---:|---:|---:|---:|
| 123 | -0.202 | +2.647 | +0.752501 | -1.333 |
| 124 | +0.303 | +2.941 | +0.754231 | -1.667 |
| 125 | -0.404 | +2.941 | +0.763209 | -3.000 |

## Paired differences versus TARGETED

| Seed | Overall delta (pp) | Many accuracy delta (pp) | Few accuracy delta (pp) |
|---:|---:|---:|---:|
| 123 | +0.101 | +0.882 | +0.000 |
| 124 | +0.303 | +1.471 | -0.333 |
| 125 | -0.404 | +1.176 | -1.333 |

## Frozen decision criteria

- Head efficacy: PASS; mean Many gain +2.843 pp, mean fixed-margin gain +0.756647, both positive in 3/3 seeds.
- Tail floor: PASS; mean Few change -2.000 pp.
- Pareto comparison against TARGETED: FAIL; TP-MAG must be no worse on both mean Many and mean Few accuracy and strictly better on at least one.

## Stop decision

NO-GO: stop this Gate line. No epsilon, threshold, coefficient, loss or reference tuning is permitted.

No threshold, epsilon, coefficient, loss or reference tuning was performed. The Gate was trained once with the frozen protocol.
