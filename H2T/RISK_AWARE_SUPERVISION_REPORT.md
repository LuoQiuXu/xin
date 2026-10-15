# Risk-aware supervision experiment (CIFAR100-LT IF100)

## Question

The balanced cRT reference improves harmful-fusion prediction without improving
the final Stage-II classifier.  This experiment tests whether the remaining
bottleneck is the supervision applied to interpolated features.

For recipient feature `f_r`, donor feature `f_d`, and interpolation ratio
`lambda`, all runs use

```
fused = (1 - lambda) * f_r + lambda * f_d
```

and compare:

- A, fused-only: `CE(C(fused), y_r)`.
- B, clean anchor: `CE(C(f_r), y_r) + CE(C(fused), y_r)`.
- C, risk-weighted anchor:
  `CE(C(f_r), y_r) + w * CE(C(fused), y_r)`.

The boundary confidence for C is

```
w = clip((lambda_critical - lambda) /
         (lambda_critical + eps), 0.1, 1)
```

where `lambda_critical` is the donor-conditioned crossing ratio from the frozen
balanced cRT reference classifier.

## Fixed controls

- Stage-I CE ResNet-32 checkpoint: epoch 184, test accuracy 40.76%.
- Frozen balanced cRT geometry head.
- CIFAR100-LT, imbalance factor 100.
- Boundary group blend `beta=0.5`, matched donor mass `mu=0.25`.
- Identical balanced recipients, long-tailed donors, image transforms, ratios,
  classifier initialization, optimizer, schedule, and Stage-II seed 42.
- Ten Stage-II epochs; Backbone parameters frozen; BN running statistics shift
  as in the existing official-style Stage-II implementation.

## Primary results

The pre-registered formulations above directly add their loss terms.

| Supervision | Last Overall | Many | Medium | Few | Best Overall (epoch) |
|---|---:|---:|---:|---:|---:|
| A fused-only | **46.38** | **58.42** | **49.97** | **27.10** | **46.45 (9)** |
| B clean anchor | 45.38 | 56.69 | 49.83 | 25.97 | 45.75 (6) |
| C risk-weighted anchor | 45.18 | 55.67 | 49.54 | 26.90 | 45.54 (1) |

The planned gates fail on seed 42:

- `B > A` is false: B is -1.00 point at the last epoch.
- `C > B` is false: C is -0.20 point at the last epoch.

## Loss-scale control

Adding two CE terms changes the effective gradient scale while retaining the
same optimizer.  Two supplementary controls normalize each objective to roughly
one-CE scale:

- B-normalized: `(clean CE + fused CE) / 2`.
- C-normalized: `(clean CE + w * fused CE) / (1 + mean(w))`.

| Supervision | Last Overall | Many | Medium | Few | Best Overall (epoch) |
|---|---:|---:|---:|---:|---:|
| A fused-only | **46.38** | **58.42** | 49.97 | 27.10 | **46.45 (9)** |
| B clean anchor, normalized | 45.99 | 56.06 | **50.20** | 28.41 | 46.19 (9) |
| C risk anchor, normalized | 45.67 | 55.42 | 49.83 | **28.55** | 45.79 (6) |

Normalization recovers much of B's raw loss, confirming that loss scale was a
real confound.  It does not reverse the conclusion: A remains +0.39 point above
B-normalized and +0.71 above C-normalized.  The anchors trade Many accuracy for
about +1.3--1.45 points of Few accuracy rather than improving Overall.

## Mechanism diagnostics

At the last epoch, all methods have the same mean boundary confidence
`mean(w)=0.3950`; 19.50% of samples hit the minimum weight 0.1.

For A, the fused-minus-clean CE difference by equal-count predicted-risk bin is:

| Risk bin | Fused CE - clean CE |
|---|---:|
| Low | -0.519 |
| Medium | +0.059 |
| High | +0.260 |

Thus the risk score has the intended ordering: high-risk fusion is harder than
the corresponding clean recipient, while low-risk fusion is substantially
easier.

However, the per-example classifier-weight gradient cosine between clean and
fused CE remains strongly positive:

| Supervision | Mean gradient cosine | Negative-cosine rate |
|---|---:|---:|
| A fused-only | 0.9301 | 0.00% |
| B clean anchor | 0.9326 | 0.00% |
| C risk anchor | 0.9329 | 0.00% |
| B normalized | 0.9320 | 0.00% |
| C normalized | 0.9324 | 0.00% |

The specific hypothesis that clean and fused recipient-label losses produce
opposing classifier-weight gradients is therefore not supported.  The problem
is better described as a change in the training distribution/regularization and
head-tail trade-off, not first-order gradient conflict at the linear head.

## Reproducibility audit

- Across A/B/C and both normalized controls, all 198 Backbone/BN tensors are
  bitwise identical (`max_abs_diff=0`).
- The ten-epoch ratio statistics, frequency-group donor masses, reference fused
  predictions, and risk weights are identical across supervision variants.
- The rerun A checkpoint is bitwise identical to the earlier balanced-reference
  boundary-group checkpoint for both Backbone and classifier
  (`max_abs_diff=0`).
- Every run stores `current.pth`, `best.pth`, `metrics.json`, and an experiment
  manifest containing the supervision and normalization settings.

## Conclusion and next gate

The analysis correctly identifies that high-risk fused samples have less
reliable recipient-only supervision.  The proposed remedy does not improve the
classifier: adding a clean anchor hurts Overall, and suppressing high-risk fused
CE hurts further.  Since the seed-42 gate fails even after controlling loss
scale, expanding these exact variants to three Stage-II seeds is not justified.

The next discriminating experiment should change label semantics rather than
only confidence.  Keep the same pairs and geometry, then compare recipient hard
labels with (1) donor-aware MixUp targets and (2) first-crossed-competitor-aware
targets.  This tests whether a boundary-crossing fused feature needs a different
target, rather than merely a smaller weight on the same target.
