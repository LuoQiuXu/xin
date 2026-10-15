# Donor-conditioned boundary fusion: full Stage-II report

## 1. Question and answer

The experiment asked whether a risk estimate that uses the complete
recipient--donor path to the nearest reference decision boundary can improve
the final CIFAR100-LT classifier, rather than merely predict harmful fusion in
the frozen-feature mini-validation.

The answer is mixed:

- **Risk/path validity is supported.** At a high matched fusion strength, the
  donor-conditioned policy beats PI-H2T center distance and recipient-only
  boundary distance, and shuffling the recipient--donor assignment removes the
  gain.
- **A robust final-accuracy gain is not established.** After calibrating fusion
  strength and repeating three Stage-II random seeds, the proposed methods,
  fixed-ratio fusion, and official H2T are effectively tied. The geometry
  consistently moves accuracy from Few toward Many, but its mean Overall gain
  is inside run-to-run variation.

The present evidence therefore supports donor-conditioned boundary geometry as
a useful control signal, but not yet as a finished performance-improving H2T
replacement.

## 2. Full training setup

- Dataset: CIFAR100-LT, imbalance factor 0.01 (IF=100).
- Training/test sizes: 10,847 / 10,000.
- Stage-I model: local 200-epoch CE ResNet-32 checkpoint, seed 42.
- Stage-I test accuracy: 40.75% in the common evaluation pipeline.
- Backbone: frozen parameters; BatchNorm running statistics shift during
  Stage-II, matching the released H2T configuration.
- Stage-II sampling: class-balanced recipient and long-tailed donor, using the
  released H2T CIFAR100-LT dataset implementation.
- Stage-II optimization: classifier only, 10 epochs, SGD, momentum 0.9,
  weight decay 5e-4, base LR 0.1 with factor 0.2 and cosine decay, batch size
  128.
- Full-image transforms and feature maps are recomputed every epoch. This is
  not the earlier frozen-vector simulation.
- Group evaluation follows the official split: Many classes 0--35, Medium
  36--70, Few 71--99.
- Stage-II randomness seeds: 42, 43, 44. These are paired sampling/augmentation
  seeds using the same Stage-I checkpoint, not three independently trained
  Stage-I models.
- Primary result: epoch 10 (Last). Test-set Best is retained only as a
  diagnostic because selecting it is test leakage.

The local Stage-I checkpoint uses CIFAR ResNet Option-B projection shortcuts,
whereas the released H2T backbone defaults to Option A. The full trainer detects
this and exactly reconstructs Option B so that no learned shortcut weights are
dropped. All compared Stage-II methods share this same backbone and
initialization.

## 3. Policy definition

For recipient class `y`, competitor `j`, feature `f`, frozen reference weights
`w` and bias `b`, the signed normalized boundary distance is

```text
d_yj(f) = ((w_y - w_j)^T f + b_y - b_j) / ||w_y - w_j||.
```

Along the recipient--donor path

```text
f(lambda) = (1 - lambda) f_recipient + lambda f_donor,
```

every `d_yj` is affine in `lambda`. The closed-form critical ratio is the first
ratio at which any dangerous `y`-versus-`j` boundary reaches the recipient
budget. With `delta=0` in these experiments:

```text
tau = min(min_j d_yj(f_recipient), 0)

lambda_critical = min over j with d_yj(f_donor) < d_yj(f_recipient) of
                  (d_yj(f_recipient) - tau) /
                  (d_yj(f_recipient) - d_yj(f_donor)).
```

Ratios are clipped to `[0, 1]`. `M_group` rescales ratios separately inside
Many/Medium/Few so that every group has the same requested mean donor mass
`mu`. The main soft policy is

```text
lambda_geometry = M_group(lambda_critical; mean=mu)
lambda_beta     = beta * lambda_geometry + (1 - beta) * mu.
```

The exploratory reliability variant leaves the geometry unchanged and uses
`beta = 0.50 / 0.25 / 0.00` for Many/Medium/Few respectively. This was added
after observing that the CE reference boundary is much less reliable for tail
classes; it is reported as a post-hoc candidate, not a confirmatory result.

## 4. High-strength transfer test (`mu = 0.49647`, seed 42)

The mean was transferred from the frozen train-feature pilot. All scalar
methods below have the same mean donor mass.

| Method | Last Overall | Last Many | Last Medium | Last Few | Best Overall |
|---|---:|---:|---:|---:|---:|
| No fusion | 45.19 | 53.75 | 50.06 | 28.69 | 45.24 |
| Fixed matched | 45.34 | 55.19 | 50.94 | 26.34 | 45.52 |
| PI-H2T center matched | 45.16 | 53.64 | 51.43 | 27.07 | 45.35 |
| Recipient-only boundary matched | 45.06 | 52.36 | 51.34 | 28.41 | 45.28 |
| Donor boundary, 25% soft weight | 45.57 | 56.69 | 50.77 | 25.48 | 45.93 |
| Donor boundary, 50% soft weight | **45.65** | **57.42** | 50.66 | 25.00 | **46.02** |
| Donor boundary 50%, shuffled path | 44.96 | 53.17 | 50.77 | 27.76 | 45.36 |
| Official H2T, 25% channel replacement | 46.24 | 57.50 | 50.17 | 27.52 | 46.46 |

At this strength, donor-conditioned 50% control improves Last Overall by 0.49
points over PI center and 0.59 points over recipient-only boundary distance.
Shuffling its path assignments while preserving frequency-group distributions
reduces Last Overall by 0.69 points. Thus the recipient--donor correspondence is
doing real work.

However, the transferred scalar strength is too aggressive for live spatial
feature maps: official H2T remains stronger, and the gain is purchased mainly
by increasing Many while reducing Few.

## 5. Strength calibration on seed 42

Reducing scalar mean donor mass from 0.49647 to 0.25 raises fixed fusion from
45.34 to 46.41 Last Overall. This is a larger effect than changing the risk
estimator.

At `mu=0.25`:

| Method | Last Overall | Last Many | Last Medium | Last Few | Best Overall |
|---|---:|---:|---:|---:|---:|
| Fixed matched | 46.41 | 57.58 | 50.14 | 28.03 | 46.41 |
| PI-H2T center matched | 46.42 | 57.72 | 50.11 | 27.93 | 46.42 |
| Recipient-only boundary matched | 46.33 | 57.50 | 49.97 | 28.07 | 46.43 |
| Donor boundary, 50% soft weight | 46.42 | 58.72 | 50.03 | 26.79 | 46.44 |
| Frequency-reliability boundary | **46.50** | 57.97 | **50.29** | 27.69 | **46.50** |
| Official H2T | 46.24 | 57.50 | 50.17 | 27.52 | 46.46 |

At `mu=0.35`, fixed/donor-boundary Last Overall is 46.33/46.25, while their
test-selected Best is 46.33/46.50. Since the donor result falls after epoch 6,
the stable Last criterion favors `mu=0.25`; no finer test-set sweep was done.

At the calibrated low strength, PI center, recipient-only boundary, and
donor-conditioned boundary are nearly indistinguishable in Overall. Geometry
still changes which groups benefit, but no longer creates the clear Overall
advantage seen at high strength.

## 6. Three Stage-II seeds at `mu = 0.25`

Mean and sample standard deviation are percentages.

### Epoch 10 (primary)

| Method | Overall | Many | Medium | Few |
|---|---:|---:|---:|---:|
| Fixed matched | 46.343 +/- 0.107 | 57.380 +/- 0.208 | **50.543 +/- 0.346** | 27.575 +/- 0.600 |
| Donor boundary, 50% soft weight | 46.327 +/- 0.095 | **58.537 +/- 0.170** | 50.143 +/- 0.114 | 26.563 +/- 0.369 |
| Frequency-reliability boundary | **46.367 +/- 0.159** | 57.769 +/- 0.208 | 50.410 +/- 0.129 | 27.333 +/- 0.530 |
| Official H2T | 46.283 +/- 0.169 | 57.213 +/- 0.267 | 50.457 +/- 0.249 | **27.678 +/- 0.502** |

Paired Last Overall differences relative to fixed fusion:

```text
Donor boundary 50%:          +0.01, +0.01, -0.07; mean -0.017 +/- 0.046
Frequency-reliability:       +0.09, -0.03, +0.01; mean +0.023 +/- 0.061
Official H2T:                -0.17, -0.08, +0.07; mean -0.060 +/- 0.121
```

The differences are too small and inconsistent to claim a final-accuracy win.
The clearest stable effect is group redistribution: uniform donor-boundary
control raises Many by about 1.16 points and lowers Few by about 1.01 points
relative to fixed fusion.

### Test-selected Best (diagnostic only)

| Method | Overall | selected epochs (seeds 42/43/44) |
|---|---:|---:|
| Fixed matched | 46.383 +/- 0.038 | 7 / 7 / 10 |
| Donor boundary, 50% soft weight | 46.437 +/- 0.006 | 7 / 7 / 6 |
| Frequency-reliability boundary | 46.440 +/- 0.052 | 10 / 7 / 6 |
| Official H2T | **46.443 +/- 0.038** | 9 / 9 / 10 |

All three fusion methods are within 0.01 points of one another here. This is a
tie, not evidence that one method is superior.

## 7. What is established

1. Class-center distance alone misses donor direction; donor-conditioned
   boundary geometry has substantially stronger harmful-fusion prediction in
   the frozen-feature mini-validation.
2. Merely avoiding all predicted harmful fusion does not improve the final
   classifier. Controlled hard boundary examples are more useful than a strict
   safety rule.
3. At high fusion strength, the donor-conditioned path produces a genuine
   pair-specific training effect: it beats PI/recipient-only controls and loses
   that gain when path assignments are shuffled.
4. Fusion strength and representation (scalar interpolation versus channel
   replacement) dominate the final result. Once strength is calibrated, the
   current risk policies are tied in Overall.
5. A CE reference head makes the boundary policy head-seeking. The current
   method needs reference reliability/tail protection before it can claim a
   long-tailed recognition improvement.

## 8. What should be tested next

The next expensive experiment should not be another test-set beta sweep. A
publication-clean continuation should:

1. Create a validation split from the long-tailed training set and select
   `mu`, policy weight, and epoch without using CIFAR100 test labels.
2. Train at least three independent Stage-I seeds; the present three seeds only
   measure Stage-II sampling/augmentation variance.
3. Replace the imbalanced CE reference boundary with a frozen calibrated or
   balanced reference classifier (for example cRT/Balanced Softmax), while
   keeping the trainable Stage-II head separate.
4. Pre-register a tail reliability rule or constrained objective, such as
   limiting the Few-class drop while maximizing Overall or harmonic group
   accuracy.
5. Compare scalar interpolation and channel-level replacement using separate,
   appropriately calibrated strength parameters; numeric `lambda=0.25` and
   channel fraction `rho=0.25` are not mathematically equivalent donor mass.

Until those checks pass, the defensible claim is **better donor-conditioned
risk modeling and controllable head--tail tradeoff**, not **state-of-the-art
classification improvement**.

