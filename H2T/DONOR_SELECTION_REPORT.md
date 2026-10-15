# Donor-selection Stage-II experiment (seed 42)

> Superseded for the primary conclusion by
> `PI_H2T_DONOR_SELECTION_VALIDATION.md`, which adds the recipient-relative
> hardest-safe rule needed when the frozen long-tailed reference already
> misclassifies a clean recipient.

## Re-centered question

The preceding experiments established that donor-conditioned boundary geometry
predicts harmful fusion better than a recipient-only center score.  This round
does **not** alter the loss, soft target, or fusion amount.  It asks one clean
question instead:

> Given the same recipient and the same K candidate donors, does using the
> pair-specific boundary score to choose the donor improve H2T training?

For every scalar-fusion method, the training target remains the hard recipient
label and the donor mass is fixed at `lambda=0.25`.

## Protocol

- CIFAR100-LT, imbalance factor 100; Stage-I CE ResNet-32 checkpoint, epoch
  184 (40.76% Stage-I test accuracy).
- Frozen balanced cRT classifier and a separate frozen/eval Stage-I backbone
  provide the pair-geometry score `lambda_critical(f_r, f_d)`.
- The train backbone is frozen as in Stage-II; its BN buffers see only the
  recipient and the donor ultimately chosen for fusion.
- Each balanced recipient receives `K=4` independently sampled long-tailed
  donor candidates.
- All methods see bitwise-identical recipient images, candidate donor images,
  target labels, and candidate labels.  Only donor identity is changed.
- Ten Stage-II epochs, same classifier initialization, CE loss, optimizer,
  schedule, and seed 42.

Policies:

| Method | Donor policy | Fusion |
|---|---|---:|
| Random | Uniform random candidate from the K-pool | scalar `lambda=0.25` |
| Best-of-K boundary | `argmax_d lambda_critical(f_r, f_d)` | scalar `lambda=0.25` |
| Shuffled ranking | Boundary scores randomly reassigned to donor slots before argmax | scalar `lambda=0.25` |
| Official H2T | Random candidate, random 25% channel replacement | official channel fusion |

The shuffled-ranking control preserves the K-candidate pool and the score
multiset while destroying score--donor alignment.

## Results

Primary selection comparison uses the last epoch; best epoch is diagnostic
only because it selects on the test set.

| Method | Last Overall | Many | Medium | Few | Best Overall (epoch) |
|---|---:|---:|---:|---:|---:|
| Random donor | 46.37 | **57.64** | 50.29 | 27.66 | 46.50 (8) |
| Best-of-4 boundary donor | 46.15 | 57.08 | 50.09 | **27.83** | 46.27 (6) |
| Shuffled boundary ranking | **46.42** | 57.47 | **50.60** | 27.66 | **46.58 (8)** |
| Official H2T | 46.33 | **57.64** | 50.51 | 27.24 | 46.49 (8) |

The key pre-registered gate fails: Best-of-4 does not exceed random donor
selection (`46.15 < 46.37` at the last epoch).  Its best epoch is also lower
than the random baseline (`46.27 < 46.50`).  Random and shuffled-ranking
policies are effectively tied within one-seed noise, which is the expected
negative-control behavior.

## Did selection work?

Yes.  The policy changes exactly the intended geometry variables:

| Quantity at epoch 10 | Random | Best-of-4 | Shuffled |
|---|---:|---:|---:|
| Selected `lambda_critical` | 0.453 | **0.654** | 0.454 |
| Selected candidate rank (1=best) | 2.414 | **1.000** | 2.415 |
| Pair safe at fixed `lambda=0.25` | 78.56% | **96.62%** | 78.84% |
| Frozen-reference flip proxy | 55.67% | **47.96%** | 55.50% |
| Fused training accuracy | 56.34% | **63.10%** | 56.49% |

Thus the boundary score successfully selects much safer, easier fused samples.
The failure is not a candidate-pool, implementation, or score-alignment error.

## Audit

- For every epoch, recipient/candidate-image checksums and recipient/candidate
  label checksums are identical across all four policies.
- All scalar-fusion policies use exactly `lambda=0.25`; no group rescaling or
  loss weighting is applied.
- Best-of-4 selects the true highest-scoring candidate on every batch.
- Random and shuffled policies have the expected mean rank of about 2.4 among
  four candidates.
- Best-of-4 selects a same-class donor 3.71% of the time versus 1.02% for
  random.  This favors the safe-selection policy, so it cannot explain its
  lower Overall score; nevertheless, future confirmation should explicitly
  exclude same-class candidates in every policy.

## Interpretation

The result is informative but narrower than “donor-conditioned geometry does
not work.”  It says that the particular rule

```
choose the safest donor = argmax lambda_critical
```

removes too much of the difficult/useful perturbation.  It makes augmented
training examples much easier (63.10% versus 56.34% training accuracy), reduces
the frozen-reference risk proxy, and slightly reduces final test accuracy.

In other words, H2T may need a **useful but boundary-safe** donor, not the
maximally safe donor.  The original donor-conditioned score remains valuable as
a safety coordinate, but safety alone is not the training objective.

## Correct next gate

Do not expand `argmax lambda_critical` to more seeds.  The next donor-selection
policy should be *nearest safe*, with a fixed fusion amount:

```
d* = argmin_d lambda_critical(f_r, f_d)
     subject to lambda_critical(f_r, f_d) >= lambda + margin
```

If no candidate is safe, use the highest-critical candidate as fallback.  This
keeps the fusion informative (near the decision boundary) without deliberately
crossing it.  It should be tested against random, max-safe, and shuffled-ranking
using a candidate pool that excludes donor class = recipient class.

The main artifacts are:

- `train_donor_selection_stage2.py` -- paired K-candidate trainer.
- `outputs/donor_selection_balref_seed42_k4_lambda025/summary.csv` -- metrics
  and candidate-stream audit values.
- One `current.pth` and `best.pth` checkpoint per policy.
