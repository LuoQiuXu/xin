# Selective Many-to-Tail Margin Guard (SMG) for CC-SAM

Status: one-shot candidate method experiment; effectiveness is not assumed.
Branch: `experiment/ccsam-selective-margin-guard-20261117`
Base experiment: `experiment/ccsam-cross-dataset-formal-20261112`

## 1. Motivation from completed evidence

The completed D0/D1/D2 official-test comparison on ImageNet-LT and Places-LT reports:

| Dataset | D1-D0 Overall | D1-D0 Many | D1-D0 Few |
|---|---:|---:|---:|
| ImageNet-LT | +1.32 pp | -2.38 pp | +9.31 pp |
| Places-LT | +2.54 pp | -1.71 pp | +9.85 pp |

D2 adds Many-input-only full-distribution KL preservation to D1. It recovers some Many accuracy relative to D1 but reduces part of the Few gain. Earlier CIFAR-100-LT experiments also found that removing direct Many-input perturbations (C3) did not restore Many performance, while full Many-only distillation (C4) traded some Few gain for Many recovery.

These results motivate, but do not prove, the hypothesis that some tail-directed classifier updates reduce useful Many-versus-tail decision margins. SMG tests a narrower alternative to full-logit distillation.

## 2. Experimental arms

All baseline arms are frozen and reused:

- D0: class-weighted CE with the CC-SAM alpha blend and exactly zero feature perturbation.
- D1: original full CC-SAM class-conditional feature-gradient perturbation.
- D2: D1 plus the existing Many-input-only full-distribution KL term.
- D3 (candidate): unchanged D1 objective plus SMG.

Do not modify D0, D1, or D2 mathematics, sampling plans, feature caches, checkpoints, or metrics.

## 3. SMG definition

Let (n_c) be the official TRAIN count for class (c). Many classes have (n_c>100); the non-Many set contains Medium and Few classes with (n_c\leq100).

For a Many input (x_i), let (y_i) be its true label, (z_i^0) the frozen same-dataset/same-seed D0 teacher logits, and (z_i^s) the current D3 student logits, both computed on the original clean cached feature.

Define the Many-to-non-Many margins:

[
m_i^0=z_{i,y_i}^0-\max_{j:n_j\leq100}z_{i,j}^0,
\qquad
m_i^s=z_{i,y_i}^s-\max_{j:n_j\leq100}z_{i,j}^s.
]

Only protect examples satisfying:

[
n_{y_i}>100,\qquad m_i^0>10^{-6}.
]

For this eligible set (\mathcal E), preserve at least 50% of the teacher margin:

[
L_{SMG}=\frac{1}{|\mathcal E|}\sum_{i\in\mathcal E}
\left[
\frac{\max(0,0.5m_i^0-m_i^s)}{m_i^0+\epsilon}
\right]^2,
\quad \epsilon=10^{-6}.
]

If (\mathcal E) is empty in a batch, return a differentiable zero. Teacher logits are detached. Gradients flow only through student logits.

The final objective is:

[
\boxed{L_{D3}=L_{D1}+1.0\,L_{SMG}.}
]

No D2 KL term is used in D3. The guard is evaluated on clean features and only on Many inputs. The original D1 feature-gradient perturbation and weighted adversarial/clean CE remain unchanged for all input groups.

## 4. Fixed protocol

- Datasets: ImageNet-LT and Places-LT.
- Seeds: 123, 124, 125.
- Stage-I: reuse the six completed epoch-90 ResNet-50-from-scratch checkpoints.
- Stage-II features: reuse the six existing frozen 2,048-D TRAIN feature caches.
- Sampling: reuse the exact official CC-SAM 30-epoch plans and match D1's per-epoch sampling hashes.
- Student initialization: matching Stage-I classifier, exactly as D1.
- Teacher: matching dataset/seed D0 final epoch-30 classifier, frozen.
- Stage-II: 30 epochs, batch size 64, FP32, SGD LR 0.01, momentum 0.9, weight decay 0.0002, no Nesterov, no MixUp, no gradient accumulation.
- D1 alpha schedule, class weights, feature perturbation normalization, and adversarial regularization remain unchanged.
- Fixed SMG retained fraction: 0.5. Fixed SMG coefficient: 1.0.
- Formal workload: six new D3 classifier runs only. No Stage-I retraining or feature re-extraction.

The new D3 outputs must use new, exclusive run directories. Existing D0/D1/D2 and Stage-I artifacts are read-only.

## 5. Required diagnostics

For each batch and epoch, record:

- Many inputs, eligible Many inputs, active guard inputs;
- eligible fraction and active fraction;
- mean teacher margin and student margin on eligible inputs;
- student/teacher margin ratio;
- D1 loss, SMG loss, total loss;
- D1 perturbation magnitude;
- optimizer steps, sample count, and exact sampling-plan hash.

Also report per-dataset/per-seed:
- D0/D1/D2/D3 Overall, Many, Medium, Few;
- paired D3-D0, D3-D1, D3-D2, plus existing D1-D0 and D2-D0 contrasts;
- per-class accuracy changes and per-class D3-D1 changes;
- guard eligibility/activation rates and margin distributions.

Three seeds are descriptive evidence; do not claim statistical significance.

## 6. Data and test isolation

- Preflight, Smoke, and D3 formal training may read only accepted TRAIN manifests, the existing sealed preflight plan, existing TRAIN feature caches, and existing Stage-I/D0/D1/D2 checkpoints.
- Do not instantiate, decode, load, or inspect any official TEST image, label, loader, or prediction during preflight, Smoke, or D3 training.
- Do not use ImageNet-LT validation to select epochs, hyperparameters, or methods.
- Places-LT `val` is the official TEST split and remains sealed until all six D3 runs and artifact checks pass.
- After formal training, run one separate fixed-checkpoint final evaluation using the already audited test adapters and the verified WNID filename reconciliation. Save a new final-evaluation directory; never overwrite the original final-evaluation records.
- No tuning or method changes after viewing D3 official-test results.

## 7. Decision criteria

SMG is supported as a useful candidate only if, on both datasets:

1. D3 improves Many over D1 in the paired mean and in at least two of three seeds.
2. D3 retains a positive Few gain over D0 in at least two of three seeds.
3. D3 improves the Many/Few trade-off over D2 without merely collapsing to D0.
4. Overall does not fall below D0 by more than 0.2 pp in the paired mean.
5. Guard diagnostics show that the loss is active on a measurable subset of eligible Many inputs, not numerically inert or dominant.

These are predeclared decision gates, not claims. If they are not met, report the failure and stop; do not start a coefficient sweep or add another module.
