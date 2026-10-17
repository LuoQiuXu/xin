# Boundary-Direction Probe | Low-VS vs Uniform (No New Training)

## What is the decisive unresolved question?

Under paired GAP L2 matching, Low-VS Tail benefits did **not** persist consistently across seed124 and seed125. Original full Low-VS causes larger pooled perturbations; matched Low-VS is attenuated substantially more than matched Uniform. Trained classifiers change W more saliently than a stand-alone replacement of their final bias. None proves that Low-VS has useful *direction*.

We therefore test a **falsifiable geometry prediction** rather than build a new adaptive sampler:

> On identical training receiver/donor pairs and *the same frozen Stage-I classifier*, do Low-VS chosen perturbations differ from random channel perturbations in their orientation relative to the receiver's most competitive existing decision boundary, even after the perturbation length is removed?

## Exact deterministic diagnostic

Use existing real **TRAIN** pairing and augmentation plans from the original Clean experiment, StageII view **epoch 1 only**, seeds124/125. The same locked Stage-I seed42 epoch184 ResNet32+BN and classifier are **inference-only**. No StageII optimizer, no loss training, no new learnable modules.

Let \`z\` be receiver GAP, \`z_h\` donor GAP, \`δ_U\`/ \`δ_L\` be the 64-d actual masked replacement vectors.

From the clean, unperturbed receiver and **fixed Stage-I W,b** choose

\`j*=argmax_{j ≠ y}(w_j^T z + b_j)\`.

Then compute for each vector

\`q_U=(w_y-w_j*)^T δ_U / max(||δ_U||2, eps)\`,
\`q_L=(w_y-w_j*)^T δ_L / max(||δ_L||2, eps)\`.

Also report signed cosine, donor-vs-receiver class projection \`q_yh\` (when h≠y), original and matched fixed-competitor margin changes, and immediate Stage-I loss/flip diagnostics.

**Sign convention:** \`q<0\` means **harder for the current frozen classifier** (lower true-vs-j* margin) per unit perturbation length. It is not evidence of better training or improved generalization. \`q>0\` means immediate margin preservation/increase, not necessarily beneficial training either.

With \`n_U=||δ_U||\`, \`n_L=||δ_L||\`, \`r=min(n_U,n_L)\` the following **identity** holds for each draw:

\`Δmargin_L_full - Δmargin_U_full = r(q_L-q_U) + [(n_L-r)q_L-(n_U-r)q_U]\`.

First term is the directional contrast at the matched norm \`r\`; second is the remaining norm-induced term. This algebra decomposes **only a fixed linear margin**, NOT training outcomes or accuracy gains. It is exact up to floating point.

## Prespecified population

- Primary: Tail receivers (classes71..99), donor label different from receiver, not a VS fallback, and different actual channel masks. This isolates places where the policy really differs.
- Secondary: All-eligible, Head-eligible, Tail receiving Head donor, Tail with correct/incorrect initial Stage-I classification.
- Keep *all* samples in saved NPZ; report sample counts, fallback and mask overlap. No post-hoc alpha/K thresholds, donor filtering or seed cherry-picking.

**Main diagnostics:** mean(qL-qU), fraction(qL<qU), mean(cosL-cosU), matched fixed-margin L-minus-U, plus donor-class normalized q. Report both seeds separately; remember both use one Stage-I model and repeated training images. Never treat 21,694 draws as independent examples for a significance claim.

## Run locally with existing source data

From repository root:

~~~text
python H2T_repro_audit/lowvs_boundary_direction/geometry.py
python H2T_repro_audit/lowvs_boundary_direction/run_probe.py run --seed 124
python H2T_repro_audit/lowvs_boundary_direction/run_probe.py run --seed 125
python H2T_repro_audit/lowvs_boundary_direction/run_probe.py report
~~~

First historical epoch has 85 minibatches per seed. The script checks all 7 original input and mask hash fields against the uploaded previous **real** norm-direction experiment, so a failed environment pairing does not silently become a different experiment. No Stage I/II training, no official TEST use.

Writes only under this directory's \`results/\`:
- \`seed124/train_epoch1_boundary_probe.npz\`, \`seed125/...\`: per-example frozen-boundary values;
- \`group_summary.csv\`, \`per_class_summary.csv\`, \`scientific_summary.json\`, \`BOUNDARY_DIRECTION_RESULTS.md\`.

## Three clear scientific outcomes

1. **Direction differs after normalization:** consistent q/cos shift across both enhancement views is evidence that Low-VS picks directions with different *existing boundary interaction*. It is **not** proof those directions are beneficial, nor a reason to add another gate immediately.
2. **No stable directional contrast:** amplitude and possibly higher-order optimization effects remain plausible; **stop** Low-VS directional-method ideation.
3. **Sign differs by group/donor:** potential dataset-condition failure mode. Investigate whether a transfer benchmark suffers harmful cross-class mixing, but do **not** tune thresholds or choose subgroups after viewing the outcome.

No separate model, no official TEST, no end-to-end representation-learning conclusion. The existing 99-class repeatedly used dev split is not involved in this probe's policy or statistics.

## Follow-up dataset stress test (NOT part of this probe)

The supplied author README provides real configs for **CIFAR-100-LT imbalance factors 50/100/200**. It notes support for other datasets is not fully integrated in released configs. Once we have an actual mechanism hypothesis, compare matched baselines **DR(no fusion), original H2T, Low-VS** across the documented ratios, then one genuinely external dataset *only after confirming its pipeline/data and 8GB VRAM feasibility*. Match stage I, sampling, BN, epoch, loss and per-group metrics; identify where H2T loses rather than indiscriminately claiming every new-dataset drop proves a flaw. Reserve an untouched test set for the final predeclared method.
