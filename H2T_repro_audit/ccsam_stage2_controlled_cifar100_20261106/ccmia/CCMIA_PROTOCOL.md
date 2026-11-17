# CC-MIA: Cross-Class Margin Influence Audit

Status: frozen diagnostic experiment. This is an audit of the existing CC-SAM C2/C3 contrast, not a new training method.

## 1. Research question

C3 sets the feature perturbation to exactly zero for input classes whose original TRAIN count is greater than 100, while retaining the original C3 objective and the shared 100-way linear classifier. C3 nevertheless shows a substantial Many accuracy reduction relative to C2 in the existing TRAIN-derived holdout.

CC-MIA asks:

1. Which source classes (the labels of perturbed training inputs) change which target classifier rows?
2. Does the direct perturbation-induced gradient change shrink Many-vs-non-Many decision margins?
3. How much of the observed C3-C2 classifier/margin difference is attributable to the direct perturbation gradient, versus the clean-gradient change caused by the two training trajectories having diverged?
4. What additional contribution comes from SGD momentum and coupled weight decay?

A nonzero off-diagonal softmax gradient is expected mathematically and is not, by itself, evidence of a harmful mechanism. The primary evidence is the signed effect on a fixed Many-vs-non-Many margin, together with the exact paired gradient/update decomposition and the observed C2/C3 margin trajectory.

## 2. Frozen experimental setup

- Dataset: CIFAR-100-LT, IF=100, original prepared official TRAIN only.
- Backbone: the sealed Clean Stage-I ResNet-32 checkpoint; frozen in eval mode.
- Representation: the existing 64-D Clean-compatible GAP features.
- Stage-II seeds: 123, 124, 125.
- Epochs: 30; batch size 64; 170 batches per epoch (the final batch has 31 examples).
- Optimizer for both heads: FP32 SGD, lr=0.01, momentum=0.9, weight_decay=0.0002, nesterov=False.
- Class weights, alpha schedule, feature perturbation formula, sampling plans and augmentations are imported unchanged from the existing CC-SAM implementation.
- The original TRAIN-derived 1,980-image holdout is used only for margin/accuracy diagnostics. It is not an independent test set. Class 0 has no holdout examples.
- Official CIFAR-100 TEST access is prohibited.

Class groups are determined only from the sealed original TRAIN counts:
- Many: n_c > 100.
- MediumN: 20 <= n_c <= 100.
- Few: n_c < 20.

## 3. Paired C2/C3 training

Both classifiers start from bitwise-identical Clean Stage-I classifier parameters. A single frozen-backbone forward pass produces each batch's GAP features; the same indices, labels, augmented images and GAP features are fed to both heads.

C2 objective:

L_C2(W2) = alpha * mean_i[w_yi CE(W2 f_i,y_i)] + (1-alpha) * mean_i[CE(W2 f_i,y_i)].

C3 objective:

L_C3(W3) = alpha * mean_i[w_yi CE(W3(f_i+delta_i),y_i)] + (1-alpha) * mean_i[CE(W3 f_i,y_i)].

C3 retains the original selective intervention: delta_i is exactly zero for Many input examples and is the unmodified CC-SAM perturbation for MediumN/Few input examples. Many samples remain in both clean and weighted adversarial CE terms. No classifier row is frozen.

Every generated batch trace is compared against the already completed C0 reference trace for the same seed. Any mismatch aborts the run.

## 4. Source-conditioned direct gradient decomposition

At a given C3 classifier state W3, define the clean C2 objective evaluated at that same state as L_C2(W3). The direct perturbation term is:

D(W3) = grad_W L_C3(W3) - grad_W L_C2(W3).

For a sample i with source class s=y_i, the per-sample direct weight-gradient difference is:

alpha*w_s/B * [(p_adv_i-e_s)(f_i+delta_i)^T - (p_clean_i-e_s)f_i^T].

The corresponding bias-gradient difference is:

alpha*w_s/B * (p_adv_i-p_clean_i).

The implementation scatters each sample contribution onto the source axis s and the target classifier-row axis k. It returns:
- source_grad_weight[s,k,d], shape [100,100,64];
- source_grad_bias[s,k], shape [100,100].

Summing the source axis must reproduce D(W3). Since C3 sets delta_i=0 for Many inputs, every Many source-class direct contribution must be exactly zero. Non-Many source classes may contribute to any of the 100 target classifier rows.

The analytic decomposition is checked against autograd on real batches and in CPU unit tests. It is not accepted merely because the softmax formula is theoretically plausible.

## 5. Separating direct perturbation from training-state feedback

At each paired step, the actual gradient difference is:

grad L_C3(W3) - grad L_C2(W2).

It is decomposed in a fixed order:

[grad L_C3(W3) - grad L_C2(W3)] + [grad L_C2(W3) - grad L_C2(W2)].

The first bracket is the direct perturbation effect at the current C3 state and is split by source class. The second bracket is the clean-gradient state/trajectory feedback caused by the current C2 and C3 heads being different.

This is an ordered path decomposition. It is exact for the chosen order, but it is not a unique, order-independent causal attribution. The report must preserve this qualification.

## 6. Propagating the decomposition through SGD

Both heads use the same constant learning rate, momentum, and coupled weight decay. The difference of their SGD momentum buffers obeys:

delta_v_t = momentum*delta_v_(t-1) + direct_gradient_t + state_feedback_gradient_t + weight_decay*(W3_t-W2_t).

The tracker maintains separate momentum and parameter-delta components for:
- direct perturbation, split by source class;
- clean-gradient state feedback;
- coupled weight decay.

The source-specific direct component is propagated through the same momentum recurrence. After each optimizer step, the sum of the tracked components is checked against the actual C3-C2 parameter difference and momentum-buffer difference. A reconstruction error above the frozen tolerance aborts the formal run.

This makes the parameter/margin attribution more informative than treating -lr * gradient_difference as the complete actual update. The report must retain the numerical reconstruction residual.

## 7. Fixed Many-vs-non-Many margins

At initialization, for every validation Many example i, choose one competitor q_i from the non-Many classes using the initial classifier's largest non-Many logit. Keep q_i fixed for the entire experiment.

The fixed margin is:

m_i(W) = z_yi(W,f_i) - z_qi(W,f_i).

Because the head is linear and f_i is fixed, this margin is linear in W and b. Therefore, projecting the tracked parameter-delta components onto this same fixed pair gives an additive decomposition of the actual C3-C2 fixed-margin change.

The experiment also records a dynamic margin using the current strongest non-Many competitor. The dynamic margin is a complementary diagnostic; it is not used for the exact additive parameter decomposition because its competitor can change over training.

## 8. Required outputs

Per seed:
- status.json, final C2/C3 classifier checkpoints (local binary files), batch_trace.csv;
- margin_metrics.csv: C2/C3 Overall, Many, MediumN, Few accuracy and fixed/dynamic margins at epoch 0 through 30;
- per_class_margin.csv: per-Many-class margins;
- step_influence.csv: per-batch direct, state-feedback and total gradient-step-equivalent margin pressure;
- source_target_group_gradient.csv: source class to Many/MediumN/Few target-row gradient norms;
- source_margin_influence.csv: source-class-resolved cumulative direct parameter contribution to each Many target class, plus per-batch direct gradient-step-equivalent margin pressure;
- margin_decomposition.csv: actual fixed-margin change and direct/state/weight-decay/residual components.

Global:
- preflight_report.json, smoke_report.json, formal_completed.json;
- analysis/summary.json, analysis/CCMIA_RESULTS.md;
- analysis/final_paired_metrics.csv;
- analysis/source_class_direct_margin_epoch30.csv;
- analysis/source_to_many_margin_pairs_epoch30.csv.

## 9. Execution gates

1. Preflight: verify the prepared manifest, Clean Stage-I hash, TRAIN counts, official sampler sequence, C0 completed statuses, and all 5,100 C0 reference trace rows per seed.
2. Smoke: two real paired batches per seed; check trace identity, Many perturbation exactly zero, analytic gradient vs autograd, source additivity, gradient path decomposition, and SGD momentum/weight-decay reconstruction. No formal output may be created.
3. Formal: only after successful preflight and smoke; three seeds x 30 epochs. No resume, overwrite, best-epoch selection, parameter sweep, or silent recovery.
4. Analysis: fixed epoch 30 is primary. Report all seeds and paired differences, source/target breakdown, reconstruction residuals, and explicit evidence limits.

## 10. Interpretation rules

- A nonzero cross-row gradient alone is not a successful result.
- Support requires the direct perturbation component to produce a consistent negative fixed Many margin contribution and the observed C3-C2 fixed Many margin shift to be negative across at least two of three seeds, with small reconstruction residuals.
- If actual margins decline but the direct component is not consistently negative, report that state feedback or another pathway may dominate.
- If direct pressure is negative but actual margins do not decline, report a local effect that did not translate into the observed training outcome.
- Do not claim independent-test generalization, statistical significance, or paper-level novelty from this audit.
