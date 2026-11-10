# CC-SAM-TPR protocol — MORE-inspired tail parameter reallocation

## 1. Research question

The sealed C0-C4 study found that CC-SAM's class-conditional feature-gradient perturbation can improve Few performance while imposing a cost on Many performance. C3 showed that simply disabling direct Many perturbations does not, by itself, restore Many accuracy. C4 added Many-only KL preservation, but it also exposed a trade-off between protecting Many and retaining the Few gain.

The next experiment tests a different level of intervention:

> Can the useful CC-SAM feature expansion be retained while allocating a dedicated low-rank parameter residual to non-Many output classes, and controlling its influence on Many predictions and competition among related tail classes?

This is a classifier-stage method iteration. It is not a full reproduction of the original CC-SAM paper or the network-wide MORE method.

## 2. MORE ideas used and exact scope

MORE decomposes a weight matrix as:

    W_i = W_i^g + W_i^t = W_i^g + B_i^t A_i^t

and guides the low-rank component with a class-prior-weighted squared logit discrepancy:

    M(x) = || f(x; theta_g + theta_t) - f(x; theta_g) ||_2^2

    L_MORE = (1/N) sum_i pi_(y_i) M(x_i)
    pi_y = N_y / N

The paper schedules its coefficient as:

    alpha_MORE(tau) = A sin(pi * tau / T)

The paper's ablation reports that the L2 discrepancy is more effective than KL for its reallocation objective. Therefore this experiment does not stack the C4 KL term on top of MORE. It uses the MORE L2 discrepancy as the reallocation/preservation objective.

The paper's Further Discussions identify two specific issues:
1. A unified low-rank space for all Few classes can create representational trade-offs among semantically disparate classes (intra-tail competition).
2. General and tail-specific parameters can overlap; a stricter orthogonality constraint may help, but hard constraints can impede convergence.

This experiment targets those two issues with:
- one shared low-rank basis A and class-specific coefficient rows B_c;
- a train-only nearest-neighbor graph between tail-class clean-GAP centroids, with a local effective-residual-row decorrelation penalty on graph edges;
- a soft, per-tail-class row-wise orthogonality penalty between the frozen general classifier row and its learned residual row.

The paper also notes static layer-wise decomposition as a limitation. Here the backbone is intentionally frozen and decomposition is restricted to the final 64-D classifier because the preceding C0-C4 evidence isolates the current failure at classifier-stage optimization. This is a scope restriction, not a claim that dynamic layer allocation has been solved. Cross-modal generalization is outside this visual-recognition project.

## 3. Model definition

For each Stage-II seed, the corresponding completed C2 classifier is the frozen general parameter set:

    z_g(x) = W_g f(x) + b_g

The Clean Stage-I ResNet-32 backbone and C2 classifier are frozen. Only a low-rank tail residual is optimized:

    W_t = D_T B A

where:
- f(x) is the existing frozen 64-D Clean Stage-I GAP feature;
- A has shape [r,64] and is shared by all non-Many classes;
- B has one class-specific coefficient row for every non-Many class;
- D_T masks the residual to output rows whose sealed TRAIN count is <=100;
- Many output rows and the classifier bias remain exactly equal to C2.

The complete classifier is:

    z_TPR(x) = (W_g + W_t) f(x) + b_g

This is a single classifier, not a multi-expert architecture. At inference, W_g + W_t is merged into one [100,64] matrix and b_g is retained.

Fixed rank: r=8. The rank is deliberately small relative to the 64-D classifier feature dimension.

This is a structured shared low-rank residual, not a collection of independent tail-class parameter subspaces: all non-Many classes share A and only their B_c rows differ. "Structural decoupling" here means separating the frozen general classifier from a masked, trainable tail residual and restricting which output rows can change. It does not mean that the complete general and tail parameter subspaces are globally orthogonal or that each tail class owns an independent basis.

## 4. CC-SAM objective and MORE reallocation objective

The original, numerically validated C1 feature-gradient perturbation and loss are reused without changing its gradient calculation, class weights, alpha schedule, normalization, or data path:

    L_C1 = alpha_C1 * mean_i[w_yi * CE(z_TPR(f_i + delta_i), y_i)]
           + (1-alpha_C1) * mean_i[CE(z_TPR(f_i), y_i)]

The feature perturbation is computed from the current effective classifier W_g + W_t, with the original validated analytic class-wise gradient. It remains active for Many, MediumN, and Few inputs.

For each original, unperturbed GAP feature, compute the MORE discrepancy:

    d_i = z_TPR(f_i) - z_g(f_i)
    L_MORE = mean_i[(N_yi / N) * ||d_i||_2^2]

The discrepancy is the squared L2 norm over all 100 output logits. The class prior is calculated only from the sealed official TRAIN counts.

The sinusoidal coefficient is applied over all 5,100 optimizer steps:

    alpha_MORE(t) = A sin(pi * t / (T-1))
    A/C = 2.0, C=100, so A=200
    T=5,100; t=0,...,5,099

The normalized amplitude A/C=2.0 follows the scale reported in the MORE amplitude ablation. It is a fixed first-run choice, not a claim that this is the CIFAR-100-LT optimum. No hyperparameter search is permitted in this experiment.

## 5. Addressing the two MORE discussion issues

### 5.1 Intra-tail competition

All non-Many classes share A, but each class has its own B_c coefficient vector. The actual classifier update for class c is the effective residual row:

       Delta W_c = B_c A

Because A is learned and is not constrained to be orthogonal, the angle between B_c and B_c' is generally not the angle between the classifier updates Delta W_c and Delta W_c'. The regularizer must therefore operate on the effective residual rows, not on B alone.

1. Extract clean GAP features for official TRAIN images using the frozen Clean Stage-I backbone and deterministic evaluation transform.
2. Compute one feature centroid per class.
3. Among non-Many classes only, construct a symmetric top-2 cosine-neighbor graph.
4. The graph is built from official TRAIN images and labels only. No validation labels or official TEST data are used.
5. Apply:

       L_div = mean_(c,c') in E [ cos(Delta W_c, Delta W_c')^2 ]

This is a local effective-residual-direction separation constraint on graph edges. The graph encodes feature proximity, not a proof that neighboring classes should be orthogonal; whether this local decorrelation helps or hurts is a testable hypothesis. It neither creates independent per-class subspaces nor guarantees that all tail classes become mutually orthogonal.

### 5.2 General/tail parameter overlap

A global W_g^T W_t orthogonality penalty is unsuitable for this 64-D classifier because W_g may span the full feature dimension. Instead, use a feasible row-wise soft constraint only on residual-enabled classes:

    L_orth = mean_(c in non-Many) [
        cos(W_g[c,:], W_t[c,:])^2
    ]

This penalizes only same-class row-wise overlap between the frozen general classifier and its residual. It is not a global cross-class subspace-orthogonality constraint. The constraint is soft because the MORE authors note that strict orthogonality can impede convergence.

### 5.3 Unified objective

    L_TPR = L_C1
            + alpha_MORE(t) * L_MORE
            + 0.1 * L_orth
            + 0.1 * L_div

No additional C4 KL term is used. The C2 general classifier itself is frozen, and the MORE discrepancy controls how much the learned tail residual changes the full output distribution, with stronger penalties on high-prior samples.

## 6. Fixed experimental protocol

- Repository: LuoQiuXu/xin
- Branch: experiment/ccsam-tpr-20261110
- Base branch: experiment/ccsam-head-preservation-distill-20261109
- Experiment root: H2T_repro_audit/ccsam_stage2_controlled_cifar100_20261106
- TPR directory: tpr_parameter_reallocation
- Dataset: CIFAR-100-LT IF=100; official TRAIN only, 10,847 images.
- Backbone: sealed Clean Stage-I ResNet-32, seed 42, best epoch 184.
- Clean Stage-I SHA256: 7ced9d07f71335357c08623573f038acd0dfdcdeadb55d95d57b332a14f83a88.
- Feature: existing 64-D clean_compatible_gap.
- Seeds: 123, 124, 125.
- Base classifier: matching seed's final epoch-30 C2 classifier.
- Trainable parameters: A and B only.
- Backbone, C2 general classifier, C2 bias, and all Many output rows: frozen.
- Stage-II schedule: 30 epochs, batch size 64, 170 steps/epoch.
- Optimizer: SGD, lr=0.01, momentum=0.9, weight_decay=0.0002, nesterov=False.
- Precision: FP32; no AMP, MixUp, gradient accumulation, scheduler changes, or early stopping.
- Sampling plans and augmentation RNG: exact existing per-seed sealed plans and the same continuous RNG convention as C1.
- Primary result: fixed epoch 30.
- Validation: existing 1,980-image holdout from official TRAIN, 20 images per class 1..99; class 0 absent.
- Official CIFAR-100 TEST: prohibited in preflight, graph extraction, Smoke, formal training, and analysis.

Only TPR is newly trained: 3 seeds × 30 epochs = 90 epochs and 15,300 optimizer steps.

## 7. Required preflight and protection

Preflight must verify:
- original C0/C1/C2 artifacts against the independent checksum index;
- archived C3 artifacts against its checksum index;
- completed C4 artifacts, including local-only checkpoints and traces, against the C4 checksum index;
- all three local C2 base checkpoints and status/checkpoint hashes;
- Clean Stage-I checkpoint, prepared arrays, transform source, and sealed sampling plans;
- original C1 and C4 batch traces;
- all source and protocol hashes.

If any required local-only artifact is missing or differs, stop. Do not retrain C0-C4, reconstruct traces from metrics, regenerate sampling plans, substitute another C2 seed, or skip a checksum gate.

## 8. Smoke requirements

For each seed, use the first two real batches from the sealed sampling plan and continuous augmentation RNG. Require exact equality with both original C1 and C4 traces for:
- sample indices and labels;
- augmented image hashes;
- clean GAP feature hashes.

Smoke must also verify:
- original C1 analytic feature gradients against direct autograd;
- finite C1, MORE, orthogonality, diversity, and total losses;
- finite nonzero gradients for A and B;
- one disposable optimizer update changes the adapter;
- C2 base weights/bias and Many output rows remain bitwise unchanged;
- backbone and BN remain bitwise unchanged;
- the official TEST guard remains clean.

Smoke also creates tail_neighbor_graph.json from TRAIN-only clean-GAP centroids. If graph extraction or Smoke fails, preserve all artifacts and stop. Do not delete and rerun.

## 9. Formal outputs

New outputs only under:
- results/seed_123/CCSAM_TPR/
- results/seed_124/CCSAM_TPR/
- results/seed_125/CCSAM_TPR/

Each seed must save:
- metrics.csv
- status.json
- batch_trace.csv (local-only)
- final_classifier.pth (local-only; includes merged classifier and low-rank factors)

The formal completion marker is formal_completed.json. No automatic resume, overwrite, or silent restart is allowed.

## 10. Analysis and decision gates

Analysis uses fixed epoch 30 and compares TPR with C0, C1, C2, C3, and C4. Required outputs:
- analysis/final_epoch_metrics.csv
- analysis/paired_deltas.csv
- analysis/per_class.csv
- analysis/summary.json
- analysis/TPR_RESULTS.md

Report per-seed values, means, sample standard deviations, paired deltas, class-wise changes, More discrepancy components, and all predeclared gates.

Exploratory gates:
1. Mean Many is within -0.2 pp of C2.
2. Mean Few is at least +2.0 pp over C2.
3. Mean Overall is within -0.2 pp of C2.
4. Many is no worse than C2 by more than 0.2 pp in at least two seeds.
5. Few gain over C2 is positive in at least two seeds.
6. Mean Few is within -0.5 pp of C4.

These are iteration gates, not statistical-significance tests. Report failed gates honestly. Three Stage-II seeds share one frozen Clean Stage-I backbone and one development holdout; do not claim independent-backbone generalization or official TEST performance.

## 11. Interpretation cautions

- This experiment isolates classifier-stage parameter reallocation; it does not test dynamic decomposition across backbone layers.
- The low-rank residual is restricted to non-Many output rows, which is a deliberate adaptation to the observed C1/C4 classifier-stage failure.
- The train-centroid neighbor graph represents feature proximity under the frozen Clean Stage-I model, not guaranteed semantic similarity.
- The orthogonality and effective-residual-row separation penalty are new hypotheses; the experiment must determine whether they help or impede learning.
- The method is MORE-inspired and not a reproduction of MORE's full layer-wise decomposition and joint optimization.
- The corrected protocol computes tail-neighbor diversity on the effective residual rows B @ A. Any results from an earlier draft that computed this term on B alone are a distinct coefficient-space variant and must not be pooled with the corrected residual-space results.
- No claim of novelty, SOTA, or paper-level significance is justified by this one controlled CIFAR-100-LT experiment.
