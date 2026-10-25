# Fusion Semantic Audit — Protocol

## Scientific question

Do historical H2T-style 16/64 channel-replacement fusions cause recipient-label inconsistency under fixed evaluators, and does that class-level risk align with the observed per-class changes from prior supervision pilots?

This is a diagnostic-only follow-up. It does not assume that the label-target hypothesis is correct and does not train a new method.

## Fixed scope

- Dataset: CIFAR-100-LT IF=100, TRAIN partition only.
- Frozen Stage-I: seed 42, epoch 184, exact checkpoint SHA verified against the existing protocol.
- Stage-II pairing seeds: 124 and 125.
- Primary endpoint: saved epoch-10 receiver/donor pairs and masks.
- Mask comparison: Uniform-PerSample versus Low-VS, with the same pair, augmentation stream, and feature tensors.
- Fusion: replace 16 of 64 complete feature channels from the donor; this is not scalar MixUp.
- Evaluators:
  1. locked Stage-I CE classifier;
  2. cosine nearest-class-centroid classifier from deterministic TRAIN-only GAP features, with the current receiver and donor identities excluded from their prototype sums for each draw.
- Prior outcome comparison: saved epoch-10 validation predictions from LowVS, LowVS-CleanCE, and LowVS-TopCompetitor. The validation set is a repeatedly used original-TRAIN holdout with 1,980 images, class 0 absent, and 20 images for each class 1–99.

## Exact replay gate

Before calculating results, replay the historical augmentation RNG through epochs 1–10. Extract feature maps only for epoch 10 because the backbone and BN are frozen/eval-mode in this protocol. The sealed reference signatures are the original LowVS-Full files under H2T_repro_audit/lowvs_norm_direction/results/seed124 and seed125; saved validation outcomes come from lowvs_disc_preserve. Load the historical masks from the original clean stability plan or its exact saved actual-mask files. For every epoch-10 batch, verify hashes for:
- receiver/donor indices and labels;
- augmented receiver/donor tensors;
- frozen receiver/donor feature maps;
- Gumbel noise;
- saved Uniform-PerSample and Low-VS masks.

Any mismatch stops the run. Do not weaken the hash gate, replace the historical masks, or use the deterministic eval-feature bank as a substitute for the exact augmented features. The eval-feature bank is used only for TRAIN-only NCM prototypes.

## Primary quantities

For each fixed evaluator:
- Harmful flip: clean augmented receiver is predicted as its true label, fused feature is not.
- Beneficial flip: clean receiver is not predicted as its true label, fused feature is.
- Net evaluator correctness change: fused correctness minus clean correctness.
- Margin change: true-label margin after fusion minus before fusion.
- Donor-label takeover: a harmful cross-class fusion is predicted as the donor label.
- Clean-competitor takeover: a harmful fusion is predicted as the clean receiver's strongest non-target competitor.

Report receiver groups Head (class IDs 0–35), OfficialMedium (36–70), Tail (71–99), and tail cross-class versus same-class donor draws. Do not merge these ID-based groups with the count-threshold Many/Medium/Few definitions used in other reports.

## Association analysis

Compute per-class semantic-risk summaries from the exact epoch-10 TRAIN draws, then correlate them descriptively with per-class accuracy changes from the saved validation predictions. Spearman rho is reported without p-values. The holdout has been reused for earlier model selection and is not independent; class-level correlation is ecological association, not causal evidence.

## Stop and interpretation rules

- No optimizer, backward pass, classifier fitting, Stage-I retraining, or hyperparameter sweep.
- No official CIFAR-100 test image or label access.
- Do not generate an automatic pass/fail verdict or choose a new method from these statistics alone.
- If exact source/input hashes or replay hashes fail, stop and report the first mismatch.
- If evaluators disagree, describe the result as evaluator-dependent or inconclusive.
- Even consistent harmful flips do not prove that a changed supervision target will improve generalization. They only establish whether that hypothesis is worth a separately controlled intervention.

## Outputs

The script writes only to this directory's results folder:
- preflight.json
- replay_verification.csv
- group_summary.csv
- class_summary.csv
- mask_contrast.csv
- tail_draws_epoch10.csv
- validation_class_deltas.csv
- class_level_association.csv
- FUSION_SEMANTIC_AUDIT_REPORT.md
- run_manifest.json
