# CC-MIA Logit-Adjustment & Classifier-Geometry Diagnostics

Status: finite offline diagnostic stage. No model training, optimizer updates, checkpoint selection, or Gate implementation is permitted.

Parent evidence: experiment/ccmia-causal-oracle-20261119. This branch is created from the completed parent branch and must remain isolated.

## 1. Questions

1. Can post-hoc prior correction explain the Many/Few trade-off observed in C2, BASE_C3, TARGETED and ORACLE_SELECTION?
2. Does BASE_C3 have a Many-vs-Few Pareto advantage over C2 after allowing the same scalar prior correction?
3. Are the unadjusted TARGETED and ORACLE_SELECTION points dominated by some point on the BASE_C3 prior-correction curve?
4. In the existing CC-MIA source-conditioned cumulative parameter displacements, how much lies along versus orthogonal to the fixed C2-final classifier-row direction? How large is the separately tracked bias contribution?
5. Do those source-conditioned row changes produce target-logit and fixed-margin shifts on Calibration and Evaluation examples?

This stage diagnoses the already trained linear heads. It does not establish that a particular gradient projection will improve accuracy.

## 2. Frozen inputs and provenance

Reuse only local artifacts from the completed parent branch:
- results/seed_{123,124,125}/status.json
- results/seed_{...}/final_classifier_{C2,BASE_C3,TARGETED,ORACLE_SELECTION}.pth
- results/seed_{...}/final_logits.npz
- calibration/seed_{...}/source_components_epoch30.npz
- calibration/seed_{...}/competitors.json
- the sealed prepared TRAIN-derived development split, feature arrays, split_manifest.json, and mask_seal.json.

Verify the parent completion ledger, all checkpoint SHA256 values, all final-logit SHA256 values, the all-seed mask seal and its dependencies, exact class counts, 990/990 Calibration/Evaluation split, and the parent test-access guard. Recompute Evaluation logits from each checkpoint and compare with the saved final logits before analysis.

The 990 Calibration and 990 Evaluation examples are both subsets of official TRAIN. Evaluation is a held-out development partition, not an independent test set. The Clean Stage-I checkpoint was historically selected using the original 1,980-example holdout; this split does not undo that historical selection. Official CIFAR-100 TEST must not be opened or evaluated.

## 3. Logit Adjustment Pareto diagnostic

For each seed and each of C2, BASE_C3, TARGETED and ORACLE_SELECTION, use the original TRAIN class prior:
pi_c = n_c / sum_j n_j.

Apply only post-hoc:
z'_c = z_c - tau * log(pi_c).

Use the fixed grid tau = -1.50, -1.45, ..., 0, ..., 1.45, 1.50 (61 points). Do not fit or change model weights.

For every tau, report on both partitions:
- Overall accuracy
- Macro accuracy
- Many accuracy (n_c > 100)
- Medium accuracy (20 <= n_c <= 100)
- Few accuracy (n_c < 20)
- sample counts for each group.

The split is exactly class-stratified (10 examples for each represented class), so Overall and Macro should agree numerically; assert this and explain it in the report rather than treating them as independent gains.

Calibration-only selection: for each seed/model, select tau_star maximizing Calibration Overall accuracy. Tie-break by smallest absolute tau, then lower tau. Report Evaluation at this locked tau. The full Evaluation sweep may be plotted descriptively, but no Evaluation value may select tau or any model. Also report the Evaluation-only peak as descriptive-only and explicitly mark it as not used for selection.

Produce Many-vs-Few Pareto curves and Overall-vs-tau plots for both partitions. Include all four model curves and visibly mark tau=0. Mark the Calibration-selected tau on both plots. Report:
- whether one curve Pareto-dominates the other;
- the fraction of each model's non-dominated frontier points dominated by the other;
- whether TARGETED and ORACLE_SELECTION at tau=0 are strictly dominated or weakly covered/matched by any BASE_C3 tau point (equality means the intervention point is covered, not a Pareto escape);
- for each unadjusted intervention point, the best BASE_C3 Few accuracy among points with at least as high Many accuracy (or report that the point is outside BASE_C3's Many range).

Do not say that a single Logit Adjustment sweep proves all classifier differences are bias-only. It tests this specific one-dimensional prior-bias family.

## 4. Radial / tangential / bias decomposition

The parent file source_components_epoch30.npz contains source-conditioned trajectory-integrated parameter displacements after propagation through the frozen SGD momentum recurrence. It does not contain per-step raw gradients.

For every eligible non-Many source s and Calibration-represented Many target row y, decompose the stored cumulative direct parameter displacement Delta_w[s,y] relative to the fixed reference direction w_ref[y] = w_C2_final[y]:

radial = dot(Delta_w, w_ref) / dot(w_ref,w_ref) * w_ref
tangential = Delta_w - radial.

Report signed first-order radial norm change, tangential norm, total displacement norm, exact norm change ||w_ref + Delta_w|| - ||w_ref||, angle change, and the separately stored Delta_b[s,y]. The radial/tangential vectors must reconstruct the original displacement and be orthogonal within numerical tolerance.

For each Many target row, also report:
- the sum of source-conditioned direct displacement;
- observed W_BASE_C3 - W_C2 displacement;
- the non-direct remainder (observed minus summed direct contribution);
- radial/tangential decomposition of all three;
- direct, observed and remainder bias changes.

The non-direct remainder combines the other tracked paths (including clean-gradient state feedback, weight decay and finite writeback residual); do not label it as state feedback alone.

For each source-target pair, project the target-row radial, tangential and bias contributions onto the target class's Calibration and Evaluation feature means. Also report the full fixed-margin contribution including the same source's contribution to the frozen non-Many competitor row from competitors.json. The competitor map must be reused unchanged from Calibration.

Only analyze the 65 non-Many sources and the 34 Many targets represented in the split (2,210 eligible pairs). Assert that direct components for Many source labels are exactly zero. Keep source (perturbed input label) and target (classifier row) axes explicit.

Terminology guardrail: describe this as a decomposition of cumulative, momentum-propagated parameter displacement, not a decomposition of instantaneous gradient vectors. A fixed reference direction is required; state that results depend on the chosen reference.

## 5. Required outputs

All new outputs must be written only under this branch's new experiment directory:
- analysis/logit_adjustment_curve.csv
- analysis/logit_adjustment_selected.csv
- analysis/pareto_comparison.csv
- analysis/pareto_many_few_calibration.png
- analysis/pareto_many_few_evaluation.png
- analysis/overall_vs_tau_calibration.png
- analysis/overall_vs_tau_evaluation.png
- analysis/geometry_source_target_pairs.csv
- analysis/geometry_target_rows.csv
- analysis/geometry_summary.json
- analysis/DIAGNOSTIC_REPORT_ZH.md
- analysis/diagnostics_completed.json
- validation/diagnostic_failure.json on failure, without overwriting an existing failure record.

Do not commit local checkpoints, cached features, NPY/NPZ data, or other large binary artifacts.

## 6. Execution and stopping rules

Run CPU unit tests and syntax checks first. Then run the single offline analysis command. It must:
- verify all parent artifacts and seal before reading Evaluation;
- perform no optimizer step and no training;
- verify checkpoint-to-logit parity;
- fail loudly on missing files, hash mismatch, invalid split, missing Medium examples, nonfinite values, nonzero Many-source direct contributions, or failed decomposition reconstruction;
- never create or overwrite parent-branch files;
- write outputs exclusively (no resume or overwrite).

A successful run completes this diagnostic stage. Do not automatically design or train radial/tangential projection interventions. Human review of the Pareto and geometry findings decides whether a narrowly targeted intervention is justified.
