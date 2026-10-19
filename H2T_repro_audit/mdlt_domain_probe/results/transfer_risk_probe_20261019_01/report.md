# OfficeHome H2T transfer-risk probe

## Purpose and boundaries

This is a validation-only, fixed-feature mechanism probe. It reuses the completed pilot's
cached train/validation GAP features, the saved epoch-10 H2T-Global linear classifier,
and the existing donor-class/mask plan. It trains no model, reads no test feature cache,
and performs no hyperparameter search.

Each validation receiver gets four paired donor-class candidates. Across Global,
DomainMatched and CrossDomain, the receiver, donor class and 128-channel batch-shared mask
are held fixed; donor identity necessarily changes with the domain constraint.

## Outcome and predictors

- Primary outcome: actual change in receiver-label cross-entropy after substitution,
  delta_CE = CE(fused) - CE(original); positive means harmful under the fixed classifier.
- Secondary outcome: change in true-label logit margin.
- Directional score: first-order Taylor estimate grad_f CE(original) dot masked(donor - receiver).
- Non-directional baselines: masked relative L2, masked cosine distance, donor-label mismatch,
  baseline CE / negative margin, and domain mismatch.
- AUROC and Spearman are descriptive prediction diagnostics, not significance tests.

## Per-arm outcome summary (mean over three seeds)

| Arm | ΔCE (nats) | Harm rate | Δmargin | Flip-to-wrong | Wrong-to-correct | Domain mismatch |
|---|---:|---:|---:|---:|---:|---:|
| H2T-Global | 0.40415 | 0.6544 | -4.77659 | 0.0992 | 0.0244 | 0.7540 |
| H2T-DomainMatched | 0.36268 | 0.6586 | -4.76565 | 0.0980 | 0.0233 | 0.0048 |
| H2T-CrossDomain | 0.41113 | 0.6612 | -4.81469 | 0.0999 | 0.0231 | 1.0000 |

## Directional risk prediction

AUROC asks whether the score ranks substitutions with positive ΔCE above substitutions with
non-positive ΔCE. Higher scores always mean greater predicted risk. A domain-mismatch score
has no within-arm variation when the arm enforces a domain constraint; its AUROC is reported
as undefined rather than interpreted as evidence for or against domain mismatch.

| Seed | Arm | Directional ΔCE AUROC | Relative-L2 AUROC | Cosine-distance AUROC | Label-mismatch AUROC | Domain-mismatch AUROC |
|---:|---|---:|---:|---:|---:|---:|
| 42 | H2T-Global | 0.835 | 0.561 | 0.493 | 0.517 | 0.494 |
| 42 | H2T-DomainMatched | 0.840 | 0.559 | 0.499 | 0.518 | 0.498 |
| 42 | H2T-CrossDomain | 0.829 | 0.568 | 0.500 | 0.520 | NA |
| 43 | H2T-Global | 0.836 | 0.562 | 0.493 | 0.518 | 0.505 |
| 43 | H2T-DomainMatched | 0.838 | 0.543 | 0.491 | 0.520 | 0.501 |
| 43 | H2T-CrossDomain | 0.837 | 0.558 | 0.494 | 0.519 | NA |
| 44 | H2T-Global | 0.821 | 0.577 | 0.498 | 0.519 | 0.494 |
| 44 | H2T-DomainMatched | 0.821 | 0.565 | 0.487 | 0.520 | 0.501 |
| 44 | H2T-CrossDomain | 0.821 | 0.569 | 0.497 | 0.520 | NA |

## Paired CrossDomain − DomainMatched contrast

Positive ΔCE difference means CrossDomain substitutions increase loss more on average.
The bootstrap resamples validation images within each domain-class cell and averages each
image's four candidates across the three fixed seed checkpoints. The interval is conditional
on these trained checkpoints; it is not a confidence interval over future training seeds.

- Mean paired ΔCE difference: 0.048448 nats (stratified image-bootstrap 95% interval 0.015305, 0.079635).
- Mean paired Δmargin difference: -0.049042 (95% interval -0.104253, 0.005862).
- Harm-rate difference: 0.002564 (95% interval -0.003141, 0.008559).

Per-seed paired contrasts are in paired_cross_minus_matched.csv; every candidate pair shares
the same validation receiver, donor class and mask, but not the same donor image.

## Predeclared exploratory go/no-go gate

A minimal directional-risk module is worth prototyping only if, in at least two of three arms,
the directional score has AUROC >= 0.60 and exceeds the masked-relative-L2 AUROC by >= 0.05
in at least two of the three seeds. This is a practical screening gate, not a statistical
significance threshold. If it fails, do not train a new module on the strength of this probe.

- Gate result: **PASS**.
- Arms meeting the gate: H2T-Global, H2T-DomainMatched, H2T-CrossDomain.

## Interpretation limits

- This measures one-step substitution effects under the fixed epoch-10 H2T-Global linear head.
  It does not establish end-to-end representation-learning effects or prove a new method works.
- Donor domain and donor identity are not separable in the conditional arms; domain effects remain
  a paired mechanism signal, not a perfectly isolated causal effect.
- Validation labels are used to evaluate the supervised training-time risk signal. Do not tune
  thresholds on this same validation set and then report these results as independent confirmation.
- No test images/pixels, cached test features, test predictions, or test metrics were read; only pinned split metadata was parsed to verify train/validation row alignment.
- If the gate passes, the next experiment should be one minimal risk-gated intervention with
  thresholds fixed on validation, followed by independent confirmation. If it fails, stop this
  branch rather than adding more heuristics.

## Artifacts

- candidate_level.csv: all receiver/candidate/arm outcomes and predictors.
- per_seed_arm_summary.csv: substitution outcome summaries.
- risk_predictor_summary.csv: AUROC and Spearman by seed, arm and score.
- paired_cross_minus_matched.csv: paired domain-condition contrasts.
- summary.json: provenance, gate and paired bootstrap intervals.
