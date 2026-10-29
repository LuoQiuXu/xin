# Scientific protocol — crossed Stage-I source × Stage-II observed-label condition

## Why continue the same scenario?

Last run on official CIFAR-100N human fine noise + CIFAR-100-LT IF100 used the actual original H2T operation and a strictly paired no-fusion control. Both Stage-II seeds showed a **Few H2T benefit attenuation**, from +3.6667/+4.5000 pp (clean) to −0.1667/+1.0000 pp (noisy). The earlier strictly preregistered `both negative under noisy` rule failed: that means *stable negative transfer unproven*, not that attenuation never occurred or the scenario should be abandoned. Existing noisy long-tail literature (RoLT, H2E, IJCAI 2024) provides relevant ideas; original novelty is not a prerequisite for a solid incremental method.

Source report: `../noisy_lt_failure_screen/results/cifar100n_lt_seed42_20261029_01/NOISY_LT_FAILURE_SCREEN_REPORT.md`.

Observed 30 original Few-class pools are heavily contaminated: 164 truly matching images among 1,234 observed-label images, weighted purity 13.29%; class-balanced receiver-weighted purity approx 15.9%. This is *data-level*, not proof of H2T's fault.

## Single research question

Does the observed decrease in the relative Few improvement of H2T stem predominantly from (A) the noisy Stage-I representation or (B) the noisy observed-label-and-sampling condition in Stage II?

Define `D(R,S) = Accuracy(H2T|R,S) − Accuracy(NoFusion|R,S)`. R=representation training labels, S=stage-II observed-label+sampling regime.

| R/S | Clean S | Noisy S |
|---|---|---|
| Clean R | Existing experiment, **do not rerun** | **New crossed comparison** |
| Noisy R | **New crossed comparison** | Existing experiment, **do not rerun** |

Both methods within any cell share identical R checkpoints, initial FC, batched receivers/donors, random masks, actual BN shift. Each crossed cell × seeds 124,125 × original H2T/NoFusion = 8 **Stage-II-only** trains, 10 epochs each. Reuse already trained R from the previous run; **zero new Stage-I epochs**.

Noisy S label targets and class-balanced recipient pools are derived from noisy observed labels, clean S uses clean observed labels. Changing S therefore changes the **combined supervised target and sampling exposure**. Factorial contrasts are conditional intervention observations, not isolation of a semantic causal module.

## Interpreting the four cells

Report every seed and group. Contrast decomposition:
- R main effect: average of `D(noisy,S)−D(clean,S)` for both S.
- S main effect: average of `D(R,noisy)−D(R,clean)` for both R.
- interaction: `D(noisy,noisy)−D(noisy,clean)−D(clean,noisy)+D(clean,clean)`.
- diagonal attenuation: `D(noisy,noisy)−D(clean,clean) = R_effect + S_effect`.

The R=noisy,S=clean or R=clean,S=noisy cells use a condition with clean ground-truth supervision. **They are counterfactual diagnostics, not a realistic deployable solution to noisy labels**. Neither metric is a publishable 'oracle-free' robustness claim.

Noisy H2T exceeding noisy NoFusion on seed125 is retained as an important counterexample; do not relabel it as an outright failure. Do not force any result into 'negative transfer'. We are examining relative benefit attenuation.

## Decision with an exit

- If moving S to noisy suppresses H2T Few gain under *both* R sources and both Stage-II seeds in the same direction, Stage II is the appropriate next intervention locus. **Next actual method**: one class-aware receiver-reliability / robust supervision recipe adapted from an established noisy long-tail principle, no new donor/mask searches. Then compare against original H2T and NoFusion under genuinely noisy data.
- If moving R to noisy suppresses H2T gain regardless of S, prioritize one existing noise-robust representation component; pairing with H2T is a legitimate incremental contribution.
- If interaction dominates or cells conflict, there is no need to exhaustively dissect: report conditionality and choose only one cheap task to resolve next.
- If cross factors do not predict attenuation consistently, stop this diagnostic, but do not pronounce the entire noisy long-tail research direction dead.

Do not require a novel theory before trying a small mechanism. But **do not train a proposed module before looking at these eight cheap crossed results**. This is an experiment-early, low-budget decision step, not engineering perfectionism.

## Safeguards that matter

Read source checkpoint hashes from earlier frozen `RUN.json`, not from guessed filenames only; refuse missing weights. Source Stage I seed42 clean/noisy is held fixed. Complete pair audit and same final BN buffers before each pair checkpoint. Use existing original H2T architecture and original code (already imported in `runtime.py`) for actual fusion. No official TEST, no best-epoch selection, no hidden oracle use in the noisy supervised branch.

Results on previously used 1980-image disjoint TRAIN holdout are development-only; class0 missing; original grouping Many>100 / Medium20..100 / Few<20. Two Stage-II seeds on **one Stage-I seed** are not independent representation replications. No PI-H2T claim.

## Code

- `cross_stage2.py`: train only crossed arms, real one-batch `--dryrun`, post hoc fixed-epoch evaluate after eight complete arms; source code/reports not modified.
- `analyze_factorial.py`: joined full 2×2 H2T−NoFusion outcomes and arithmetic checks against original diagonal CSV.
- `test_factorial.py`: synthetic factor/provenance tests; these are only contract tests.

The cross results must be reported without cherry-picking epochs, groups or individual classes. A next one-module iteration is justified by stable practical signal, *not* by demands for absolute originality or unrealistic perfection.
