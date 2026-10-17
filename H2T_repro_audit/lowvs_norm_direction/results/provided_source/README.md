# Low-VS: GAP perturbation magnitude vs. selected channel/direction

**Science question, not a new method proposal.** Low-VS beat Uniform-PerSample by +0.2778 percentage points overall and +0.9483 Tail on the two Clean Stage-II development seeds (124/125), but Low-VS also produced stronger GAP perturbations and affected classifier head/tail tradeoffs. Which effect is driving the result?

## Prespecified intervention

Use exactly the historical Clean H2T source, same frozen seed42 epoch184 ResNet-32/BN, BS receiver / IS donor plans, identical augmentation, Gumbel, online VS score, donor fallback, 16 full spatial channels, Stage-II 100-way FC, CE, SGD and 10-epoch training.

For one matched receiver/donor feature draw let
`d_U = F_U_full - F_receiver` and `d_L = F_LowVS_full - F_receiver`.
Compute `n_U = ||GAP(d_U)||2`, `n_L = ||GAP(d_L)||2`, and
`r = min(n_U,n_L)`. The two matched variants use
`d_U* = (r/n_U) d_U` and `d_L* = (r/n_L) d_L` (zero-norm safe).

- `Uniform-Full`: original Uniform16/64 hard replacement.
- `LowVS-Full`: original Low-VS Bottom48 random16 hard replacement.
- `Uniform-Matched`: Uniform identity/direction with pooled norm attenuated to r.
- `LowVS-Matched`: Low-VS identity/direction with pooled norm attenuated to r.

**Important limitations:** We match *per-example GAP L2*, not full spatial L2, logits, perturbation covariance, or semantic category directions. Matched controls use attenuated full channel replacements (not original hard swaps); a matched Low-VS win means GAP magnitude alone is insufficient, **not** proof of semantic useful channels. A matched tie does not prove magnitude is the unique mechanism.

## Minimal protocol

From local repository root:

```text
python H2T_repro_audit/lowvs_norm_direction/norm_policy.py
python H2T_repro_audit/lowvs_norm_direction/run_experiment.py preflight
python H2T_repro_audit/lowvs_norm_direction/run_experiment.py train --seed 124 --arm Uniform-Full
python H2T_repro_audit/lowvs_norm_direction/run_experiment.py train --seed 124 --arm LowVS-Full
python H2T_repro_audit/lowvs_norm_direction/run_experiment.py train --seed 124 --arm Uniform-Matched
python H2T_repro_audit/lowvs_norm_direction/run_experiment.py train --seed 124 --arm LowVS-Matched
python H2T_repro_audit/lowvs_norm_direction/run_experiment.py train --seed 125 --arm Uniform-Full
python H2T_repro_audit/lowvs_norm_direction/run_experiment.py train --seed 125 --arm LowVS-Full
python H2T_repro_audit/lowvs_norm_direction/run_experiment.py train --seed 125 --arm Uniform-Matched
python H2T_repro_audit/lowvs_norm_direction/run_experiment.py train --seed 125 --arm LowVS-Matched
python H2T_repro_audit/lowvs_norm_direction/run_experiment.py report
```

8 new Stage-II FC-only runs, no Stage-I retrain. Strict same-seed pairing across all 850 minibatches: images, indices, fixed features, scores, Gumbel, and **both candidate masks**. Preflight uses genuine train images, not synthetic substitute. Official CIFAR-100 TEST remains prohibited.

Main endpoint: fixed epoch10, reused 1980-image **TRAIN** validation split, 99 classes class1..99; class0 absent; all models retain 100 logits. Best-epoch checkpoint is a diagnostic only. No selected-seed or alpha tuning.

Outputs in this directory's `results/`: `epoch10.csv`, `paired.csv`, `classifier_hybrids.csv`, `scientific_summary.json`, `NORM_DIRECTION_RESULTS.md`, plus per-arm signatures, final/best FC checkpoints, predictions and training curves.

The `classifier_hybrids.csv` offline diagnostic evaluates combinations of independently learned FC weights W and biases b (original/full and matched contrasts) on **the same frozen validation features**. The swap isolates a particular parameter intervention, not an identified causal explanation or a substitute for retraining.

### Decision

Compare `LowVS-Full - Uniform-Full` with `LowVS-Matched - Uniform-Matched` *within each seed*. If matched Low-VS retains improvements in both seeds, simple GAP magnitude is insufficient. If the matched result vanishes, magnitude becomes plausible but not proven. If both results depend on a few images, cease Low-VS method optimization rather than tune on the reused holdout.

Only two Stage-II seeds, a single Stage-I backbone, and a repeatedly used development split; **not publication-grade external validation**. All Git/version-control actions remain user's responsibility; Codex should only run the tasks in the downloadable instruction file.
