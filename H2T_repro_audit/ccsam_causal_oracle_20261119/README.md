# CC-MIA Causal Attribution & Oracle Upper Bound

This directory is the next finite validation stage after CC-MIA. It does not implement Margin-aware Gate.

Read `CAUSAL_ORACLE_PROTOCOL.md` before execution. Codex implementation instructions: `CODEX_CAUSAL_ORACLE_20261119.txt`.

Parent evidence is on `experiment/ccsam-cross-class-margin-influence-audit-20261117`. All new code/results must remain isolated under this directory. The 1,980-image holdout is split into calibration/evaluation partitions; evaluation is still TRAIN-derived development data, not an independent test set.

Formal training is prohibited until Codex has completed source audit, CPU tests, preflight and real-data smoke.

## Frozen execution

The user-approved clarification is recorded in `CAUSAL_ORACLE_PROTOCOL.md`.
Calibration uses complete, freshly replayed parent BASE_C3 trajectories,
30 epochs for each seed, and must reproduce the actual parent C2/C3 final
classifier hashes. Every epoch's Calibration pair influence is retained
locally. Class groups remain fixed from the original TRAIN frequencies.

`isolated_data.py` reads only Calibration rows until the all-seed mask seal
exists and verifies. Legacy full-holdout loaders, including mmap, are
blocked. All 3,960 split row/raw-ID entries are in `split_manifest.json`;
this is TRAIN-derived development data, not an independent test set.

The six actual arms are C2, BASE_C3, TARGETED, RANDOM_MATCHED,
FREQUENCY_MATCHED, and ORACLE_SELECTION. ORACLE_SUPPRESSION aliases
TARGETED exactly. Sources are non-Many input labels; targets are
Calibration-represented Many classifier rows. There are 2,210 eligible
pairs, K=221. Selection retains the 221 largest row-local Calibration
contributions; suppression removes the 221 smallest. Neither masks the
clean gradient, class weights, C2, source examples, or the feature delta.

Run from `D:\xiangmuxunlian\xin` in this order (each process must exit 0):

```powershell
& 'D:\app\minicondaaa\envs\xiaoju\python.exe' -B -u 'H2T_repro_audit\ccsam_causal_oracle_20261119\validate_cpu.py'
& 'D:\app\minicondaaa\envs\xiaoju\python.exe' -B -u 'H2T_repro_audit\ccsam_causal_oracle_20261119\run_causal_oracle.py' --phase preflight
& 'D:\app\minicondaaa\envs\xiaoju\python.exe' -B -u 'H2T_repro_audit\ccsam_causal_oracle_20261119\run_causal_oracle.py' --phase calibrate
& 'D:\app\minicondaaa\envs\xiaoju\python.exe' -B -u 'H2T_repro_audit\ccsam_causal_oracle_20261119\run_causal_oracle.py' --phase smoke
& 'D:\app\minicondaaa\envs\xiaoju\python.exe' -B -u 'H2T_repro_audit\ccsam_causal_oracle_20261119\run_causal_oracle.py' --phase formal
& 'D:\app\minicondaaa\envs\xiaoju\python.exe' -B -u 'H2T_repro_audit\ccsam_causal_oracle_20261119\run_causal_oracle.py' --phase analyze
```

Calibration first performs a real baseline parity probe. After all masks
are sealed, production Smoke checks one real paired batch for every arm
and seed. Formal then starts again from the common Clean initialization,
with 5,100 updates per arm/seed and frozen cached, trace-verified inputs.
There is no resume, overwrite, early selection, or official TEST access.
Any failed gate stops execution and preserves `validation/*_FAILURE.json`.

Checkpoints, feature caches, complete 30-epoch influence CSVs, detailed
batch traces, and logs are local and gitignored. Small manifests contain
their actual SHA256. Review `formal_completed.json` and each real
`results/seed_*/status.json`, not configured epoch budgets, for completion.
Analysis reports per-seed effects, sample SD and class-stratified 95% CIs.
Practical thresholds are pre-registered; Gate admission is human review.

## Actual completion

All phases completed on 2026-11-19. Formal ledgers confirm 91,800 actual
optimizer updates and 18 final classifiers. All C2/BASE_C3 final states
match the parent exactly. Official TEST access remained zero.

Read `SCIENTIFIC_REVIEW_ZH.md`, `CAUSAL_ORACLE_REPORT.md`,
`analysis/SCIENTIFIC_REVIEW_DECISION_ZH.json`, and `VALIDATION_REPORT.md`.
TARGETED beats both matched controls on both primaries in 3/3 seeds,
while Few and Overall tradeoffs remain. Oracle Selection's mean practical
flag passes, but seed 125 loses 2.6667pp Few accuracy. Gate admission
remains **HUMAN_REVIEW_ONLY**; no additional experiment or Gate starts.

`validation/finalize_reports.py` is a post-run reporting/reconciliation
utility with a separate SHA. It verified the actual update rows, saved
model tensors, frozen mask hashes and every Calibration epoch. The seven
experimental scripts and frozen protocol still match the pre-GPU CPU
receipt; no experiment code was changed after validation.
