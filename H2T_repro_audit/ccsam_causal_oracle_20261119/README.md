# CC-MIA Causal Attribution & Oracle Upper Bound

This directory is the next finite validation stage after CC-MIA. It does not implement Margin-aware Gate.

Read `CAUSAL_ORACLE_PROTOCOL.md` before execution. Codex implementation instructions: `CODEX_CAUSAL_ORACLE_20261119.txt`.

Parent evidence is on `experiment/ccsam-cross-class-margin-influence-audit-20261117`. All new code/results must remain isolated under this directory. The 1,980-image holdout is split into calibration/evaluation partitions; evaluation is still TRAIN-derived development data, not an independent test set.

Formal training is prohibited until Codex has completed source audit, CPU tests, preflight and real-data smoke.
