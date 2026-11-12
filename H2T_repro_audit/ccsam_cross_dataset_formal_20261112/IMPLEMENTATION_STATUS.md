# Implementation status

Branch: experiment/ccsam-cross-dataset-formal-20261112

## Committed

- Frozen experiment protocol and machine-readable configuration.
- Dimension-agnostic D0/D1/D2 CC-SAM objectives.
- ImageNet-LT Parquet dataset adapter with explicit split selection and lazy row-group image loading.
- Places-LT manifest-backed adapter with train-derived class-folder mapping and no directory-scan fallback.
- Common ResNet-50-from-scratch model builder and frozen encoder/BN integrity checks.
- CPU unit tests for objective mathematics and both dataset adapters.
- Full Codex task instructions.

## Still required before any formal training

- Run all unit tests in the user's conda environment.
- Complete preflight, deterministic sampler-plan preparation, Stage-I training, feature extraction, Stage-II training, final-only test evaluation, and analysis scripts.
- Run real-data Smoke on both datasets and pass every required check.
- Only then execute the six Stage-I and eighteen Stage-II formal runs.

## Important

The committed code has not yet been executed against the user's local Windows dataset paths or RTX 5070 Laptop GPU. Do not interpret this branch as a completed training implementation or as experimental evidence. Codex must validate the current modules and finish the remaining pipeline before formal training.
