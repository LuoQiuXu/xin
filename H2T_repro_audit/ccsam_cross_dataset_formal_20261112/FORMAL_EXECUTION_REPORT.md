# CC-SAM cross-dataset execution

Recorded 2026-11-12T20:31:30+08:00.

- Branch: experiment/ccsam-cross-dataset-formal-20261112
- Git HEAD: a3977ae07f3fd33c1cac9673909a300792818c7f
- Implementation files with worktree changes: 13
- The per-run configs contain exact SHA256 hashes for the implementation and protocol.
- Python: 3.12.12. PyTorch: 2.14.1+cu132. torchvision: 0.29.1+cu132.
- PyArrow: 26.0.1. CUDA: 13.2. cuDNN: 92400.
- GPU: NVIDIA GeForce RTX 5070 Laptop GPU. Device memory: 8546484224 bytes.
- CPU unit tests: Ran 15 tests · exit 0
- Syntax check: Compiled 15 Python files · exit 0
- Preflight: completed. Smoke: incomplete. Final evaluation: not started.
- Official test split access started: False.

Worktree source changes at execution time

- M H2T_repro_audit/ccsam_cross_dataset_formal_20261112/cross_dataset_data.py
- M H2T_repro_audit/ccsam_cross_dataset_formal_20261112/cross_dataset_models.py
- M H2T_repro_audit/ccsam_cross_dataset_formal_20261112/test_cross_dataset_data.py
- ?? H2T_repro_audit/ccsam_cross_dataset_formal_20261112/analyze_results.py
- ?? H2T_repro_audit/ccsam_cross_dataset_formal_20261112/cross_dataset_smoke.py
- ?? H2T_repro_audit/ccsam_cross_dataset_formal_20261112/evaluate_final.py
- ?? H2T_repro_audit/ccsam_cross_dataset_formal_20261112/experiment_common.py
- ?? H2T_repro_audit/ccsam_cross_dataset_formal_20261112/extract_features.py
- ?? H2T_repro_audit/ccsam_cross_dataset_formal_20261112/prepare_experiment.py
- ?? H2T_repro_audit/ccsam_cross_dataset_formal_20261112/run_cross_dataset.py
- ?? H2T_repro_audit/ccsam_cross_dataset_formal_20261112/test_pipeline.py
- ?? H2T_repro_audit/ccsam_cross_dataset_formal_20261112/train_stage1.py
- ?? H2T_repro_audit/ccsam_cross_dataset_formal_20261112/train_stage2.py

## Execution stopped

RuntimeError: Invalid device argument 

## Stage I

| Dataset | Seed | State | Completed epochs | Checkpoint SHA256 |
| --- | ---: | --- | ---: | --- |
| imagenet_lt | 123 | not started | 0 | — |
| imagenet_lt | 124 | not started | 0 | — |
| imagenet_lt | 125 | not started | 0 | — |
| places_lt | 123 | not started | 0 | — |
| places_lt | 124 | not started | 0 | — |
| places_lt | 125 | not started | 0 | — |

## Stage II

| Dataset | Seed | Arm | State | Completed epochs | Checkpoint SHA256 |
| --- | ---: | --- | --- | ---: | --- |
| imagenet_lt | 123 | D0 | not started | 0 | — |
| imagenet_lt | 123 | D1 | not started | 0 | — |
| imagenet_lt | 123 | D2 | not started | 0 | — |
| imagenet_lt | 124 | D0 | not started | 0 | — |
| imagenet_lt | 124 | D1 | not started | 0 | — |
| imagenet_lt | 124 | D2 | not started | 0 | — |
| imagenet_lt | 125 | D0 | not started | 0 | — |
| imagenet_lt | 125 | D1 | not started | 0 | — |
| imagenet_lt | 125 | D2 | not started | 0 | — |
| places_lt | 123 | D0 | not started | 0 | — |
| places_lt | 123 | D1 | not started | 0 | — |
| places_lt | 123 | D2 | not started | 0 | — |
| places_lt | 124 | D0 | not started | 0 | — |
| places_lt | 124 | D1 | not started | 0 | — |
| places_lt | 124 | D2 | not started | 0 | — |
| places_lt | 125 | D0 | not started | 0 | — |
| places_lt | 125 | D1 | not started | 0 | — |
| places_lt | 125 | D2 | not started | 0 | — |

Maximum recorded PyTorch CUDA allocation: 0 bytes.
