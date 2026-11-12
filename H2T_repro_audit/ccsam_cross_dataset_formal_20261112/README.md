# CC-SAM cross-dataset formal transfer

This experiment is isolated from the previous CIFAR-100-LT implementation and from the accepted read-only dataset audit.

The selected arms are fixed:

- D0 = existing C2 weighted CE without feature perturbation.
- D1 = existing C1 full CC-SAM class-conditional feature-gradient perturbation.
- D2 = existing C4 full perturbation plus Many-input-only head-preservation distillation from the matching same-dataset/same-seed D0 teacher.

Datasets: ImageNet-LT and Places-LT. Seeds: 123, 124, 125. The formal workload is six independent Stage-I ResNet-50 runs and eighteen Stage-II classifier runs.

Read PROTOCOL.md before running. The protocol deliberately uses a common ResNet-50-from-scratch Stage-I setup for both datasets to keep model capacity and initialization policy matched and avoid potential ImageNet-1K pretraining overlap with ImageNet-LT. This is a controlled transfer experiment, not a strict reproduction of the official H2T recipes for those datasets.

Already implemented on this branch:

- cross_dataset_core.py: dimension-agnostic D0/D1/D2 losses.
- cross_dataset_data.py: audited ImageNet-LT Parquet and Places-LT official-manifest dataset adapters.
- cross_dataset_models.py: common ResNet-50-from-scratch encoder and frozen-state checks.
- test_cross_dataset_core.py and test_cross_dataset_data.py: objective and dataset-adapter unit tests.
- experiment_protocol.json and PROTOCOL.md: frozen protocol.
- CODEX_CROSS_DATASET_FORMAL_20261112.txt: full Codex implementation and execution task.

Codex must complete the remaining training orchestration, preflight, feature caching, Stage-I/Stage-II runners, final-only test evaluator, and analysis pipeline. It must run the CPU tests and real-data Smoke before formal training.

Dataset roots expected on the user's Windows machine:

- D:\datasets\ImageNet-LT
- D:\datasets\places-lt

Do not redownload or repeat the full data audit. Places-LT must be loaded only through the official TXT manifests; the 7,300 unreferenced images must be excluded. The Places-LT val folder is the official test split and must remain sealed until final evaluation.
