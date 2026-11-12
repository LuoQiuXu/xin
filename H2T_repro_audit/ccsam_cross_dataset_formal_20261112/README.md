# CC-SAM cross-dataset formal transfer

This experiment is isolated from the previous CIFAR-100-LT implementation and from the accepted read-only dataset audit.

The selected arms are fixed:

- D0 = existing C2 weighted CE without feature perturbation.
- D1 = existing C1 full CC-SAM class-conditional feature-gradient perturbation.
- D2 = existing C4 full perturbation plus Many-input-only head-preservation distillation from the matching same-dataset/same-seed D0 teacher.

Datasets: ImageNet-LT and Places-LT. Seeds: 123, 124, 125. The formal workload is six independent Stage-I ResNet-50 runs and eighteen Stage-II classifier runs.

Read PROTOCOL.md before running. The protocol deliberately uses a common ResNet-50-from-scratch Stage-I setup for both datasets to keep model capacity and initialization policy matched and avoid potential ImageNet-1K pretraining overlap with ImageNet-LT. This is a controlled transfer experiment, not a strict reproduction of the official H2T recipes for those datasets.

The cross_dataset_core.py module implements the dimension-agnostic D0/D1/D2 objective. Its CPU unit tests are in test_cross_dataset_core.py. The remaining data adapters, training orchestration, feature caching, final test evaluator, and result analyzer are to be completed and verified by Codex under this directory only.

Dataset roots expected on the user's Windows machine:

- D:\datasets\ImageNet-LT
- D:\datasets\places-lt

Do not redownload or repeat the full data audit. Places-LT must be loaded only through the official TXT manifests; the 7,300 unreferenced images must be excluded. The Places-LT val folder is the official test split and must remain sealed until final evaluation.

The final training and evaluation entry points will be documented here after implementation. The Codex task file is CODEX_CROSS_DATASET_FORMAL_20261112.txt.
