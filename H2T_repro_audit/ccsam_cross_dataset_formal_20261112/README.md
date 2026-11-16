# CC-SAM cross-dataset formal transfer

This experiment is isolated from the previous CIFAR-100-LT implementation and from the accepted read-only dataset audit.

The selected arms are fixed:

- D0 = existing C2 weighted CE without feature perturbation.
- D1 = existing C1 full CC-SAM class-conditional feature-gradient perturbation.
- D2 = existing C4 full perturbation plus Many-input-only head-preservation distillation from the matching same-dataset/same-seed D0 teacher.

Datasets: ImageNet-LT and Places-LT. Seeds: 123, 124, 125. The formal workload is six independent Stage-I ResNet-50 runs and eighteen Stage-II classifier runs.

Read PROTOCOL.md before running. The protocol deliberately uses a common ResNet-50-from-scratch Stage-I setup for both datasets to keep model capacity and initialization policy matched and avoid potential ImageNet-1K pretraining overlap with ImageNet-LT. This is a controlled transfer experiment, not a strict reproduction of the official H2T recipes for those datasets.

Implementation on this branch

- cross_dataset_core.py: dimension-agnostic D0/D1/D2 losses.
- cross_dataset_data.py: audited ImageNet-LT Parquet and Places-LT official-manifest dataset adapters.
- cross_dataset_models.py: common ResNet-50-from-scratch encoder and frozen-state checks.
- test_cross_dataset_core.py and test_cross_dataset_data.py: objective and dataset-adapter unit tests.
- experiment_protocol.json and PROTOCOL.md: frozen protocol.
- CODEX_CROSS_DATASET_FORMAL_20261112.txt: full Codex implementation and execution task.

Run the commands in order from the repository root in PowerShell.

    $python = 'D:/app/minicondaaa/envs/xiaoju/python.exe'
    $runner = 'H2T_repro_audit/ccsam_cross_dataset_formal_20261112/run_cross_dataset.py'
    $env:CCSAM_CROSS_DATASET_ATTEMPT = 'fresh_attempt'
    & $python -B -m unittest discover -s H2T_repro_audit/ccsam_cross_dataset_formal_20261112 -p 'test_*.py' -v
    & $python -B -u $runner preflight --all
    & $python -B -u $runner cuda-self-check --all
    & $python -B -u $runner smoke --all
    & $python -B -u $runner run-formal --all

Choose a fresh attempt name and keep it fixed for every phase. Its outputs go under runs/<attempt>. The CUDA self-check allocates two small tensors and writes its device and peak memory report. Run each command after the previous command exits successfully. The formal command verifies the preflight, CUDA self-check and Smoke certificates. The ordered phase commands after Smoke are stage1, extract, stage2, evaluate-final and analyze. The default is four loader workers. Pass --workers 0 for Windows debugging.

Dataset roots expected on the user's Windows machine:

- D:\datasets\ImageNet-LT
- D:\datasets\places-lt

Do not redownload or repeat the full data audit. Places-LT must be loaded only through the official TXT manifests; the 7,300 unreferenced images must be excluded. The Places-LT val folder is the official test split and must remain sealed until final evaluation.

## Run records

- [Official test reconciliation and final ImageNet-LT and Places-LT results](diagnostics/imagenet_test_reconciliation/IMAGENET_TEST_RECONCILIATION_REPORT.md)
- [Training records and initial final-evaluation failure on 2026-11-16](publication_20261116/README.md)
- [Stage-I AMP gradient diagnosis](diagnostics/AMP_GRADIENT_DIAGNOSIS_REPORT.md)
