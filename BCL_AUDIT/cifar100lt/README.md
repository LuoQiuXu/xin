# CIFAR-100-LT BCL implementation

This is a standalone CIFAR-100 implementation for the BCL audit. It does not modify
the older H2T / CC-SAM / TP-MAG code.

## Environment

Required packages:
- Python
- PyTorch
- torchvision
- NumPy
- Pillow

The first pilot is intentionally FP32. Do not enable AMP or gradient accumulation.

## Strict TRAIN-only data access

The training and offline-probe paths use `CIFAR100TrainOnly`. Do not replace
it with stock `torchvision.datasets.CIFAR100(train=True)`: torchvision 0.29.1
checks both train and test file integrity even for `train=True`. The static
topology/DINO audit reads only the original TRAIN and meta pickle files.
The balanced test dataset is instantiated only after both training runs.

## Run the one-seed paired pilot

First run the static data audit so the exact IF=100 indices are written:

\`\`\`powershell
python BCL_AUDIT/scripts/audit_cifar100lt_topology.py --data-root "D:\datasets\cifar100" --output-dir "BCL_AUDIT\outputs\data_audit"
\`\`\`

Then run the two training groups sequentially. The test split is instantiated only
after both 200-epoch training runs have completed:

\`\`\`powershell
python -m BCL_AUDIT.cifar100lt.train --data-root "D:\datasets\cifar100" --output-root "BCL_AUDIT\outputs" --indices-file "BCL_AUDIT\outputs\data_audit\cifar100lt_if100_train_indices_seed0.npy" --num-workers 0
\`\`\`

Fixed pilot protocol:
- seed 42
- 200 epochs
- batch size 256
- FP32
- SGD, momentum 0.9, weight decay 5e-4
- LR 0.15, 5-epoch warmup, drops after 160 and 180 completed epochs
- LC-only: 2.0 × LC
- BCL: 2.0 × LC + 0.6 × BCL
- BCL temperature 0.1
- sample/prototype MLP dimensions 64 → 512 → 128

## Tests

Run the implementation checks before training:

\`\`\`powershell
python -m pytest BCL_AUDIT/cifar100lt/tests -q
\`\`\`

These tests validate implementation wiring only. They are not evidence that a
scientific gradient conflict exists.

## Outputs

All generated outputs are local and ignored by Git:
- BCL_AUDIT/outputs/LC_only_seed42/
- BCL_AUDIT/outputs/BCL_seed42/
- BCL_AUDIT/outputs/paired_pilot_summary.json

Both LC-only and BCL use the same three-view dataset output. All three views are
concatenated through the shared backbone in both groups; LC-only applies its
classification loss to v1 logits only, while BCL additionally uses v2/v3
projections and classifier-derived prototypes. The training script records
per-view BatchNorm input moments and running statistics at checkpoint epochs.

The training script saves complete epoch 40, 100, 160-pre-decay, 180-pre-decay,
and 200 checkpoints, including model, optimizer, protocol, and RNG states. Do not
upload checkpoints, arrays, or raw predictions unless asked.


## Preflight status after TRAIN-only loader hardening

The previous READY report establishes that the fixed 3-view FP32 batch fits
the RTX 5070 Laptop 8 GB GPU, but it was generated against an earlier source
commit. After the permanent TRAIN-only loader change, rerun
`BCL_AUDIT/CODEX_PREFLIGHT_ONLY.txt` and obtain a fresh READY report before
starting either 200-epoch run.
