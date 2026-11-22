# BCL Audit: CIFAR-100-LT

This is an independent research line for auditing **Balanced Contrastive Learning (BCL)** on CIFAR-100-LT. It does not assume that BCL has a defect and does not include a proposed replacement method.

## Current scope

1. Paper / public-code protocol audit: `BCL_PROTOCOL_AUDIT.md`
2. Leakage-safe CIFAR-100-LT superclass and DINOv2 topology statistics:
   `scripts/audit_cifar100lt_topology.py`
3. Paired one-seed, 200-epoch FP32 pilot: LC-only versus original BCL.
4. One merged offline diagnostic pass: checkpoint margins, dual-layer gradients,
   Head-to-Tail / Tail-to-Head confusion, and superclass/DINO paired DID.

## Critical paired-control definition

Both LC-only and BCL receive the same three independently sampled views
(v1, v2, v3), concatenate all three through the shared ResNet-32, and use only
v1 logits for the LC objective. BCL additionally uses v2/v3 sample embeddings
and classifier-derived prototypes. This matches the shared-backbone BatchNorm2d
input distribution. The BCL-only sample projection MLP has its own BatchNorm1d
which mixes all three views; its per-view input and running statistics are
recorded separately.

## Data audit commands

PowerShell example:

```powershell
python BCL_AUDIT/scripts/audit_cifar100lt_topology.py `
  --data-root "D:\datasets\cifar100" `
  --output-dir "BCL_AUDIT\outputs\data_audit"
```

Then compute the DINOv2 class topology using the original balanced CIFAR-100
**training** split only:

```powershell
python BCL_AUDIT/scripts/audit_cifar100lt_topology.py `
  --data-root "D:\datasets\cifar100" `
  --output-dir "BCL_AUDIT\outputs\data_audit" `
  --compute-dino --dino-model dinov2_vits14 --dino-batch-size 16 --topk 5
```

The script loads only `CIFAR100(train=True)`; it does not instantiate the test
dataset. DINOv2 uses Bicubic resize to 224x224 and ImageNet normalization. The
default model is ViT-S/14 to keep extraction practical on an 8 GB GPU.

## Training command

After the static audit writes the fixed IF=100 indices, run the paired groups
sequentially:

```powershell
python -m BCL_AUDIT.cifar100lt.train --data-root "D:\datasets\cifar100" --output-root "BCL_AUDIT\outputs" --indices-file "BCL_AUDIT\outputs\data_audit\cifar100lt_if100_train_indices_seed0.npy" --num-workers 0
```

Fixed protocol: seed 42, 200 epochs, batch 256, FP32, SGD, base LR 0.15,
5-epoch warmup, LR drops after 160 and 180 completed epochs, weight decay
5e-4, LC coefficient 2.0, BCL coefficient 0.6, temperature 0.1.
Checkpoints include model/optimizer/RNG state at epochs 40, 100, 160-pre-decay,
180-pre-decay, and 200.

## Merged offline diagnostics

After both training runs and their final epoch-200 test evaluation complete:

```powershell
python -m BCL_AUDIT.scripts.analyze_paired_bcl --data-root "D:\datasets\cifar100" --indices-file "BCL_AUDIT\outputs\data_audit\cifar100lt_if100_train_indices_seed0.npy" --output-root "BCL_AUDIT\outputs" --probe-per-class 5 --probe-seed 777 --batch-size 128
```

This creates one fixed five-per-class probe set from the selected LT training
subset. It reports margins at all five checkpoints; gradients at classifier
rows W_y and the same probe feature f(x) on BCL checkpoints; and paired
Head->Tail / Tail->Head confusion and DID from the saved final benchmark
predictions. The DINO relation is reported separately from the exogenous
CIFAR-100 coarse-superclass relation.

The first run is one-seed reconnaissance only. The script writes
`stage_gate.json` and does not run any Oracle. Do not proceed to an Oracle
until the mechanism evidence is coherent and repeatable across paired seeds.

## Expected outputs

- `outputs/data_audit/class_counts.csv`
- `outputs/data_audit/superclass_counts.csv`
- `outputs/data_audit/head_tail_pair_topology.csv`
- `outputs/data_audit/DINO_CLASS_SIMILARITY.npy` (after DINO pass)
- `outputs/LC_only_seed42/` and `outputs/BCL_seed42/`
- `outputs/*_seed42/batchnorm_audit.jsonl`
- `outputs/offline_probe_margin_metrics.csv`
- `outputs/offline_dual_gradient_metrics.csv`
- `outputs/head_tail_pair_effects.csv`
- `outputs/head_tail_did_summary.csv`
- `outputs/stage_gate.json`

Do not commit datasets, DINO weights, checkpoints, logs, or generated arrays.
Keep run outputs local and upload only compact reports / CSV summaries when
requested.

## Guardrails

- No 400-epoch run or extra seeds in this first pilot.
- No new method, gradient projection, prototype correction, or semantic gate.
- No test images or labels for DINO/topology construction.
- No test-based checkpoint selection or hyperparameter tuning.
- If batch 256 FP32 OOMs, stop and report the trace; do not silently reduce
  batch size, enable AMP, or add gradient accumulation.
