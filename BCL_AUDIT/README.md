# BCL Audit: CIFAR-100-LT

This directory starts a separate research line for auditing **Balanced Contrastive Learning (BCL)** on CIFAR-100-LT. It does not assume that BCL has a defect and does not contain a proposed replacement method.

## Current scope

1. Paper / public-code protocol audit: `BCL_PROTOCOL_AUDIT.md`
2. Leakage-safe CIFAR-100-LT superclass and DINOv2 topology statistics:
   `scripts/audit_cifar100lt_topology.py`
3. One-seed, 200-epoch FP32 pilot: LC-only versus BCL (implementation is the next Codex task described in `CODEX_STAGE_MINUS1.txt`).

## Data audit command

PowerShell example:

```powershell
python BCL_AUDIT/scripts/audit_cifar100lt_topology.py `
  --data-root "D:\datasets\cifar100" `
  --output-dir "BCL_AUDIT\outputs\data_audit"
```

Then compute the DINOv2 class topology using the original balanced CIFAR-100 **training** split only:

```powershell
python BCL_AUDIT/scripts/audit_cifar100lt_topology.py `
  --data-root "D:\datasets\cifar100" `
  --output-dir "BCL_AUDIT\outputs\data_audit" `
  --compute-dino --dino-model dinov2_vits14 --dino-batch-size 16 --topk 5
```

The script loads only `CIFAR100(train=True)`; it does not instantiate the test dataset. DINOv2 uses Bicubic resize to 224x224 and ImageNet normalization. The default DINO model is ViT-S/14 to keep the offline feature extraction practical on an 8 GB GPU.

## Expected data-audit outputs

- `class_counts.csv`
- `superclass_counts.csv`
- `head_tail_pair_topology.csv`
- `cifar100lt_if100_train_indices_seed0.npy` (filename reflects the selected split seed)
- `data_audit_summary.json`
- `DINO_CLASS_MEAN_FEATURES.npy`
- `DINO_CLASS_SIMILARITY.npy`
- `dino_metadata.json`

Do not commit dataset files, checkpoints, logs, or generated arrays to GitHub. Keep run outputs local and upload only compact reports / CSV summaries when requested.

## Guardrails

- The first training comparison is only LC-only versus BCL, one seed, 200 epochs.
- Do not start the 400-epoch run or multi-seed experiments yet.
- Do not add a new method or gradient-conflict probe in Stage -1.
- Keep the 3-view BatchNorm caveat explicit: BCL concatenates all three views before the shared backbone forward; LC-only uses one view in this first pilot.
- Use the same fixed IF=100 training subset and pilot seed for both groups.
