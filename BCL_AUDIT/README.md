# BCL Audit: CIFAR-100-LT

This is an independent research line for auditing **Balanced Contrastive Learning (BCL)** on CIFAR-100-LT. It does not assume that BCL has a defect and does not include a proposed replacement method.

## Current scope

1. Paper / public-code protocol audit: `BCL_PROTOCOL_AUDIT.md`
2. Leakage-safe CIFAR-100-LT superclass and DINOv2 topology statistics:
   `scripts/audit_cifar100lt_topology.py`
3. Paired one-seed, 200-epoch FP32 pilot: LC-only versus original BCL.
4. Human-gated two-phase diagnostics: first final TEST benchmark; only after
   approval, LT-train gradients/margins plus read-only TEST margins and DID.

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
  --compute-dino --reuse-static-audit --dino-model dinov2_vits14 --dino-batch-size 16 --topk 5
```

The topology script reads and checksum-validates only the original TRAIN and
meta pickle files. Do not use stock `CIFAR100(train=True)` for this audit:
torchvision 0.29.1 checks both train and test file integrity even when
`train=True`. The DINO path receives only the TRAIN images. The test split is
not instantiated or read during topology construction. DINOv2 uses Bicubic
resize to 224x224 and ImageNet normalization. If preflight used a fresh output
directory, replace the example path with its recorded data_audit_output_dir.
The default model is ViT-S/14 to keep extraction practical on an 8 GB GPU.

`--reuse-static-audit` validates the five original static artifacts listed below
and leaves them byte-identical. The DINO pass writes its pair table, summary,
arrays, and metadata separately. It skips a complete set of DINO outputs and
stops if it finds a partial set.

## Mandatory current-source GPU preflight — run this first

Before any 200-epoch training, run
`CODEX_PREFLIGHT_REVALIDATE_CURRENT.txt` and require a fresh
`READY FOR FULL PILOT` result for the exact current Python source. The older
dc150d1 / 0617c2 READY entries are historical and do not authorize the newly
hardened code. If batch 256 FP32 OOMs, do not reduce batch size, enable AMP or
add gradient accumulation; stop and review the capacity report first.

## Training command

After the exact saved IF=100 indices from the newest preflight report are
available, run the paired groups sequentially with a dedicated output root:

```powershell
python -m BCL_AUDIT.cifar100lt.train --data-root "D:\\datasets\\cifar100" --output-root "BCL_AUDIT\\outputs\\stage_minus1_seed42_20261122_01" --indices-file "BCL_AUDIT\\outputs\\data_audit_recheck_20261122_01\\cifar100lt_if100_train_indices_seed0.npy" --num-workers 0
```

Fixed protocol: seed 42, 200 epochs, batch 256, FP32, SGD, base LR 0.15,
5-epoch warmup, LR drops after 160 and 180 completed epochs, weight decay
5e-4, LC coefficient 2.0, BCL coefficient 0.6, temperature 0.1.
Both groups use the same saved initial state; each epoch has the same
group-independent epoch seed, full sample-order hash and first-minibatch
three-view fingerprint. The trainer refuses to regenerate missing indices and
does not open TEST until both histories pass the paired data-stream audit.

## Two-phase Stage -1 execution

Phase 1 is DINOv2 train-only topology + the paired 200-epoch runs + final
epoch-200 TEST metrics. It must stop for manual acceptance. Do not run the
offline analyzer in the same task.

The DINO pass additionally writes `DINO_CLASS_TOPOLOGY.csv`, containing all
100 classes' symmetric-union degrees and directed Top-5 superclass overlap.
The summary reports degree min/median/max and same-superclass enrichment
against the random-neighbor expectation.

## Phase 2 offline diagnostics — after explicit approval only

After reviewing and approving Phase 1, run
`CODEX_STAGE_MINUS1_PHASE2.txt`. It uses the SAME dedicated training output
root, not the generic outputs parent:

```powershell
python -m BCL_AUDIT.scripts.analyze_paired_bcl --data-root "D:\\datasets\\cifar100" --indices-file "BCL_AUDIT\\outputs\\data_audit_recheck_20261122_01\\cifar100lt_if100_train_indices_seed0.npy" --data-audit-dir "BCL_AUDIT\\outputs\\data_audit_recheck_20261122_01" --output-root "BCL_AUDIT\\outputs\\stage_minus1_seed42_20261122_01" --probe-per-class 5 --probe-seed 777 --batch-size 128
```

Training margins and both gradient probes use only the fixed LT TRAIN probe.
The common-input probe runs in eval mode; the native three-view probe runs
with train-mode BN and restores all BN running buffers. TEST margins and
Head/Tail confusion/DID are read-only calculations from saved final logits
and predictions. No TEST gradient is computed.

## Expected outputs

- `outputs/data_audit/class_counts.csv`
- `outputs/data_audit/superclass_counts.csv`
- `outputs/data_audit/head_tail_pair_topology.csv`
- `outputs/data_audit/data_audit_summary.json`
- `outputs/data_audit/cifar100lt_if100_train_indices_seed0.npy`
- `outputs/data_audit/head_tail_pair_topology_dino.csv`
- `outputs/data_audit/DINO_CLASS_TOPOLOGY.csv`
- `outputs/data_audit/data_audit_dino_summary.json`
- `outputs/data_audit/DINO_CLASS_MEAN_FEATURES.npy`
- `outputs/data_audit/DINO_CLASS_SIMILARITY.npy`
- `outputs/data_audit/dino_metadata.json`
- `<stage_output_root>/environment.json`
- `<stage_output_root>/LC_only_seed42/` and `<stage_output_root>/BCL_seed42/`
- `<stage_output_root>/*_seed42/batchnorm_audit.jsonl`
- `<stage_output_root>/paired_data_stream_audit.json`
- `<stage_output_root>/offline_probe_margin_metrics.csv`
- `<stage_output_root>/offline_dual_gradient_metrics.csv`
- `<stage_output_root>/offline_native_three_view_gradient_class_metrics.csv`
- `<stage_output_root>/offline_native_three_view_gradient_global_metrics.csv`
- `<stage_output_root>/offline_test_margin_metrics.csv`
- `<stage_output_root>/offline_test_margin_summary.json`
- `<stage_output_root>/head_tail_pair_effects.csv`
- `<stage_output_root>/head_tail_did_summary.csv`
- `<stage_output_root>/stage_gate.json`

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
