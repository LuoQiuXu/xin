# BCL_LTN — isolated label-noise sensitivity baseline

This directory is a separate experiment implementation. **Do not modify anything under
`BCL_AUDIT/`**. The implementation imports the already-audited BCL model, loss,
transforms and strict TRAIN-only CIFAR loader from `BCL_AUDIT.cifar100lt`, but keeps
all new scripts, noise arrays and outputs under `BCL_LTN/`.

Locked protocol:
- CIFAR-100-LT IF=100; reuse `cifar100lt_if100_train_indices_seed0.npy` exactly
- NR=0.4, independent per-example Bernoulli flips, seed 42
- replacement uniformly sampled from the other 99 classes
- Original BCL ResNet-32, three views, FP32, batch 256, seed 42, 200 epochs
- SGD 0.15, momentum 0.9, weight decay 5e-4, 5 epoch warmup, drops after 160/180
- objective `2.0*LC + 0.6*BCL`; LC prior from observed noisy-label counts
- checkpoints 40, 100, 160, 180, 200
- train-only diagnostics at all five checkpoints; balanced TEST only once after epoch 200

The clean labels and `is_noisy` mask are persisted separately for offline diagnostics.
They must never enter the training dataset return tuple, LC/BCL losses, prior
calculation, optimizer, scheduler, checkpoint selection or model update.

Run the exact commands in `CODEX_RUN.txt`. All arrays/checkpoints/logs stay local and
are ignored by Git. Commit only source, tests and the final compact Markdown report.

## Rebuild the existing baseline

Use the saved JSON and CSV artifacts to rebuild the report and check the 20 TRAIN
diagnostic rows. The command preserves the original report and an existing
matching CSV. Checkpoint paths and SHA256 values come from the recorded receipts.

```powershell
python -B -m BCL_LTN.postprocess_existing_run --output-root BCL_LTN/outputs/BCL_sym04_seed42 --noise-dir BCL_LTN/data --execution-dir BCL_LTN/outputs/BCL_sym04_seed42_execution_20261123_01
```

The reconstructed report is `BCL_sym04_baseline_report_rebuilt.md` in the outputs
directory. Use `--report-path` for a fresh report path and
`--pre-edit-trainer-sha256` when the trainer was fingerprinted before editing it.
Source verification uses the historical execution checkout when it is available.

## Single-batch Smoke

Add `--smoke-only` to the training command and choose a fresh `--output-root`.
Smoke verifies the existing TRAIN inputs and noise artifacts, then performs one
FP32 GPU optimizer update with batch size 256. `--smoke-report` sets the output
JSON path in this mode and defaults to `<output-root>/smoke_report.json`.

In formal mode `--smoke-report` selects a passed Smoke receipt as input. The
trainer validates its source, FP32 settings, and the index, noisy-label, and
manifest SHA256 values before starting the epoch loop.
