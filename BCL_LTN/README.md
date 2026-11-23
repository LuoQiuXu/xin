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
