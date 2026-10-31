# Fish16 B0 AMP recovery: Gate 1 protocol

Date: 2026-10-31  
Branch: `experiment/fish16-h2t-pih2t-cross-domain-20261031`  
Scope: numerical recovery validation only; no algorithm change and no formal training.

## 1. Evidence being acted on

The controlled replay archived under `diagnostics/results/b0_grad_diag_20261031_01/` identifies:

- zero-based batch 6, after six successful SGD updates;
- finite forward logits and loss;
- no missing parameter gradients and no NaN;
- exactly one positive Inf at `extractor.pif.fuse.weight[0,1,0]`, the shared PIF coefficient b;
- the same input and forward hashes pass at AMP scales 32768, 16384, 8192, 4096, and in FP32;
- the finite unscaled b-gradient at lower scales is 1.2802734375, which exceeds the FP16 finite range after multiplication by 65536;
- a disposable clone confirms native GradScaler behavior: zero optimizer.step calls and scale 65536 -> 32768.

The diagnosis does **not** justify changing PIF, its initialization, the backbone, the loss, or the data.

## 2. Implementation contract

`amp_recovery.py` is intentionally fail-closed:

1. The optimizer parameter set must exactly match the named gradient-audit set.
2. Every optimized parameter must have a gradient; all-zero gradients are rejected.
3. NaN is always fatal.
4. Inf is fatal except for exactly one Inf at the audited PIF tensor coordinate `extractor.pif.fuse.weight[0,1,0]`, shape `(1,2,1)`, during PIF Stage-I.
5. No gradient is clamped, replaced, clipped, or silently discarded.
6. The code calls the ordinary `scaler.step(optimizer)` and `scaler.update()`. It then verifies:
   - the optimizer post-step hook did not fire;
   - the GradScaler scale decreased;
   - optimized parameter hashes are unchanged.
7. Finite gradients must produce exactly one actual optimizer.step call.
8. More than two consecutive recoverable overflow skips is fatal.

The explicit GradScaler values reproduce the PyTorch defaults used by the archived run: init scale 65536, backoff 0.5, growth factor 2, growth interval 2000.

Important: a skipped SGD update is not a rollback of the forward pass. BatchNorm running statistics can still update during that training forward. The skip/hash assertion is measured after forward and immediately before/after scaler.step/update.

## 3. Gate 1: exact 100-batch B0 recovery

Run only `b0_recovery_smoke.py`:

- Fish4Knowledge-16 locked manifest; Stage-I training split only;
- seed 42, PIF-ResNet50, original ImageNet weight SHA;
- micro/effective batch 16, no accumulation;
- SGD lr 0.002, momentum 0.9, weight decay 1e-4;
- original H2T MixUp implementation, alpha 1.0;
- FP16 autocast and initial GradScaler scale 65536;
- exactly the first 100 batches (1600 images).

The runner verifies the archived batch-6 replay before accepting recovery: row indices, raw tensor SHA256, pre-MixUp RNG SHA256, MixUp lambda, and PIF a/b values. It requires the expected batch-6 skip (65536 -> 32768), no optimizer.step call on that batch, and a successful optimizer step on batch 7 at scale 32768.

It also requires finite forward outputs, no missing gradients, no NaNs, no other non-finite parameter/coordinate, no more than three total skips, and no more than two consecutive skips. Partial CSV traces are retained on failure.

No validation or TEST images are loaded. No checkpoint is produced. No Stage-II epoch, additional seed, or formal 90+10 pilot is authorized.

## 4. Gate sequence

- **Gate 1 (this task):** exact B0 first-100-batch recovery.
- **Gate 2 (separate approval):** B0 complete one-epoch smoke, then the complete six-condition smoke. The existing `smoke.py` now uses the same strict helper and records actual optimizer updates separately from skipped attempts.
- **Gate 3 (separate approval):** only after review of Gate 2, consider the seed-42 90+10 pilot.

A Gate 1 pass is evidence that the known numerical failure is handled safely in a short real-data run. It is not evidence of classification performance, PI-H2T cross-domain effectiveness, or permission to begin formal training.
