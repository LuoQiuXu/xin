# Fish16 seed-42 clean-launch and formal protocol

Execution branch: `experiment/fish16-clean-launch-fix-20261103`.

## Numerical protocol

The earlier FP16 attempts are preserved unchanged on the preceding branches.
FP16 failed during backward even after scale growth was disabled: the A0
formal attempt at epoch 66 produced 147 NaNs in backbone conv1 gradients and
two Infs in BN1 gradients while the forward loss/logits were finite. The clean
launch therefore changes the numerical protocol for **all six conditions**
to full FP32:

- Autocast disabled in every training and validation forward.
- GradScaler disabled; the retained GradScaler object is a no-op compatibility
  adapter only.
- No gradient clipping, replacement, clamping, skip, scale backoff or retry.
- Any non-finite forward or gradient is fatal and must be saved as a failure.
- The same numerical policy is used by A0, B0, A1, A2, B1 and B2.

This is an explicit FP32 protocol change, not a claim that FP16 was repaired.
Every run manifest must record `NUMERIC_MODE=float32_no_autocast_no_grad_scaler`,
`amp=false`, `autocast_enabled=false`, and `grad_scaler_enabled=false`.

## Data and split isolation

The locked manifest is `D:\h2t_cross_domain_audit\fish4knowledge\manifest_primary.csv`
with SHA256
`62d9a2e9fffaaca22ef0c99c0ce4134b4d0e1ecc308333a7cb57b10cf95f0f30`.

- TRAIN: 16,269 images / 5,208 groups.
- VAL: 5,396 images / 1,736 groups.
- TEST: 5,468 images / 1,736 groups.

Preflight reads and hashes image bytes only for TRAIN and VAL (21,665 files).
For TEST it validates only manifest metadata and split assignment. It must not
resolve TEST paths, stat/open TEST files, hash TEST image bytes, create TEST
views, or evaluate TEST. Both TEST image-byte reads and TEST model evaluations
must remain exactly zero. Cross-split leakage checks use the locked manifest
metadata only.

## Model and source identity

The approved model adapter is `models.FishModel`, loading the locally cached
official ResNet-50 `19c8e357` initialization through the strict adapter
provenance file. Preflight verifies the official weight SHA, converted tensor
state SHA, adapter provenance, common A/B initialization, PIF parameter count,
and the immutable upstream H2T/PI-H2T source fingerprints. It must not call a
legacy `ResNet50_FE` API or download weights.

The locked upstream fingerprints are centralized in `protocol_config.py`.
CPU validation, preflight and Smoke bind their records to the exact source
fingerprints. No external historical `ORIGINAL_SOURCE_HASHES.json` or
`CPU_TEST_RESULTS_FINAL.txt` is a runtime dependency.

## Smoke protocol

The one-epoch real-data Smoke executes six conditions in this exact order:

1. A0 — Stage-I baseline.
2. B0 — Stage-I PIF.
3. A1 — Stage-II cRT from A0.
4. A2 — Stage-II H2T from A0.
5. B1 — Stage-II cRT from B0.
6. B2 — Stage-II PI-H2T from B0.

Seed 42, micro/effective batch 16, four workers. Stage-I and Stage-II use
the existing locked data schedules, transforms, objectives, optimizer and
learning-rate values. Smoke uses the first epoch of the formal schedule; it
does not rescale the learning rate to a one-epoch schedule. A0/B0 use the
same Stage-I order and MixUp random stream. A1/A2 and B1/B2 use matched
recipient views; A2/B2 use matched donor views.

Stage-II optimizes only the classifier. The extractor and all BatchNorm
parameters and buffers must remain unchanged. Every condition must complete
one full training epoch and validation. No condition may be silently skipped.
All attempted windows must have finite gradients and actual optimizer-step
counts must match attempted windows. No optimizer skip is allowed under FP32.

The Smoke runner accepts the current CPU validation and preflight JSON records
and verifies their branch, HEAD, status, test-split isolation, and source
fingerprints before any GPU work. Runtime outputs remain outside Git.

## Formal 90+10 pilot (not authorized by this Smoke task)

A0/B0 each start from the unchanged official initialization and run 90 complete
Stage-I epochs. A1/A2 use the same final epoch-90 A0 state; B1/B2 use the same
final epoch-90 B0 state. Stage-II budgets remain 10 epochs. Smoke-only
checkpoints and earlier Stage-I epochs are rejected as formal Stage-II sources.

An independent Stage-II invocation is supported only with an explicit final
source checkpoint and its SHA256. It does not authorize extra conditions.

For zero-based epoch `e`, Stage-I uses
`0.01*(e+1)/5` for e=0..4, then
`0.005*(1+cos(pi*(e-4)/85))` for e=5..89. The five warmup rates are
0.002, 0.004, 0.006, 0.008, 0.01; cosine ends at zero in epoch 90.
Stage-II uses `0.005*(1+cos(pi*e/10))` for e=0..9, beginning at 0.01.
Scheduling occurs once per epoch. The explicit LR tables are stored in the
run and condition manifests.

Stage-I weight decay is 1e-4; Stage-II is 5e-4. Stage-I uses the existing
two-target MixUp CE and alpha=1. Stage-II uses ordinary CE. The existing
sample-count multiplication/division of the objective is retained, including
the final partial batch. SGD momentum is 0.9. The micro/effective batch is 16.

Formal training requires a PASS CPU validation record produced by
`validate_cpu.py`, with matching source fingerprints for
`formal_train.py`, `models.py`, `data.py`, `amp_recovery.py`,
`preflight.py`, and `protocol_config.py`. It verifies the approved locked
upstream hashes and environment before training. It has no TEST evaluation
entry point.

## Failure and checkpoint policy

An OOM, mismatched pairing, invalid source, non-finite forward or rejected
gradient ends the run. The current model/optimizer/scaler, RNGs, failing
input/targets and gradients are saved when accessible. There is no automatic
altered-setting retry or continuation to other conditions.

Formal checkpoints preserve model, optimizer, no-op scaler state, scheduler,
Python/NumPy/CPU/CUDA RNG state, metric history and actual optimizer-update
totals. Resume verifies the source/protocol fingerprint and sequential epoch
cursor. An interrupted uncommitted epoch is explicitly reported and replayed
from its saved state; no completed epoch is silently restarted or skipped.

The clean-launch branch must pass CPU validation, preflight and the six
one-epoch Smoke conditions before any 90+10 formal run is considered. One
seed is not a cross-domain generalization claim; no TEST result is produced.
