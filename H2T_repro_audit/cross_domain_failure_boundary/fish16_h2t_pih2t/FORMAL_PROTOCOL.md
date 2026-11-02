# Fish16 seed-42 formal 90+10 pilot

Original authorization: `CODEX_FISH16_FORMAL_90X10_TRAINING.txt`.
Numerical-stability correction: `CODEX_FISH16_AMP_SCALE_STABILITY.txt` on
`experiment/fish16-amp-scale-stability-20261102`. The failed first run is
preserved unchanged; the corrected run uses a new output directory. This runner
has no smoke mode and does not call the smoke runner.

`formal_train.py` defaults to A0, B0, A1, A2, B1, B2, in that order. A0/B0
each start from the unchanged official 19c8e357 adapter initialization and run
90 complete epochs. Stage-II sources are immutable copies of the corresponding
formal epoch-90 checkpoint; smoke-only checkpoints and earlier epochs are
rejected. A1/A2 use the same A0 state and B1/B2 use the same B0 state.

An independent Stage-II invocation is supported by `--condition A1` (or
A2/B1/B2), `--source-checkpoint` and the required `--source-sha256`. Budgets
remain fixed at 10 epochs. This interface is not used to add any extra run to
the authorized six-condition pilot.

## Epoch learning rates

For zero-based epoch `e`, Stage-I uses `0.01*(e+1)/5` for e=0..4, then
`0.005*(1+cos(pi*(e-4)/85))` for e=5..89. Thus the five warmup rates are
0.002, 0.004, 0.006, 0.008, 0.01; cosine decreases over the remaining epochs,
ending at zero in epoch 90. Stage-II uses
`0.005*(1+cos(pi*e/10))` for e=0..9, following the original zero-based
10-epoch cosine convention, with first rate 0.01. The final Stage-II rate is
approximately 0.0002447174. Scheduling occurs once per epoch, never per batch.
The entire explicit LR table is stored in the run and condition manifests.

Both stages use native FP16 autocast/GradScaler, micro/effective batch 16,
one optimizer window per batch, four workers and SGD momentum 0.9. The formal
runner uses init_scale=65,536, growth_factor=2.0, backoff_factor=0.5 and
growth_interval=100,000. This exceeds the maximum 91,530 attempted windows in
a 90-epoch Stage-I condition, so automatic scale growth cannot occur during a
condition. Native backoff remains enabled: if the permitted B0 PIF-b Inf occurs,
GradScaler must skip the update and lower the scale to 32,768; the scale then
remains at that value for the rest of that condition. The generic
amp_recovery.make_grad_scaler() and smoke-runner policy are unchanged.

Stage-I weight decay is 1e-4; Stage-II is 5e-4. Stage-I reuses the exact existing
two-target MixUp CE and alpha1; Stage-II reuses ordinary CE. The original
sample-count multiplication/division of the scaled objective is retained,
including the final partial batch. Only classifier parameters are optimized
in Stage-II; extractor parameters and all BN statistics remain frozen.

## Pairing, data access and numerical failures

The original `data.schedules` and `Views` are reused. All Stage-I conditions
use the same fixed instance permutation and per-epoch/position views; all
Stage-II conditions share the class-balanced recipient schedule, and A2/B2
share the donor schedule. Run seed42 is reset before each condition. Every
epoch records recipient/MixUp/donor control hashes and checks the available
paired reference. TEST indices are rejected before any image access, both
when constructing a dataset and in its image-loading method.

`amp_recovery.py` is imported unchanged. Only B0 may recover the single Inf
of either sign at PIF b [0,1,0], with zero NaNs and no missing gradients. Native
GradScaler must skip SGD and lower scale without changing parameters. The
next optimizer window must make a real update at that scale; the requirement
can cross an epoch boundary, and an unresolved final skip is fatal. The
existing consecutive-skip bound is retained. Later formal epochs have no
hard-coded batch locations or total skip count taken from the old smoke.
The full signed PIF event and native actual-step counter are logged.

An OOM, mismatched pairing, invalid source, nonfinite forward or rejected
gradient ends the entire run with `FAILURE.json`. The current model/optimizer/
scaler, RNGs, failing input/targets and gradients are saved when accessible.
There is no automatic altered-setting retry or continuation to other conditions.

## Checkpoints, exact state restoration and interruption semantics

Each condition writes an initial epoch-zero resume checkpoint, then an atomic
`last.pth` after every committed epoch, including model, optimizer, scaler,
epoch scheduler, complete Python/NumPy/CPU/CUDA RNG states, metric history,
actual-update totals and any pending overflow recovery. Checkpoint data is
compatible with `weights_only=True`. `epoch_090.pth` or `epoch_010.pth` is the
separate immutable final source/result. `best_validation.pth` is selected
strictly by validation overall, ties retaining the first epoch. It is never
substituted for the fixed epoch-90 Stage-II source.

`--resume` restores the authoritative atomic committed-epoch checkpoint,
verifies the protocol/source/environment fingerprint and sequential epoch
cursor, and does not restart or skip any completed epoch. If interruption
happened inside an uncommitted epoch, that partial attempt is explicitly
reported and replayed from the saved state. Its old JSONL is retained and a
new exclusive attempt log is created. This is committed-epoch resumption,
not a claim to checkpoint the last in-flight GPU instruction. All physical
calls, including rolled-back attempts, remain in raw attempt logs; aggregate
epoch counts describe the committed checkpoint lineage. No automatic resume
is attempted after a true runtime failure.

All runtime files remain outside Git. Summary comparisons emphasize final
A2-A1 and B2-B1; best-validation metrics are separate reporting information.
One seed does not establish generalization, and no TEST result is produced.

## Recorded execution limit and numerical correction

The first attempt stopped at A0 epoch4/batch976 because of one negative Inf at
classifier.bias[0] when the default GradScaler had grown from 65,536 to 131,072
to 262,144 at approximately 2,000-update intervals. The loss and logits were
finite, and the failed update was not submitted to the optimizer. See the
compact formal report. The correction retains FP16 autocast and the original
65,536 initial scale, but raises the growth interval above the maximum number
of attempted updates in a 90-epoch condition. It does not alter the optimizer,
learning rate, batch size, model, loss, data or transfer methods. A0 is
restarted from the locked initialization in a new run directory so all 90
epochs use one numerical policy; the prior partial run is retained as evidence.
The exact failed A0 batch must be replayed from the preserved failure snapshot
at scales 65,536, 131,072 and 262,144 without calling optimizer.step(), before
the corrected formal run begins. Expected evidence is finite gradients at
65,536 and the recorded classifier.bias[0] Inf at 262,144; record the actual
131,072 result without assuming it. If the expected boundary does not
reproduce, stop rather than guessing another setting.

The first implementation also compared checkpoint records containing tuples
directly with JSON records containing lists. The corrected runner normalizes
both representations before comparison. This CPU-tested code fix does not
establish GPU resume correctness by itself. The failed _01 run and all its
checkpoints remain preserved; the corrected run starts fresh in a new _02
output directory so one AMP policy is used throughout all 90 epochs.
