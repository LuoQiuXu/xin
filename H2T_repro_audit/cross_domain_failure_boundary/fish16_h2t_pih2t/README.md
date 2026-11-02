# CURRENT EXECUTION STATUS — 2026-11-02

**Historical execution record: the first formal seed-42 pilot is FAILED_STOPPED. Do not resume the _01 run or continue on its original branch. The correction branch and its exact-snapshot replay gate below define the only next step.**

Execution code: c6805db4b1a806b2a7830a48d131f9d1acfa617e on experiment/fish16-formal-90x10-20261102. Two syntax checks and 34 CPU tests passed, then the formal runner started directly without another GPU smoke.

A0 completed 3 of 90 epochs, then stopped in epoch4 at zero-based batch976: one negative Inf at classifier.bias[0], scale262144, no NaN or missing gradients. This is outside the allowed B0-only PIF-b recovery signature. The failed window called no scaler.step/update; observed actual SGD calls=4027. There were zero native AMP skips.

B0, A1, A2, B1 and B2 were not started. The 90+10 pilot is incomplete, and neither A2-A1 nor B2-B1 can be calculated. Checkpoints, per-epoch metrics, partial batch journals and the complete failure snapshot remain outside Git. No TEST image or seed43/44 ran.

See [the formal run report](diagnostics/results/formal_seed42_90x10_20261102_01/FORMAL_RUN_REPORT.md). Runtime failure evidence and implementation/resume limits are explicit there. No numerical workaround, modified setting, extra diagnostic or automatic retry followed this failure.

The failure record above is immutable historical evidence. The corrected plan below supersedes the old stop instruction: run only the exact saved-batch AMP replay on the correction branch, and start a fresh formal run only if the expected scale-dependent failure is reproduced. Do not use the old partial checkpoint or old output directory for continuation.

---

# CURRENT CORRECTION PLAN — 2026-11-02

The first formal attempt remains preserved as `FAILED_STOPPED` on
`experiment/fish16-formal-90x10-20261102`. Its A0 failure occurred at epoch 4,
batch 976, with one negative Inf at `classifier.bias[0]` after GradScaler
automatically grew to 262,144. The loss and logits were finite; no optimizer
step was submitted for the failed window.

A correction branch has been created:
`experiment/fish16-amp-scale-stability-20261102`.

The formal runner now has a separate scaler policy: FP16 autocast remains
unchanged, initial scale remains 65,536, but the growth interval is 100,000
updates. This exceeds the maximum 91,530 attempted windows in a 90-epoch
Stage-I condition, preventing automatic scale escalation during a condition.
Native backoff remains active for the one previously diagnosed and permitted
B0 PIF-b overflow. The original `amp_recovery.py` and smoke scaler are
unchanged.

The runner also normalizes tuple/list representations when comparing restored
epoch records with JSON metric files, fixing a documented resume edge case.

**Next action:** run one no-optimizer-step replay of the exact saved A0 failure
snapshot at scales 65,536 and 262,144 (optionally 131,072 for the threshold
check). If the recorded signature is reproduced—finite gradients at 65,536
and the same classifier-bias Inf at 262,144—run the corrected six-condition
formal pilot from fresh initialization in a new output directory ending
`_02`. Do not resume the failed run or reuse its partial checkpoint. If the
exact replay does not reproduce, stop and report; do not guess another scale.

No new smoke, no additional mechanism audit, no TEST access, and no seed 43/44
are authorized. See `CODEX_FISH16_AMP_SCALE_STABILITY.txt` for the execution
instructions. The earlier failure report remains historical evidence and must
not be overwritten.

---

# Fish4Knowledge-16: H2T / PI-H2T six-condition smoke

Current implementation branch: `experiment/fish16-six-condition-smoke-20261101`.

## Purpose

The next and only authorized run is the **complete six-condition, one-epoch-per-condition real-data smoke**. It checks that the shared Stage-I representations and all four Stage-II paths can execute on the locked Fish4Knowledge-16 split. It is not a performance experiment and does not authorize formal training.

The research question remains whether H2T and PI-H2T head-to-tail transfer gains generalize from CIFAR long-tail recognition to a real underwater visual domain. No new loss, donor selector, channel selector, fusion rule, or backbone is being proposed in this stage.

## Locked data

- Fish4Knowledge official release: 23 classes, 27,370 images, 8,725 tracking trajectories.
- Primary exploratory subset: 16 classes, 27,133 images, 8,680 trajectories.
- Locked split, seed 42, trajectory-grouped: train 16,269 images / 5,208 groups; validation 5,396 / 1,736; test 5,468 / 1,736.
- Train-set imbalance factor: 253.17.
- Head / Medium / Tail are frequency tertiles of 6 / 5 / 5 classes in this 16-class subset. They are not the original CIFAR Many / Medium / Few thresholds.
- The remaining seven classes stay in the full audit but are outside this primary training scope.

The manifest is external to Git and is pinned in `preflight.py`. The image data, manifest, downloaded weights, checkpoints, and raw run outputs must remain outside Git.

## Six matched conditions

| ID | Stage-I representation | Stage-II | Purpose |
|---|---|---|---|
| A0 | Plain ResNet50 | none | Plain Stage-I reference |
| A1 | Plain ResNet50 | cRT | Matched no-fusion control for H2T |
| A2 | Plain ResNet50 | H2T channel replacement | H2T transfer |
| B0 | PIF-ResNet50 | none | PIF Stage-I reference |
| B1 | PIF-ResNet50 | cRT | Matched no-fusion control for PI-H2T |
| B2 | PIF-ResNet50 | PI-H2TF | PI-H2T transfer |

A2 must be compared with A1, and B2 with B1. Do not compare only A2 with B2 because PIF changes the Stage-I representation.

## Known numerical event and handling

The previous B0 smoke stopped because FP16 GradScaler scale 65,536 produced exactly one positive Inf at `extractor.pif.fuse.weight[0,1,0]`. The controlled same-batch diagnosis found finite forward outputs, no missing gradients or NaNs, finite gradients at lower scales, and verified that native GradScaler skips the optimizer update and backs off the scale to 32,768.

`amp_recovery.py` permits only that exact audited event during B0 Stage-I. It does not clamp, replace, clip, or silently discard gradients. Every other non-finite event, missing gradient, unexpected optimizer skip, or unverified scale transition is fatal.

At micro-batch 16, the archived deterministic replay must occur exactly once at zero-based batch 6 and be followed by a successful update at scale 32,768. If and only if CUDA OOM occurs, a fresh full run may use micro-batch 8 with accumulation 2. The batch-16 MixUp grouping is not identical at micro-batch 8, so no batch-6 overflow is required there; any overflow must still match the one audited PIF-b event and be followed by a real update.

## Run

Use the validated interpreter:

```powershell
$PY = "D:\app\minicondaaa\envs\xiaoju\python.exe"

& $PY -B -X utf8 -m unittest discover -s H2T_repro_audit\cross_domain_failure_boundary\fish16_h2t_pih2t -p "test_amp_recovery.py" -v
& $PY -B -X utf8 -m unittest discover -s H2T_repro_audit\cross_domain_failure_boundary\fish16_h2t_pih2t -p "test_contract.py" -v
& $PY -B -X utf8 -m unittest discover -s H2T_repro_audit\cross_domain_failure_boundary\fish16_h2t_pih2t\diagnostics -p "test_*.py" -v

& $PY -B -X utf8 -u H2T_repro_audit\cross_domain_failure_boundary\fish16_h2t_pih2t\smoke.py --output-dir "D:\h2t_cross_domain_results\fish16_six_condition_smoke_seed42_20261101_01" --micro-batch 16
```

All output directories must be new and outside the repository. If the first run fails specifically with CUDA OOM, preserve it and retry the entire six-condition smoke in a fresh process and a new output directory with `--micro-batch 8`. Do not retry other errors by changing settings.

## Output and interpretation

The external run directory contains the sampling and donor-pair audit, per-condition one-epoch checkpoints/results, and:

- `run_manifest.json`: locked inputs and intended smoke budgets (not proof of completion).
- `ACTUAL_EXECUTION_STATUS.json`: conditions actually completed, or the exact stopped state.
- `SMOKE_SUMMARY.json`: created only after all six conditions and matching checks pass.
- `FAILURE.json`: created if execution stops.

Validation metrics are smoke diagnostics only. No test images are evaluated. A passing smoke means only that the six paths execute under the locked protocol; it does not establish a method gain, a negative-transfer effect, or a paper result.

The current local PI-H2T adaptation uses the reviewed PIF / PI-H2TF implementation in `PI_H2T_review/H2T-main/`; it is not verified author training code. The paper specifies the two-parameter PIF formula but does not specify exact numeric initialization of those parameters; the local implementation retains PyTorch Conv1d default initialization. Stage-II freezes extractor parameters and BN statistics for all four conditions; this is a controlled comparison and differs from original H2T configurations that use `shift_bn=True`. These limitations must remain explicit in any report.

After the six-condition smoke is reviewed, the next possible experiment is the already specified seed-42 90-epoch Stage-I + 10-epoch Stage-II pilot. That formal pilot is **not authorized by this smoke instruction**. Do not start it, evaluate TEST, or start seeds 43/44 without a separate decision.
