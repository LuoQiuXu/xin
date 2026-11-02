# B0 PIF Stage-I gradient diagnosis

Historical diagnosis, seed 42, micro/effective batch 16, official
`resnet50-19c8e357.pth` initialization. The original forensic report and
captured evidence remain immutable. A subsequent review identified that the
recovery gate incorrectly distinguished +Inf from -Inf at the same PIF-b
coordinate; the source-level correction is documented below.

The controlled replay fails at zero-based batch 6. Loss scaling by 65536
overflows the FP16 Conv1d gradient for the shared PIF coefficient b. All
backbone/classifier gradients are finite and present. Four lower diagnostic
AMP scales and FP32 pass on identical pre-forward model/BN/optimizer state
and the exact same mixed input. This establishes a local numerical failure;
it does not establish a PI-H2T method accuracy defect.

The scientific review starts at
[PIF_GRAD_DIAG_REPORT.md](results/b0_grad_diag_20261031_01/PIF_GRAD_DIAG_REPORT.md).
See the [archive notes](results/b0_grad_diag_20261031_01/README.md) for evidence
coverage, source-byte hashes, and the distinction between pre-publication
validation snapshots and this later Git publication.

## Source and execution scope

- `diagnose_b0.py`: bounded B0 replay, named parameter gradients, forward
  ranges, pre-forward snapshot, six fresh same-batch diagnostic clones, and
  one disposable normal GradScaler skip/update check.
- `summarize_b0.py`: CPU-only summary from saved diagnostics; refuses to
  overwrite its report/CSV outputs. It needs the externally retained tensors.
- `test_diagnostic_stats.py`: five CPU tests covering finite statistics,
  None versus zero gradients, NaN/Inf reporting, grouping and snapshot copies.

CPU verification from the repository root:

```powershell
& 'D:\app\minicondaaa\envs\xiaoju\python.exe' -B -X utf8 -m unittest discover -s H2T_repro_audit/cross_domain_failure_boundary/fish16_h2t_pih2t/diagnostics -p 'test_*.py' -v
& 'D:\app\minicondaaa\envs\xiaoju\python.exe' -B -X utf8 -m unittest discover -s H2T_repro_audit/cross_domain_failure_boundary/fish16_h2t_pih2t -p 'test_contract.py' -v
```

The completed diagnostic logs record 5/5 new and 10/10 existing CPU tests
passing, plus syntax checks for all three new Python files. No additional
GPU computation was performed for publication.

Do not interpret the diagnostic script as approval to resume smoke or formal
training. Its invocation requires a fresh external directory. Scientific
review of the AMP nonfinite handling contract is pending; the actual smoke
gate is unchanged. No Stage-II epochs, TEST evaluation, seed 43/44, or formal
training were executed in this diagnosis.

All snapshots, weights, complete gradient tensors, paper page renders and
original outputs remain under
`D:\h2t_cross_domain_results\pif_grad_diagnosis_seed42_20261031_01\`.
The existing repository `.gitignore` excludes model tensors, raw logs and
Python caches.


## 2026-11-02 AMP recovery-contract correction

The 2026-11-01 forensic report remains the historical record of the observed
negative Inf at `extractor.pif.fuse.weight[0,1,0]`, batch 13, scale 32768.
The correction does **not** claim that the negative Inf's first internal
producer has been isolated. It corrects only the recovery contract:

- one and only one Inf (either sign) at the exact PIF-b parameter coordinate
  may be passed to an enabled native GradScaler skip/backoff path;
- NaN, missing gradients, any other non-finite parameter/coordinate, multiple
  Inf values, disabled GradScaler, and exceeded consecutive-skip limits remain
  fail-closed;
- the guard still verifies that the optimizer did not execute, model
  parameters did not change, and GradScaler reduced its scale;
- the full B0 Stage-I one-epoch runner is separately selectable with
  `--b0-stage1-only`; it skips A0, all Stage-II conditions, and TEST.

The smoke invariant now preserves the historically observed first overflow at
batch 6 while allowing further individually audited PIF-b signed overflow
events within the existing total/consecutive skip limits. The historical
forensic report is not rewritten.

Codex execution instructions:
[`CODEX_B0_AMP_RECOVERY_VALIDATION.txt`](../CODEX_B0_AMP_RECOVERY_VALIDATION.txt).
