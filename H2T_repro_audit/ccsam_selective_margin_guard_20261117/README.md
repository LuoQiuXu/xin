# CC-SAM Selective Many-to-Tail Margin Guard

This branch adds one targeted candidate to the completed CC-SAM D0/D1/D2 cross-dataset study.

- D0/D1/D2: reused, frozen baselines from `experiment/ccsam-cross-dataset-formal-20261112`.
- D3: original D1 objective plus a selective teacher-relative Many-to-non-Many margin guard.
- Datasets: ImageNet-LT and Places-LT.
- Seeds: 123, 124, 125.
- New formal work: six D3 classifier runs only. Stage-I and TRAIN feature caches are reused.

Read `PROTOCOL.md`, `smg_run_protocol.json`, and `CODEX_EXECUTE_SELECTIVE_MARGIN_GUARD_20261117.txt` before running.

The implementation is now present:

- `margin_guard_core.py`: pure selective-margin loss and diagnostics.
- `train_margin_guard.py`: read-only baseline preflight, paired TRAIN Smoke, six D3 jobs, protected-artifact verification, separate fixed-checkpoint evaluation and analysis.
- `smg_run_protocol.json`: machine-readable frozen method and execution settings.
- `test_margin_guard_core.py` and `test_margin_guard_integration.py`: loss semantics and D3 objective integration tests.
- `CODEX_EXECUTE_SELECTIVE_MARGIN_GUARD_20261117.txt`: Codex validation and execution instructions.

The D3 training uses the unchanged D1 objective plus the fixed SMG term. It does not retrain Stage I, re-extract features, regenerate sampling plans, modify D0/D1/D2, or access official TEST before the dedicated evaluation phase.

No claim is made that SMG improves accuracy until the experiment completes.
