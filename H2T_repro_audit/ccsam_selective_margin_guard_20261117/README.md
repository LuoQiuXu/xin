# CC-SAM Selective Many-to-Tail Margin Guard

This branch adds one targeted candidate to the completed CC-SAM D0/D1/D2 cross-dataset study.

- D0/D1/D2: reused, frozen baselines from `experiment/ccsam-cross-dataset-formal-20261112`.
- D3: original D1 objective plus a selective teacher-relative Many-to-non-Many margin guard.
- Datasets: ImageNet-LT and Places-LT.
- Seeds: 123, 124, 125.
- New formal work: six D3 classifier runs only. Stage-I and TRAIN feature caches are reused.

Read `PROTOCOL.md` and `CODEX_SELECTIVE_MARGIN_GUARD_20261117.txt` before running.

The pure loss implementation and CPU tests are already present:

- `margin_guard_core.py`
- `test_margin_guard_core.py`

The Codex task wires the module into the existing paired Stage-II pipeline, adds strict provenance and no-TEST gates, performs Smoke, runs the six D3 jobs, and evaluates/analyzes fixed checkpoints once.

No claim is made that SMG improves accuracy until the experiment completes.
