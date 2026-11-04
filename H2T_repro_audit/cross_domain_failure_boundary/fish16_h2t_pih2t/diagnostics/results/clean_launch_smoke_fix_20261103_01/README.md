# Clean-launch Smoke evidence

**PASS_SMOKE_ONLY: six conditions completed one epoch each; formal training did not start.**

- [Complete report and commands](CLEAN_LAUNCH_REPORT.md)
- [Actual condition results](CONDITION_RESULTS.csv)
- [Consolidated evidence](EXECUTION_RESULT.json)
- [Pairing and frozen-state checks](PAIRING_CHECKS.json)
- [Startup repairs and provenance](STARTUP_FIXES.md)
- [Original actual execution status](ACTUAL_EXECUTION_STATUS.json)
- [External checkpoint paths and SHA256](CHECKPOINT_SHA256.csv)

Execution HEAD is960b1be97058202ee97db88f97a424bf1e6c37fb, distinct from the later report/archive commit. The auxiliary raw batch_smoke field retains RUNNING although its four successful proofs and all six completed epochs are present; the report discloses this stale metadata field without rewriting execution records.

All source data and checkpoints remain in the three external directories from the task. The archive contains requested compact evidence, including the complete small CPU log, original per-batch hashes and optimizer events. Full runtime files are covered byOUTPUT_SHA256.csv; root labels specify which directory each relative path uses.
