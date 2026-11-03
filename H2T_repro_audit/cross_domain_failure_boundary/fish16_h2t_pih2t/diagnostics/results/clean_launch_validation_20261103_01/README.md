# Fish16 clean-launch validation archive

Actual status: **STOPPED_CPU_IMPORT_GATE**. No GPU Smoke or formal training ran.

Execution branch: `experiment/fish16-clean-launch-20261103`. Execution HEAD: `9f1a7ac8373f087274b6f919b048c45a14b46e2e`.

The11 required syntax checks passed. The CPU command passed36 loaded test methods, then reported one module-import error for `diagnostics.test_diagnostic_stats`; its five diagnostic methods were not loaded. The recorded package-style command could not resolve the top-level `diagnose_b0` import. This is not an assertion result for those five methods. No source/environment patch, alternate test invocation or retry followed the failed gate.

A0/B0/A1/A2/B1/B2 each completed0/1 Smoke epochs and made zero training updates. Full preflight, GPU pairing checks, validation metrics and GPU BN/freeze verification were not run.

- [Full report and exact commands](CLEAN_LAUNCH_VALIDATION_REPORT.md)
- [Structured execution result](EXECUTION_RESULT.json)
- [Actual status](ACTUAL_EXECUTION_STATUS.json)
- [Complete import traceback](FAILURE.json)
- [Source fingerprints before](EXECUTION_IDENTITY_BEFORE.json) and [after](SOURCE_FINGERPRINTS_AFTER.json)
- [External output SHA256 index](OUTPUT_SHA256.csv)

These11 JSON/CSV/Markdown artifacts are byte-for-byte copies from `D:\h2t_cross_domain_results\fish16_clean_launch_validation_20261103_01`. The output-index paths are relative to that external directory, and its23 indexed outputs include raw CPU/syntax logs and bytecode caches that remain local. No datasets, weights, checkpoints or raw logs were added to Git. The original external artifacts remain unchanged.
