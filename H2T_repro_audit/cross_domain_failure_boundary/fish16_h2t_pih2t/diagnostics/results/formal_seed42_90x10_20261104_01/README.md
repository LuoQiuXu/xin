# Fish16 formal Seed-42 results

Full 90+90+10+10+10+10 epochs completed. Start with [POSTRUN_ANALYSIS.md](POSTRUN_ANALYSIS.md) and [VALIDATION_REPORT.md](VALIDATION_REPORT.md).

- raw/: exact small execution records and per-condition metrics; OUTPUT_SHA256.csv indexes the untouched external run, including large artifacts not uploaded.
- analysis/: CPU-derived complete metric vectors, per-class changes, pairing/freeze checks, actual fusion summaries and PNG/PDF figures.
- derive_postrun.py: one-shot read-only CPU derivation locked to the recorded execution branch/HEAD and source fingerprints. Never starts a model forward or image loader; refuses to overwrite analysis/.
- finalize_report.py: report generation and exact small-file archive copying; refuses to overwrite outputs.
- TASK_SPEC.txt: exact user-supplied current task.

All original checkpoints, datasets, NPZ plans and batch logs remain outside Git. No training implementation changed. This controlled frozen-BN single-seed result is not a claim of full author-protocol reproduction or a universal cross-domain failure.
