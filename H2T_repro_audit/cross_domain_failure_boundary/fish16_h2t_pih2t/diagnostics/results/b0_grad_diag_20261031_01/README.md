# Completed B0 diagnostic evidence archive

This is a small, read-only publication copy of the completed external run:

`D:\h2t_cross_domain_results\pif_grad_diagnosis_seed42_20261031_01\`

It was packaged after the user explicitly requested a local commit and push.
Packaging performs no GPU computation or training and does not overwrite the
external outputs or historical experimental results.

## Evidence navigation

- [Scientific report](PIF_GRAD_DIAG_REPORT.md): established root cause,
  controlled comparisons, paper/local implementation distinction and limits.
- [Six-condition comparison](SCALE_COMPARISON.csv): AMP 65536, 32768, 16384,
  8192, 4096 and FP32, all on the same failing batch.
- [Root-cause evidence](ROOT_CAUSE_EVIDENCE.json) and
  [complete diagnostic summaries](DIAGNOSIS_RESULTS.json).
- [All named parameter gradients](ALL_PARAMETER_GRADIENTS.csv): all 4,536
  rows from seven replay batches and seven same-batch diagnostic clones,
  before/after unscale or the corresponding FP32 phases.
- [Gradient groups](GRADIENT_GROUP_SUMMARY.csv): PIF, classifier, stem and
  backbone layers 1–4, with missing and nonfinite gradients distinguished.
- [Failing batch](FAILING_BATCH.json), [sample/track pairing](FAILING_SAMPLES.csv)
  and [actual batch log](batch_log.csv).
- `reproduction/`, `amp_*/`, `fp32/`, `normal_scaler_disposable/`: per-batch
  forward ranges, hashes, backward summaries, groups and activation-gradient
  statistics. The normal-scaler clone alone calls step/update, proves that
  SGD was skipped and the scale halved, and is then discarded.
- [Paper formula and initialization audit](PAPER_FORMULA_INITIALIZATION_AUDIT.json)
  and [evidence file/line locators](EVIDENCE_LOCATORS.json). Exact author PIF
  initialization remains UNVERIFIED.
- [Final pre-publication validation](FINAL_VALIDATION.json),
  [protected-file baseline](PROTECTED_BEFORE.json), CPU test transcripts,
  and `DIAGNOSTIC_RUN.txt` (byte-identical copy of the small external `.log`).

## Provenance and retained local files

[PUBLICATION_MANIFEST.json](PUBLICATION_MANIFEST.json) maps all 88 copied files
to their original relative paths, byte counts and SHA256 values. Copies were
verified against the original files before staging. Git text normalization
may change CSV line endings on another checkout; source-byte hashes describe
the original external files.

[OUTPUT_SHA256.csv](OUTPUT_SHA256.csv) indexes 125 external files and excludes
the index itself. It includes the local-only 208 MB pre-forward snapshot,
complete finite-gradient tensor files, two-element PIF gradient tensors and
paper page renders. These binary artifacts are not committed. Per-case
`parameter_gradients.csv` files are also omitted as duplicates: every row is
included in `ALL_PARAMETER_GRADIENTS.csv`.

The original report and `FINAL_VALIDATION.json` describe diagnostic execution
before this publication commit. Their HEAD, clean/staged worktree and
`committed_or_pushed=false` fields intentionally retain that original
point-in-time meaning. They are not the post-publication Git status. This
archive does not alter old `FAILED_SMOKE_STOPPED` records or claim that the
original full B0 smoke has passed.

The diagnosed replay stops at batch 6 without updating the failed batch. The
old original B0 log did not record its batch index. Subsequent B0 stability
and any accuracy conclusion remain untested. Stop for scientific review;
there is no authorization here to modify the finite-gradient gate or start
formal training.
