# Git delivery

Date: 2026-10-31 (Asia/Shanghai).
Branch: `experiment/h2t-cross-domain-data-gate-20261030`.

The user separately authorized a local commit and GitHub push after the CPU
audit finished. Statements about no commit/push and the unchanged starting HEAD
in the audit reports describe the audit phase at
`e65f9a9d42f42fd5cbbd7b0841f77e9fe1981fe7`.
Those reports and all scientific results are preserved.

Delivery contains only audit tools, reports and small summary tables, including
the earlier directory-inspection report. It does not contain datasets, detailed
image manifests, raw image hash banks, large logs, checkpoints or Python caches.
Existing ignore rules cover the local datasets and caches; detailed outputs
remain outside the repository under `D:\h2t_cross_domain_audit\`.

`DELIVERY_INDEX.csv` preserves checksums of the original local files before Git
staging. Some generated text uses CRLF, while this checkout has
`core.autocrlf=true`. Consequently, Git normalizes text to LF in its blobs.
`GIT_DELIVERY_INDEX.csv` records both the original local SHA256 and the SHA256
after CRLF-to-LF normalization for all delivery files except itself. The Git
blob bytes are checked against the LF column before committing.

Validation remains 15/15 CPU tests passed and five Python syntax checks passed.
No additional dataset audit, model training or scientific changes were performed
for Git delivery. ISIC remains `NO_GO_DATA_INTEGRITY`; Fish4Knowledge remains
`PASS_LIMITED_SCOPE` for 16 classes.
