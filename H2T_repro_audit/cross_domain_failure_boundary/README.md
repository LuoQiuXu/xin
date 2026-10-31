# Cross-domain H2T / PI-H2T failure-boundary study

This is the next outcome-first research branch after the completed CIFAR-100N-LT screen and crossed Stage-II diagnosis. The goal is to find a real external visual domain where H2T / PI-H2T benefits weaken or reverse, not to keep searching for arbitrary internal-module defects.

## Current iteration: data-readiness gate only

The first run is intentionally CPU-only. It validates the two candidate datasets, group identifiers, independent tail support, exact duplicates, class counts, and leakage-safe splits. It does not train a model.

Read in this order:

1. SCIENTIFIC_PROTOCOL.md
2. CODEX_CROSS_DOMAIN_DATA_AUDIT.txt
3. Run the synthetic contract tests and the audits.

## Candidate datasets

### ISIC 2019 (medical dermoscopy)

- Official training set: 25,331 labelled images, 8 diagnostic categories.
- Official metadata contains a common lesion identifier.
- License: CC-BY-NC; use only for non-commercial academic research.
- Official source: https://challenge.isic-archive.com/data/

Expected training class counts used for release-integrity checking:
MEL=4522, NV=12875, BCC=3323, AK=867, BKL=2624, DF=239, VASC=253, SCC=628.

### Fish4Knowledge FRGT (underwater fish species)

- Official release: 27,370 images, 23 species.
- The filename format is tracking_id_fish_id; same tracking ID means the same fish trajectory.
- The official release has 23 class counts and trajectory counts encoded in audit_external_domains.py.
- Official source: https://homepages.inf.ed.ac.uk/rbf/Fish4Knowledge/GROUNDTRUTH/RECOG/
- Keep the copyright notice and cite Boom et al. (ICPR 2012).

## Run

PowerShell commands:

    python -m unittest discover -s H2T_repro_audit/cross_domain_failure_boundary -p "test_*.py" -v

    python H2T_repro_audit/cross_domain_failure_boundary/audit_external_domains.py --dataset isic2019 --images-root "D:\datasets\ISIC2019\Training_Input" --metadata-csv "D:\datasets\ISIC2019\Training_Metadata.csv" --ground-truth-csv "D:\datasets\ISIC2019\Training_GroundTruth.csv" --output-dir "D:\h2t_cross_domain_audit\isic2019"

    python H2T_repro_audit/cross_domain_failure_boundary/audit_external_domains.py --dataset fish4knowledge --images-root "D:\datasets\Fish4Knowledge\fish_images" --output-dir "D:\h2t_cross_domain_audit\fish4knowledge"

The script writes:
- audit_summary.json: PASS / PASS_LIMITED_SCOPE / NO-GO decision and reasons.
- audit_issues.json: missing images, invalid rows, missing lesion IDs, etc.
- class_counts.csv: image counts and independent group counts for every class, with primary split support.
- manifest_all.csv: all valid images and group-aware descriptive splits.
- manifest_primary.csv: only classes that pass the predeclared independent-group minimum.

Outputs should remain outside the Git repository; do not commit datasets, image hashes, manifests, checkpoints, or local audit results.

## Interpretation

A passing dataset is only eligible for the next training iteration. It is not evidence that H2T or PI-H2T fails. A failure to meet group support is a dataset/protocol NO-GO, not a method failure.

Do not start a backbone comparison yet. Keep one common architecture fixed when the later H2T / PI-H2T training begins so that domain and architecture effects are not confounded.

## Local execution on 2026-10-31

See [CROSS_DOMAIN_DATA_AUDIT_REPORT.md](CROSS_DOMAIN_DATA_AUDIT_REPORT.md) and
[DATASET_DIRECTORY_INSPECTION.md](DATASET_DIRECTORY_INSPECTION.md).
The branch was kept unchanged and no training, dataset download, commit, or push
was performed. CPU tests passed 11/11; all three Python tools passed compilation.

New local audit outputs are outside the repository at
`D:\h2t_cross_domain_audit\audit_20261031_01\`.
Fish4Knowledge passed only for the predeclared 16-class primary subset (the
complete 23-class manifest is retained). ISIC remains NO-GO for training because
both required ISIC CSV files were absent in a complete recursive inspection of
`data`; the only CSV found belongs to Office-Home. Its empty failure manifests
are diagnostics, not training inputs. All data remain in their original paths.

Audit output directories must now be **new, nonexistent paths outside both the
repository and the source image tree**; existing outputs are rejected. Failed
audits persist an explicit error and unknown measurements before exiting.
Pixel decoding uses both `verify()` and `load()`; raw trajectory counts are
checked separately from duplicate-merged groups. Split ratios and minimum group
support are unchanged. An ISIC image-ID fallback is explicitly marked and does
not verify independence of missing lesion identifiers.

The directory inspection can be reproduced with new output filenames:

    D:\app\minicondaaa\envs\xiaoju\python.exe -B -X utf8 H2T_repro_audit/cross_domain_failure_boundary/inspect_dataset_directory.py --data-root D:\xiangmuxunlian\xin\data --inspection-date 2026-10-31 --output-json D:\h2t_cross_domain_audit\NEW_INSPECTION.json --output-report H2T_repro_audit/cross_domain_failure_boundary/NEW_INSPECTION.md

This command only reads the datasets; its JSON and Markdown reports are created
exclusively and must be outside the dataset directory.

## Formal audit after CSV supply (2026-10-31)

The earlier reports above remain snapshots from before the three CSV files were
supplied. See the new
[formal audit report](results/formal_data_audit_20261031_01/CROSS_DOMAIN_DATA_AUDIT_REPORT.md)
and [validation](results/formal_data_audit_20261031_01/VALIDATION_REPORT.md).

All 15 CPU tests and syntax checks passed. ISIC decoded and hashed all 25,331
images, but stopped with `NO_GO_DATA_INTEGRITY`: two byte-identical pairs have
conflicting MEL/NV labels and connect two lesion IDs. No ISIC splits were
created. Its original failed attempt is retained; actual per-class measurements
are supplemented in `ISIC_INTEGRITY_DIAGNOSIS.json` and the new report tables.
The upstream origin of the local conflict remains unverified.

Fish4Knowledge passed as `PASS_LIMITED_SCOPE` for 16 classes. The supplied
`class_id.csv` has headers `file_name,cluster_id`; all 27,370 rows matched both
decimal tracking/fish IDs and class folders. Numeric leading-zero normalization
is used only for mapping comparison; original tracking IDs and split rules
remain unchanged. The optional `--class-id-csv` flag enables this strict gate.

Full new outputs are in `D:\h2t_cross_domain_audit\isic2019\` and
`D:\h2t_cross_domain_audit\fish4knowledge\`; logs, conflict proofs and provenance
are in `D:\h2t_cross_domain_audit\formal_data_audit_20261031_01\`.
Old reports and `audit_20261031_01` outputs are preserved. No GPU training,
dataset download, Git commit/push or branch change was performed.
