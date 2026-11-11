# Cross-dataset data audit: ImageNet-LT + Places-LT

This directory adds a read-only local data audit for the two datasets prepared by the user.

## Scope

- ImageNet-LT is audited from the downloaded Hugging Face Parquet shards.
- Places-LT is audited from the extracted image folder and the repository's official annotation manifests:
  - H2T/datasets/data_txt/Places_LT_train.txt
  - H2T/datasets/data_txt/Places_LT_test.txt
- The repository's ImageNet-LT official TXT manifests are used as class-distribution references:
  - H2T/datasets/data_txt/ImageNet_LT_train.txt
  - H2T/datasets/data_txt/ImageNet_LT_test.txt
- No training code, CC-SAM method, C0/C1/C2/C3/C4 logic, model, optimizer, checkpoint, or dataset file is modified.
- The script does not download, extract, move, rename, or delete dataset files.

## Reference protocol encoded in the audit

The counts below were computed from the checked-in official annotation lists on the parent experiment branch:

| Dataset | Split | Images | Classes | Class count |
|---|---|---:|---:|---:|
| ImageNet-LT | train | 115,846 | 1,000 | 5–1,280 |
| ImageNet-LT | test | 50,000 | 1,000 | exactly 50/class |
| Places-LT | train | 62,500 | 365 | 5–4,980 |
| Places-LT | test | 36,500 | 365 | exactly 100/class |

The Hugging Face ImageNet-LT repository contains three splits and the local download plan used 25 train, 5 validation, and 15 test Parquet shards. Validation is recorded separately; it must not be silently relabeled as the official test set.

## Dependencies

The script uses Python standard library, PyArrow, and Pillow. It does not require datasets, torch, or CUDA.

Install only if missing in the active conda environment:

    python -m pip install pyarrow pillow

## Run on Windows PowerShell

    conda activate 小菊

    python .\H2T_repro_audit\cross_dataset_data_audit_20261111\audit_cross_dataset_inputs.py --imagenet-root "D:\datasets\ImageNet-LT" --places-root "D:\datasets\places-lt" --repo-root "." --output-dir ".\H2T_repro_audit\cross_dataset_data_audit_20261111\results"

Run the unit tests:

    python -m unittest discover -s .\H2T_repro_audit\cross_dataset_data_audit_20261111 -p "test_*.py" -v

## Outputs

The run writes only small audit artifacts to results/:

- audit_report.md: human-readable summary and warnings.
- audit_report.json: full machine-readable details, shard counts, schema, labels, class counts, and path checks.
- imagenet_lt_class_counts.csv: class counts for train/validation/test.
- places_lt_class_counts.csv: class counts for train/test.
- ARTIFACT_SHA256.csv: hashes for the generated report artifacts.

## Interpretation

A PASS means the checks implemented here passed. It does not certify that every embedded ImageNet-LT image was decoded: Parquet labels are scanned for every row, while image decoding uses a deterministic sample from selected shards. Places-LT annotation paths are checked against local files for every listed row, while image decoding uses a deterministic sample.

If ImageNet-LT's label column is not recognized or PyArrow is unavailable, the script records a BLOCKED result and prints the exact schema/dependency issue. Do not convert or rewrite the Parquet files to work around an audit failure. Resolve the reported dependency or schema issue first.

A successful data audit is a data-readiness gate only. It does not start or authorize model training, choose a backbone, define the cross-dataset training protocol, or establish compatibility with the current CIFAR CC-SAM training code.
