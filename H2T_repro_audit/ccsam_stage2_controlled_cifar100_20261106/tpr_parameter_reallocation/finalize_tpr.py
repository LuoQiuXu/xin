"""Create the final TPR artifact index without uploading local-only weights/traces."""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path

from ccsam_common import SEEDS, read_json, sha256_file, STAGE2_EPOCHS

TPR_DIR = Path(__file__).resolve().parent
ARM = "CCSAM_TPR"


def retention_for(path: Path) -> str:
    if path.name in {"final_classifier.pth", "batch_trace.csv"}:
        return "local_only"
    if path.suffix.lower() in {".log", ".pth", ".pt", ".npz", ".npy"}:
        return "local_only"
    return "github"


def main():
    completion = TPR_DIR / "formal_completed.json"
    analysis = TPR_DIR / "analysis" / "summary.json"
    if not completion.is_file() or not analysis.is_file():
        raise FileNotFoundError("TPR formal completion and fixed-epoch analysis are required")
    marker = read_json(completion)
    summary = read_json(analysis)
    if (
        marker.get("passed") is not True
        or marker.get("arm") != ARM
        or marker.get("seeds") != list(SEEDS)
        or marker.get("epochs_per_seed") != STAGE2_EPOCHS
        or marker.get("official_TEST_used") is not False
        or summary.get("arm") != ARM
        or summary.get("official_TEST_used") is not False
    ):
        raise RuntimeError("TPR completion/analysis provenance is invalid")

    for seed in SEEDS:
        run = TPR_DIR / "results" / f"seed_{seed}" / ARM
        status_path = run / "status.json"
        checkpoint_path = run / "final_classifier.pth"
        trace_path = run / "batch_trace.csv"
        if not all(path.is_file() for path in (status_path, checkpoint_path, trace_path, run / "metrics.csv")):
            raise FileNotFoundError(f"Incomplete TPR run artifacts for seed {seed}")
        status = read_json(status_path)
        if (
            status.get("state") != "completed"
            or status.get("seed") != seed
            or status.get("arm") != ARM
            or status.get("final_epoch") != STAGE2_EPOCHS
            or status.get("official_TEST_used") is not False
            or sha256_file(checkpoint_path) != status.get("final_classifier_sha256")
        ):
            raise RuntimeError(f"TPR run validation failed for seed {seed}")

    index_path = TPR_DIR / "ARTIFACT_SHA256.csv"
    if index_path.exists():
        raise FileExistsError("Refusing to overwrite existing TPR artifact index")
    rows = []
    for path in sorted(TPR_DIR.rglob("*")):
        if not path.is_file() or path == index_path:
            continue
        if "__pycache__" in path.parts or path.suffix == ".pyc" or path.name.endswith(".writing"):
            continue
        relative = path.relative_to(TPR_DIR).as_posix()
        rows.append({
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "retention": retention_for(path),
            "github_blob_sha256": "",
        })
    if not rows:
        raise RuntimeError("No TPR artifacts found for indexing")
    with index_path.open("x", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=["path", "bytes", "sha256", "retention", "github_blob_sha256"],
        )
        writer.writeheader()
        writer.writerows(rows)
    local_only = sum(row["retention"] == "local_only" for row in rows)
    print(
        f"TPR artifact index created: {len(rows)} files "
        f"({len(rows) - local_only} GitHub-retained, {local_only} local-only)."
    )
    print("Do not force-add local-only checkpoints, traces, or logs to Git.")


if __name__ == "__main__":
    main()
