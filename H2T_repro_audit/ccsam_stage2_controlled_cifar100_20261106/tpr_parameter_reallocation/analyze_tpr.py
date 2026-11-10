"""Fixed epoch-30 paired analysis for CC-SAM-TPR against sealed C0-C4."""
from __future__ import annotations

import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

TPR_DIR = Path(__file__).resolve().parent
EXPERIMENT_ROOT = TPR_DIR.parents[0]
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from ccsam_common import ROOT, SEEDS, STAGE2_EPOCHS, read_json, sha256_file  # noqa: E402
from train_tpr import TPR_ARM, C2_ARM, C4_ARM, compare_protected  # noqa: E402

ANALYSIS_DIR = TPR_DIR / "analysis"
ARMS = (
    "C0_BALANCED_CE",
    "C1_CCSAM_FEATURE_GRAD",
    C2_ARM,
    "C3_NO_MANY_PERTURB",
    C4_ARM,
    TPR_ARM,
)
METRICS = (
    "Overall", "Macro", "Many", "MediumN", "Few",
    "Head", "OfficialMedium", "Tail",
)
COMPARISONS = (
    ("TPR_minus_C1", TPR_ARM, "C1_CCSAM_FEATURE_GRAD"),
    ("TPR_minus_C2", TPR_ARM, C2_ARM),
    ("TPR_minus_C3", TPR_ARM, "C3_NO_MANY_PERTURB"),
    ("TPR_minus_C4", TPR_ARM, C4_ARM),
    ("TPR_minus_C0", TPR_ARM, "C0_BALANCED_CE"),
    ("C1_minus_C2_reference", "C1_CCSAM_FEATURE_GRAD", C2_ARM),
    ("C4_minus_C2_reference", C4_ARM, C2_ARM),
)


def stamp():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def exclusive_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as fp:
        json.dump(value, fp, ensure_ascii=False, indent=2, allow_nan=False)
        fp.write("\n")


def exclusive_csv(path: Path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_directory(seed: int, arm: str) -> Path:
    if arm == TPR_ARM:
        return TPR_DIR / "results" / f"seed_{seed}" / arm
    if arm == C4_ARM:
        return ROOT / "ab_head_preservation" / "results" / f"seed_{seed}" / arm
    if arm == "C3_NO_MANY_PERTURB":
        candidates = (
            ROOT / "results" / f"seed_{seed}" / arm,
            ROOT / "c3_selective" / "results" / f"seed_{seed}" / arm,
        )
        for candidate in candidates:
            if (candidate / "status.json").is_file():
                return candidate
        raise FileNotFoundError(f"Cannot locate archived C3 result for seed {seed}")
    return ROOT / "results" / f"seed_{seed}" / arm


def load_run_status(seed: int, arm: str):
    directory = run_directory(seed, arm)
    status_path = directory / "status.json"
    checkpoint_path = directory / "final_classifier.pth"
    metrics_path = directory / "metrics.csv"
    if not status_path.is_file() or not metrics_path.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError(f"Incomplete run artifacts for seed={seed}, arm={arm}: {directory}")
    status = read_json(status_path)
    if (
        status.get("state") != "completed"
        or status.get("seed") != seed
        or status.get("arm") != arm
        or status.get("final_epoch") != STAGE2_EPOCHS
        or len(status.get("epochs", [])) != STAGE2_EPOCHS
        or status.get("official_TEST_used") is not False
    ):
        raise RuntimeError(f"Run is not a completed fixed-epoch TEST-free result: {directory}")
    if sha256_file(checkpoint_path) != status.get("final_classifier_sha256"):
        raise RuntimeError(f"Final classifier SHA mismatch for seed={seed}, arm={arm}")
    final_epoch = status["epochs"][-1]
    if final_epoch.get("epoch") != STAGE2_EPOCHS:
        raise RuntimeError(f"Status does not contain fixed epoch {STAGE2_EPOCHS}: {directory}")
    validation = final_epoch.get("validation")
    if not isinstance(validation, dict):
        raise RuntimeError(f"Missing validation metrics: {directory}")
    return status, validation, directory


def main():
    completion_path = TPR_DIR / "formal_completed.json"
    preflight_path = TPR_DIR / "preflight_report.json"
    smoke_path = TPR_DIR / "smoke_report.json"
    graph_path = TPR_DIR / "tail_neighbor_graph.json"
    if not all(path.is_file() for path in (completion_path, preflight_path, smoke_path, graph_path)):
        raise FileNotFoundError("TPR formal, preflight, Smoke and train-only graph artifacts are required")
    completion = read_json(completion_path)
    preflight = read_json(preflight_path)
    smoke = read_json(smoke_path)
    graph = read_json(graph_path)
    if (
        completion.get("passed") is not True
        or completion.get("arm") != TPR_ARM
        or completion.get("seeds") != list(SEEDS)
        or completion.get("epochs_per_seed") != STAGE2_EPOCHS
        or completion.get("official_TEST_used") is not False
        or completion.get("original_c0_c1_c2_c3_c4_unchanged") is not True
    ):
        raise RuntimeError("TPR formal completion marker is invalid")
    if (
        preflight.get("passed") is not True
        or preflight.get("official_TEST_used") is not False
        or smoke.get("passed") is not True
        or smoke.get("graph_sha256") != graph.get("graph_sha256")
        or smoke.get("official_TEST_used") is not False
    ):
        raise RuntimeError("TPR preflight/Smoke/graph provenance is invalid")
    compare_protected(preflight["protected_inputs"])

    train_counts = np.asarray(
        read_json(ROOT / "prepared" / "prepared_manifest.json")["train_class_counts"],
        dtype=np.int64,
    )
    if train_counts.shape != (100,):
        raise RuntimeError("Invalid sealed TRAIN class counts")

    all_status = {}
    final_rows = []
    per_class_rows = []
    validation_counts = None
    for seed in SEEDS:
        all_status[seed] = {}
        for arm in ARMS:
            status, validation, directory = load_run_status(seed, arm)
            all_status[seed][arm] = (status, validation)
            row = {
                "seed": seed,
                "arm": arm,
                **{metric: float(validation[metric]) for metric in METRICS},
            }
            if arm == TPR_ARM:
                metrics = status["epochs"][-1]["metrics"]
                row.update({
                    "base_checkpoint_sha256": status["base_checkpoint_sha256"],
                    "train_more_loss": float(metrics["train_more_loss"]),
                    "train_weighted_more_loss": float(metrics["train_weighted_more_loss"]),
                    "train_orthogonality_loss": float(metrics["train_orthogonality_loss"]),
                    "train_tail_diversity_loss": float(metrics["train_tail_diversity_loss"]),
                    "mean_logit_delta_l2_many": float(metrics["mean_logit_delta_l2_many"]),
                    "mean_logit_delta_l2_medium": float(metrics["mean_logit_delta_l2_medium"]),
                    "mean_logit_delta_l2_few": float(metrics["mean_logit_delta_l2_few"]),
                    "base_validation_Overall": float(metrics["base_validation_Overall"]),
                    "base_validation_Many": float(metrics["base_validation_Many"]),
                    "base_validation_Few": float(metrics["base_validation_Few"]),
                })
            else:
                row.update({
                    "base_checkpoint_sha256": "",
                    "train_more_loss": "",
                    "train_weighted_more_loss": "",
                    "train_orthogonality_loss": "",
                    "train_tail_diversity_loss": "",
                    "mean_logit_delta_l2_many": "",
                    "mean_logit_delta_l2_medium": "",
                    "mean_logit_delta_l2_few": "",
                    "base_validation_Overall": "",
                    "base_validation_Many": "",
                    "base_validation_Few": "",
                })
            final_rows.append(row)
            counts = np.asarray(validation["class_counts"], dtype=np.int64)
            if validation_counts is None:
                validation_counts = counts
            elif not np.array_equal(validation_counts, counts):
                raise RuntimeError("Validation class counts differ across arms/seeds")
            per_class = validation.get("per_class_accuracy")
            if not isinstance(per_class, list) or len(per_class) != 100:
                raise RuntimeError(f"Missing 100-class accuracy vector: seed={seed}, arm={arm}")

        for class_id in range(100):
            record = {
                "seed": seed,
                "class_id": class_id,
                "train_count": int(train_counts[class_id]),
                "validation_count": int(validation_counts[class_id]),
            }
            for arm in ARMS:
                validation = all_status[seed][arm][1]
                record[arm] = validation["per_class_accuracy"][class_id]
            for comparison, left, right in (
                ("TPR_minus_C2", TPR_ARM, C2_ARM),
                ("TPR_minus_C4", TPR_ARM, C4_ARM),
                ("TPR_minus_C1", TPR_ARM, "C1_CCSAM_FEATURE_GRAD"),
            ):
                a, b = record[left], record[right]
                record[comparison] = "" if a is None or b is None else float(a) - float(b)
            per_class_rows.append(record)

    by_seed = {
        seed: {
            arm: {metric: float(all_status[seed][arm][1][metric]) for metric in METRICS}
            for arm in ARMS
        }
        for seed in SEEDS
    }
    paired_rows = []
    paired_summary = {}
    for comparison, left, right in COMPARISONS:
        paired_summary[comparison] = {}
        for metric in METRICS:
            deltas = [
                by_seed[seed][left][metric] - by_seed[seed][right][metric]
                for seed in SEEDS
            ]
            paired_summary[comparison][metric] = {
                "per_seed_delta_pp": deltas,
                "mean_delta_pp": float(np.mean(deltas)),
                "sample_sd_pp": float(np.std(deltas, ddof=1)),
                "positive_seed_count": int(sum(value > 0 for value in deltas)),
                "nonnegative_seed_count": int(sum(value >= 0 for value in deltas)),
            }
            paired_rows.append({
                "comparison": comparison,
                "metric": metric,
                "mean_delta_pp": float(np.mean(deltas)),
                "sample_sd_pp": float(np.std(deltas, ddof=1)),
                "positive_seed_count": int(sum(value > 0 for value in deltas)),
                "seed_123_delta_pp": deltas[0],
                "seed_124_delta_pp": deltas[1],
                "seed_125_delta_pp": deltas[2],
            })

    delta_many = paired_summary["TPR_minus_C2"]["Many"]
    delta_few = paired_summary["TPR_minus_C2"]["Few"]
    delta_overall = paired_summary["TPR_minus_C2"]["Overall"]
    delta_few_vs_c4 = paired_summary["TPR_minus_C4"]["Few"]
    gates = {
        "many_within_0_2pp_of_C2_mean": bool(delta_many["mean_delta_pp"] >= -0.2),
        "few_gain_vs_C2_at_least_2pp_mean": bool(delta_few["mean_delta_pp"] >= 2.0),
        "overall_within_0_2pp_of_C2_mean": bool(delta_overall["mean_delta_pp"] >= -0.2),
        "many_not_worse_than_C2_by_more_than_0_2pp_in_at_least_2_seeds": bool(
            sum(value >= -0.2 for value in delta_many["per_seed_delta_pp"]) >= 2
        ),
        "few_gain_positive_vs_C2_in_at_least_2_seeds": bool(
            delta_few["positive_seed_count"] >= 2
        ),
        "few_within_0_5pp_of_C4_mean": bool(delta_few_vs_c4["mean_delta_pp"] >= -0.5),
    }

    summary = {
        "analyzed_at": stamp(),
        "arm": TPR_ARM,
        "primary_epoch": STAGE2_EPOCHS,
        "seeds": list(SEEDS),
        "base_arm": C2_ARM,
        "base_is_frozen": True,
        "rank": completion["rank"],
        "more_peak": completion["more_peak"],
        "orthogonality_lambda": completion["orthogonality_lambda"],
        "tail_diversity_lambda": completion["tail_diversity_lambda"],
        "tail_neighbor_graph_sha256": graph["graph_sha256"],
        "tail_neighbor_graph_centroid_sha256": graph["centroid_sha256"],
        "tail_neighbor_graph_source": graph["source"],
        "many_rule": "sealed TRAIN count strictly greater than 100",
        "official_TEST_used": False,
        "validation_partition": "1980-image official-TRAIN holdout; 20 images per class 1..99; class 0 absent",
        "original_c0_c1_c2_c3_c4_unchanged": True,
        "paired_delta_summary": paired_summary,
        "predeclared_exploratory_gates": gates,
        "caveats": [
            "This is a classifier-stage MORE-inspired adaptation, not a reproduction of the full MORE network-wide method.",
            "The Clean Stage-I backbone and each seed's C2 general classifier are frozen; only the low-rank tail residual is optimized.",
            "The low-rank residual modifies only non-Many output rows. Its effect on Many input distributions is constrained by the MORE discrepancy loss.",
            "The tail-neighbor graph is computed from clean GAP class centroids using official TRAIN images only; no validation labels or official TEST data are used to build it.",
            "Three Stage-II seeds share one frozen Clean Stage-I backbone and one reused TRAIN-derived development holdout; these are not independent-backbone replications.",
            "Class 0 is a Many training class but is absent from the validation holdout.",
            "Fixed epoch 30 is primary; no best-epoch selection, hyperparameter search, or official TEST evaluation is used.",
            "The row-wise orthogonality and neighbor-coefficient separation terms are proposed responses to MORE's discussion of parameter overlap and intra-tail competition; their effectiveness must be judged empirically.",
        ],
    }

    output_paths = (
        ANALYSIS_DIR / "final_epoch_metrics.csv",
        ANALYSIS_DIR / "paired_deltas.csv",
        ANALYSIS_DIR / "per_class.csv",
        ANALYSIS_DIR / "summary.json",
        ANALYSIS_DIR / "TPR_RESULTS.md",
    )
    for path in output_paths:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing analysis output: {path}")

    metric_fields = list(final_rows[0].keys())
    per_class_fields = list(per_class_rows[0].keys())
    exclusive_csv(output_paths[0], final_rows, metric_fields)
    exclusive_csv(output_paths[1], paired_rows, [
        "comparison", "metric", "mean_delta_pp", "sample_sd_pp",
        "positive_seed_count", "seed_123_delta_pp", "seed_124_delta_pp",
        "seed_125_delta_pp",
    ])
    exclusive_csv(output_paths[2], per_class_rows, per_class_fields)
    exclusive_json(output_paths[3], summary)

    def fmt(comparison, metric):
        item = paired_summary[comparison][metric]
        return (
            f"{item['mean_delta_pp']:+.3f} ± {item['sample_sd_pp']:.3f} pp "
            f"(seeds {item['per_seed_delta_pp'][0]:+.3f}, "
            f"{item['per_seed_delta_pp'][1]:+.3f}, {item['per_seed_delta_pp'][2]:+.3f})"
        )

    lines = [
        "# CC-SAM-TPR Results",
        "",
        f"- Analyzed: {summary['analyzed_at']}",
        f"- Primary epoch: {STAGE2_EPOCHS}",
        "- Seeds: 123, 124, 125",
        "- Base: frozen matching-seed C2 classifier",
        f"- Rank: {completion['rank']}; MORE peak: {completion['more_peak']}",
        "- Official CIFAR-100 TEST used: no",
        "",
        "## Paired results",
        "",
        "| Comparison | Overall | Many | MediumN | Few |",
        "|---|---:|---:|---:|---:|",
    ]
    for comp in ("TPR_minus_C2", "TPR_minus_C4", "TPR_minus_C1", "TPR_minus_C3", "TPR_minus_C0"):
        lines.append(
            f"| {comp} | {fmt(comp, 'Overall')} | {fmt(comp, 'Many')} | "
            f"{fmt(comp, 'MediumN')} | {fmt(comp, 'Few')} |"
        )
    lines += ["", "## Exploratory gates", ""]
    for key, value in gates.items():
        lines.append(f"- {'PASS' if value else 'NOT MET'} — {key}")
    lines += [
        "",
        "## Interpretation constraints",
        "",
        "- The validation set is a 1,980-image holdout from official TRAIN; class 0 has no validation examples.",
        "- The three seeds share one frozen Clean Stage-I backbone and one development holdout.",
        "- This is a classifier-stage MORE-inspired adaptation, not a full MORE reproduction or a statistical-significance claim.",
        "- Read summary.json, final_epoch_metrics.csv, paired_deltas.csv, and per_class.csv together; do not select a favorable metric in isolation.",
        "",
    ]
    output_paths[4].write_text("\n".join(lines), encoding="utf-8")
    print("TPR fixed-epoch analysis completed.")
    print(f"Results: {ANALYSIS_DIR}")


if __name__ == "__main__":
    main()
