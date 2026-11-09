"""Fixed-epoch paired analysis for C4 against sealed C0-C3."""
from __future__ import annotations

import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

AB_DIR = Path(__file__).resolve().parent
EXPERIMENT_ROOT = AB_DIR.parents[0]
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from ccsam_common import ROOT as EXPERIMENT_ROOT, SEEDS, STAGE2_EPOCHS, read_json, sha256_file  # noqa: E402
from train_c3 import baseline_file_snapshot  # noqa: E402
from train_ab import (  # noqa: E402
    C4_ARM, C3_DIR, AB_DIR as TRAIN_AB_DIR, compare_protected_inputs,
)

ANALYSIS_DIR = AB_DIR / "analysis"
ARMS = (
    "C0_BALANCED_CE",
    "C1_CCSAM_FEATURE_GRAD",
    "C2_WEIGHTED_CE_NO_PERTURB",
    "C3_NO_MANY_PERTURB",
    C4_ARM,
)
METRICS = ("Overall", "Macro", "Many", "MediumN", "Few", "Head", "OfficialMedium", "Tail")
COMPARISONS = (
    ("C4_minus_C1", C4_ARM, "C1_CCSAM_FEATURE_GRAD"),
    ("C4_minus_C2", C4_ARM, "C2_WEIGHTED_CE_NO_PERTURB"),
    ("C4_minus_C3", C4_ARM, "C3_NO_MANY_PERTURB"),
    ("C4_minus_C0", C4_ARM, "C0_BALANCED_CE"),
    ("C1_minus_C2_reference", "C1_CCSAM_FEATURE_GRAD", "C2_WEIGHTED_CE_NO_PERTURB"),
    ("C3_minus_C2_reference", "C3_NO_MANY_PERTURB", "C2_WEIGHTED_CE_NO_PERTURB"),
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


def load_run_status(seed: int, arm: str):
    if arm == C4_ARM:
        directory = AB_DIR / "results" / f"seed_{seed}" / arm
    else:
        directory = EXPERIMENT_ROOT / "results" / f"seed_{seed}" / arm
    status_path = directory / "status.json"
    checkpoint_path = directory / "final_classifier.pth"
    metrics_path = directory / "metrics.csv"
    if not status_path.is_file() or not metrics_path.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError(f"Incomplete run artifacts for seed={seed}, arm={arm}")
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
    completion_path = AB_DIR / "formal_completed.json"
    preflight_path = AB_DIR / "preflight_report.json"
    if not completion_path.is_file() or not preflight_path.is_file():
        raise FileNotFoundError("C4 formal completion and preflight reports are required")
    completion = read_json(completion_path)
    preflight = read_json(preflight_path)
    if (
        completion.get("passed") is not True
        or completion.get("arm") != C4_ARM
        or completion.get("seeds") != list(SEEDS)
        or completion.get("epochs_per_seed") != STAGE2_EPOCHS
        or completion.get("official_TEST_used") is not False
        or completion.get("original_c0_c1_c2_and_c3_unchanged") is not True
    ):
        raise RuntimeError("C4 formal completion marker is invalid")
    if preflight.get("passed") is not True or preflight.get("official_TEST_used") is not False:
        raise RuntimeError("C4 preflight marker is invalid")

    compare_protected_inputs(preflight)
    if baseline_file_snapshot() != preflight["original_c0_c1_c2_artifact_sha256"]:
        raise RuntimeError("Original C0/C1/C2 archive changed before analysis")

    all_status = {}
    final_rows = []
    per_class_rows = []
    validation_counts = None
    train_counts = np.asarray(
        read_json(EXPERIMENT_ROOT / "prepared" / "prepared_manifest.json")["train_class_counts"],
        dtype=np.int64,
    )
    if train_counts.shape != (100,):
        raise RuntimeError("Invalid sealed TRAIN class counts")

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
            if arm == C4_ARM:
                diag = status["epochs"][-1].get("teacher_diagnostics", {})
                row.update({
                    "teacher_validation_Many": float(diag["teacher_validation_many_accuracy"]),
                    "teacher_student_Many_KL": float(diag["teacher_student_many_kl"]),
                    "teacher_student_Many_top1_agreement": float(
                        diag["teacher_student_many_top1_agreement"]
                    ),
                    "train_head_kd_loss": float(status["epochs"][-1]["metrics"]["train_head_kd_loss"]),
                })
            else:
                row.update({
                    "teacher_validation_Many": "",
                    "teacher_student_Many_KL": "",
                    "teacher_student_Many_top1_agreement": "",
                    "train_head_kd_loss": "",
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
            if arm == ARMS[0]:
                pass

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
            c4 = record[C4_ARM]
            for comparison, left, right in (
                ("C4_minus_C1", C4_ARM, "C1_CCSAM_FEATURE_GRAD"),
                ("C4_minus_C2", C4_ARM, "C2_WEIGHTED_CE_NO_PERTURB"),
                ("C4_minus_C3", C4_ARM, "C3_NO_MANY_PERTURB"),
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
            mean_delta = float(np.mean(deltas))
            sample_sd = float(np.std(deltas, ddof=1))
            paired_summary[comparison][metric] = {
                "per_seed_delta_pp": deltas,
                "mean_delta_pp": mean_delta,
                "sample_sd_pp": sample_sd,
                "positive_seed_count": int(sum(value > 0 for value in deltas)),
            }
            paired_rows.append({
                "comparison": comparison,
                "metric": metric,
                "mean_delta_pp": mean_delta,
                "sample_sd_pp": sample_sd,
                "positive_seed_count": int(sum(value > 0 for value in deltas)),
                "seed_123_delta_pp": deltas[0],
                "seed_124_delta_pp": deltas[1],
                "seed_125_delta_pp": deltas[2],
            })

    many_recovery = paired_summary["C4_minus_C1"]["Many"]["mean_delta_pp"]
    original_many_gap = -paired_summary["C1_minus_C2_reference"]["Many"]["mean_delta_pp"]
    few_retained = paired_summary["C4_minus_C2"]["Few"]["mean_delta_pp"]
    original_few_gain = paired_summary["C1_minus_C2_reference"]["Few"]["mean_delta_pp"]
    overall_vs_c2 = paired_summary["C4_minus_C2"]["Overall"]["mean_delta_pp"]
    recovery_fraction = None if original_many_gap <= 0 else many_recovery / original_many_gap
    few_retention_fraction = None if abs(original_few_gain) < 1e-12 else few_retained / original_few_gain

    gates = {
        "many_recovers_at_least_half_of_C1_to_C2_gap": bool(
            many_recovery >= 0.5 * original_many_gap
        ),
        "few_gain_vs_C2_at_least_2pp": bool(few_retained >= 2.0),
        "overall_within_0_2pp_of_C2": bool(overall_vs_c2 >= -0.2),
        "many_recovery_positive_in_at_least_2_seeds": bool(
            paired_summary["C4_minus_C1"]["Many"]["positive_seed_count"] >= 2
        ),
        "few_gain_positive_in_at_least_2_seeds": bool(
            paired_summary["C4_minus_C2"]["Few"]["positive_seed_count"] >= 2
        ),
    }

    result = {
        "analyzed_at": stamp(),
        "arm": C4_ARM,
        "primary_epoch": STAGE2_EPOCHS,
        "seeds": list(SEEDS),
        "teacher_arm": "C2_WEIGHTED_CE_NO_PERTURB",
        "teacher_temperature": 2.0,
        "head_preservation_lambda": 1.0,
        "many_rule": "sealed TRAIN count strictly greater than 100",
        "official_TEST_used": False,
        "validation_partition": "1980-image official-TRAIN holdout; 20 images per class 1..99; class 0 absent",
        "original_c0_c1_c2_c3_unchanged": True,
        "paired_delta_summary": paired_summary,
        "interpretation_ratios": {
            "many_recovery_fraction_of_C1_to_C2_gap": recovery_fraction,
            "few_gain_retention_fraction_relative_to_C1_vs_C2": few_retention_fraction,
        },
        "predeclared_exploratory_gates": gates,
        "teacher_diagnostics": {
            str(seed): all_status[seed][C4_ARM][0]["epochs"][-1]["teacher_diagnostics"]
            for seed in SEEDS
        },
        "caveats": [
            "Three Stage-II seeds share one frozen Clean Stage-I backbone and one reused TRAIN-derived development holdout.",
            "These results are not independent-backbone replications and do not establish statistical significance.",
            "Class 0 is a Many training class but is absent from the validation holdout.",
            "The C2 teacher is a frozen classifier trained on the same fixed representation; this is teacher-guided classifier preservation, not a new backbone or representation.",
            "The fixed epoch 30 is primary; no best-epoch selection or official TEST evaluation is used.",
        ],
    }

    output_files = (
        ANALYSIS_DIR / "final_epoch_metrics.csv",
        ANALYSIS_DIR / "paired_deltas.csv",
        ANALYSIS_DIR / "per_class.csv",
        ANALYSIS_DIR / "summary.json",
        ANALYSIS_DIR / "AB_RESULTS.md",
    )
    existing = [str(path) for path in output_files if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite existing C4 analysis artifacts: {existing}")

    final_fields = [
        "seed", "arm", *METRICS, "teacher_validation_Many",
        "teacher_student_Many_KL", "teacher_student_Many_top1_agreement",
        "train_head_kd_loss",
    ]
    paired_fields = [
        "comparison", "metric", "mean_delta_pp", "sample_sd_pp",
        "positive_seed_count", "seed_123_delta_pp", "seed_124_delta_pp",
        "seed_125_delta_pp",
    ]
    per_class_fields = [
        "seed", "class_id", "train_count", "validation_count", *ARMS,
        "C4_minus_C1", "C4_minus_C2", "C4_minus_C3",
    ]
    exclusive_csv(output_files[0], final_rows, final_fields)
    exclusive_csv(output_files[1], paired_rows, paired_fields)
    exclusive_csv(output_files[2], per_class_rows, per_class_fields)
    exclusive_json(output_files[3], result)

    means = {}
    for arm in ARMS:
        means[arm] = {}
        for metric in METRICS:
            values = [by_seed[seed][arm][metric] for seed in SEEDS]
            means[arm][metric] = {
                "mean": float(np.mean(values)),
                "sample_sd": float(np.std(values, ddof=1)),
            }

    lines = [
        "# C4 — CC-SAM with Many-only head-preservation distillation",
        "",
        "Primary results are fixed Stage-II epoch 30 on the sealed TRAIN-derived holdout. Official CIFAR-100 TEST was never accessed.",
        "",
        "## Per-seed final metrics (%)",
        "",
        "| Seed | Arm | Overall | Many | MediumN | Few |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for row in final_rows:
        lines.append(
            f"| {row['seed']} | {row['arm']} | {row['Overall']:.3f} | "
            f"{row['Many']:.3f} | {row['MediumN']:.3f} | {row['Few']:.3f} |"
        )
    lines += [
        "",
        "## Mean ± sample SD (%)",
        "",
        "| Arm | Overall | Many | MediumN | Few |",
        "|---|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        m = means[arm]
        lines.append(
            f"| {arm} | {m['Overall']['mean']:.3f} ± {m['Overall']['sample_sd']:.3f} | "
            f"{m['Many']['mean']:.3f} ± {m['Many']['sample_sd']:.3f} | "
            f"{m['MediumN']['mean']:.3f} ± {m['MediumN']['sample_sd']:.3f} | "
            f"{m['Few']['mean']:.3f} ± {m['Few']['sample_sd']:.3f} |"
        )
    lines += [
        "",
        "## C4 paired deltas (percentage points)",
        "",
        "| Comparison | Overall | Many | MediumN | Few |",
        "|---|---:|---:|---:|---:|",
    ]
    for comparison in ("C4_minus_C1", "C4_minus_C2", "C4_minus_C3", "C4_minus_C0"):
        item = paired_summary[comparison]
        lines.append(
            f"| {comparison} | {item['Overall']['mean_delta_pp']:+.3f} | "
            f"{item['Many']['mean_delta_pp']:+.3f} | "
            f"{item['MediumN']['mean_delta_pp']:+.3f} | "
            f"{item['Few']['mean_delta_pp']:+.3f} |"
        )
    lines += [
        "",
        "## Predeclared exploratory gates",
        "",
    ]
    for name, passed in gates.items():
        lines.append(f"- {'PASS' if passed else 'NOT MET'}: {name}")
    lines += [
        "",
        f"- Mean Many recovery relative to the C1–C2 gap: {recovery_fraction if recovery_fraction is not None else 'N/A'}",
        f"- Few gain retained relative to the C1–C2 gain: {few_retention_fraction if few_retention_fraction is not None else 'N/A'}",
        f"- Overall delta versus C2: {overall_vs_c2:+.3f} pp",
        "",
        "## Interpretation guardrails",
        "",
        "- C4−C1 measures the added effect of Many-only C2 teacher distillation while retaining the full C1 perturbation.",
        "- C4−C2 measures whether C4 preserves a Few benefit over the no-perturbation weighted-CE teacher.",
        "- C4−C3 compares teacher-guided preservation against direct Many-perturbation removal.",
        "- KL is applied only to Many inputs, but it compares the full 100-class output distribution.",
        "- Class 0 is excluded from validation; Many validation metrics cover only observed Many classes.",
        "- Three Stage-II seeds share one frozen Stage-I representation; no significance claim is justified.",
        "- No official TEST data, labels, or test loader are used.",
        "",
        "Machine-readable details: summary.json, paired_deltas.csv, per_class.csv, final_epoch_metrics.csv.",
    ]
    with output_files[4].open("x", encoding="utf-8") as fp:
        fp.write("\n".join(lines) + "\n")

    compare_protected_inputs(preflight)
    print("C4 analysis PASS. Original C0-C3 artifacts remain unchanged.")


if __name__ == "__main__":
    main()
