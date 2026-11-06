from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from ccsam_common import ROOT, SEEDS, ARMS, STAGE2_EPOCHS, sha256_file, read_json, write_json

METRICS = [
    "Overall", "Macro", "Head", "OfficialMedium", "Tail", "Many", "MediumN", "Few"
]
PAIRS = [
    ("C1_CCSAM_FEATURE_GRAD", "C0_BALANCED_CE", "C1_minus_C0_full_transfer"),
    ("C1_CCSAM_FEATURE_GRAD", "C2_WEIGHTED_CE_NO_PERTURB", "C1_minus_C2_feature_perturbation"),
    ("C2_WEIGHTED_CE_NO_PERTURB", "C0_BALANCED_CE", "C2_minus_C0_weighting_and_curriculum"),
]


def load_final(seed, arm):
    run_dir = ROOT / "results" / f"seed_{seed}" / arm
    status_path = run_dir / "status.json"
    metrics_path = run_dir / "metrics.csv"
    checkpoint_path = run_dir / "final_classifier.pth"
    if not status_path.is_file() or not metrics_path.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError(f"Incomplete formal run: {run_dir}")
    status = read_json(status_path)
    if status.get("state") != "completed" or status.get("official_TEST_used") is not False:
        raise RuntimeError(f"Run is not complete or TEST-free: {run_dir}")
    if len(status.get("epochs", [])) != STAGE2_EPOCHS or status.get("final_epoch") != STAGE2_EPOCHS:
        raise RuntimeError(f"Unexpected epoch count in {run_dir}")
    if sha256_file(checkpoint_path) != status.get("final_classifier_sha256"):
        raise RuntimeError(f"Final classifier checksum mismatch: {checkpoint_path}")
    with metrics_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != STAGE2_EPOCHS or int(rows[-1]["epoch"]) != STAGE2_EPOCHS:
        raise RuntimeError(f"Final epoch metrics are missing: {metrics_path}")
    return status["final_validation"], rows[-1], status


def main():
    completed_path = ROOT / "formal_completed.json"
    guard_path = ROOT / "formal_test_guard.json"
    if not completed_path.is_file() or not guard_path.is_file():
        raise FileNotFoundError("Formal completion marker and TEST guard seal are required before analysis.")
    completed = read_json(completed_path)
    guard = read_json(guard_path)
    if completed.get("official_TEST_used") is not False or guard.get("official_TEST_used") is not False:
        raise RuntimeError("Formal run does not certify TEST-free execution.")
    if guard.get("blocked_attempts") or guard.get("official_test_file_open_attempts") != 0:
        raise RuntimeError("TEST guard recorded an access attempt.")

    all_results = {}
    final_rows = []
    for seed in SEEDS:
        all_results[str(seed)] = {}
        for arm in ARMS:
            validation, csv_row, status = load_final(seed, arm)
            all_results[str(seed)][arm] = validation
            final_rows.append({
                "seed": seed,
                "arm": arm,
                **{metric: float(validation[metric]) for metric in METRICS},
                "train_loss": float(csv_row["train_loss"]),
                "train_clean_accuracy": float(csv_row["train_clean_accuracy"]),
                "mean_delta_l2_per_sample": float(csv_row["mean_delta_l2_per_sample"]),
                "mean_relative_delta_l2": float(csv_row["mean_relative_delta_l2"]),
                "mean_adv_minus_clean_ce": float(csv_row["mean_adv_minus_clean_ce"]),
            })

    final_table_path = ROOT / "final_epoch_metrics.csv"
    with final_table_path.open("w", encoding="utf-8", newline="") as f:
        fields = list(final_rows[0])
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(final_rows)

    paired_rows = []
    delta_summary = {}
    for arm_a, arm_b, label in PAIRS:
        delta_summary[label] = {}
        for metric in METRICS:
            values = []
            for seed in SEEDS:
                a = float(all_results[str(seed)][arm_a][metric])
                b = float(all_results[str(seed)][arm_b][metric])
                delta = a - b
                values.append(delta)
                paired_rows.append({
                    "seed": seed,
                    "comparison": label,
                    "metric": metric,
                    "arm_a": arm_a,
                    "arm_b": arm_b,
                    "arm_a_value": a,
                    "arm_b_value": b,
                    "delta_pp": delta,
                })
            delta_summary[label][metric] = {
                "per_seed_delta_pp": values,
                "mean_delta_pp": float(np.mean(values)),
                "sample_sd_pp": float(np.std(values, ddof=1)),
                "positive_seeds": int(sum(x > 0 for x in values)),
                "num_seeds": len(values),
            }

    paired_path = ROOT / "paired_deltas.csv"
    with paired_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(paired_rows[0]))
        writer.writeheader()
        writer.writerows(paired_rows)

    summary = {
        "analyzed_at": __import__("datetime").datetime.now().astimezone().isoformat(timespec="seconds"),
        "primary_epoch": STAGE2_EPOCHS,
        "seeds": list(SEEDS),
        "arms": list(ARMS),
        "clean_stage1_sha256": "7ced9d07f71335357c08623573f038acd0dfdcdeadb55d95d57b332a14f83a88",
        "fixed_backbone_across_all_arms": True,
        "fixed_validation_partition": "1980 images from official CIFAR-100 TRAIN; 20/class for IDs 1..99; class 0 absent",
        "official_TEST_used": False,
        "stage2_seed_count_is_not_independent_backbone_count": True,
        "final_epoch_metrics": all_results,
        "paired_delta_summary": delta_summary,
        "interpretation": {
            "C1_minus_C2": "Primary estimate of the added contribution of CC-SAM-style class-conditional feature-gradient perturbation, with class weights, alpha schedule, sampler, initialization, and optimizer held fixed.",
            "C2_minus_C0": "Contribution of the CC-SAM Stage-II class weighting and alpha blend without feature perturbation, relative to balanced CE.",
            "C1_minus_C0": "Overall effect of transferring the CC-SAM Stage-II objective and feature perturbation together.",
            "limitations": [
                "All Stage-II seeds share one Clean Stage-I backbone and one fixed TRAIN-derived validation partition.",
                "The validation partition has no class-0 examples and has been used in prior development; it is not an independent test set.",
                "Three Stage-II seeds quantify optimization/sampling variability conditional on this fixed representation, not representation-level or cross-dataset generalization.",
                "This is a controlled transfer of CC-SAM Stage II, not a full reproduction of the original two-stage CC-SAM training pipeline."
            ],
        },
    }
    write_json(ROOT / "summary.json", summary)

    lines = [
        "# CC-SAM Stage II transfer: CIFAR-100-LT controlled results",
        "",
        f"Primary comparison uses fixed epoch {STAGE2_EPOCHS}. Official CIFAR-100 TEST was not loaded or evaluated.",
        "",
        "## Final-epoch metrics (%)",
        "",
        "| Seed | Arm | Overall | Head | Official Medium | Tail | Many | Medium-n | Few |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in final_rows:
        lines.append(
            f"| {row['seed']} | {row['arm']} | {row['Overall']:.3f} | {row['Head']:.3f} | "
            f"{row['OfficialMedium']:.3f} | {row['Tail']:.3f} | {row['Many']:.3f} | "
            f"{row['MediumN']:.3f} | {row['Few']:.3f} |"
        )
    lines.extend([
        "",
        "## Paired deltas (percentage points)",
        "",
        "| Comparison | Metric | Mean Δ | Sample SD | Positive seeds | Per-seed Δ |",
        "|---|---|---:|---:|---:|---|",
    ])
    for _, _, label in PAIRS:
        for metric in METRICS:
            item = delta_summary[label][metric]
            per_seed = ", ".join(f"{x:+.3f}" for x in item["per_seed_delta_pp"])
            lines.append(
                f"| {label} | {metric} | {item['mean_delta_pp']:+.3f} | "
                f"{item['sample_sd_pp']:.3f} | {item['positive_seeds']}/{item['num_seeds']} | {per_seed} |"
            )
    lines.extend([
        "",
        "## Interpretation guardrails",
        "",
        "- C1 minus C2 is the primary feature-perturbation comparison. The sampler, class weights, alpha schedule, classifier initialization, optimizer, training budget, and augmented inputs are paired.",
        "- C2 minus C0 estimates the contribution of class weighting and the alpha blend without feature perturbation.",
        "- C1 minus C0 measures the combined transferred Stage-II objective.",
        "- Do not claim statistical significance from three Stage-II seeds.",
        "- All three seeds share one Clean Stage-I backbone; they are not three independent representation seeds.",
        "- The validation partition is drawn from official TRAIN, contains no class-0 examples, and is not an independent test set.",
        "- This experiment transfers the feature-gradient part of CC-SAM Stage II; it is not a full reproduction of original CC-SAM.",
        "",
        "See summary.json and paired_deltas.csv for machine-readable results.",
    ])
    (ROOT / "CCSAM_STAGE2_RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("CC-SAM analysis completed: final_epoch_metrics.csv, paired_deltas.csv, summary.json, CCSAM_STAGE2_RESULTS.md")


if __name__ == "__main__":
    main()
