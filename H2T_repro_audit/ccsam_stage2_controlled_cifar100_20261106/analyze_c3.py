"""C3-only paired Stage-II analysis. Read-only original C0/C1/C2 results.

All primary results are fixed epoch 30 on the sealed TRAIN holdout; never
read or instantiate official CIFAR-100 TEST data. No best-epoch cherry-picking.
"""
from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from ccsam_common import ROOT, SEEDS, ARMS, STAGE2_EPOCHS, read_json, sha256_file
from train_ccsam import C3_ARM, read_trace

OUT = ROOT / "c3_selective"
METRICS = ("Overall", "Macro", "Many", "MediumN", "Few",
           "Head", "OfficialMedium", "Tail")
PAIRS = (
    (C3_ARM, "C1_CCSAM_FEATURE_GRAD", "C3_minus_C1_remove_Many_perturbation"),
    (C3_ARM, "C2_WEIGHTED_CE_NO_PERTURB", "C3_minus_C2_nonMany_perturbation"),
    (C3_ARM, "C0_BALANCED_CE", "C3_minus_C0_total_difference"),
    ("C1_CCSAM_FEATURE_GRAD", "C2_WEIGHTED_CE_NO_PERTURB", "C1_minus_C2_original_effect"),
)


def exclusive_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as fp:
        json.dump(value, fp, ensure_ascii=False, indent=2)
        fp.write("\n")


def exclusive_csv(path, rows):
    if not rows:
        raise RuntimeError(f"No rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_one(seed, arm):
    directory = ROOT / "results" / f"seed_{seed}" / arm
    path = directory / "status.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing run: {path}")
    data = read_json(path)
    if (data.get("state") != "completed"
            or data.get("seed") != seed or data.get("arm") != arm
            or data.get("final_epoch") != STAGE2_EPOCHS
            or len(data.get("epochs", [])) != STAGE2_EPOCHS
            or data.get("official_TEST_used") is not False):
        raise RuntimeError(f"Run is not complete or TEST-free: {directory}")
    checkpoint = directory / "final_classifier.pth"
    if sha256_file(checkpoint) != data.get("final_classifier_sha256"):
        raise RuntimeError(f"Classifier SHA mismatch: {checkpoint}")
    rows = read_trace(directory / "metrics.csv")
    if len(rows) != STAGE2_EPOCHS or int(rows[-1]["epoch"]) != STAGE2_EPOCHS:
        raise RuntimeError(f"Unsealed fixed-epoch metrics: {directory}")
    validation = data["final_validation"]
    if validation.get("samples") != 1980 or validation.get("missing_validation_classes") != [0]:
        raise RuntimeError(f"Validation protocol differs: {directory}")
    return {
        "status": data,
        "last_metric": rows[-1],
        "trace": read_trace(directory / "batch_trace.csv"),
        "validation": validation,
    }


def main():
    completed_file = OUT / "formal_completed.json"
    if not completed_file.is_file():
        raise FileNotFoundError("C3 three-seed formal completion marker missing")
    done = read_json(completed_file)
    if (done.get("passed") is not True
            or done.get("only_c3_new_training") is not True
            or done.get("original_nine_runs_unchanged") is not True
            or done.get("official_TEST_used") is not False
            or done.get("seeds") != list(SEEDS)):
        raise RuntimeError("C3 formal marker not clean")

    original_summary = ROOT / "summary.json"
    if not original_summary.is_file():
        raise FileNotFoundError("Original C0/C1/C2 summary must remain available")
    original_statuses = read_json(original_summary)
    if original_statuses.get("official_TEST_used") is not False:
        raise RuntimeError("Original analysis is not TEST-free")

    all_data, pair_checks = {}, []
    all_arms = tuple(ARMS) + (C3_ARM,)
    for seed in SEEDS:
        items = {arm: load_one(seed, arm) for arm in all_arms}
        baseline = items["C0_BALANCED_CE"]
        expected_rows = STAGE2_EPOCHS * 170
        for arm in all_arms:
            if len(items[arm]["trace"]) != expected_rows:
                raise RuntimeError(f"Wrong trace count: seed={seed} arm={arm}")
            if items[arm]["trace"] != baseline["trace"]:
                raise RuntimeError(f"Unpaired augment/features: seed={seed} arm={arm}")
            if items[arm]["status"]["initial_classifier_sha256"] != baseline["status"]["initial_classifier_sha256"]:
                raise RuntimeError(f"Initial classifier mismatch: seed={seed} arm={arm}")
            if items[arm]["status"]["initial_backbone_sha256"] != baseline["status"]["initial_backbone_sha256"]:
                raise RuntimeError(f"Initial backbone mismatch: seed={seed} arm={arm}")
            if items[arm]["status"]["clean_stage1_sha256"] != baseline["status"]["clean_stage1_sha256"]:
                raise RuntimeError(f"Stage I checkpoint mismatch: seed={seed} arm={arm}")
        for arm in ARMS:
            base_metrics = original_statuses["final_epoch_metrics"][str(seed)][arm]
            for m in METRICS:
                if abs(float(items[arm]["validation"][m]) - float(base_metrics[m])) > 1e-12:
                    raise RuntimeError(f"Original baseline metric changed: {seed}/{arm}/{m}")
        if float(items[C3_ARM]["last_metric"]["mean_delta_l2_per_sample"]) <= 0:
            raise RuntimeError(f"C3 never perturbed eligible classes: seed={seed}")
        all_data[str(seed)] = items
        pair_checks.append({
            "seed": seed, "batch_trace_rows_per_arm": expected_rows,
            "C3_C0_C1_C2_augments_features_match_bitwise": True,
            "initial_classifier_and_backbone_match": True,
            "frozen_stage1_sha_match": True,
            "c3_final_classifier_sha256": items[C3_ARM]["status"]["final_classifier_sha256"],
        })

    final_rows, paired_rows = [], []
    for seed in SEEDS:
        for arm in all_arms:
            item = all_data[str(seed)][arm]
            final_rows.append({
                "seed": seed, "arm": arm,
                **{m: float(item["validation"][m]) for m in METRICS},
                "train_clean_accuracy": float(item["last_metric"]["train_clean_accuracy"]),
                "mean_delta_l2_per_sample": float(item["last_metric"]["mean_delta_l2_per_sample"]),
                "mean_relative_delta_l2": float(item["last_metric"]["mean_relative_delta_l2"]),
            })
    delta_summary = {}
    for arm_a, arm_b, name in PAIRS:
        delta_summary[name] = {}
        for metric in METRICS:
            values = []
            for seed in SEEDS:
                a = float(all_data[str(seed)][arm_a]["validation"][metric])
                b = float(all_data[str(seed)][arm_b]["validation"][metric])
                delta = a-b
                values.append(delta)
                paired_rows.append({
                    "seed": seed, "contrast": name, "metric": metric,
                    "arm_a": arm_a, "arm_b": arm_b,
                    "a_percent": a, "b_percent": b, "delta_pp": delta,
                })
            delta_summary[name][metric] = {
                "per_seed_delta_pp": values,
                "mean_delta_pp": float(np.mean(values)),
                "sample_sd_pp": float(np.std(values, ddof=1)),
                "positive_seed_count": sum(x > 0 for x in values),
            }

    # Class-wise changes are descriptive only: the TRAIN-derived validation has
    # 20 images per observed class, no class 0, and shares one Stage I backbone.
    class_rows = []
    observed_classes = np.flatnonzero(
        np.asarray(all_data[str(SEEDS[0])][C3_ARM]["validation"]["class_counts"]) > 0
    )
    counts = np.asarray(read_json(ROOT / "prepared" / "prepared_manifest.json")["train_class_counts"])
    for cls in observed_classes.tolist():
        if cls == 0:
            raise RuntimeError("Class 0 is not part of this validation partition")
        vals = {}
        for arm in all_arms:
            accuracies = []
            for seed in SEEDS:
                val = all_data[str(seed)][arm]["validation"]
                if int(val["class_counts"][cls]) != 20:
                    raise RuntimeError(f"Class {cls}: unexpected validation count")
                accuracies.append(5.0 * int(val["class_correct"][cls]))
            vals[arm] = accuracies
        c3_c1 = [a-b for a,b in zip(vals[C3_ARM], vals["C1_CCSAM_FEATURE_GRAD"])]
        c3_c2 = [a-b for a,b in zip(vals[C3_ARM], vals["C2_WEIGHTED_CE_NO_PERTURB"])]
        class_rows.append({
            "class_id": cls, "train_count": int(counts[cls]),
            "group": "Many" if counts[cls] > 100 else
                     "Few" if counts[cls] < 20 else "MediumN",
            "c3_avg_percent": float(np.mean(vals[C3_ARM])),
            "c1_avg_percent": float(np.mean(vals["C1_CCSAM_FEATURE_GRAD"])),
            "c2_avg_percent": float(np.mean(vals["C2_WEIGHTED_CE_NO_PERTURB"])),
            "c3_minus_c1_avg_pp": float(np.mean(c3_c1)),
            "c3_minus_c2_avg_pp": float(np.mean(c3_c2)),
            "c3_minus_c1_all3_positive": all(x > 0 for x in c3_c1),
            "c3_minus_c1_all3_negative": all(x < 0 for x in c3_c1),
        })

    delta_c1_c2_few = delta_summary["C1_minus_C2_original_effect"]["Few"]["mean_delta_pp"]
    delta_c3_c2_few = delta_summary["C3_minus_C2_nonMany_perturbation"]["Few"]["mean_delta_pp"]
    c1_many_damage = delta_summary["C1_minus_C2_original_effect"]["Many"]["mean_delta_pp"]
    c3_many_damage = delta_summary["C3_minus_C2_nonMany_perturbation"]["Many"]["mean_delta_pp"]
    ratios = {
        "few_benefit_retention_relative_to_C1_vs_C2":
            None if abs(delta_c1_c2_few) < 1e-12 else delta_c3_c2_few/delta_c1_c2_few,
        "many_damage_fraction_relative_to_C1_vs_C2":
            None if abs(c1_many_damage) < 1e-12 else c3_many_damage/c1_many_damage,
    }
    result = {
        "analyzed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "arm": C3_ARM, "primary_epoch": STAGE2_EPOCHS, "seeds": list(SEEDS),
        "official_TEST_used": False, "original_nine_runs_untouched": True,
        "c3_only_training": True,
        "validation_partition": "TRAIN holdout 20/class for classes 1-99, no class0",
        "paired_trace_validation": pair_checks,
        "paired_delta_summary": delta_summary,
        "interpretation_ratios": ratios,
        "caveats": [
            "Removing direct Many perturbation does not freeze Many classifier weights.",
            "A C3 Many recovery demonstrates the contribution of Many perturbations, not semantic sufficiency.",
            "Three seeds share the same Clean Stage I backbone and development validation set.",
            "This is not independent official TEST performance or a significance test.",
            "No best-epoch selection was used; primary results are fixed epoch 30.",
        ],
    }
    for p in (OUT / "c3_paired_deltas.csv", OUT / "c3_final_epoch_metrics.csv",
              OUT / "c3_per_class.csv", OUT / "c3_summary.json", OUT / "C3_RESULTS.md"):
        if p.exists():
            raise FileExistsError(f"Do not overwrite C3 analysis: {p}")
    exclusive_csv(OUT / "c3_paired_deltas.csv", paired_rows)
    exclusive_csv(OUT / "c3_final_epoch_metrics.csv", final_rows)
    exclusive_csv(OUT / "c3_per_class.csv", class_rows)
    exclusive_json(OUT / "c3_summary.json", result)

    summary_lines = [
        "# C3 — Exclude Many-class feature perturbation, fixed Stage II",
        "",
        "C3: delta=0 exactly when original TRAIN class count >100; all remaining delta and weighted/clean CE are unchanged.",
        "Original C0/C1/C2 are reused, not rerun. Only C3 for seeds 123,124,125 was trained.",
        "All primary values use sealed TRAIN-holdout, epoch 30; official TEST was never accessed.",
        "",
        "## Per-seed final metrics (%)",
        "",
        "| Seed | Arm | Overall | Many | MediumN | Few |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for r in final_rows:
        summary_lines.append(
            f"| {r['seed']} | {r['arm']} | {r['Overall']:.3f} | "
            f"{r['Many']:.3f} | {r['MediumN']:.3f} | {r['Few']:.3f} |")
    summary_lines += [
        "", "## Paired changes (percentage points; 3 Stage-II seeds)", "",
        "| Contrast | Overall | Many | MediumN | Few |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in delta_summary:
        row = delta_summary[name]
        summary_lines.append(
            f"| {name} | {row['Overall']['mean_delta_pp']:+.3f} | "
            f"{row['Many']['mean_delta_pp']:+.3f} | "
            f"{row['MediumN']['mean_delta_pp']:+.3f} | "
            f"{row['Few']['mean_delta_pp']:+.3f} |")
    summary_lines += [
        "", "## Interpretation limits", "",
        "- C3-C1 tests the direct contribution of Many feature perturbations in the presence of shared classifier updates.",
        "- C3-C2 measures the remaining effect when ONLY Medium/Few feature perturbations are active.",
        "- Even with no direct Many perturbation, the shared Softmax classifier can move Many decision boundaries.",
        "- Class 0 is absent from validation, so 'Many' describes observed Many classes only.",
        "- Three Stage-II seeds share a single fixed Stage-I representation; no statistical significance claim.",
        "- Validation derives from official TRAIN and was previously used for development; not independent TEST.",
        "", "Files: c3_summary.json, c3_final_epoch_metrics.csv, c3_paired_deltas.csv, c3_per_class.csv.",
    ]
    with (OUT / "C3_RESULTS.md").open("x", encoding="utf-8") as fp:
        fp.write("\n".join(summary_lines)+"\n")
    print("C3 analysis PASS. Original C0/C1/C2 files were only read.", flush=True)


if __name__ == "__main__":
    main()
