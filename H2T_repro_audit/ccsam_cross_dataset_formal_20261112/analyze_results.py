"""Summarize fixed test results and within-seed paired differences."""
from __future__ import annotations

import csv
import statistics

import numpy as np

from evaluate_final import DISPLAY, FINAL, STAGE1, classification_metrics
from experiment_common import METRICS, OUTPUT_ROOT, record_run, require_complete, write_csv, write_json

ANALYSIS = OUTPUT_ROOT / "analysis"
CONTRASTS = (("D1", "D0"), ("D2", "D0"), ("D2", "D1"))


def aggregate(rows: list[dict]) -> dict:
    return {
        metric: {"mean": statistics.mean(row[metric] for row in rows),
                 "sample_sd": statistics.stdev(row[metric] for row in rows)}
        for metric in METRICS
    }


def read_dataset_results(exp, name, seal):
    with (FINAL / f"{DISPLAY[name]}_metrics.csv").open(encoding="utf-8", newline="") as handle:
        raw = list(csv.DictReader(handle))
    results = {}
    per_class = []
    counts = np.asarray(seal["train_class_counts"], dtype=np.int64)
    expected = {(seed, arm) for seed in exp.seeds for arm in (STAGE1, *exp.arms)}
    reference_labels = None
    for row in raw:
        seed, arm = int(row["seed"]), row["arm"]
        if (seed, arm) not in expected or (seed, arm) in results or row["dataset"] != name:
            raise RuntimeError(f"Unexpected or duplicate final result in {name}: {seed} {arm}")
        with np.load(FINAL / f"{name}_seed{seed}_{arm}_predictions.npz", allow_pickle=False) as data:
            labels, predictions = data["labels"], data["predictions"]
        test_rows = exp.datasets[name]["test_images"]
        epoch = exp.protocol["stage1" if arm == STAGE1 else "stage2"]["epochs"]
        if int(row["epoch"]) != epoch or int(row["test_rows"]) != test_rows or len(labels) != test_rows:
            raise RuntimeError(f"Final result epoch or row count mismatch in {name}: {seed} {arm}")
        if reference_labels is None:
            reference_labels = labels
        elif not np.array_equal(reference_labels, labels):
            raise RuntimeError(f"Test label vector differs among paired results in {name}")
        metrics, support, correct, accuracy = classification_metrics(predictions, labels, counts)
        if any(float(row[metric]) != value for metric, value in metrics.items()):
            raise RuntimeError(f"Prediction and metric files disagree in {name}: {seed} {arm}")
        results[seed, arm] = {"dataset": name, "seed": seed, "arm": arm, "epoch": epoch,
                              "test_rows": len(labels), **metrics}
        for class_id, train_count in enumerate(counts.tolist()):
            group = "Many" if train_count > 100 else "Few" if train_count < 20 else "Medium"
            per_class.append({
                "dataset": name, "seed": seed, "arm": arm, "class_id": class_id,
                "train_count": train_count, "group": group, "test_support": int(support[class_id]),
                "correct": int(correct[class_id]), "accuracy": float(accuracy[class_id]),
            })
    if results.keys() != expected:
        raise RuntimeError(f"Incomplete fixed-epoch test results for {name}")
    return results, per_class


def dataset_markdown(name: str, results: dict) -> str:
    lines = [
        f"# {DISPLAY[name].replace('_', '-')} fixed test results", "",
        f"{results['test_rows']:,} test images. Metrics are percentages. Mean and sample SD use three independent seeds.",
        "", "## Fixed checkpoints", "",
        "| Arm | Overall | Many | Medium | Few |", "| --- | ---: | ---: | ---: | ---: |",
    ]
    for arm, summary in results["mean_sample_sd"].items():
        cells = [f"{summary[metric]['mean']:.4f} ± {summary[metric]['sample_sd']:.4f}" for metric in METRICS]
        lines.append(f"| {arm} | " + " | ".join(cells) + " |")
    lines.extend(["", "## Paired differences in percentage points", "",
                  "| Contrast | Overall | Many | Medium | Few |", "| --- | ---: | ---: | ---: | ---: |"])
    for contrast, summary in results["paired_mean_sample_sd"].items():
        cells = [f"{summary[metric]['mean']:+.4f} ± {summary[metric]['sample_sd']:.4f}" for metric in METRICS]
        lines.append(f"| {contrast} | " + " | ".join(cells) + " |")
    lines.extend(["", "Groups use official TRAIN counts. Many exceeds 100. Medium includes 20 through 100. Few is below 20."])
    return "\n".join(lines) + "\n"


def analyze(exp, prepared):
    evaluation = require_complete(FINAL, phase="evaluate-final", expected={
        "official_test_access": True, "stage2_verified": len(exp.datasets) * len(exp.seeds) * len(exp.arms),
    })
    with record_run(ANALYSIS, exp.config("analyze")) as status:
        summaries = {}
        all_deltas = []
        all_classes = []
        for name in exp.datasets:
            seal = prepared["datasets"][name]
            rows, per_class = read_dataset_results(exp, name, seal)
            write_csv(ANALYSIS / f"{name}_per_seed.csv", list(rows.values()))
            deltas = []
            for first, second in CONTRASTS:
                for seed in exp.seeds:
                    deltas.append({
                        "dataset": name, "seed": seed, "contrast": f"{first}-{second}",
                        **{metric: rows[seed, first][metric] - rows[seed, second][metric] for metric in METRICS},
                    })
            summary = {
                "test_rows": exp.datasets[name]["test_images"], "train_class_groups": seal["group_class_counts"],
                "per_seed": list(rows.values()), "paired_per_seed": deltas,
                "mean_sample_sd": {
                    arm: aggregate([rows[seed, arm] for seed in exp.seeds]) for arm in (STAGE1, *exp.arms)
                },
                "paired_mean_sample_sd": {
                    f"{first}-{second}": aggregate([row for row in deltas if row["contrast"] == f"{first}-{second}"])
                    for first, second in CONTRASTS
                },
            }
            summaries[name] = summary
            all_deltas.extend(deltas)
            all_classes.extend(per_class)
            with (ANALYSIS / f"{DISPLAY[name].upper()}_RESULTS.md").open("x", encoding="utf-8") as handle:
                handle.write(dataset_markdown(name, summary))
        write_csv(ANALYSIS / "paired_deltas.csv", all_deltas)
        write_csv(ANALYSIS / "per_class_accuracy.csv", all_classes)
        write_json(ANALYSIS / "cross_dataset_summary.json", {
            "datasets": summaries, "seeds": exp.seeds, "official_test_splits": evaluation["test_splits"],
            "fixed_stage1_epoch": exp.protocol["stage1"]["epochs"], "fixed_stage2_epoch": exp.protocol["stage2"]["epochs"],
            "statistical_significance_claimed": False,
        })
        header = (
            "# CC-SAM cross-dataset results\n\n"
            "Three independent Stage-I initializations per dataset and three paired Stage-II arms per seed. "
            "The tables describe fixed-checkpoint effects in percentage points. Inference is descriptive with three seeds.\n\n"
        )
        sections = [dataset_markdown(name, summaries[name]).replace("# ", "## ", 1) for name in exp.datasets]
        with (ANALYSIS / "CROSS_DATASET_RESULTS.md").open("x", encoding="utf-8") as handle:
            handle.write(header + "\n".join(sections))
        status.update(official_test_access=False, datasets=list(summaries), paired_rows=len(all_deltas))
