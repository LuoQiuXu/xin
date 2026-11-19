"""Paired development-data effects, frozen practical flags, and uncertainty."""
from __future__ import annotations

from itertools import combinations
import json
from pathlib import Path

import numpy as np
import torch

from causal_core import (ARMS, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, canonical_hash)
from oracle_runtime import (ROOT, SEEDS, code_hashes, guard_pass, inspect_parent, load_context,
                            new_csv, new_json, now, read_csv, read_json)
from isolated_data import file_sha

METRICS = ("overall_accuracy", "macro_accuracy", "many_accuracy", "few_accuracy", "fixed_many_margin")
CONTRASTS = (("BASE_C3", "C2"), ("TARGETED", "BASE_C3"),
             ("TARGETED", "RANDOM_MATCHED"), ("TARGETED", "FREQUENCY_MATCHED"),
             ("RANDOM_MATCHED", "BASE_C3"), ("FREQUENCY_MATCHED", "BASE_C3"),
             ("ORACLE_SELECTION", "BASE_C3"), ("ORACLE_SELECTION", "TARGETED"))


def stratified_draws(labels, repetitions=BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED):
    labels = np.asarray(labels)
    if labels.ndim != 1 or len(labels) == 0 or repetitions < 1:
        raise ValueError("Nonempty labels and positive bootstrap count required")
    rng = np.random.default_rng(seed)
    parts = []
    for cls in np.unique(labels):
        rows = np.flatnonzero(labels == cls)
        parts.append(rows[rng.integers(0, len(rows), size=(repetitions, len(rows)))])
    return np.concatenate(parts, axis=1)


def metric_samples(logits, labels, counts, competitors, draws):
    labels = np.asarray(labels, dtype=np.int64)
    logits = np.asarray(logits)
    if not np.isfinite(logits).all() or logits.shape != (len(labels), len(counts)):
        raise ValueError("Invalid logits")
    correct = (logits.argmax(1) == labels).astype(float) * 100
    group = np.asarray(counts)[labels]
    q = np.array([competitors.get(str(int(y)), 0) for y in labels], dtype=np.int64)
    margins = logits[np.arange(len(labels)), labels].astype(float) - logits[np.arange(len(labels)), q].astype(float)
    values, boot = {}, {}
    for metric, data, selected in (
        ("overall_accuracy", correct, np.ones(len(labels), bool)),
        ("many_accuracy", correct, group > 100),
        ("few_accuracy", correct, group < 20),
        ("fixed_many_margin", margins, group > 100),
    ):
        if not selected.any():
            raise ValueError(f"No examples for {metric}")
        values[metric] = float(data[selected].mean())
        # Class-stratified draws have fixed labels per column across repeats.
        columns = selected[draws[0]]
        boot[metric] = data[draws[:, columns]].mean(1)
    class_means, class_boot = [], []
    for cls in np.unique(labels):
        selected = labels == cls
        class_means.append(correct[selected].mean())
        class_boot.append(correct[draws[:, labels[draws[0]] == cls]].mean(1))
    values["macro_accuracy"] = float(np.mean(class_means))
    boot["macro_accuracy"] = np.mean(class_boot, axis=0)
    return values, boot


def _rank(values):
    values = np.asarray(values)
    order = np.argsort(values, kind="stable")
    result = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        result[order[start:stop]] = (start + stop - 1) / 2
        start = stop
    return result


def _score_agreement(context, eval_x, eval_y, destination):
    from run_causal_oracle import _source_score_rows
    records = []
    for seed in SEEDS:
        folder = ROOT / "calibration" / f"seed_{seed}"
        with np.load(folder / "source_components_epoch30.npz", allow_pickle=False) as components:
            sw, sb = torch.from_numpy(components["weight"]), torch.from_numpy(components["bias"])
        mapping = read_json(folder / "competitors.json")
        y = torch.from_numpy(eval_y)
        many = torch.as_tensor(context["counts"])[y] > 100
        q = torch.tensor([mapping.get(str(int(label)), 0) for label in eval_y])
        rows = _source_score_rows(seed, 30, sw, sb, torch.from_numpy(eval_x), y, q, many, context["counts"])
        for row in rows:
            row["evaluation_target_examples"] = row.pop("calibration_target_examples")
        new_csv(destination / f"evaluation_pair_scores_seed{seed}.csv", rows)
        cal = read_csv(folder / "pair_scores_epoch30.csv")
        lookup = {(int(r["source_class"]), int(r["target_many_class"])):
                  float(r["row_local_direct_fixed_margin_delta"]) for r in cal}
        a = np.array([lookup[(r["source_class"], r["target_many_class"])] for r in rows])
        b = np.array([r["row_local_direct_fixed_margin_delta"] for r in rows])
        corr = float(np.corrcoef(_rank(a), _rank(b))[0, 1]) if a.std() and b.std() else None
        records.append({"seed": seed, "pairs": len(a), "spearman_rank_correlation": corr,
                        "sign_agreement": float((np.sign(a) == np.sign(b)).mean()),
                        "used_to_modify_masks": False})
    return records


def run_analysis():
    from run_causal_oracle import _verify_postseal_data, metrics, require_preflight
    require_preflight()
    complete = read_json(ROOT / "formal_completed.json")
    if (complete.get("state") != "completed" or complete.get("seeds") != list(SEEDS)
            or complete.get("epochs") != 30 or complete.get("actual_optimizer_steps") != 91800
            or complete.get("code_sha256") != code_hashes()):
        raise RuntimeError("Formal completion gate failed; no analysis of incomplete formal runs")
    context = load_context(read_json(ROOT / "split_manifest.json"))
    eval_x, eval_y = _verify_postseal_data(context)
    draws = stratified_draws(eval_y)
    destination = ROOT / "analysis"
    destination.mkdir(exist_ok=False)
    per_seed, per_class, estimates, samples = [], [], {}, {}
    logits_all = {}
    for seed in SEEDS:
        folder = ROOT / "results" / f"seed_{seed}"
        status = read_json(folder / "status.json")
        if status.get("state") != "completed" or status.get("actual_optimizer_steps") != {a: 5100 for a in ARMS}:
            raise RuntimeError("Per-seed actual completion ledger invalid")
        for info in status["checkpoints"].values():
            if file_sha(ROOT / info["path"]) != info["sha256"]:
                raise RuntimeError("Final checkpoint SHA changed")
        path = folder / "final_logits.npz"
        if file_sha(path) != status["final_logits_sha256"]:
            raise RuntimeError("Final logits changed")
        mapping = read_json(ROOT / "calibration" / f"seed_{seed}" / "competitors.json")
        with np.load(path, allow_pickle=False) as saved:
            if not np.array_equal(saved["evaluation_labels"], eval_y):
                raise RuntimeError("Evaluation labels differ across seeds / arms")
            for arm in ARMS:
                logits = saved[arm]
                values, bootstrap = metric_samples(logits, eval_y, context["counts"], mapping, draws)
                _, pc = metrics(logits, eval_y, context["counts"], mapping)
                per_seed.append({"seed": seed, "arm": arm, **values})
                per_class.extend({"seed": seed, "arm": arm, **row} for row in pc)
                estimates[(seed, arm)], samples[(seed, arm)] = values, bootstrap
                logits_all[(seed, arm)] = logits.copy()
    deltas, intervals, summaries = [], [], []
    effects = [(arm, None) for arm in ARMS] + list(CONTRASTS)
    for treatment, control in effects:
        name = treatment if control is None else treatment + "_minus_" + control
        for metric in METRICS:
            points, boots = [], []
            for seed in SEEDS:
                point = estimates[(seed, treatment)][metric]
                boot = samples[(seed, treatment)][metric]
                if control is not None:
                    point -= estimates[(seed, control)][metric]
                    boot = boot - samples[(seed, control)][metric]
                points.append(point)
                boots.append(boot)
                lo, hi = np.quantile(boot, [0.025, 0.975])
                intervals.append({"contrast": name, "metric": metric, "seed": seed,
                                  "estimate": point, "bootstrap_low": float(lo), "bootstrap_high": float(hi)})
                if control is not None:
                    deltas.append({"seed": seed, "treatment": treatment, "control": control,
                                   "metric": metric, "difference": point})
            mean, sd = float(np.mean(points)), float(np.std(points, ddof=1))
            lo, hi = np.quantile(np.mean(boots, axis=0), [0.025, 0.975])
            summaries.append({"contrast": name, "metric": metric, "mean": mean, "sample_SD": sd,
                              "bootstrap_low": float(lo), "bootstrap_high": float(hi)})
            intervals.append({"contrast": name, "metric": metric, "seed": "fixed_three_seed_mean",
                              "estimate": mean, "bootstrap_low": float(lo), "bootstrap_high": float(hi)})
    control_checks = []
    combined_positive = []
    for seed in SEEDS:
        both = True
        for control in ("RANDOM_MATCHED", "FREQUENCY_MATCHED"):
            acc = estimates[(seed, "TARGETED")]["many_accuracy"] - estimates[(seed, control)]["many_accuracy"]
            margin = estimates[(seed, "TARGETED")]["fixed_many_margin"] - estimates[(seed, control)]["fixed_many_margin"]
            passed = acc > 0 and margin > 0
            both &= passed
            control_checks.append({"seed": seed, "control": control, "many_gain_pp": acc,
                                   "fixed_margin_gain": margin, "both_primary_positive": passed})
        combined_positive.append(both)
    summary_lookup = {(r["contrast"], r["metric"]): r for r in summaries}
    mean_directions = all(summary_lookup[("TARGETED_minus_" + control, metric)]["mean"] > 0
                          for control in ("RANDOM_MATCHED", "FREQUENCY_MATCHED")
                          for metric in ("many_accuracy", "fixed_many_margin"))
    few_guard = summary_lookup[("TARGETED_minus_BASE_C3", "few_accuracy")]["mean"] >= -2
    supported = sum(combined_positive) >= 2 and mean_directions and few_guard
    oracle = {}
    for oracle_name, actual in (("ORACLE_SUPPRESSION", "TARGETED"), ("ORACLE_SELECTION", "ORACLE_SELECTION")):
        contrast = actual + "_minus_BASE_C3"
        raw = [{"seed": seed, **{metric: estimates[(seed, actual)][metric] - estimates[(seed, "BASE_C3")][metric]
                                 for metric in METRICS}} for seed in SEEDS]
        means = {m: summary_lookup[(contrast, m)]["mean"] for m in METRICS}
        positive = sum(r["many_accuracy"] > 0 and r["fixed_many_margin"] > 0 for r in raw)
        reaches = (means["many_accuracy"] >= 2.0 and means["fixed_many_margin"] >= 0.10
                   and means["few_accuracy"] >= -2.0 and positive >= 2)
        oracle[oracle_name] = {"actual_arm": actual, "same_result_as_TARGETED": actual == "TARGETED",
                              "per_seed": raw, "mean_effects": means,
                              "seeds_both_primary_positive": positive,
                              "preregistered_practical_effect_flag": bool(reaches),
                              "headroom": "substantial_under_preregistered_practical_cutoffs" if reaches else "small_or_inconsistent_under_preregistered_cutoffs",
                              "automatic_Gate_admission": False}
    class_checks = []
    for seed in SEEDS:
        for control in ("RANDOM_MATCHED", "FREQUENCY_MATCHED"):
            target_pc = {r["class_id"]: r for r in per_class if r["seed"] == seed and r["arm"] == "TARGETED"}
            control_pc = {r["class_id"]: r for r in per_class if r["seed"] == seed and r["arm"] == control}
            acc = np.array([target_pc[c]["accuracy"] - control_pc[c]["accuracy"] for c in sorted(target_pc)])
            margin = np.array([target_pc[c]["fixed_margin"] - control_pc[c]["fixed_margin"] for c in sorted(target_pc)])
            leave_acc = (acc.sum() - acc) / (len(acc) - 1)
            leave_margin = (margin.sum() - margin) / (len(margin) - 1)
            class_checks.append({"seed": seed, "control": control,
                "many_classes": len(acc), "class_accuracy_effects_positive": int((acc > 0).sum()),
                "class_margin_effects_positive": int((margin > 0).sum()),
                "leave_one_class_out_accuracy_min": float(leave_acc.min()),
                "leave_one_class_out_margin_min": float(leave_margin.min()),
                "all_leave_one_out_primary_positive": bool((leave_acc > 0).all() and (leave_margin > 0).all())})
    stability = []
    for a, b in combinations(SEEDS, 2):
        ba = read_json(ROOT / "calibration" / f"seed_{a}" / "mask_bundle.json")
        bb = read_json(ROOT / "calibration" / f"seed_{b}" / "mask_bundle.json")
        for arm in ("TARGETED", "RANDOM_MATCHED", "FREQUENCY_MATCHED", "ORACLE_SELECTION"):
            sa, sb = set(map(tuple, ba["suppressed_pairs"][arm])), set(map(tuple, bb["suppressed_pairs"][arm]))
            stability.append({"seed_a": a, "seed_b": b, "arm": arm, "overlap": len(sa & sb),
                              "jaccard": len(sa & sb) / len(sa | sb) if sa | sb else 1.0})
    agreement = _score_agreement(context, eval_x, eval_y, destination)
    new_csv(destination / "per_seed_final_metrics.csv", per_seed)
    new_csv(destination / "per_class_many_final_metrics.csv", per_class)
    new_csv(destination / "paired_deltas.csv", deltas)
    new_csv(destination / "bootstrap_intervals.csv", intervals)
    new_csv(destination / "mean_SD_effects.csv", summaries)
    new_csv(destination / "class_robustness.csv", class_checks)
    new_csv(destination / "mask_stability.csv", stability)
    # Recheck parent fingerprints at completion, including read-only historical
    # artifacts, rather than claiming immutability from intention alone.
    parent_check = inspect_parent()
    decision = {
        "causal_evidence": "supported_under_predeclared_controlled_intervention_rule" if supported else "not_supported",
        "targeted_beats_both_controls_both_primary_seeds": int(sum(combined_positive)),
        "targeted_control_mean_directions_positive": bool(mean_directions),
        "targeted_few_guard": bool(few_guard), "per_seed_control_checks": control_checks,
        "oracle": oracle, "Gate_admission": "HUMAN_REVIEW_ONLY",
        "supporting_evidence": [r for r in control_checks if r["both_primary_positive"]],
        "opposing_evidence": [r for r in control_checks if not r["both_primary_positive"]],
        "class_robustness": class_checks,
        "recommendation": "Review effects, controls, oracle headroom and class robustness together; no automatic Gate admission or implementation.",
        "limitations": ["TRAIN-derived development split; parent work already examined the original 1980 holdout examples",
                        "Clean Stage-I checkpoint was originally selected on that 1980-example holdout; this split cannot undo historical selection",
                        "class 0 absent; Many metrics cover 34 of 35 TRAIN-Many classes",
                        "conditional bootstrap uncertainty given three fixed training seeds, not population inference",
                        "local analytic direct-gradient intervention; no new end-to-end objective or global upper-bound guarantee",
                        "Oracle Selection retains K, while suppression removes K; different constraints must not be confused",
                        "BASE_C3 autograd and intervention analytic decomposition have measured FP32 rounding residuals"],
    }
    report = {"completed_at": now(), "dataset": "CIFAR-100-LT IF100 TRAIN-derived development",
              "seeds": list(SEEDS), "epochs": 30, "actual_arms": list(ARMS),
              "alias": {"ORACLE_SUPPRESSION": "TARGETED"}, "per_seed_final_metrics": per_seed,
              "mean_SD_effects": summaries, "decision": decision, "calibration_evaluation_agreement": agreement,
              "bootstrap": {"replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED,
                            "stratification": "within each Evaluation class; same draw across paired arms and seeds",
                            "mean_CI_scope": "conditional on these three training seeds"},
              "official_TEST_used": False, "test_guard": guard_pass(),
              "parent_unchanged_verified": parent_check["passed"], "mask_seal_sha256": file_sha(ROOT / "mask_seal.json")}
    new_json(destination / "summary.json", report)
    new_json(destination / "DECISION.json", decision)
    lines = ["# CC-MIA Causal / Oracle Report", "", f"Completed: {now()}", "",
             "This is a held-out TRAIN-derived development analysis. No official TEST was accessed.", "",
             "All three seeds completed 30 epochs for six paired actual arms (91,800 optimizer steps). "
             "ORACLE_SUPPRESSION is exactly TARGETED and shares its result.", "",
             "| Seed | Arm | Overall % | Macro % | Many % | Few % | Fixed Many margin |",
             "|---|---|---|---|---|---|---|"]
    for r in per_seed:
        lines.append(f"| {r['seed']} | {r['arm']} | {r['overall_accuracy']:.4f} | {r['macro_accuracy']:.4f} | "
                     f"{r['many_accuracy']:.4f} | {r['few_accuracy']:.4f} | {r['fixed_many_margin']:.6f} |")
    lines += ["", "Full mean, sample SD, paired effects and class-stratified bootstrap 95% intervals: "
              "`analysis/mean_SD_effects.csv`, `analysis/paired_deltas.csv`, `analysis/bootstrap_intervals.csv`.", "",
              f"TARGETED beats BOTH matched controls on BOTH primaries in {sum(combined_positive)}/3 seeds. "
              f"Controlled-intervention decision: **{decision['causal_evidence']}**.", ""]
    for name, item in oracle.items():
        effect = item["mean_effects"]
        lines += [f"{name}: mean Many {effect['many_accuracy']:+.4f}pp; fixed margin "
                  f"{effect['fixed_many_margin']:+.6f}; Few {effect['few_accuracy']:+.4f}pp; "
                  f"both primary directions positive in {item['seeds_both_primary_positive']}/3 seeds. "
                  f"Preregistered practical-effect flag: {item['preregistered_practical_effect_flag']}.", ""]
    lines += ["Oracle Suppression has zero distance from TARGETED by construction; this is not "
              "independent confirmation. Oracle Selection is a fixed-K retain objective, not a "
              "theoretical global bound. `ORACLE_SELECTION_minus_TARGETED` reports its observed gap.", "",
              "Gate admission: **HUMAN_REVIEW_ONLY**. No Gate module was designed or trained.", "",
              "## Evidence limits", ""]
    lines.extend("- " + item for item in decision["limitations"])
    lines += ["", "All parent source and artifact fingerprints were rechecked at completion. "
              "Masks were sealed before Evaluation was opened and were never revised.", ""]
    path = ROOT / "CAUSAL_ORACLE_REPORT.md"
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines))
    print(f"ANALYSIS COMPLETE: {decision['causal_evidence']}; Gate admission requires human review", flush=True)


if __name__ == "__main__":
    run_analysis()
