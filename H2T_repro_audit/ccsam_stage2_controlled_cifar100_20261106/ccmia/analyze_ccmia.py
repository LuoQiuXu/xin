from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SEEDS = (123, 124, 125)
FINAL_EPOCH = 30


def read_csv(path):
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, fieldnames, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def mean_sd(values):
    values = [float(value) for value in values]
    if not values:
        return None, None
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, 0.0
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return mean, math.sqrt(variance)


def fmt(mean, sd, digits=4):
    if mean is None:
        return "NA"
    return f"{mean:.{digits}f} ± {sd:.{digits}f}"


def final_rows_by_seed(rows, epoch, arm=None):
    selected = [row for row in rows if int(row["epoch"]) == epoch and (arm is None or row["arm"] == arm)]
    return selected


def analyze():
    completion_path = ROOT / "formal_completed.json"
    if not completion_path.is_file():
        raise FileNotFoundError("Missing formal_completed.json; formal training is not complete.")
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    if completion.get("state") != "completed" or completion.get("official_TEST_used") is not False:
        raise RuntimeError("Formal completion record is incomplete or does not certify TEST-free execution.")
    if sorted(completion.get("seeds", [])) != list(SEEDS):
        raise RuntimeError("Formal completion record does not contain exactly the frozen three seeds.")
    test_guard = completion.get("test_guard", {})
    if int(test_guard.get("official_test_file_open_attempts", -1)) != 0:
        raise RuntimeError("Formal completion record reports official TEST file access.")
    if int(test_guard.get("CIFAR100_train_false_attempts", -1)) != 0:
        raise RuntimeError("Formal completion record reports CIFAR100(train=False) access.")

    all_metrics = []
    all_decomp = []
    all_source = []
    all_group_grad = []
    all_steps = []
    for seed in SEEDS:
        run_dir = ROOT / "results" / f"seed_{seed}"
        status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
        if status.get("state") != "completed" or status.get("official_TEST_used") is not False:
            raise RuntimeError(f"Seed {seed} is not completed and TEST-free.")
        if int(status.get("paired_batch_steps_completed", -1)) != 5100:
            raise RuntimeError(f"Seed {seed} does not contain the expected 5,100 paired batch steps.")
        if int(status.get("paired_C0_trace_batches_matched", -1)) != 5100:
            raise RuntimeError(f"Seed {seed} did not match all 5,100 C0 reference batches.")
        if status.get("backbone_and_bn_bitwise_unchanged") is not True:
            raise RuntimeError(f"Seed {seed} did not certify bitwise-frozen backbone/BN state.")

        seed_metrics = read_csv(run_dir / "margin_metrics.csv")
        seed_per_class = read_csv(run_dir / "per_class_margin.csv")
        seed_decomp = read_csv(run_dir / "margin_decomposition.csv")
        seed_source = read_csv(run_dir / "source_margin_influence.csv")
        seed_group_grad = read_csv(run_dir / "source_target_group_gradient.csv")
        seed_steps = read_csv(run_dir / "step_influence.csv")
        seed_trace = read_csv(run_dir / "batch_trace.csv")
        seed_competitors = read_csv(run_dir / "fixed_competitors.csv")
        expected_rows = {
            "margin_metrics": (len(seed_metrics), 62),
            "per_class_margin": (len(seed_per_class), 2108),
            "margin_decomposition": (len(seed_decomp), 30),
            "source_margin_influence": (len(seed_source), 66300),
            "source_target_group_gradient": (len(seed_group_grad), 5850),
            "step_influence": (len(seed_steps), 5100),
            "batch_trace": (len(seed_trace), 5100),
            "fixed_competitors": (len(seed_competitors), 680),
        }
        for name, (actual_count, expected_count) in expected_rows.items():
            if actual_count != expected_count:
                raise RuntimeError(
                    f"Seed {seed} {name} row count is {actual_count}, expected {expected_count}."
                )
        for row in seed_metrics:
            for field in (
                "overall", "many_accuracy", "mediumN_accuracy", "few_accuracy",
                "fixed_margin_mean", "dynamic_margin_mean",
            ):
                if not math.isfinite(float(row[field])):
                    raise RuntimeError(f"Seed {seed} has a non-finite {field} in margin_metrics.csv.")
        for row in seed_decomp:
            for field in (
                "actual_c3_minus_c2_fixed_margin_mean", "direct_perturbation_component_mean",
                "state_feedback_component_mean", "weight_decay_component_mean",
                "reconstruction_residual_component_mean", "parameter_reconstruction_relative",
            ):
                if not math.isfinite(float(row[field])):
                    raise RuntimeError(f"Seed {seed} has a non-finite {field} in margin_decomposition.csv.")

        all_metrics.extend(seed_metrics)
        all_decomp.extend(seed_decomp)
        all_source.extend(seed_source)
        all_group_grad.extend(seed_group_grad)
        all_steps.extend(seed_steps)

    (ROOT / "analysis").mkdir(exist_ok=True)
    final_metrics = final_rows_by_seed(all_metrics, FINAL_EPOCH)
    expected_metric_rows = len(SEEDS) * 2
    if len(final_metrics) != expected_metric_rows:
        raise RuntimeError(f"Expected {expected_metric_rows} final C2/C3 metric rows, got {len(final_metrics)}.")
    metric_lookup = {(int(row["seed"]), row["arm"]): row for row in final_metrics}
    paired_rows = []
    metrics_to_compare = (
        "overall", "macro", "many_accuracy", "mediumN_accuracy", "few_accuracy",
        "fixed_margin_mean", "fixed_margin_median", "fixed_margin_p10",
        "fixed_margin_negative_fraction", "dynamic_margin_mean", "dynamic_margin_negative_fraction",
    )
    for seed in SEEDS:
        c2 = metric_lookup[(seed, "C2")]
        c3 = metric_lookup[(seed, "C3")]
        row = {"seed": seed}
        for key in metrics_to_compare:
            row[f"C2_{key}"] = float(c2[key])
            row[f"C3_{key}"] = float(c3[key])
            row[f"C3_minus_C2_{key}"] = float(c3[key]) - float(c2[key])
        paired_rows.append(row)
    write_csv(
        ROOT / "analysis" / "final_paired_metrics.csv",
        tuple(paired_rows[0].keys()),
        paired_rows,
    )

    decomp_final = [row for row in all_decomp if int(row["epoch"]) == FINAL_EPOCH]
    if len(decomp_final) != len(SEEDS):
        raise RuntimeError("Final margin decomposition is missing one or more seeds.")
    decomp_by_seed = {int(row["seed"]): row for row in decomp_final}
    for seed in SEEDS:
        paired_fixed_delta = float(metric_lookup[(seed, "C3")]["fixed_margin_mean"]) - float(
            metric_lookup[(seed, "C2")]["fixed_margin_mean"]
        )
        projected_fixed_delta = float(decomp_by_seed[seed]["actual_c3_minus_c2_fixed_margin_mean"])
        if abs(paired_fixed_delta - projected_fixed_delta) > 1e-4:
            raise RuntimeError(
                f"Seed {seed} fixed-margin evaluation and parameter projection disagree: "
                f"{paired_fixed_delta} vs {projected_fixed_delta}"
            )

    step_summary = []
    for seed in SEEDS:
        rows = [row for row in all_steps if int(row["seed"]) == seed]
        if len(rows) != 5100:
            raise RuntimeError(f"Seed {seed} step influence trace length is {len(rows)}, expected 5,100.")
        step_summary.append({
            "seed": seed,
            "direct_step_mean": sum(float(r["direct_gradient_margin_step_mean"]) for r in rows) / len(rows),
            "direct_step_negative_batch_fraction": sum(
                float(r["direct_gradient_margin_step_negative_fraction"]) for r in rows
            ) / len(rows),
            "state_step_mean": sum(float(r["state_feedback_margin_step_mean"]) for r in rows) / len(rows),
            "total_gradient_step_mean": sum(float(r["total_gradient_margin_step_mean"]) for r in rows) / len(rows),
            "max_gradient_decomposition_error": max(float(r["gradient_decomposition_max_abs"]) for r in rows),
            "max_parameter_reconstruction_relative": max(float(r["parameter_reconstruction_relative"]) for r in rows),
            "max_momentum_buffer_reconstruction_relative": max(
                float(r["momentum_buffer_reconstruction_relative"]) for r in rows
            ),
        })

    # Source-to-target direct margin effects at the fixed epoch-30 classifier state.
    pair_values = defaultdict(list)
    pair_step_values = defaultdict(list)
    pair_step_negative = defaultdict(list)
    source_values = defaultdict(list)
    for row in all_source:
        if int(row["epoch"]) != FINAL_EPOCH:
            continue
        key = (int(row["source_class"]), int(row["target_many_class"]))
        pair_values[key].append(float(row["cumulative_direct_margin_delta_mean"]))
        pair_step_values[key].append(float(row["mean_direct_gradient_step_margin"]))
        pair_step_negative[key].append(float(row["mean_direct_gradient_step_negative_fraction"]))
        source_values[int(row["source_class"])].append(
            (float(row["cumulative_direct_margin_delta_mean"]), int(row["target_validation_samples"]))
        )
    pair_rows = []
    for (source_id, target_id), values in sorted(pair_values.items()):
        mean, sd = mean_sd(values)
        step_mean, step_sd = mean_sd(pair_step_values[(source_id, target_id)])
        neg_mean, neg_sd = mean_sd(pair_step_negative[(source_id, target_id)])
        pair_rows.append({
            "source_class": source_id,
            "target_many_class": target_id,
            "seed_count": len(values),
            "mean_cumulative_direct_margin_delta": mean,
            "sd_cumulative_across_seeds": sd,
            "mean_direct_gradient_step_margin": step_mean,
            "sd_direct_gradient_step_margin": step_sd,
            "mean_direct_gradient_step_negative_fraction": neg_mean,
            "sd_direct_gradient_step_negative_fraction": neg_sd,
            "direction_consistent_negative": all(value < 0 for value in values),
        })
    pair_rows.sort(key=lambda row: row["mean_cumulative_direct_margin_delta"])
    write_csv(
        ROOT / "analysis" / "source_to_many_margin_pairs_epoch30.csv",
        ("source_class", "target_many_class", "seed_count", "mean_cumulative_direct_margin_delta",
         "sd_cumulative_across_seeds", "mean_direct_gradient_step_margin",
         "sd_direct_gradient_step_margin", "mean_direct_gradient_step_negative_fraction",
         "sd_direct_gradient_step_negative_fraction", "direction_consistent_negative"),
        pair_rows,
    )

    source_rows = []
    for source_id, pairs in source_values.items():
        weighted_sum = sum(value * n for value, n in pairs)
        total_n = sum(n for _, n in pairs)
        seed_level = []
        for seed in SEEDS:
            vals = [
                float(row["cumulative_direct_margin_delta_mean"])
                for row in all_source
                if int(row["epoch"]) == FINAL_EPOCH
                and int(row["seed"]) == seed
                and int(row["source_class"]) == source_id
            ]
            if vals:
                # Equal weighting across target Many classes is retained for seed consistency.
                seed_level.append(sum(vals) / len(vals))
        mean, sd = mean_sd(seed_level)
        source_rows.append({
            "source_class": source_id,
            "source_train_count": next(
                int(row["source_train_count"]) for row in all_source
                if int(row["epoch"]) == FINAL_EPOCH and int(row["source_class"]) == source_id
            ),
            "source_group": next(
                row["source_group"] for row in all_source
                if int(row["epoch"]) == FINAL_EPOCH and int(row["source_class"]) == source_id
            ),
            "seed_count": len(seed_level),
            "mean_direct_margin_contribution_across_many_targets": mean,
            "sd_across_seeds": sd,
            "negative_in_all_seeds": len(seed_level) == len(SEEDS) and all(value < 0 for value in seed_level),
            "target_weighted_mean_all_seed_pairs": weighted_sum / total_n if total_n else None,
        })
    source_rows.sort(key=lambda row: row["mean_direct_margin_contribution_across_many_targets"])
    write_csv(
        ROOT / "analysis" / "source_class_direct_margin_epoch30.csv",
        ("source_class", "source_train_count", "source_group", "seed_count",
         "mean_direct_margin_contribution_across_many_targets", "sd_across_seeds",
         "negative_in_all_seeds", "target_weighted_mean_all_seed_pairs"),
        source_rows,
    )

    group_final = [row for row in all_group_grad if int(row["epoch"]) == FINAL_EPOCH]
    group_aggregate = []
    for group_name in ("Many", "MediumN", "Few"):
        values = [
            float(row["mean_batch_gradient_frobenius_all_steps"])
            for row in group_final if row["target_classifier_group"] == group_name
        ]
        mean, sd = mean_sd(values)
        group_aggregate.append({
            "target_classifier_group": group_name,
            "source_class_rows": len(values),
            "mean_source_to_target_group_gradient_frobenius": mean,
            "sd_across_source_seed_rows": sd,
        })

    paired_summary = {}
    for key in metrics_to_compare:
        mean, sd = mean_sd([row[f"C3_minus_C2_{key}"] for row in paired_rows])
        paired_summary[key] = {"mean": mean, "sample_sd": sd}

    decomp_summary = {}
    for key in (
        "actual_c3_minus_c2_fixed_margin_mean",
        "direct_perturbation_component_mean",
        "state_feedback_component_mean",
        "weight_decay_component_mean",
        "reconstruction_residual_component_mean",
        "direct_negative_fraction",
        "state_negative_fraction",
        "actual_negative_fraction",
        "margin_reconstruction_rmse",
    ):
        mean, sd = mean_sd([float(row[key]) for row in decomp_final])
        decomp_summary[key] = {"mean": mean, "sample_sd": sd}

    direct_negative_seeds = sum(float(row["direct_perturbation_component_mean"]) < 0 for row in decomp_final)
    actual_negative_seeds = sum(float(row["actual_c3_minus_c2_fixed_margin_mean"]) < 0 for row in decomp_final)
    many_accuracy_negative_seeds = sum(
        float(row["C3_minus_C2_many_accuracy"]) < 0 for row in paired_rows
    )
    max_reconstruction = max(
        float(row["parameter_reconstruction_relative"]) for row in decomp_final
    )
    if actual_negative_seeds >= 2 and direct_negative_seeds >= 2 and max_reconstruction <= 5e-3:
        interpretation = "SUPPORTIVE: the direct perturbation component and the observed fixed-pair Many margin shift are negative in at least two of three seeds; inspect source/target consistency before making a causal claim."
    elif actual_negative_seeds >= 2 and direct_negative_seeds < 2:
        interpretation = "PARTIAL: Many margins decline, but the direct perturbation component is not consistently negative; state/trajectory feedback or another pathway may dominate."
    elif direct_negative_seeds >= 2 and actual_negative_seeds < 2:
        interpretation = "LOCAL-ONLY: the perturbation produces negative local margin pressure, but it does not consistently translate into an epoch-30 Many margin decline."
    else:
        interpretation = "NOT SUPPORTED: the prespecified paired margin evidence does not consistently support the proposed harmful cross-class perturbation effect."

    summary = {
        "state": "analyzed",
        "seeds": list(SEEDS),
        "primary_epoch": FINAL_EPOCH,
        "official_TEST_used": False,
        "validation_partition": "1,980-image TRAIN-derived holdout; development diagnostic, not an independent test set",
        "missing_validation_class": 0,
        "paired_C2_C3_accuracy_and_margin_deltas": paired_summary,
        "epoch30_parameter_margin_decomposition": decomp_summary,
        "step_level_gradient_summary": step_summary,
        "source_to_target_group_gradient_summary": group_aggregate,
        "direct_negative_component_seeds": direct_negative_seeds,
        "actual_negative_margin_seeds": actual_negative_seeds,
        "many_accuracy_negative_seeds": many_accuracy_negative_seeds,
        "max_parameter_reconstruction_relative": max_reconstruction,
        "interpretation": interpretation,
        "important_scope": [
            "The direct component is defined at the current C3 classifier state: grad(L_C3 at W3)-grad(L_C2 at W3).",
            "The state-feedback component is grad(L_C2 at W3)-grad(L_C2 at W2).",
            "Momentum and coupled weight decay are propagated separately; reconstruction residual is reported.",
            "The decomposition is an ordered path decomposition, not a unique causal attribution independent of decomposition order.",
            "The audit does not establish independent-test generalization, statistical significance or paper-level novelty.",
        ],
    }
    (ROOT / "analysis").mkdir(exist_ok=True)
    (ROOT / "analysis" / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )

    lines = [
        "# CC-MIA 实验结果",
        "",
        f"状态：{interpretation}",
        "",
        "## 1. 固定第 30 轮 C3 − C2 配对结果",
        "",
        "| 指标 | 均值 ± 样本标准差 |",
        "|---|---:|",
    ]
    display_names = {
        "overall": "Overall accuracy (pp)",
        "many_accuracy": "Many accuracy (pp)",
        "mediumN_accuracy": "MediumN accuracy (pp)",
        "few_accuracy": "Few accuracy (pp)",
        "fixed_margin_mean": "固定竞争类别 Many margin",
        "dynamic_margin_mean": "动态竞争类别 Many margin",
        "fixed_margin_negative_fraction": "固定 margin 负值比例",
    }
    for key, label in display_names.items():
        value = paired_summary[key]
        lines.append(f"| {label} | {fmt(value['mean'], value['sample_sd'])} |")

    lines += [
        "",
        "## 2. Many 固定决策间隔的参数路径分解",
        "",
        "| 分量 | 均值 ± 样本标准差 |",
        "|---|---:|",
    ]
    for key, label in (
        ("actual_c3_minus_c2_fixed_margin_mean", "实际 C3 − C2 margin"),
        ("direct_perturbation_component_mean", "直接扰动分量"),
        ("state_feedback_component_mean", "训练状态反馈分量"),
        ("weight_decay_component_mean", "权重衰减分量"),
        ("reconstruction_residual_component_mean", "重构残差"),
    ):
        value = decomp_summary[key]
        lines.append(f"| {label} | {fmt(value['mean'], value['sample_sd'], 6)} |")

    lines += [
        "",
        "## 3. 来源类别与目标 Many 类别",
        "",
        "最负的来源类别—目标类别组合（按三个 seed 的均值排序）：",
        "",
        "| 来源类 | 目标 Many 类 | 累积直接分量 | 每步直接压力 | 每步负向比例 |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in pair_rows[:15]:
        lines.append(
            f"| {row['source_class']} | {row['target_many_class']} | "
            f"{row['mean_cumulative_direct_margin_delta']:.6f} | "
            f"{row['mean_direct_gradient_step_margin']:.6g} | "
            f"{row['mean_direct_gradient_step_negative_fraction']:.3f} |"
        )

    lines += [
        "",
        "## 4. 梯度指向的目标分类器行",
        "",
        "| 目标分类器行组 | 平均来源梯度 Frobenius 范数 |",
        "|---|---:|",
    ]
    for row in group_aggregate:
        lines.append(
            f"| {row['target_classifier_group']} | "
            f"{row['mean_source_to_target_group_gradient_frobenius']:.6f} ± "
            f"{row['sd_across_source_seed_rows']:.6f} |"
        )

    lines += [
        "",
        "## 5. 解释边界",
        "",
        "- C2/C3 使用同一 Clean Stage-I 初始化、同一采样计划、同一增强图像和同一 GAP 特征轨迹。",
        "- 来源类别是产生扰动的训练标签；目标类别是被梯度更新的分类器参数行；二者没有混为一谈。",
        "- 直接扰动分量按来源类别分解；训练状态反馈与 SGD 动量、权重衰减分别记录。",
        "- 固定竞争类别在初始分类器上选择，保证 C2/C3 margin 差值使用同一组类别对；动态 margin 作为补充。",
        "- 验证集是 1,980 张 TRAIN-derived holdout，不是独立测试集；class 0 在 holdout 中没有样本。",
        "- 不从本实验单独声称统计显著性、独立测试集泛化或论文创新性。",
        "",
        "## 6. 逐种子结果",
        "",
        "| Seed | ΔOverall | ΔMany | ΔFew | 实际固定 margin Δ | 直接扰动分量 | 状态反馈 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for seed in SEEDS:
        paired = next(row for row in paired_rows if row["seed"] == seed)
        dec = decomp_by_seed[seed]
        lines.append(
            f"| {seed} | {paired['C3_minus_C2_overall']:+.3f} | "
            f"{paired['C3_minus_C2_many_accuracy']:+.3f} | "
            f"{paired['C3_minus_C2_few_accuracy']:+.3f} | "
            f"{float(dec['actual_c3_minus_c2_fixed_margin_mean']):+.6f} | "
            f"{float(dec['direct_perturbation_component_mean']):+.6f} | "
            f"{float(dec['state_feedback_component_mean']):+.6f} |"
        )

    (ROOT / "CCMIA_RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("CC-MIA analysis complete.")
    print(interpretation)


if __name__ == "__main__":
    analyze()
