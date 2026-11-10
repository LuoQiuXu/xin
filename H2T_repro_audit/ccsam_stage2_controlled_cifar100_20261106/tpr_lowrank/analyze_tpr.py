"""Read-only fixed-epoch analysis of TPR against sealed C0-C4 controls."""
from __future__ import annotations

import csv
import math
import statistics

import torch

from tpr_inputs import ARM, C1, C2, HISTORICAL_ARMS, OUT, STEPS, c, csv_rows, historical_dir, rel, require, run_dir
from train_tpr import BATCH_SCALARS, METRICS, PREPARATION, RANK, check_smoke_marker, exclusive_json, guard_record

ARMS = HISTORICAL_ARMS + (ARM,)
LABELS = dict(zip(ARMS, ("C0", "C1", "C2", "C3", "C4", "TPR"), strict=True))
CONTROLS = (C2, C1, HISTORICAL_ARMS[-1])


def describe(values):
    return {"per_seed": list(values), "mean": statistics.mean(values), "sample_sd": statistics.stdev(values)}


def write_csv(path, records):
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def load_result(seed, arm, snapshot):
    folder = run_dir(seed) if arm == ARM else historical_dir(seed, arm)
    status = c.read_json(folder / "status.json")
    require(status["state"] == "completed" and status["seed"] == seed and status["arm"] == arm
            and status["final_epoch"] == c.STAGE2_EPOCHS and len(status["epochs"]) == c.STAGE2_EPOCHS
            and status["official_TEST_used"] is False, f"Unsealed final result: {folder}")
    checkpoint = folder / "final_classifier.pth"
    require(c.sha256_file(checkpoint) == status["final_classifier_sha256"], f"Checkpoint SHA changed: {folder}")
    if arm != ARM:
        require(status["final_classifier_sha256"] == snapshot["protected"]["files"][rel(checkpoint)],
                f"Historical checkpoint SHA changed: {folder}")
    validation = status["final_validation"]
    require(validation == status["epochs"][-1]["validation"] and validation["samples"] == c.VALIDATION_SAMPLES
            and validation["class_counts"] == [0] + [20] * 99 and validation["missing_validation_classes"] == [0],
            f"Unexpected fixed-epoch holdout: {folder}")
    return status


def verify_new_result(seed, status, completion, inputs, snapshot, prepared):
    folder = run_dir(seed)
    records = csv_rows(folder / "metrics.csv")
    batches = csv_rows(folder / "batch_diagnostics.csv")
    expected_steps = c.STAGE2_EPOCHS * STEPS
    reference_path = historical_dir(seed, C1) / "batch_trace.csv"
    require(len(records) == c.STAGE2_EPOCHS and len(batches) == expected_steps
            and len(csv_rows(folder / "batch_trace.csv")) == expected_steps, f"Incomplete TPR CSVs for seed {seed}")
    require(c.sha256_file(folder / "batch_trace.csv") == c.sha256_file(reference_path)
            == status["batch_trace_sha256"] == snapshot["protected"]["files"][rel(reference_path)],
            f"Formal TPR/C1 trace SHA mismatch for seed {seed}")
    require([int(row["optimizer_step"]) for row in batches] == list(range(1, expected_steps + 1)),
            f"Optimizer steps are not consecutive for seed {seed}")
    for index, (row, epoch) in enumerate(zip(records, status["epochs"], strict=True)):
        require(int(row["epoch"]) == epoch["epoch"] == index + 1
                and all(math.isfinite(float(value)) for value in row.values())
                and int(row["steps"]) == STEPS and int(row["samples"]) == c.TRAIN_SAMPLES
                and all(float(value) == float(epoch["metrics"][key]) for key, value in row.items())
                and epoch["frozen_bitwise"] is True, f"Epoch evidence mismatch for seed {seed}, epoch {index + 1}")
    require(all(math.isfinite(float(row[key])) for row in batches for key in BATCH_SCALARS)
            and int(batches[0]["optimizer_step"]) == 1
            and float(batches[0]["grad_norm_a"]) == 0.0 and float(batches[0]["grad_norm_b"]) > 0.0,
            f"Invalid batch diagnostics for seed {seed}")
    require(all(float(row[f"weighted_{key}"]) <= 0.10 * float(row["c1"])
                for row in batches[:10] for key in ("sub", "div")),
            f"First-ten-batch regularizer scale gate failed for seed {seed}")
    require(status["optimizer_steps"] == expected_steps and status["samples_seen"] == c.TRAIN_SAMPLES * c.STAGE2_EPOCHS
            and status["step0_bitwise_equal_to_C2"] is True and status["frozen_bitwise_all_epochs"] is True
            and status["trainable_parameters"] == ["A", "B"], f"TPR formal invariant failed for seed {seed}")
    require(all(completion[key] == status[key] for key in completion),
            f"Completion marker disagrees with TPR status for seed {seed}")
    checkpoint = torch.load(folder / "final_classifier.pth", map_location="cpu", weights_only=True)
    require(checkpoint["epoch"] == c.STAGE2_EPOCHS and checkpoint["seed"] == seed and checkpoint["arm"] == ARM
            and checkpoint["base_checkpoint_sha256"] == snapshot["teachers"][str(seed)]["sha256"]
            == prepared["runs"][str(seed)]["base_checkpoint_sha256"]
            and checkpoint["clean_stage1_sha256"] == c.EXPECTED_CLEAN_SHA256
            and checkpoint["protocol_sha256"] == snapshot["protocol_sha256"]
            and checkpoint["source_sha256"] == snapshot["sources"]
            and checkpoint["optimizer"] == status["optimizer"] and checkpoint["official_TEST_used"] is False,
            f"Final TPR checkpoint provenance failed for seed {seed}")
    a, b = checkpoint["A"], checkpoint["B"]
    require(a.shape == (RANK, 64) and b.shape == (100, RANK)
            and a.dtype == b.dtype == torch.float32
            and torch.isfinite(a).all() and torch.isfinite(b).all()
            and torch.count_nonzero(b) > 0
            and torch.equal(checkpoint["tail_mask"], torch.from_numpy(inputs.counts <= 100).float()),
            f"Invalid final TPR factors for seed {seed}")
    residual = status["residual_diagnostics"]
    require(0 < residual["numerical_rank"] <= RANK
            and all(residual["row_norms"][i] == 0 for i in range(100) if inputs.counts[i] > 100),
            f"TPR residual rank or Many row invariant failed for seed {seed}")
    return {"matched_C1_batches": len(batches), "actual_steps": expected_steps, "actual_epochs": len(records),
            "actual_samples": sum(int(row["samples"]) for row in records),
            "checkpoint_sha256": status["final_classifier_sha256"], "trace_sha256": status["batch_trace_sha256"],
            "batch_diagnostics_sha256": c.sha256_file(folder / "batch_diagnostics.csv"),
            "teacher": status["final_teacher"], "residual": residual}


def analyze(inputs, snapshot, preparation):
    completed = c.read_json(OUT / "formal_completed.json")
    check_smoke_marker(snapshot)
    require(completed["passed"] is True and completed["arm"] == ARM and completed["seeds"] == list(c.SEEDS)
            and completed["official_TEST_used"] is False and completed["protected_C0_C4_unchanged"] is True
            and completed["preflight_sha256"] == c.sha256_file(OUT / "preflight_report.json")
            and completed["smoke_sha256"] == c.sha256_file(OUT / "smoke_report.json")
            and completed["preparation_sha256"] == c.sha256_file(PREPARATION)
            and completed["protocol_sha256"] == snapshot["protocol_sha256"]
            and completed["test_guard"] == c.read_json(OUT / "formal_test_guard.json")
            and [item["seed"] for item in completed["runs"]] == list(c.SEEDS),
            "Incomplete or stale TPR formal completion marker")
    formal_guard = completed["test_guard"]
    require(formal_guard["official_TEST_used"] is False and not formal_guard["blocked_attempts"]
            and formal_guard["official_test_file_open_attempts"] == formal_guard["CIFAR100_train_false_attempts"] == 0,
            "Invalid formal TEST guard")
    output = OUT / "analysis"
    require(not output.exists(), "Analysis already exists; refusing to overwrite")
    final_rows, class_rows, paired_rows = [], [], []
    scores, diagnostics = {}, {}
    for seed in c.SEEDS:
        runs = {arm: load_result(seed, arm, snapshot) for arm in ARMS}
        tpr = runs[ARM]
        scores[seed] = {arm: {metric: runs[arm]["final_validation"][metric] for metric in METRICS} for arm in ARMS}
        diagnostics[str(seed)] = verify_new_result(
            seed, tpr, next(item for item in completed["runs"] if item["seed"] == seed),
            inputs, snapshot, preparation
        )
        require(tpr["final_teacher"]["teacher_many_accuracy"] == scores[seed][C2]["Many"],
                f"TPR teacher and historical C2 Many accuracy differ for seed {seed}")
        for arm in ARMS:
            final_rows.append({"seed": seed, "arm": arm, **scores[seed][arm]})
        for class_id, count in enumerate(inputs.counts.tolist()):
            accuracies = {LABELS[arm]: runs[arm]["final_validation"]["per_class_accuracy"][class_id] for arm in ARMS}
            record = {
                "seed": seed, "class_id": class_id, "train_count": count,
                "validation_count": 0 if class_id == 0 else 20, **accuracies,
                "tpr_residual_row_l2": tpr["residual_diagnostics"]["row_norms"][class_id],
            }
            for arm in CONTROLS:
                previous = accuracies[LABELS[arm]]
                record[f"TPR_minus_{LABELS[arm]}_pp"] = None if previous is None else accuracies["TPR"] - previous
            class_rows.append(record)
    actual = {key: sum(item[key] for item in diagnostics.values())
              for key in ("actual_epochs", "actual_steps", "actual_samples", "matched_C1_batches")}
    require(tuple(actual.values()) == (90, 15300, 976230, 15300)
            and (completed["epochs_completed"], completed["optimizer_steps"], completed["train_samples_seen"])
            == (90, 15300, 976230), "Aggregate execution evidence differs from actual runs")

    averages = {LABELS[arm]: {metric: describe([scores[seed][arm][metric] for seed in c.SEEDS])
                             for metric in METRICS} for arm in ARMS}
    paired = {}
    for control in CONTROLS:
        name = f"TPR_minus_{LABELS[control]}"
        paired[name] = {}
        for metric in METRICS:
            values = [scores[seed][ARM][metric] - scores[seed][control][metric] for seed in c.SEEDS]
            stat = describe(values)
            paired[name][metric] = stat
            paired_rows.append({"contrast": name, "metric": metric, "mean_pp": stat["mean"],
                                "sample_sd_pp": stat["sample_sd"],
                                **{f"seed_{seed}_pp": value for seed, value in zip(c.SEEDS, values, strict=True)}})
    c1_vs_c2_few = averages["C1"]["Few"]["mean"] - averages["C2"]["Few"]["mean"]
    c1_vs_c2_many_gap = averages["C2"]["Many"]["mean"] - averages["C1"]["Many"]["mean"]
    require(c1_vs_c2_few > 0 and c1_vs_c2_many_gap > 0, "Unexpected sealed C1/C2 reference contrasts")
    result = {
        "analyzed_at": c.now(), "primary_epoch": c.STAGE2_EPOCHS, "seeds": list(c.SEEDS),
        "validation_partition": snapshot["validation_partition"], "official_TEST_used": False,
        "protected_C0_C4_unchanged": True, "actual_execution": actual,
        "arm_metrics": averages, "paired_delta_pp": paired, "tpr_diagnostics": diagnostics,
        "few_gain_fraction_vs_C1": paired["TPR_minus_C2"]["Few"]["mean"] / c1_vs_c2_few,
        "many_gap_fraction_recovered_vs_C1": paired["TPR_minus_C1"]["Many"]["mean"] / c1_vs_c2_many_gap,
        "formal_marker_sha256": c.sha256_file(OUT / "formal_completed.json"),
        "preparation_sha256": c.sha256_file(PREPARATION), "protocol_sha256": snapshot["protocol_sha256"],
        "source_sha256": snapshot["sources"], "analysis_test_guard": guard_record(),
    }

    output.mkdir(exist_ok=False)
    write_csv(output / "final_epoch_metrics.csv", final_rows)
    write_csv(output / "paired_deltas.csv", paired_rows)
    write_csv(output / "per_class.csv", class_rows)
    exclusive_json(output / "summary.json", result)
    with (output / "TPR_RESULTS.md").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(render_report(result, preparation) + "\n")
    print("TPR ANALYSIS PASS: fixed epoch 30, original C0-C4 seals intact, TEST=0.", flush=True)


def render_report(result, preparation):
    lines = [
        "# CC-SAM-TPR 第 30 轮结果",
        "",
        "三个 seed 各完成 30 轮，共 15,300 次更新。每批样本、标签、增强图像和 GAP 特征均与同 seed C1 配对。"
        "校验记录显示冻结 Backbone、BN、C2 基座及 Many 参数行在训练期间逐位一致。",
        "",
        "准确率来自同一 1,980 张 TRAIN holdout。class 0 参与训练，验证集该类样本数为零。"
        "三个 seed 共享同一冻结 Stage I Backbone。逐类变化用于描述。",
        "",
        "| 条件 | " + " | ".join(METRICS) + " |",
        "|---|" + "---:|" * len(METRICS),
    ]
    for label, metrics in result["arm_metrics"].items():
        lines.append("| " + " | ".join([label, *(f'{metrics[key]["mean"]:.3f} ± {metrics[key]["sample_sd"]:.3f}'
                                                  for key in METRICS)]) + " |")
    lines += ["", "## 配对差值", "", "单位为百分点，均值 ± 样本标准差。",
              "", "| 对照 | " + " | ".join(METRICS) + " |", "|---|" + "---:|" * len(METRICS)]
    for label, metrics in result["paired_delta_pp"].items():
        lines.append("| " + " | ".join([label, *(f'{metrics[key]["mean"]:+.3f} ± {metrics[key]["sample_sd"]:.3f}'
                                                  for key in METRICS)]) + " |")
    lines += [
        "",
        f'相对 C2 的 Few 收益占原 C1 收益的 {result["few_gain_fraction_vs_C1"]:.2%}。'
        f'相对 C1，Many 恢复了原 C1 到 C2 差距的 {result["many_gap_fraction_recovered_vs_C1"]:.2%}。'
        "这些是固定协议下的描述性变化。Many 参数行固定时，其他类别的 logits 仍能改变 Many 预测。",
        "",
        "## 残差和 Many 保持",
        "",
        "| Seed | 数值秩 | 残差 L2 | C2 top-8 能量比例 | 残差投影比例 | Many KL | Many top-1 一致率 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for seed, record in result["tpr_diagnostics"].items():
        graph = preparation["runs"][seed]
        residual, teacher = record["residual"], record["teacher"]
        lines.append(f'| {seed} | {residual["numerical_rank"]} | {residual["delta_weight_norm"]:.6f} | '
                     f'{graph["top8_singular_energy_fraction"]:.2%} | {residual["dominant_projection_ratio"]:.6f} | '
                     f'{teacher["many_kl"]:.6f} | {teacher["many_top1_agreement"]:.2%} |')
    lines += [
        "",
        "混淆图只使用全部 10,847 张未增强 Stage II TRAIN 图像。分数为双向行归一化混淆的平均值。"
        "L_div 对选中对的平方余弦取等权均值。",
        "",
        "逐 seed 指标与差值、逐类准确率及残差行范数见 [末轮指标](final_epoch_metrics.csv)、"
        "[配对差值](paired_deltas.csv) 和 [逐类结果](per_class.csv)。",
        "",
        "混淆对与 C2 SVD 见 [TRAIN 图与子空间](../train_confusion_graphs.json)。"
        "最终残差与有效权重奇异值、混淆对余弦、教师诊断及核验哈希见 [摘要](summary.json)。",
        "",
        "损失和梯度的逐轮记录见 [seed 123](../results/seed_123/TPR_LOW_RANK_TAIL/metrics.csv)、"
        "[seed 124](../results/seed_124/TPR_LOW_RANK_TAIL/metrics.csv) 和 "
        "[seed 125](../results/seed_125/TPR_LOW_RANK_TAIL/metrics.csv)。损失及梯度范数按批次等权汇总。",
        "",
        "封存 C0–C4 的完整性与 TEST 零访问见 [Preflight](../preflight_report.json) 和 "
        "[正式完成记录](../formal_completed.json)。",
    ]
    return "\n".join(lines)
