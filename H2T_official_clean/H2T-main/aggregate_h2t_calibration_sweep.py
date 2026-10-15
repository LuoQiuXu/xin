"""Aggregate the preregistered H2T rho sweep (three paired Stage-II seeds).

Inputs are produced only by ``audit_h2t_calibration.py``.  This script does
not train models or tune hyperparameters; it calculates descriptive summaries
and figures for the fixed rho grid {0, 0.10, 0.25, 0.40}.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SEEDS = (42, 43, 44)
CONDITIONS = (
    {
        "rho": 0.00,
        "effective_channels": 0,
        "label": "matched no-fusion",
        "directory": "calibration_audit_official_h2t_seed{seed}",
        "arm": "nofusion",
    },
    {
        "rho": 0.10,
        "effective_channels": 6,
        "label": "H2T",
        "directory": "calibration_audit_official_h2t_rho010_seed{seed}",
        "arm": "h2t",
    },
    {
        "rho": 0.25,
        "effective_channels": 16,
        "label": "H2T (official)",
        "directory": "calibration_audit_official_h2t_seed{seed}",
        "arm": "h2t",
    },
    {
        "rho": 0.40,
        "effective_channels": 25,
        "label": "H2T",
        "directory": "calibration_audit_official_h2t_rho040_seed{seed}",
        "arm": "h2t",
    },
)
GROUPS = ("Overall", "Many", "Medium", "Few")
METRICS = (
    "accuracy_pct",
    "ece15_pct",
    "nll",
    "brier",
    "wrong_confidence_pct",
    "wrong_rate_pct",
)


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / "calibration_audit_official_h2t_multiseed_rho_sweep",
    )
    return parser.parse_args()


def load_runs(root: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for condition in CONDITIONS:
        for seed in SEEDS:
            path = root / condition["directory"].format(seed=seed) / condition["arm"] / "metrics.json"
            if not path.is_file():
                raise FileNotFoundError(f"Missing audit result: {path}")
            payload = json.loads(path.read_text(encoding="utf-8"))
            if abs(float(payload["rho"]) - float(condition["rho"])) > 1e-9:
                raise ValueError(f"Unexpected rho in {path}: {payload['rho']}")
            records.append(
                {
                    "seed": seed,
                    "rho": float(condition["rho"]),
                    "effective_channels": int(condition["effective_channels"]),
                    "label": condition["label"],
                    "clean_test": payload["clean_test"],
                    "heldout_fusion_probe": payload["heldout_fusion_probe"],
                    "source_path": str(path),
                }
            )
    return records


def flatten(records: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for record in records:
        for scope in ("clean_test", "heldout_fusion_probe"):
            data = record[scope]
            if not data:
                continue
            for group in GROUPS:
                row = {
                    "seed": record["seed"],
                    "rho": record["rho"],
                    "effective_channels": record["effective_channels"],
                    "scope": scope,
                    "group": group,
                }
                for metric in METRICS:
                    row[metric] = data[group][metric]
                rows.append(row)
    return rows


def mean_sd(values: Sequence[float]) -> Tuple[float, float]:
    values = np.asarray(values, dtype=float)
    return float(values.mean()), float(values.std(ddof=1))


def summary_by_rho(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, Dict[float, Dict[str, float]]]]:
    summary: Dict[str, Dict[str, Dict[float, Dict[str, float]]]] = {}
    for scope in ("clean_test", "heldout_fusion_probe"):
        summary[scope] = {}
        for group in GROUPS:
            summary[scope][group] = {}
            filtered = [row for row in rows if row["scope"] == scope and row["group"] == group]
            for condition in CONDITIONS:
                rho = float(condition["rho"])
                matches = [row for row in filtered if float(row["rho"]) == rho]
                metrics: Dict[str, float] = {
                    "n_seeds": float(len(matches)),
                    "effective_channels": float(condition["effective_channels"]),
                }
                for metric in METRICS:
                    mean, sd = mean_sd([float(row[metric]) for row in matches])
                    metrics[f"{metric}_mean"] = mean
                    metrics[f"{metric}_sd"] = sd
                summary[scope][group][rho] = metrics
    return summary


def rankdata(values: np.ndarray) -> np.ndarray:
    # There are no ties in this small experiment, but average ranks keep this
    # well-defined if a tie occurs.
    order = np.argsort(values)
    ranks = np.empty_like(values, dtype=float)
    ranks[order] = np.arange(1, len(values) + 1, dtype=float)
    for value in np.unique(values):
        matching = values == value
        if matching.sum() > 1:
            ranks[matching] = ranks[matching].mean()
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])


def trend_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    rhos = np.asarray([float(item["rho"]) for item in CONDITIONS], dtype=float)
    for scope, group, metric in (
        ("clean_test", "Overall", "accuracy_pct"),
        ("clean_test", "Overall", "ece15_pct"),
        ("clean_test", "Overall", "nll"),
        ("clean_test", "Overall", "brier"),
        ("clean_test", "Overall", "wrong_confidence_pct"),
        ("clean_test", "Few", "accuracy_pct"),
        ("clean_test", "Few", "ece15_pct"),
        ("clean_test", "Few", "wrong_confidence_pct"),
        ("heldout_fusion_probe", "Few", "accuracy_pct"),
        ("heldout_fusion_probe", "Few", "ece15_pct"),
    ):
        slopes, correlations, shapes = [], [], []
        per_seed = []
        for seed in SEEDS:
            values = []
            for rho in rhos:
                matching = [
                    row
                    for row in rows
                    if row["scope"] == scope
                    and row["group"] == group
                    and row["seed"] == seed
                    and float(row["rho"]) == float(rho)
                ]
                assert len(matching) == 1
                values.append(float(matching[0][metric]))
            values_array = np.asarray(values)
            slopes.append(float(np.polyfit(rhos, values_array, 1)[0]))
            correlations.append(spearman(rhos, values_array))
            diffs = np.diff(values_array)
            if np.all(diffs > 0):
                shapes.append("递增")
            elif np.all(diffs < 0):
                shapes.append("递减")
            else:
                shapes.append("非单调")
            per_seed.append(values)
        signs = ["+" if value > 0 else "−" if value < 0 else "0" for value in slopes]
        results.append(
            {
                "scope": scope,
                "group": group,
                "metric": metric,
                "slope_mean_per_rho": float(np.mean(slopes)),
                "slope_by_seed": slopes,
                "slope_signs": " ".join(signs),
                "spearman_mean": float(np.mean(correlations)),
                "spearman_by_seed": correlations,
                "all_seed_slopes_same_sign": len(set(signs)) == 1,
                "shape_by_seed": shapes,
                "all_seed_same_monotonic_shape": len(set(shapes)) == 1 and shapes[0] != "非单调",
                "per_seed_values": per_seed,
            }
        )
    return results


def paired_delta_summary(rows: Sequence[Mapping[str, Any]], scope: str, group: str) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}
    for metric in METRICS:
        deltas = []
        for seed in SEEDS:
            base = next(
                row
                for row in rows
                if row["scope"] == scope and row["group"] == group and row["seed"] == seed and row["rho"] == 0.0
            )
            treatment = next(
                row
                for row in rows
                if row["scope"] == scope and row["group"] == group and row["seed"] == seed and row["rho"] == 0.25
            )
            deltas.append(float(treatment[metric]) - float(base[metric]))
        mean, sd = mean_sd(deltas)
        summary[metric] = {"mean": mean, "sd": sd, "values": deltas}
    return summary


def group_stratified_ece(records: Sequence[Mapping[str, Any]], scope: str) -> Dict[float, Dict[str, Any]]:
    """Frequency-group weighted ECE, separate from the usual pooled ECE.

    Pooled ECE takes an absolute gap only after examples from every frequency
    group have been mixed within each confidence bin.  It therefore cannot be
    attributed by adding Many/Medium/Few ECE values.  This diagnostic instead
    computes ECE within each group and weights by the group test-sample count.
    It aids attribution; it does not replace standard pooled ECE.
    """
    output: Dict[float, Dict[str, Any]] = {}
    for condition in CONDITIONS:
        rho = float(condition["rho"])
        values = []
        for seed in SEEDS:
            record = next(
                item for item in records if item["seed"] == seed and float(item["rho"]) == rho
            )
            data = record[scope]
            frequency_groups = ("Many", "Medium", "Few")
            total = sum(int(data[group]["n"]) for group in frequency_groups)
            values.append(
                sum(
                    int(data[group]["n"]) / total * float(data[group]["ece15_pct"])
                    for group in frequency_groups
                )
            )
        mean, sd = mean_sd(values)
        output[rho] = {"mean": mean, "sd": sd, "values": values}
    return output


def plot_curves(
    summary: Mapping[str, Any],
    output: Path,
    scope: str,
    group: str,
    specs: Sequence[Tuple[str, str]],
    title: str,
) -> None:
    rhos = np.asarray([float(item["rho"]) for item in CONDITIONS])
    channels = [int(item["effective_channels"]) for item in CONDITIONS]
    cols = 2
    rows = int(np.ceil(len(specs) / cols))
    figure, axes = plt.subplots(rows, cols, figsize=(10, 3.65 * rows), dpi=180)
    axes = np.asarray(axes).reshape(-1)
    for axis, (metric, label) in zip(axes, specs):
        means = [summary[scope][group][float(rho)][f"{metric}_mean"] for rho in rhos]
        sds = [summary[scope][group][float(rho)][f"{metric}_sd"] for rho in rhos]
        axis.errorbar(rhos, means, yerr=sds, marker="o", linewidth=1.8, capsize=3, color="#3366A8")
        axis.set_xticks(rhos)
        axis.set_xticklabels([f"{rho:g}\n({channel}/64)" for rho, channel in zip(rhos, channels)])
        axis.set_xlabel("H2T rho (effective replaced channels)")
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
    for axis in axes[len(specs) :]:
        axis.axis("off")
    figure.suptitle(title, y=1.02, fontsize=13)
    figure.tight_layout()
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def compact_table(summary: Mapping[str, Any], scope: str, group: str, metrics: Sequence[str]) -> List[str]:
    header_map = {
        "accuracy_pct": "Acc (%)",
        "ece15_pct": "ECE-15 (%)",
        "nll": "NLL",
        "brier": "Brier",
        "wrong_confidence_pct": "错误置信度 (%)",
    }
    headings = ["rho（替换通道）"] + [header_map[metric] for metric in metrics]
    lines = ["| " + " | ".join(headings) + " |", "|" + "---:|" * len(headings)]
    for condition in CONDITIONS:
        rho = float(condition["rho"])
        prefix = f"{rho:g} ({condition['effective_channels']}/64)"
        cells = [prefix]
        for metric in metrics:
            data = summary[scope][group][rho]
            mean = data[f"{metric}_mean"]
            sd = data[f"{metric}_sd"]
            precision = 3 if metric in {"nll", "brier"} else 2
            cells.append(f"{mean:.{precision}f} ± {sd:.{precision}f}")
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def create_report(
    output: Path,
    summary: Mapping[str, Any],
    delta_overall: Mapping[str, Any],
    delta_few: Mapping[str, Any],
    delta_probe_few: Mapping[str, Any],
    stratified_clean_ece: Mapping[float, Mapping[str, Any]],
    trends: Sequence[Mapping[str, Any]],
) -> None:
    lines = [
        "# H2T 原始代码：accuracy–calibration 先验审计（3 seed × rho sweep）",
        "",
        "## 固定设计",
        "",
        "- 共同 Stage-I 起点：作者发布的 CIFAR-100-LT IF=100 checkpoint；每个 Stage-II arm 都从该 checkpoint 重新加载。",
        "- 发布版 Stage-II：ResNet-32 feature extractor、CE、weighted balanced recipient sampling、long-tailed donor sampling、shift-BN、SGD、10 epoch、batch 128。",
        "- 配对 rho=0 control 仍然抽样、变换、前向 donor，并调用官方 `H2T`；只是 selected-channel set 为空。因此不把 donor forward / BN shift / RNG 消耗误归因给 fusion。",
        "- rho 网格为 0、0.10、0.25、0.40；由于官方实现使用 `int(rho * 64)`，实际替换通道为 0、6、16、25。",
        "- 主终点是固定第 10 epoch 的正常（无 test-time fusion）CIFAR-100 test。ECE 为 15-bin top-label ECE；NLL 与 multiclass Brier 同时报告。",
        "- synthetic fusion probe 是辅助诊断：测试 recipient + 依照训练长尾频率抽取的测试 donor。它不替代普通 clean-test 校准。",
        "",
        "## 1. H2T 提升 Accuracy 时，ECE 是否稳定恶化？",
        "",
        *compact_table(summary, "clean_test", "Overall", ("accuracy_pct", "ece15_pct", "nll", "brier")),
        "",
        "### rho=0.25 − matched rho=0（每个数值为 3 seed mean ± SD）",
        "",
        "| 指标 | 差值 | 每 seed 差值 |",
        "|---|---:|---:|",
    ]
    for metric, label in (
        ("accuracy_pct", "ΔOverall Acc (pp)"),
        ("ece15_pct", "ΔOverall ECE-15 (pp)"),
        ("nll", "ΔOverall NLL"),
        ("brier", "ΔOverall Brier"),
        ("wrong_confidence_pct", "ΔOverall 错误置信度 (pp)"),
    ):
        stats = delta_overall[metric]
        values = ", ".join(f"{value:+.3f}" for value in stats["values"])
        lines.append(f"| {label} | {stats['mean']:+.3f} ± {stats['sd']:.3f} | {values} |")
    lines.extend(
        [
            "",
            "### 频组加权的分层 ECE（归因诊断，不替代 pooled ECE）",
            "",
            "| rho（替换通道） | Many/Medium/Few 先各自计算 ECE 后按测试样本数加权 (%) |",
            "|---:|---:|",
        ]
    )
    for condition in CONDITIONS:
        rho = float(condition["rho"])
        stats = stratified_clean_ece[rho]
        lines.append(
            f"| {rho:g} ({condition['effective_channels']}/64) | {stats['mean']:.3f} ± {stats['sd']:.3f} |"
        )
    stratified_delta = [
        stratified_clean_ece[0.25]["values"][index] - stratified_clean_ece[0.0]["values"][index]
        for index in range(len(SEEDS))
    ]
    stratified_mean, stratified_sd = mean_sd(stratified_delta)
    lines.extend(
        [
            "",
            f"rho=0.25 − 0 的分层 ECE：{stratified_mean:+.3f} ± {stratified_sd:.3f}pp（每 seed：{', '.join(f'{value:+.3f}' for value in stratified_delta)}）。",
        ]
    )
    lines.extend(
        [
            "",
            "结论：在这三个 Stage-II seed 中，rho=0.25 的 Overall Accuracy 与 top-label ECE 都相对于 matched rho=0 同向上升；因此 **“官方 rho=0.25 时 Overall 15-bin ECE 上升”在本配置下可重复**。但 NLL 和 Brier 同时、且三 seed 一致地下降，故不能把这句话扩大成“整体概率质量或校准全面恶化”。分层 ECE 还降低，说明 pooled ECE 的上升不能直接分摊给频组、很可能受不同频组在同一 confidence bin 中混合/抵消关系变化影响。更重要的是，rho=0.40 的 Accuracy 继续上升而 ECE 回落到低于 rho=0；故在整个 rho 网格上不存在“强度越大、Accuracy 提升越多、ECE 越恶化”的一般规律。",
            "",
            "## 2. 恶化是否主要发生在 Tail / fusion samples？",
            "",
            "### Few clean test",
            "",
            *compact_table(summary, "clean_test", "Few", ("accuracy_pct", "ece15_pct", "nll", "brier", "wrong_confidence_pct")),
            "",
            "### Few held-out fusion probe",
            "",
            *compact_table(summary, "heldout_fusion_probe", "Few", ("accuracy_pct", "ece15_pct", "nll", "brier", "wrong_confidence_pct")),
            "",
            "| 对比 rho=0.25 − 0 | Few clean ECE (pp) | Few probe ECE (pp) | Few clean Acc (pp) | Few probe Acc (pp) |",
            "|---|---:|---:|---:|---:|",
            f"| 3-seed mean ± SD | {delta_few['ece15_pct']['mean']:+.3f} ± {delta_few['ece15_pct']['sd']:.3f} | {delta_probe_few['ece15_pct']['mean']:+.3f} ± {delta_probe_few['ece15_pct']['sd']:.3f} | {delta_few['accuracy_pct']['mean']:+.3f} ± {delta_few['accuracy_pct']['sd']:.3f} | {delta_probe_few['accuracy_pct']['mean']:+.3f} ± {delta_probe_few['accuracy_pct']['sd']:.3f} |",
            "",
            "结论：**否**。Few clean ECE 的平均变化不是恶化，辅助 fusion probe 的 Few ECE 也下降。融合后 Tail 的 Accuracy 在两个评估域均上升。因而目前证据反驳“ECE 的 Overall 上升主要来自 Tail 或 fusion samples”的版本；不要把 Overall ECE 增量归因给 Tail。",
            "",
            "## 3. fusion 强度与错误置信度是否存在稳定关系？",
            "",
            *compact_table(summary, "clean_test", "Overall", ("wrong_confidence_pct",)),
            "",
            *compact_table(summary, "clean_test", "Few", ("wrong_confidence_pct",)),
            "",
            "下表的斜率单位是“指标 / 1.0 rho”；Spearman 仅由四个预先指定的强度点计算，作为单调性描述，不能视为精确函数拟合。",
            "",
            "| 域 / 组别 / 指标 | 平均线性斜率 | seed 42/43/44 曲线形状 | 平均 Spearman | 三 seed 同一单调方向？ |",
            "|---|---:|---|---:|---|",
        ]
    )
    for item in trends:
        if item["metric"] not in {"wrong_confidence_pct", "ece15_pct", "accuracy_pct"}:
            continue
        display_scope = "clean" if item["scope"] == "clean_test" else "fusion probe"
        lines.append(
            f"| {display_scope} / {item['group']} / {item['metric']} | "
            f"{item['slope_mean_per_rho']:+.3f} | {' / '.join(item['shape_by_seed'])} | "
            f"{item['spearman_mean']:+.2f} | {'是' if item['all_seed_same_monotonic_shape'] else '否'} |"
        )
    lines.extend(
        [
            "",
            "结论：**不存在稳定的单调 strength → 错误置信度关系**。Overall 与 Few 的错误置信度均从 rho=0 上升到 rho=0.25，随后在 rho=0.40 明显回落；三条 seed 曲线均为非单调。因而不能以‘融合越强，错误越自信’作为 H2T 的机制假设。",
            "",
            "解释约束：错误样本条件平均置信度会随着‘哪些样本仍然错’而改变，所以它不是独立的过度自信证明。只有当它与 ECE、NLL/Brier、reliability diagram 的方向一致时，才可以讨论 confidence mechanism。",
            "",
            "## 当前可写、不可写的结论",
            "",
            "可写：在发布版 H2T 的 CIFAR-100-LT IF=100 Stage-II 配对控制下，官方 rho=0.25 相比 matched no-fusion 稳定提高 Overall/Few Accuracy，并稳定提高 Overall 15-bin ECE；这种 ECE 变化没有伴随 NLL/Brier 的恶化，也不集中于 Few 或 held-out fusion probe。完整 rho sweep 又显示该 ECE 效应非单调，rho=0.40 同时获得更高 Accuracy 和更低 ECE。",
            "",
            "不可写：H2T ‘导致 Tail calibration 恶化’、H2T ‘总体校准变差’，或错误置信度已证明为 H2T 性能提升的因果机制。要写后两项，至少还需要独立 Stage-I seed，以及不依赖单一 ECE binning 的校准分析。",
            "",
            "## 图文件",
            "",
            "- `rho_sweep_clean_overall.png`：Overall Accuracy / ECE / NLL / Brier / 错误置信度。",
            "- `rho_sweep_clean_few_and_wrong_confidence.png`：Few Accuracy / ECE / 错误置信度。",
            "- `rho_sweep_heldout_fusion_probe.png`：辅助 fusion probe 的 Overall/Few Accuracy 与 ECE。",
            "",
        ]
    )
    (output / "H2T_CALIBRATION_AUDIT_MULTISEED_ZH.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = load_runs(args.root)
    rows = flatten(records)
    summary = summary_by_rho(rows)
    trends = trend_rows(rows)
    delta_overall = paired_delta_summary(rows, "clean_test", "Overall")
    delta_few = paired_delta_summary(rows, "clean_test", "Few")
    delta_probe_few = paired_delta_summary(rows, "heldout_fusion_probe", "Few")
    stratified_clean_ece = group_stratified_ece(records, "clean_test")

    with (args.output_dir / "rho_sweep_seed_level_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with (args.output_dir / "rho_sweep_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "seeds": SEEDS,
                "conditions": CONDITIONS,
                "summary": summary,
                "trends": trends,
                "rho025_minus_rho0": {
                    "clean_overall": delta_overall,
                    "clean_few": delta_few,
                    "fusion_probe_few": delta_probe_few,
                },
                "group_stratified_clean_ece15_pct": stratified_clean_ece,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
    plot_curves(
        summary,
        args.output_dir / "rho_sweep_clean_overall.png",
        "clean_test",
        "Overall",
        (
            ("accuracy_pct", "Accuracy (%)"),
            ("ece15_pct", "ECE-15 (%)"),
            ("nll", "NLL"),
            ("brier", "Brier"),
            ("wrong_confidence_pct", "Wrong confidence (%)"),
        ),
        "Clean test: overall, mean ± SD across 3 Stage-II seeds",
    )
    plot_curves(
        summary,
        args.output_dir / "rho_sweep_clean_few_and_wrong_confidence.png",
        "clean_test",
        "Few",
        (("accuracy_pct", "Few accuracy (%)"), ("ece15_pct", "Few ECE-15 (%)"), ("wrong_confidence_pct", "Few wrong confidence (%)")),
        "Clean test: Few classes, mean ± SD across 3 Stage-II seeds",
    )
    plot_curves(
        summary,
        args.output_dir / "rho_sweep_heldout_fusion_probe.png",
        "heldout_fusion_probe",
        "Few",
        (("accuracy_pct", "Few accuracy (%)"), ("ece15_pct", "Few ECE-15 (%)"), ("nll", "Few NLL"), ("brier", "Few Brier")),
        "Held-out fusion probe: Few recipients, mean ± SD across 3 seeds",
    )
    create_report(
        args.output_dir,
        summary,
        delta_overall,
        delta_few,
        delta_probe_few,
        stratified_clean_ece,
        trends,
    )
    print(args.output_dir.resolve())


if __name__ == "__main__":
    main()
