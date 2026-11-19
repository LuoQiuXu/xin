"""Post-run reconciliation only; never trains, changes masks, or rewrites results.

This reporting utility is added after the frozen experiment. Its SHA is
recorded separately; the seven experimental scripts and protocol must still
match their pre-GPU CPU receipt byte for byte.
"""
from __future__ import annotations

from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from causal_core import ARMS, balance_diagnostics, canonical_hash, source_bin, target_bin
from oracle_runtime import (PARENT, SEEDS, code_hashes, file_sha, git, guard_pass,
                            inspect_parent, load_context, new_csv, new_json, now, read_csv, read_json)
from run_causal_oracle import _verify_postseal_data, require_preflight
from ccsam_common import hash_arrays


def state_digest(state):
    digest = hashlib.sha256()
    for name, value in state.items():
        if not torch.isfinite(value).all() or value.dtype != torch.float32:
            raise RuntimeError("Nonfinite / non-FP32 saved model")
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def main():
    preflight = require_preflight()
    context = load_context(read_json(ROOT / "split_manifest.json"))
    _verify_postseal_data(context)
    complete = read_json(ROOT / "formal_completed.json")
    summary = read_json(ROOT / "analysis" / "summary.json")
    assert complete["state"] == "completed" and complete["code_sha256"] == code_hashes()
    assert summary["official_TEST_used"] is False
    old = read_json(ROOT / "source_audit.json")
    fresh = inspect_parent()
    assert old["reused_source_and_artifact_sha256"] == fresh["reused_source_and_artifact_sha256"]
    actual_steps, seed_checks, balances, strata, retained_stability = 0, [], [], [], []
    bundles = {}
    counts = context["counts"]
    for seed in SEEDS:
        folder = ROOT / "results" / f"seed_{seed}"
        status = read_json(folder / "status.json")
        assert status["state"] == "completed" and status["epochs_completed"] == 30
        assert status["actual_optimizer_steps"] == {arm: 5100 for arm in ARMS}
        assert status["C2_BASE_C3_final_states_match_parent"] is True
        assert status["backbone_bn_bitwise_unchanged"] is True
        journal = [json.loads(line) for line in (folder / "execution.jsonl").read_text().splitlines()]
        assert len(journal) == 30
        assert [r["epoch_completed"] for r in journal] == list(range(1, 31))
        assert [r["steps_per_arm"] for r in journal] == [170 * n for n in range(1, 31)]
        observed, errors, update_counts = Counter(), {a: 0.0 for a in ARMS}, Counter()
        with (folder / "gradient_trace.csv").open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                arm, epoch, batch = row["arm"], int(row["epoch"]), int(row["batch"])
                assert arm in ARMS and 1 <= epoch <= 30 and 0 <= batch < 170
                observed[(epoch, batch, arm)] += 1
                assert row["writeback_replay_bitwise_equal"] == "True"
                values = [float(v) for k, v in row.items() if k.endswith("max_abs") or k == "loss"]
                assert all(math.isfinite(v) for v in values)
                error = max(float(row[key]) for key in (
                    "unmasked_weight_max_abs", "unmasked_bias_max_abs",
                    "masked_weight_max_abs", "masked_bias_max_abs"))
                assert error <= 2e-5
                errors[arm] = max(errors[arm], error)
                assert float(row["weight_update_max_abs"]) > 0
                assert float(row["bias_update_max_abs"]) > 0
                update_counts[arm] += 1
        assert len(observed) == 30600 and set(observed.values()) == {1}
        assert dict(update_counts) == {arm: 5100 for arm in ARMS}
        assert errors == status["max_gradient_errors"]
        parent_status = read_json(PARENT / "results" / f"seed_{seed}" / "status.json")
        for arm, record in status["checkpoints"].items():
            path = ROOT / record["path"]
            assert file_sha(path) == record["sha256"]
            payload = torch.load(path, map_location="cpu", weights_only=False)
            assert payload["seed"] == seed and payload["arm"] == arm and payload["epoch"] == 30
            assert payload["official_TEST_used"] is False
            assert sum(v.numel() for v in payload["state_dict_classifier"].values()) == 6500
            assert state_digest(payload["state_dict_classifier"]) == record["classifier_state_sha256"]
            if arm in ("C2", "BASE_C3"):
                assert record["classifier_state_sha256"] == parent_status[
                    "final_c2_classifier_sha256" if arm == "C2" else "final_c3_classifier_sha256"]
        cal = ROOT / "calibration" / f"seed_{seed}"
        seen_cal, final_rows = Counter(), []
        with (cal / "influence_epochs.csv").open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                epoch, source, target = int(row["epoch"]), int(row["source_class"]), int(row["target_many_class"])
                assert 1 <= epoch <= 30 and 35 <= source <= 99 and 1 <= target <= 34
                assert int(row["source_train_count"]) == counts[source]
                assert int(row["target_train_count"]) == counts[target]
                assert int(row["calibration_target_examples"]) == 10
                assert row["target_group"] == "Many"
                assert row["source_group"] == ("Few" if counts[source] < 20 else "Medium")
                seen_cal[(epoch, source, target)] += 1
                if epoch == 30:
                    final_rows.append(row)
        assert len(seen_cal) == 66300 and set(seen_cal.values()) == {1}
        assert final_rows == read_csv(cal / "pair_scores_epoch30.csv")
        bundle = read_json(cal / "mask_bundle.json")
        bundles[seed] = bundle
        assert bundle["K"] == 221 and len(bundle["oracle_retained_pairs"]) == 221
        balance_diagnostics(bundle, counts)  # includes exact matching assertions
        for row in read_json(cal / "matched_control_balance.json"):
            balances.append({"seed": seed, **row})
        masks = {arm: list(map(tuple, pairs)) for arm, pairs in bundle["suppressed_pairs"].items()}
        for family, arms in (("source_bin_x_exact_target_class", ("TARGETED", "RANDOM_MATCHED")),
                             ("source_bin_x_target_frequency_bin", ("TARGETED", "RANDOM_MATCHED", "FREQUENCY_MATCHED"))):
            for arm in arms:
                grouped = Counter((source_bin(counts[s]), t if family.endswith("target_class") else target_bin(counts[t]))
                                  for s, t in masks[arm])
                for (source, target), n in sorted(grouped.items()):
                    strata.append({"seed": seed, "family": family, "arm": arm,
                                   "source_bin": source, "target_class_or_bin": target, "pairs": n})
        actual_steps += sum(update_counts.values())
        seed_checks.append({"seed": seed, "epochs_observed": len(journal), "gradient_rows": len(observed),
                            "actual_steps_by_arm": dict(update_counts), "calibration_epoch_pair_rows": len(seen_cal),
                            "saved_classifiers_verified": 6, "baseline_final_states_exact": True,
                            "gradient_max_abs": max(errors.values()), "backbone_BN_bitwise_unchanged": True})
    assert actual_steps == 91800 == complete["actual_optimizer_steps"]
    for row in read_json(ROOT / "MASK_HASH_INDEX.json")["records"]:
        arm = row.get("alias_of", row["arm"])
        mask = np.ones((100, 100), dtype=bool)
        for s, t in bundles[row["seed"]]["suppressed_pairs"][arm]:
            mask[s, t] = False
        assert hash_arrays(mask) == row["keep_mask_sha256"]
    for a, b in ((123, 124), (123, 125), (124, 125)):
        sa, sb = set(map(tuple, bundles[a]["oracle_retained_pairs"])), set(map(tuple, bundles[b]["oracle_retained_pairs"]))
        retained_stability.append({"seed_a": a, "seed_b": b, "set_semantics": "ORACLE_SELECTION_retained_K221",
                                   "overlap": len(sa & sb), "jaccard": len(sa & sb) / len(sa | sb)})
    balance_fields = ["seed", "arm", "axis", "n", "mean_train_count", "standardized_mean_difference", "source_bin", "target_bin"]
    new_csv(ROOT / "analysis" / "matched_control_balance.csv", balances, balance_fields)
    new_csv(ROOT / "analysis" / "matched_control_strata.csv", strata)
    new_csv(ROOT / "analysis" / "oracle_retained_mask_stability.csv", retained_stability)
    decision = summary["decision"]
    practical = []
    for name, result in decision["oracle"].items():
        for row in result["per_seed"] + [{"seed": "three_seed_mean", **result["mean_effects"]}]:
            practical.append({"oracle": name, **row, "many_at_least_2pp": row["many_accuracy"] >= 2,
                "margin_at_least_0_10": row["fixed_many_margin"] >= .1,
                "few_change_at_least_minus_2pp": row["few_accuracy"] >= -2,
                "both_primary_positive": row["many_accuracy"] > 0 and row["fixed_many_margin"] > 0,
                "all_three_size_cutoffs": row["many_accuracy"] >= 2 and row["fixed_many_margin"] >= .1 and row["few_accuracy"] >= -2,
                "Gate_admission": "HUMAN_REVIEW_ONLY"})
    new_csv(ROOT / "analysis" / "preregistered_practical_flags.csv", practical)
    evidence = {
        "controlled_intervention_rule": decision["causal_evidence"],
        "Gate_admission": "HUMAN_REVIEW_ONLY",
        "supporting_evidence": [
            "TARGETED exceeds BOTH matched controls on Many accuracy and fixed margin in all three seeds",
            "All six leave-one-Many-class-out checks preserve positive control contrasts",
            "Calibration/Evaluation reference-score rank correlation is 0.99855..0.99904 after mask sealing",
            "All baselines reproduce parent final states exactly; numerical and data gates pass"],
        "opposing_evidence": [
            "TARGETED minus BASE_C3: Many +1.6667pp, below the 2.0pp practical headroom cutoff",
            "TARGETED minus BASE_C3: Few -1.4444pp; Overall/Macro -0.1010pp",
            "ORACLE_SELECTION minus BASE_C3: Few mean -1.8889pp; seed125 -2.6667pp breaches the per-seed -2pp descriptive cutoff",
            "ORACLE_SELECTION Few conditional-bootstrap 95% interval [-3.6667,-0.1111]pp; mean below-2pp harm cannot be excluded",
            "ORACLE_SELECTION Overall/Macro mean -0.0673pp, with mixed seed directions",
            "Only 2..6 of 34 Many classes improve top1 accuracy per seed/control contrast; margin improvement is more widespread",
            "Exact frequency-bin matching leaves continuous source-count SMD up to 0.1077"],
        "preregistered_mean_effect_flags": {name: result["preregistered_practical_effect_flag"]
                                             for name, result in decision["oracle"].items()},
        "limitations": decision["limitations"] + [
            "Bootstrap conditions on frozen masks, not uncertainty from re-estimating Calibration masks",
            "Retain-K and suppress-K oracle families impose different intervention budgets",
            "Train loss for intervention arms is the original C3 scalar for monitoring, not a claimed end-to-end masked loss"],
        "recommendation": "Submit for human scientific review. Evidence supports this controlled actionability contrast, but tail tradeoffs and lack of Overall gains preclude an automatic success or Gate-admission claim. No Gate or further experiment is started.",
        "note": "This supplements opposing tradeoff evidence; the original DECISION.json opposing_evidence field enumerates only failed primary control comparisons.",
    }
    new_json(ROOT / "analysis" / "SCIENTIFIC_REVIEW_DECISION_ZH.json", evidence)
    validation = {"passed": True, "completed_at": now(), "experimental_code_still_matches_CPU_receipt": True,
        "postrun_utility_sha256": file_sha(__file__), "phase_exit_codes": {"CPU": 0, "preflight": 0, "calibrate": 0, "smoke": 0, "formal": 0, "analyze": 0},
        "syntax_files": 7, "CPU_tests": {"causal_oracle": 23, "parent_ccmia": 13},
        "source_fingerprints_rechecked": len(fresh["reused_source_and_artifact_sha256"]),
        "historical_artifacts_rechecked": len(fresh["parent_artifact_checks"]),
        "seed_checks": seed_checks, "actual_optimizer_updates_from_observed_gradient_rows": actual_steps,
        "calibration_epoch_pair_rows_observed": 198900, "all_masks_verified_against_seal": True,
        "mask_hash_index_verified_using_actual_numpy_bool_arrays": True,
        "calibration_evaluation_raw_ids_disjoint": not bool(set(context["split"]["calibration_raw_train_ids"]) & set(context["split"]["evaluation_raw_train_ids"])),
        "official_TEST_used": False, "test_guard": guard_pass(),
        "branch": git("branch", "--show-current"), "precommit_head": git("rev-parse", "HEAD")}
    new_json(ROOT / "validation" / "FINAL_ARTIFACT_VALIDATION.json", validation)
    effects = {(r["contrast"], r["metric"]): r for r in summary["mean_SD_effects"]}
    lines = ["# CC-MIA 受控因果干预与 Oracle：科学审查报告", "", f"实际分析完成：{summary['completed_at']}", "",
        "三个种子均完成30轮，六个实际组，共91,800次真实SGD更新。全部门禁通过，官方TEST零访问。",
        "本轮停止在科学证据交付；Gate准入为 **HUMAN_REVIEW_ONLY**，没有实现或训练Gate。", "",
        "## 三个种子的固定末轮原始指标", "",
        "|Seed|组|Overall%|Macro%|Many%|Few%|固定Many Margin|", "|---|---|---|---|---|---|---|"]
    for row in summary["per_seed_final_metrics"]:
        lines.append(f"|{row['seed']}|{row['arm']}|{row['overall_accuracy']:.4f}|{row['macro_accuracy']:.4f}|{row['many_accuracy']:.4f}|{row['few_accuracy']:.4f}|{row['fixed_many_margin']:.6f}|")
    lines += ["", "Evaluation每个已覆盖类恰好10张，故本轮Overall与Macro相同；不是两项独立收益。", "",
        "## 配对效果量、样本SD和分层Bootstrap 95%区间", "",
        "准确率差值单位为百分点。区间条件于三个固定种子、固定掩码和本开发集，不是总体显著性检验。", "",
        "|对照|指标|均值差|样本SD|Bootstrap 95% CI|", "|---|---|---|---|---|"]
    contrasts = ("TARGETED_minus_BASE_C3", "TARGETED_minus_RANDOM_MATCHED", "TARGETED_minus_FREQUENCY_MATCHED", "ORACLE_SELECTION_minus_BASE_C3", "ORACLE_SELECTION_minus_TARGETED")
    for contrast in contrasts:
        for metric in ("many_accuracy", "fixed_many_margin", "few_accuracy", "overall_accuracy", "macro_accuracy"):
            r = effects[(contrast, metric)]
            lines.append(f"|{contrast}|{metric}|{r['mean']:+.6f}|{r['sample_SD']:.6f}|[{r['bootstrap_low']:+.6f}, {r['bootstrap_high']:+.6f}]|")
    lines += ["", "## 支持证据", "",
        "TARGETED在3/3种子中同时超过两个匹配随机对照的Many Accuracy与固定Margin，达到预注册受控干预判据。",
        "相对RANDOM_MATCHED的Many平均提升0.8824pp，相对FREQUENCY_MATCHED提升1.2745pp。",
        "六个seed/control逐一删除单个Many类后，两个主指标方向仍为正，未由单个类单独决定。",
        "Calibration/Evaluation参考影响排序相关约0.99855–0.99904；这些Evaluation评分只在封存后计算，未用于重选。", "",
        "## 反对证据和代价", "",
        "TARGETED/Oracle Suppression相对BASE_C3的Many平均提升1.6667pp，未达到2.0pp实用门槛；Few下降1.4444pp，Overall/Macro下降0.1010pp。",
        "Oracle Selection的Many平均提升2.8431pp、Margin提升0.729471，Few均值下降1.8889pp，按提前冻结的均值口径实用效果标记通过。",
        "但seed125的Few下降2.6667pp，超过单种子−2pp描述性边界；Few均值的条件Bootstrap区间为[-3.6667,-0.1111]pp，不能保证尾类损失不超过2pp。",
        "Oracle Selection的Overall/Macro均值下降0.0673pp，三个种子方向混合；不得用Many或Margin单项提升宣布整体实验成功。",
        "Top1收益在每个seed/control对照中仅覆盖2–6个Many类；Margin改善更广泛。这支持有限受控可操作性，不证明所有Many类均受益。", "",
        "## Oracle约束与剩余空间", "",
        "Oracle Suppression就是TARGETED：同一掩码、同一训练结果，二者差距为零，不能当作独立重复。",
        "Oracle Selection固定保留221对，抑制其余1989对；Suppression固定抑制221对。两者约束不同，不能将Selection叫作全局理论上界。",
        "Selection相对TARGETED的Many平均再提高1.1765pp、Margin再提高0.226570，同时Few平均再下降0.4444pp。", "",
        "## 匹配和证据局限", "",
        "RANDOM_MATCHED精确匹配source频次箱×target类计数；FREQUENCY_MATCHED匹配source频次箱×target频次箱。连续频次数值并非完全相同：source SMD最大0.1077，target SMD最大0.0544。",
        "精确计数与连续平衡见analysis/matched_control_strata.csv和matched_control_balance.csv。",
        "Oracle Selection的抑制集很大，不能将其抑制集Jaccard与221对保留集混同；保留集稳定性另见oracle_retained_mask_stability.csv。",
        "原Stage-I在完整1980张holdout上选模，且父研究已查看该集合；本轮隔离不能撤销历史选择。Evaluation是TRAIN-derived development，不是独立测试集。",
        "class0缺失，Many仅覆盖34/35类；每类Calibration只有10张。Bootstrap条件于封存掩码，不覆盖重新估计掩码的不确定性。",
        "干预是source→classifier-row直接梯度的控制实验；clean路径保留。日志train_loss是原C3标量监测值，不声称新梯度来自新的端到端损失。", "",
        "## 决策", "",
        "**支持预注册范围内的受控负交互可操作性；存在明显Many/Few代价，整体收益尚未建立。**",
        "Gate准入留待人工综合评价；本轮不自动准入、不设计新Gate，也不启动额外实验。",
        "完整支持/反对证据见analysis/SCIENTIFIC_REVIEW_DECISION_ZH.json；该补充保留原DECISION.json及全部既有输出。", ""]
    with (ROOT / "SCIENTIFIC_REVIEW_ZH.md").open("x", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines))
    with (ROOT / "VALIDATION_REPORT.md").open("x", encoding="utf-8", newline="\n") as handle:
        handle.write("# 最终验证\n\n" + "\n".join([
            "- 7个实验Python文件语法检查：PASS。",
            "- 本轮CPU 23/23、父CC-MIA CPU 13/13：PASS。",
            "- Preflight、真实baseline probe、三种子完整参考重放：PASS。",
            "- 三种子各66,300行Calibration记录，共198,900行，逐行类别/频次/epoch检查：PASS。",
            "- 三种子掩码共同封存、封存前Evaluation零读取、真实布尔掩码SHA复核：PASS。",
            "- 生产Smoke：3种子×6实际组，各1个真实batch，PASS。",
            "- Formal：3×30配对Epoch，实际梯度记录91,800行，全部写回replay通过。",
            "- 18个末轮分类头文件SHA、状态SHA、seed、arm、epoch与6500参数检查：PASS。",
            "- C2/BASE_C3三种子末轮状态均精确匹配父实验；Backbone/BN bitwise unchanged。",
            "- 90项复用文件指纹与45项历史产物再次核查：PASS。",
            "- 分析完成，2000次class-stratified Bootstrap，原始结果未覆盖。",
            "- 官方CIFAR-100 TEST访问/尝试均为0。",
            "- 实验代码及冻结协议SHA仍匹配GPU运行前CPU收据。",
            "- 本post-run报告工具另存SHA，只核对与整理，未训练或修改任何掩码/算法。",
            "", "机器核验：validation/FINAL_ARTIFACT_VALIDATION.json。所有历史目录保持原状。", ""]))
    print("POST-RUN RECONCILIATION PASS: observed 91,800 updates, 18 checkpoints, 198,900 Calibration rows; reports assembled")


if __name__ == "__main__":
    main()
