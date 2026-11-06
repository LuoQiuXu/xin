"""Report completed GAP repair and the independent preparation blocker; no execution."""
import json
import subprocess
from pathlib import Path

OUT=Path(__file__).resolve().parent
ROOT=OUT.parent
def read(name):
    return json.loads((OUT/name).read_text(encoding="utf-8"))
def save(name,value):
    with (OUT/name).open("x",encoding="utf-8",newline="\n") as f:
        if isinstance(value,str):f.write(value)
        else:json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False)
before=read("GAP_BEFORE.json")
after=read("GAP_AFTER.json")
prep=read("prepare_execution.json")
sampler=read("SAMPLER_DIAGNOSIS.json")
protection=read("PROTECTION_CHECKS.json")
original=before["batches"][-1]
baseline=next(x["values"]["baseline_metrics"] for x in prep["captured_failure_context"] if "baseline_metrics" in x["values"])
assert before["status"]=="POOLING_NUMERIC_CAUSE_CONFIRMED" and after["status"]=="PASS"
assert baseline["Overall"]==40.05050505050505
guards={name:read(name)["test_guard"] for name in ("GAP_BEFORE.json","GAP_AFTER.json","prepare_execution.json","SAMPLER_DIAGNOSIS.json")}
assert all(g["official_test_file_open_attempts"]==g["CIFAR100_train_false_attempts"]==0 and not g["blocked_attempts"] for g in guards.values())
status=dict(status="GAP_FIXED_BASELINE_REPRODUCED_PREPARATION_BLOCKED_BY_INDEPENDENT_SAMPLER_VALIDATOR",
    branch="experiment/ccsam-stage2-controlled-cifar100-20261106",
    execution_head="b6235cc9fe45be479e7dffeb5cefdd06f4402cbd",
    gap_root_cause_confirmed=True,
    root_cause="Different FP32 pooling reduction paths: adaptive_avg_pool2d vs sealed avg_pool2d; same feature map/head. Original avg path is bitwise identical to Clean, adaptive logits fail unchanged tolerance.",
    feature_shape=original["feature_shape"],feature_dtype=original["feature_dtype"],
    before_pool_max_abs=original["pool_C_vs_B"]["max_abs"],
    before_pool_max_relative=original["pool_C_vs_B"]["max_relative"],
    before_logits_max_abs=original["logits_C_vs_A"]["max_abs"],
    before_logits_max_relative=original["logits_C_vs_A"]["max_relative"],
    after_pool_max_abs=after["original_avg_vs_fixed_gap_max_abs"],
    after_logits_max_abs=after["original_A_vs_fixed_B_logits_max_abs"],
    after_logits_bitwise_equal=True,allclose_rtol=1e-5,allclose_atol=1e-6,
    static_python_files_passed=8,checkpoint_sha256=after["checkpoint_sha256"],
    clean_stage1_holdout_overall=baseline["Overall"],validation_samples=1980,
    validation_feature_shape=[1980,64],validation_reproduction="PASS",
    preparation="FAILED_STOPPED",preparation_failure=prep["error"],
    sampler_diagnosis=sampler,smoke="NOT_RUN_BLOCKED",smoke_arms_completed=[],
    gradient_equivalence_check="NOT_RUN",gradient_equivalence_max_error=None,
    backbone_BN_bitwise_unchanged_in_gap_diagnostics=True,
    backbone_BN_smoke_check="NOT_RUN",formal_training_started=False,formal_runs=0,
    optimizer_updates=0,test_guards=guards,
    original_classifier_and_checkpoint_modified=False,protocol_modified=False,
    sampling_strategy_modified=False,CCSAM_formula_or_arm_definitions_modified=False,
    old_run_stage_has_no_pooling_and_delegates_to_repaired_entrypoints=True,
    awaiting_user_scope_decision_for_independent_sampler_validator=True)
save("VALIDATION_STATUS.json",status)
save("GIT_DIFF.patch",subprocess.check_output(["git","diff","--no-ext-diff","--no-color"],text=True))
save("GIT_STATUS_REPORT_TIME.txt",subprocess.check_output(["git","status","--porcelain=v1","--untracked-files=all"],text=True))
report=f"""# CC-SAM 64-D GAP 一致性修复报告

日期：2026-11-06（Asia/Shanghai）。

**GAP 原始故障已定位并修复；40.0505050505% TRAIN-holdout 基线已成功复现。数据准备随后被独立的采样合法性断言阻断，Smoke 尚未运行，正式训练 0 组。未修改该采样断言，等待用户范围确认。**

## 1. 原始故障的实证根因

同一个锁定 checkpoint、同一首批 128 张验证图像、同一个实际 feature_map，比较：

- A：原 Classifier(feature_map)；
- B：原式 F.avg_pool2d(feature_map, feature_map.size(3))，再使用原式 view 和同一 fc；
- C：adaptive_avg_pool2d(feature_map,1)，再接同一 fc。

实际张量为 [128,64,8,8]、torch.float32。feature_map、两种 GAP、A/B/C logits 全部有限；backbone/BN 和 classifier 状态前后逐位不变，optimizer 更新 0。

| 比较 | 最大绝对误差 | 最大相对误差 | 原 allclose | 逐位相同 |
|---|---:|---:|---|---|
| GAP C vs B | {original["pool_C_vs_B"]["max_abs"]:.16g} | {original["pool_C_vs_B"]["max_relative"]:.16g} | True | False |
| logits B vs A | 0 | 0 | True | True |
| logits C vs A | {original["logits_C_vs_A"]["max_abs"]:.16g} | {original["logits_C_vs_A"]["max_relative"]:.16g} | False | False |

相对误差分母统一为 max(abs(reference),1e-12)，不把不同元素的绝对/相对最大值当作同一个元素。检查保持 rtol=1e-5、atol=1e-6。

128 个样本中有 1 个样本的 1 个 logit 违反原容差。具体为批内样本 8、logit 22：A=0.008202150464057922，C=0.00820334255695343；绝对误差 1.1920928955078125e-6，大于允许的 1.082021504640579e-6。两种 GAP 虽自身通过 allclose，线性投影后仍可能触发近零 logit 的阈值。

逐样本 logits 最大绝对误差分布：min/P50/P95/P99/max = {before["sample_logit_error_quantiles"]}。该批 argmax 无变化，也未因此删除数值门禁。

因此，**本次原始失败由两条池化实现的 FP32 数值差异造成**：控制了输入特征、模型状态和分类头，B 与 A 完全一致，仅替换为 adaptive 后 C 失败。这里证明的是原始一致性故障的原因，没有声称 CUDA 硬件故障，也没有据此推断训练机制。

证据：[GAP_BEFORE.json](GAP_BEFORE.json)、[逐样本误差](GAP_BEFORE_PER_SAMPLE.csv)、[超限元素](TOLERANCE_VIOLATIONS.json)。特征快照保留本地 tensor_snapshots/，SHA256 为 {after["feature_snapshot_sha256"]}，不纳入 Git。

## 2. 最小代码修改

| 文件 | 修改 |
|---|---|
| ../ccsam_common.py | 新增 clean_compatible_gap；强制 [B,64,8,8]，原式 avg_pool2d + view，保持 dtype，输出 [B,64]；extract_gap 统一调用 |
| ../train_ccsam.py | Smoke 和未来正式训练的两个特征提取位置均调用 helper；未执行正式分支 |
| ../prepare_ccsam.py | 验证特征仍经 extract_gap，现间接使用统一 helper；增加基线成功的真实打印，便于后续准备失败时保留证据 |
| ../README.md | 明确 Clean-compatible 池化和原容差，保留 64-D 扰动定义 |
| ../.gitignore | 仅增加本轮 tensor_snapshots 和 syntax_cache 的本地忽略规则 |

旧 validation_20261106_01/run_stage.py **没有独立池化代码**，它调用 prepare/train 原入口，因此已经间接统一到 helper；为保留旧诊断资产，未修改其字节。新 validation_20261106_02/run_stage.py 使用相同入口并在新目录记录日志、失败上下文，不覆盖旧结果，也不接受 formal 参数。

原始 Clean 模型、checkpoint、split 均未改。分类头权重仅用于只读对照，没有 optimizer 更新。CC-SAM analytic/reference 梯度函数、compute_arm_loss 和数值等价性检验函数经 AST 比较未变；协议 JSON、sampler 函数、class weights 和 alpha 均未改。

原 allclose 断言原样保留，rtol/atol 未变。没有把扰动改到 64×8×8 feature map。

## 3. 修复后验证

1. 8 个相关 Python 文件 py_compile 全部通过，缓存写到本轮目录。
2. 对保存的原失败 feature_map 重放：helper GAP 与原 avg GAP 最大绝对误差 **0**；helper fc logits 与 A 最大绝对/相对误差 **0/0**，逐位一致；原 A logits 也与修复前快照逐位一致。非法输入形状均被拒绝。
3. checkpoint SHA256 仍为 7ced9d07f71335357c08623573f038acd0dfdcdeadb55d95d57b332a14f83a88。
4. 完整 1,980 张 TRAIN-holdout 的 GAP 路径检查通过，Overall={baseline["Overall"]:.10f}%，与封存值完全一致；validation_gap shape=[1980,64]。
5. 数据准备继续到生成 class-aware 计划时失败，详见下一节。
6. 因准备未完成，没有 prepared_manifest.json，没有运行 C0/C1/C2 Smoke，也没有生成虚假的成功 smoke_report.json。

证据：[STATIC_CHECKS.json](STATIC_CHECKS.json)、[GAP_AFTER.json](GAP_AFTER.json)、[prepare.log](prepare.log)、[prepare_execution.json](prepare_execution.json)。

## 4. 独立的采样校验阻断

实际异常为 “Class-aware sampler did not produce a balanced epoch”，位置为 common.make_class_aware_plan 的原 max_count-min_count>1 检查。seed=123、第 2 轮（零基 epoch=1）实际每类 107–109 次，总数 10,847。

以已审查的官方 datasets/Samplers.py 原文为只读参考，对相同 TRAIN labels、seed=123 的前两轮逐项比较。**本地实际生成的所有索引与官方完全相同**；两者数组哈希均为 {sampler["local_sequence_sha256"]}。官方前两轮类计数范围分别为 [108,109]、[107,109]。

这说明独立阻断来自本地附加的每轮差≤1 校验，与官方跨 Epoch 连续类轮转不相容。10,847 非 100 整数倍，边界可跨越两次类排列。此问题并非 GAP 修复引入：sampler 函数与修复前 AST 完全相同。

没有移除或放宽该断言，没有修改采样序列，没有写出伪造的 sampling_plan 文件。上述诊断只在 CPU 内存中对照官方两轮序列并捕获原函数真实异常；不是训练。仅此两轮被验证，不能据此宣称三个 seed、全部 30 轮已经通过。

现有 prepared 内有 5 个数组，均保留原字节：[PARTIAL_PREPARED_SHA256.json](PARTIAL_PREPARED_SHA256.json)。不存在完整准备 manifest，不删除目录，不覆盖数组，也不通过人工补 manifest 绕过门禁。

证据：[SAMPLER_DIAGNOSIS.json](SAMPLER_DIAGNOSIS.json)。已请求用户确认是否允许保持全部采样序列和参数不变，仅修正合法性验证并无覆盖地补全准备；确认前不开展依赖该修正的工作。

## 5. Smoke、梯度、BN 和 TEST 的真实状态

| 项目 | 状态 |
|---|---|
| C0/C1/C2 Smoke | NOT_RUN，准备被独立校验阻断 |
| 各 arm 梯度等价性最大误差 | 未执行，不填 0 |
| C1 有限非零扰动 / C0,C2 零扰动 | 未执行 Smoke，不能报告通过 |
| 分类头梯度与真实更新 | 未执行；本轮总 optimizer 更新 0 |
| Backbone/BN bitwise unchanged | GAP 诊断和单批修复验证均通过；Smoke 项仍未执行 |
| TEST 打开 / train=False / blocked attempts | 四个诊断/验证进程全部为 0 |
| Stage-I 重训 / 正式训练 | 0 / 0 |

所有 TEST 守卫快照均保存在结构化状态中。没有读取官方 TEST 图像或标签。

## 6. Git 与历史保护

保持当前分支及 HEAD。Git diff 见 [GIT_DIFF.patch](GIT_DIFF.patch)；Git 状态见 [GIT_STATUS_REPORT_TIME.txt](GIT_STATUS_REPORT_TIME.txt)。未提交或推送本轮修复，供用户审查。

原验证目录 24 个文件、140 个历史未跟踪文件及 38 个受保护输入均核验未变。[PROTECTION_CHECKS.json](PROTECTION_CHECKS.json) 保存核验结果。新增代码/证据均在本次实验目录；快照和 prepared 大数组已忽略，不会混入常规 Git 暂存。

本轮已证明并修复 GAP 根因，当前未解决项是独立采样合法性校验的修复范围，以及其后尚未执行的 Smoke。无论后续决定如何，均不启动 9 组正式训练。
"""
save("GAP_FIX_REPORT.md",report)
print(json.dumps(dict(report=str(OUT/"GAP_FIX_REPORT.md"),gap_repair="PASS",
    baseline_overall=baseline["Overall"],preparation="BLOCKED_BY_INDEPENDENT_VALIDATOR",
    smoke="NOT_RUN",formal_runs=0),ensure_ascii=False))
