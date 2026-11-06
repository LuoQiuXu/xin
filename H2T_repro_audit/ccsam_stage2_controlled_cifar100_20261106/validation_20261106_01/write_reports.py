"""Summarize the stopped preparation; no model execution or dataset loading."""
import csv
import hashlib
import json
import subprocess
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
def read(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))
def write(name, value):
    with (OUT / name).open("x", encoding="utf-8") as stream:
        if isinstance(value, str):
            stream.write(value)
        else:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)

execution = read("prepare_execution.json")
inputs = read("INPUT_VALIDATION.json")
syntax = read("STATIC_SYNTAX.json")
protection = read("POSTRUN_PROTECTION.json")
official = read("official_sources.json")
assert execution["status"] == "FAILED_STOPPED"
assert syntax["status"] == inputs["status"] == protection["status"] == "PASS"
assert not (ROOT / "prepared").exists()
assert not (ROOT / "smoke_report.json").exists()
assert not (ROOT / "results").exists()
status = dict(
    status="FAILED_STOPPED_AT_PREPARATION_GAP_EQUIVALENCE",
    branch="experiment/ccsam-stage2-controlled-cifar100-20261106",
    execution_head="b6235cc9fe45be479e7dffeb5cefdd06f4402cbd",
    python=inputs["interpreter"], syntax={"status": "PASS", "files": 4},
    sealed_input_checks="PASS", preparation="FAILED_STOPPED", smoke="NOT_RUN_BLOCKED",
    formal_training_started=False, formal_runs=0, optimizer_steps=0,
    sealed_checkpoint_sha256=inputs["checkpoint_sha256"], clean_seed=42, clean_epoch=184,
    expected_validation_overall=inputs["expected_validation_overall"],
    recomputed_validation_overall=None, validation_reproduction="NOT_COMPLETED",
    data_checks_reached_before_failure=dict(train_samples=10847, validation_samples=1980,
        class_counts_match_sealed_protocol=True, LT_pixels_match_sealed_protocol=True,
        raw_id_overlap=0, exact_pixel_overlap=0,
        evidence="Unmodified prepare_ccsam.py control flow reached extract_gap at line 122 after all these fatal gates."),
    actual_gap_equivalence_max_abs_error=None,
    gap_logits_finite_status="NOT_PERSISTED", original_map_logits_finite_status="NOT_PERSISTED",
    failing_validation_chunk="NOT_PERSISTED",
    sampling_plans_generated=0, paired_arm_hashes=None, gradient_equivalence_max_error=None,
    C1_perturbation_magnitude=None, classifier_gradient_checks="NOT_RUN",
    classifier_update_checks="NOT_RUN", backbone_BN_smoke_bitwise_checks="NOT_RUN",
    failure=execution["error"], test_guard=execution["test_guard"],
    root_cause_confirmed=False,
    confirmed_source_path_difference="New adaptive_avg_pool2d(map,1) vs sealed Classifier.forward F.avg_pool2d(map,map.size()[3]); same linear head.",
    no_tolerance_or_algorithm_change=True, no_retry=True,
    protected_history=protection,
    implementation_sources_modified=[], outputs_confined_to_new_experiment=True)
write("VALIDATION_STATUS.json", status)
with (OUT / "SEALED_CLASS_COUNTS.csv").open("x", encoding="utf-8-sig", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=["class_id", "sealed_train_count", "sealed_validation_count", "counts_source"])
    writer.writeheader()
    for c, count in enumerate(inputs["LT_class_counts"]):
        writer.writerow(dict(class_id=c, sealed_train_count=count, sealed_validation_count=0 if c == 0 else 20,
                             counts_source="sealed Clean Stage-I protocol; equality check was reached before the later GAP gate failed"))

rows = [
    ["Checkpoint and split", "PASS_INPUT", "Sealed checkpoint 7ced9d...; seed42 epoch184; original frozen split", "INPUT_VALIDATION.json"],
    ["Perturbation variable", "STATIC_MATCH_ONLY", "64-D GAP; official ResNet32Feature.forward avgpool then fea=x; local runtime validation blocked", "models/ResNet32Feature.py; ccsam_common.py:259"],
    ["Class-conditional gradient", "STATIC_FORMULA_MATCH; NUMERICAL_NOT_RUN", "Mean class CE gradient; one joint L2 norm per class subset; linear-head analytic derivative", "train_cifar.py:train_sample; train_ccsam.py:analytic_classwise_feature_delta"],
    ["Class weights", "STATIC_MATCH", "log(max_count/count+0.01)/2.0", "utils/utils.py:88-105; configs/Cifar100.json train.div"],
    ["Alpha ramp", "STATIC_MATCH", "min((epoch_index+1)/30,0.3)", "train_cifar.py:train_sample; configs/Cifar100.json"],
    ["CIFAR Stage-II LR", "STATIC_MATCH", "0.1 * 0.1 = 0.01, constant for epoch indices 0..29", "utils/lr_scheduler.py:adjust_learning_rate"],
    ["Class-aware sampler", "STATIC_CONTINUOUS_CYCLE; EXECUTION_NOT_REACHED", "Official default num_samples_cls=1; local random-cycle stream; additional local max-min<=1 check may conflict with epoch-crossing cycles", "datasets/Samplers.py; ccsam_common.py:321-357"],
    ["C0/C1/C2 controls", "STATIC_MATCH; SMOKE_NOT_RUN", "All start from Clean head; C0 unweighted CE; C1 weighted perturbation; C2 weighted zero delta", "experiment_protocol.json; train_ccsam.py:compute_arm_loss"],
    ["BN and backbone", "INTENTIONAL_CONTROLLED_TRANSFER", "Local eval/freeze including buffers; official freeze_backbone freezes parameter grads but train_sample calls model.train()", "utils/utils.py:61-66; train_cifar.py:train_sample; ccsam_common.py"],
    ["Dataset and Stage-I", "INTENTIONAL_CONTROLLED_TRANSFER", "Sealed local IF100/Stage-I; official CIFAR100 config uses imbalance 0.005 and its own Stage-I", "experiment_protocol.json; configs/Cifar100.json"],
    ["Zero class-gradient behavior", "LOCAL_SAFETY_CHOICE_NOT_EXERCISED", "Local <=1e-12 leaves zero delta; official unconditionally divides by norm", "train_ccsam.py:analytic_classwise_feature_delta; train_cifar.py:train_sample"],
    ["TEST access", "PASS_OBSERVED_GUARD", "0 TEST opens; 0 CIFAR100(train=False); 0 blocked attempts", "prepare_execution.json:test_guard"],
]
with (OUT / "SOURCE_CONSISTENCY_MATRIX.csv").open("x", encoding="utf-8-sig", newline="") as stream:
    writer = csv.writer(stream)
    writer.writerow(["item", "status", "evidence", "source"])
    writer.writerows(rows)
sources = "\n".join(
    "| " + " | ".join([s["path"], s["sha"], s["display_url"]]) + " |"
    for s in official["sources"])
counts = inputs["LT_class_counts"]
report = f"""# CC-SAM Stage-II 静态检查、数据准备与 Smoke 准入报告

日期：2026-11-06（Asia/Shanghai）。

**结论：FAILED_STOPPED_AT_PREPARATION_GAP_EQUIVALENCE。静态语法及锁定输入检查通过，但数据准备的 GAP 分类路径等价性门禁失败，按任务说明停止。Smoke 未运行，9 组正式训练未启动。**

## 1. 分支与保护

- 当前及远端分支均为 experiment/ccsam-stage2-controlled-cifar100-20261106，HEAD=b6235cc9fe45be479e7dffeb5cefdd06f4402cbd；没有切换或创建分支。
- GitHub 任务文件的 Git blob 为 b73fdfa343825dab405d706d7575198c17da18cb，与本地 HEAD 文件一致。
- 开始时没有已跟踪修改或冲突；有 140 个历史未跟踪文件。逐一记录大小与 SHA256，结束后全部未变。
- 两个 checkpoint、TRAIN pickle、split、封存协议及 frozen runtime 共 38 个受保护输入文件核验未变。四个执行脚本和协议/README/TXT 共 7 个文件 SHA 未变。
- 所有新增文件位于本次新实验目录；没有修改原 H2T、PI-H2T、Clean Stage-I、历史结果或仓库根配置。没有提交或推送本轮待审查结果。

## 2. 已实际执行的阶段

| 项目 | 结果 |
|---|---|
| 固定解释器 | D:\\app\\minicondaaa\\envs\\xiaoju\\python.exe |
| 四文件 py_compile | PASS，4/4，均 exit 0 |
| 官方 TRAIN pickle SHA256 | PASS：735e79b04f092ca3d2e6d07f368c0a7d70d48c48d28865950cc24454cf45129b |
| 两个 Clean checkpoint 副本 SHA256 | PASS：7ced9d07f71335357c08623573f038acd0dfdcdeadb55d95d57b332a14f83a88 |
| checkpoint 元数据 | seed=42、epoch=184、TEST_used=False、selection_partition=official TRAIN holdout |
| frozen split SHA256 | c8579feef60d7705472ec00aeac85fa00a66c243761e0a8d76c462c663286ca1 |
| 数据准备 | 实际执行 1 次，exit 1，未重试 |
| TRAIN / TRAIN-holdout | 10,847 / 1,980；重建类频、LT 像素哈希、raw-ID 和像素重叠门禁均先于失败位置通过 |
| raw-ID / exact-pixel 重叠 | 0 / 0，证据来自到达后续失败位置的未修改控制流 |
| 40.050505% 基线复现 | 未完成；这是封存期望值，不能报告成当前实测值 |
| 30-epoch sampling plans | 未生成，3 个 seed 均未到达该阶段 |
| Smoke | 未运行，依赖准备成功 |
| 正式训练 / optimizer 更新 | 0 / 0 |
| TEST 守卫 | TEST 文件打开尝试 0；CIFAR100(train=False) 0；blocked attempts 0 |

四个语法命令及返回码见 [STATIC_SYNTAX.json](STATIC_SYNTAX.json)。输入 SHA、封存类频及 checkpoint 元数据见 [INPUT_VALIDATION.json](INPUT_VALIDATION.json)。

封存 TRAIN 类频最大/最小为 {max(counts)}/{min(counts)}，100 类合计 {sum(counts)}，列表及 VAL 计数见 [SEALED_CLASS_COUNTS.csv](SEALED_CLASS_COUNTS.csv)。VAL 为 class 1..99 各 20 张，class 0 缺失，不能当作完整 100 类独立 TEST；这些统计的来源和验证阶段已在结构化记录中区分。

## 3. 实际失败与根因状态

原始错误：

    RuntimeError: GAP(64-D) classifier path is not equivalent to the frozen Clean classifier

调用路径为 prepare_ccsam.py:122 → ccsam_common.py:extract_gap → 第 262–263 行的 allclose 检查，固定 rtol=1e-5、atol=1e-6。完整异常见 [prepare.log](prepare.log)，同进程守卫和实际环境见 [prepare_execution.json](prepare_execution.json)。

源码事实：

1. 本次新路径为 adaptive_avg_pool2d(feature_map,1).flatten(1) → classifier.module.fc(gap)。
2. 封存 Clean 的 runtime/models/resnet_cifar.py:176–180 使用 F.avg_pool2d(x,x.size()[3]) → flatten → 同一 fc。
3. 对预期的 64×8×8 map，两种池化在数学上都是全局均值；实现路径不同是已确认事实，不能直接当作已证实根因。
4. 当前异常未保存失败 chunk 的 start/index、两路 logits、最大绝对/相对差异及有限性检查。**无法判断是浮点归约/内核差异、非有限值，还是实际执行语义存在其他问题。根因未确定。**

未更改阈值、池化、公式、初始化、学习率或超参数；没有为使检查通过而换 checkpoint 或数据。依照任务第 5 节 “If any check fails, stop and report it. Do not bypass the assertion.” 停止，未调用 smoke。

prepared 目录只在全部验证后创建，本次在此之前失败，因此 prepared/prepared_manifest.json 和 smoke_report.json 均不存在；没有伪造任何成功报告。

## 4. 官方代码—本地迁移核对

检查了官方 train_cifar.py、configs/Cifar100.json、utils/lr_scheduler.py、datasets/Samplers.py、models/ResNet32Feature.py 和 utils/utils.py，原文、URL、Git blob 均保存于 [official_sources.json](official_sources.json)。

官方 ResNet32Feature.forward 在 avgpool 后返回 64-D fea；本轮的扰动变量选取符合该变量位置。[官方模型](https://github.com/zzpustc/CC-SAM/blob/main/models/ResNet32Feature.py)

官方 train_sample 对每个类的 clean CE 求 feature 梯度，按该类整个子集梯度的联合 L2 范数归一化，权重 log(max_count/count+0.01)/2，alpha=min((e+1)/30,0.3)。本地 linear-head 解析梯度和联合归一化从源码看对应相同数学式，**实际数值等价性测试尚未运行**。[官方 Stage-II](https://github.com/zzpustc/CC-SAM/blob/main/train_cifar.py)、[权重](https://github.com/zzpustc/CC-SAM/blob/main/utils/utils.py)

官方 CIFAR Stage-II 在 0..29 使用常数 0.01，和本地设置一致。[官方 LR](https://github.com/zzpustc/CC-SAM/blob/main/utils/lr_scheduler.py)

本地 C0/C1/C2 与协议的静态定义一致：共用 Clean 分类头初始化；C0 无权 CE；C1 class-weighted perturbation；C2 相同加权/alpha 但 delta=0。C0 是保持原 head 的匹配 cRT-style 对照，不是随机重置分类头的 canonical cRT。

明确的受控迁移差异包括：固定本地 IF100 Clean Stage-I、严格冻结 BN，以及零特征梯度时本地 <=1e-12 的防护分支；官方代码和本地完整训练协议并不相同，不能声称完整复现原始 CC-SAM。

额外静态待复核项：本地 make_class_aware_plan 在跨 Epoch 保留连续类轮转状态，同时要求每个 10,847 样本 Epoch 的类计数 max-min<=1；官方连续 RandomCycleIter 并没有该额外断言。因 10,847 不是 100 的整数倍，Epoch 边界可能跨越两次不同的类排列，需后续核实该断言是否会拒绝合法的连续采样。**本轮未执行该函数，不把此风险认定为当前 GAP 异常的原因，也未改断言。**

完整条目见 [SOURCE_CONSISTENCY_MATRIX.csv](SOURCE_CONSISTENCY_MATRIX.csv)。

## 5. Smoke 必需结果的真实状态

| 要求 | 本轮结果 |
|---|---|
| C0/C1/C2 索引、标签、增强图像、GAP 哈希一致 | NOT_RUN；没有实际哈希 |
| 同一初始分类头 | 静态路径对应；Smoke 运行时比对未执行 |
| 解析梯度 vs autograd 最大误差 | NOT_RUN；不可填 0 |
| C1 有限且非零扰动及其幅度 | NOT_RUN |
| C0/C2 delta 精确为零 | 代码定义如此，运行时未验证 |
| 分类头有限梯度与真实更新 | NOT_RUN |
| backbone / BN 运行时逐位不变 | NOT_RUN；受保护磁盘文件的 SHA 未变不能替代此项 |
| TEST 守卫 | 准备失败进程仍完整记录 0 次访问尝试 |

## 6. 待审查与最小后续动作

当前不具备 Smoke 或正式训练准入条件。下一步应在另行确认范围后，仅对原失败验证 chunk 持久化输入/特征标识、两种池化输出、两路 logits、有限性以及最大绝对/相对差异，保持当前容差和全部原配置；先区分数值与语义原因，再决定修复。本轮没有执行该额外诊断、Smoke、正式训练、TEST 评价或新的 seed 实验。

环境：{json.dumps(execution["environment"], ensure_ascii=False)}。

## 7. 文件与完整性

新增仅为实验目录下的 .gitignore 和 validation_20261106_01/ 审查产物。四个实现脚本和已有协议文件未改。原有 140 个未跟踪历史文件与 38 个受保护输入 SHA 均未变，结果见 [POSTRUN_PROTECTION.json](POSTRUN_PROTECTION.json)。

运行前/后 Git 状态分别见 [GIT_STATUS_BEFORE.txt](GIT_STATUS_BEFORE.txt)、[GIT_STATUS_AFTER.txt](GIT_STATUS_AFTER.txt)。.gitignore 仅忽略本实验的 Python 缓存和未来 prepared 大数组，不改动历史资产或根目录规则。

官方来源的抓取记录：

| 文件 | Git blob SHA | 来源 |
|---|---|---|
{sources}
"""
write("CCSAM_STAGE2_PRE_SMOKE_REPORT.md", report)
write("CHANGED_FILES.json", dict(
    existing_implementation_modified=[],
    added_files=[p.relative_to(ROOT).as_posix() for p in sorted(OUT.rglob("*")) if p.is_file()] + [".gitignore"],
    generated_python_cache="Ignored within experiment; no outside cache written by Python -B",
    generated_dataset_arrays="None; preparation stopped before creating prepared/",
    unrelated_historical_untracked_files=140))
print(json.dumps(dict(report=str(OUT / "CCSAM_STAGE2_PRE_SMOKE_REPORT.md"),
                      status=status["status"], formal_runs=0, smoke="NOT_RUN"), ensure_ascii=False))
