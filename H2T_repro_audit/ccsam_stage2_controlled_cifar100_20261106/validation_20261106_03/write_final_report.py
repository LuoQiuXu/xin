"""Build the final review package from existing evidence; never executes an experiment."""
import json
import subprocess
from pathlib import Path

OUT=Path(__file__).resolve().parent
ROOT=OUT.parent
def read(path):
    return json.loads(path.read_text(encoding="utf-8"))
def save(name,value):
    with (OUT/name).open("x",encoding="utf-8",newline="\n") as stream:
        if isinstance(value,str):
            stream.write(value)
        else:
            json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)
before=read(OUT/"FULL_SAMPLER_BEFORE_FIX.json")
after=read(OUT/"FULL_SAMPLER_AFTER_FIX.json")
pre=read(OUT/"PRE_SMOKE_VERIFICATION.json")
prepare=read(OUT/"prepare_execution.json")
smoke=read(OUT/"smoke_execution.json")
failure=read(OUT/"SMOKE_FAILURE_SUMMARY.json")
protection=read(OUT/"POSTRUN_PROTECTION.json")
manifest=read(ROOT/"prepared/prepared_manifest.json")
gap_before=read(ROOT/"validation_20261106_02/GAP_BEFORE.json")
gap_after=read(ROOT/"validation_20261106_02/GAP_AFTER.json")
assert before["total_mismatches"]==after["total_mismatches"]==0
assert prepare["status"]=="PASS" and smoke["status"]=="FAILED_STOPPED"
assert manifest["validation_baseline_metrics"]["Overall"]==40.05050505050505
guards={name:read(OUT/name)["test_guard"] for name in (
    "FULL_SAMPLER_BEFORE_FIX.json","FULL_SAMPLER_AFTER_FIX.json","prepare_execution.json","smoke_execution.json")}
assert all(g["official_test_file_open_attempts"]==g["CIFAR100_train_false_attempts"]==0 and not g["blocked_attempts"] for g in guards.values())
status=dict(status="PREPARATION_PASS_SINGLE_SMOKE_FAILED_ACTUAL_GRADIENT_EQUIVALENCE",
    branch="experiment/ccsam-stage2-controlled-cifar100-20261106",
    execution_head="b6235cc9fe45be479e7dffeb5cefdd06f4402cbd",
    checkpoint_sha256=pre["checkpoint_sha256"],clean_stage1_seed=42,clean_stage1_epoch=184,
    validation_overall=manifest["validation_baseline_metrics"]["Overall"],
    validation_samples=1980,validation_feature_shape=manifest["validation_feature_shape"],
    full_sampler_validation_before=before,full_sampler_validation_after=after,
    exact_sampling_sequences_unchanged=True,prepared_status="PASS",
    original_prepared_arrays_unchanged=5,prepared_mode=manifest["preparation_mode"],
    raw_id_overlap=0,exact_pixel_overlap=0,
    smoke_invocations=1,smoke_status="FAIL",smoke_failure=failure,
    no_success_smoke_report_created=not (ROOT/"smoke_report.json").exists(),
    gradient_check_rtol=1e-5,gradient_check_atol=1e-7,
    synthetic_gradient_equivalence="passed by unchanged control flow; numerical maximum not persisted",
    actual_gradient_equivalence="FAIL",actual_delta_max_abs_difference=5.364418029785156e-7,
    root_cause_of_actual_gradient_mismatch="UNDETERMINED",
    C0_single_update_and_freeze_checks="passed by control flow before entering C1",
    C1_optimizer_update="NOT_REACHED",C2_smoke="NOT_REACHED",
    formal_training_started=False,formal_runs=0,formal_training_authorized=False,
    test_guards=guards,integrity=protection)
save("VALIDATION_STATUS.json",status)
save("GIT_DIFF.patch",subprocess.check_output(["git","diff","--no-ext-diff","--no-color"],text=True))
scope_diff=subprocess.run(
    ["git","diff","--no-index","--no-color",str(OUT/"sources_before/ccsam_common.py"),str(ROOT/"ccsam_common.py")],
    stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
if scope_diff.returncode not in (0,1):
    raise RuntimeError(scope_diff.stderr)
save("GIT_DIFF_SAMPLER_SCOPE.patch",scope_diff.stdout)
save("GIT_STATUS_REPORT_TIME.txt",subprocess.check_output(["git","status","--porcelain=v1","--untracked-files=all"],text=True))
seed_rows="\n".join(
    f"| {r['seed']} | 30 | {r['indices_compared']:,} | 0 | {r['local_array_sha256']} |"
    for r in before["seeds"])
plan_rows="\n".join(
    f"| {r['seed']} | {r['plan_file_sha256']} |"
    for r in pre["new_sampling_plans"])
report=f"""# CC-SAM Stage-II 采样校验修复、数据准备和单次 Smoke 报告

日期：2026-11-06（Asia/Shanghai）。分支 experiment/ccsam-stage2-controlled-cifar100-20261106；HEAD=b6235cc9fe45be479e7dffeb5cefdd06f4402cbd。

**结论：完整采样序列一致性与数据准备均通过；严格只执行了一次 Smoke，该次在 C1 真实批次的解析/autograd 扰动等价性检查失败，已停止。没有启动 9 组正式训练，没有重跑 Smoke、改梯度公式或放宽容差。**

## 1. 官方采样机制及完整实证

逐行审查了官方 RandomCycleIter 和 ClassAwareSampler：类循环和各类样本循环在跨 Epoch 时保留状态；默认 num_samples_cls=1；随机排列仅在循环耗尽时刷新。每个 epoch 取 10,847 个样本，不要求边界落在 100 类循环的起点。

官方原文：[datasets/Samplers.py](https://github.com/zzpustc/CC-SAM/blob/main/datasets/Samplers.py)。锁定 Git blob=9711d4d60ac41da5d01b3b2214d3f07d4ba4ac23，源码 SHA256=c84dfcb46c747a2ed1131df533184b2f9bf1724bc855a3d774830e795f60e174。字节未修改的副本及来源信息保存在 ../official_reference/。

在修改生产校验前，独立诊断副本仅省略原有已知错误计数断言，以取得原生成器完整输出；RNG、迭代器、样本选择和顺序均不变，生产源码当时未改。该输出与独立执行的官方原文逐个索引比较，覆盖三个种子、全部 30 epoch、每轮完整顺序：

| Seed | Epoch | 比较索引数 | 差异数 | 本地和官方相同的数组哈希 |
|---|---:|---:|---:|---|
{seed_rows}

合计 **976,230 个位置，0 差异**。诊断副本未用于生成正式 prepared 产物；生产修复后又对相同 976,230 个位置核验一次，仍与修改前封存的官方参考逐项一致，随机数外部状态也未改变。

完整实测轮内类计数范围包括 [107,109]、[108,109]、[108,110]。这些均来自官方序列，不能用“每个 epoch 最大差≤1”判断该连续采样是否有效。以前 seed123 第2轮的107–109就是一个实际反例。

证据：[修复前完整对照](FULL_SAMPLER_BEFORE_FIX.json)、[修复后完整对照](FULL_SAMPLER_AFTER_FIX.json)、[修复前全部轮计数](FULL_SAMPLER_EPOCH_COUNTS_BEFORE.csv)、[修复后全部轮计数](FULL_SAMPLER_EPOCH_COUNTS_AFTER.csv)。每轮保留全部 100 类计数，不仅是 min/max。

## 2. 修改范围和校验依据

1. ../ccsam_common.py：保留本地生成器的 RNG、shuffle、class/data iterator 和索引构建；移除语义错误的单轮差≤1 断言。新的 validate_class_aware_plan 会重新运行锁定官方实现，要求所有索引按顺序完全相同，并保存每轮 100 类计数。
2. ../official_reference/Samplers.py：官方原文字节副本；每次调用先检查固定 SHA256。参考计算隔离并恢复 Python random 状态。局部 .gitattributes 保留该源码的换行字节，使固定 SHA 在 Git 检出后仍有效。
3. ../prepare_ccsam.py：新增显式 --complete-existing-preparation。默认仍拒绝已有目录；完成模式拒绝未知文件或已完成 manifest。已有数组须与重新计算的 sealed 数据在 shape、dtype、值上精确相同，且文件 SHA 不变；缺失文件使用排他创建。
4. prepared_manifest.json 记录完整官方序列核验、每轮计数和各文件动作，区分官方连续采样与原错误的单轮“均衡”解释。
5. README 更新校验语义与无覆盖补全命令；.gitignore 仅忽略本轮本地参考计划和编译缓存。

保持不变：官方算法文件、实际采样序列、C0/C1/C2、loss、alpha、optimizer、seeds 123/124/125、30 epoch、batch size、Clean checkpoint 和 TRAIN split。train_ccsam.py 相对本次范围批准前的字节未变；CC-SAM 公式与各 arm 定义未改。协议 JSON 也未改。

另做了一个 CPU 拒绝测试：仅在内存测试副本交换两个位置，保持类别计数相同，新校验仍在 [0,0] 拒绝。该测试副本未保存或用于训练，原计划不变。这验证校验的是完整顺序。

## 3. Checkpoint、数据完整性和无覆盖准备

- 固定解释器 D:\\app\\minicondaaa\\envs\\xiaoju\\python.exe。
- checkpoint SHA256：7ced9d07f71335357c08623573f038acd0dfdcdeadb55d95d57b332a14f83a88；seed42、epoch184、TRAIN-holdout 来源均保持不变。
- split SHA256：c8579feef60d7705472ec00aeac85fa00a66c243761e0a8d76c462c663286ca1。
- 官方 TRAIN pickle SHA256：735e79b04f092ca3d2e6d07f368c0a7d70d48c48d28865950cc24454cf45129b。
- TRAIN=10,847，VAL=1,980；raw-ID 重叠=0，exact-pixel 重叠=0。VAL 仍为 class1..99 各20张、class0缺失。
- 完整 TRAIN-holdout Overall 再次复现 **40.0505050505%**，validation GAP shape=[1980,64]。
- 原有五个 NPY 文件全部被复用，文件 SHA 与失败时记录完全相同；没有删除、移动或覆盖。仅创建缺失的三个采样计划、prepared_manifest.json 和 test_guard_prepare.json。
- 三个新计划都与生产修复前封存的官方参考文件逐字节同 SHA：

| Seed | NPY 文件 SHA256 |
|---|---|
{plan_rows}

证据：[补全过程日志](prepare.log)、[实际执行记录](prepare_execution.json)、[Smoke 前全部验证](PRE_SMOKE_VERIFICATION.json)、[准备 manifest](../prepared/prepared_manifest.json)、[TEST 守卫](../prepared/test_guard_prepare.json)。

## 4. 只执行一次 Smoke：真实失败

调用 train_ccsam.py --smoke-only 一次，2026-11-06 18:07:02 开始、18:07:07 停止，exit=1。没有重试。

错误：

    Vectorized CC-SAM gradient does not match autograd reference; max_abs=5.364418029785156e-07

栈位于 train_ccsam.py:155，来自 smoke_test 的 C1 真实 Clean 批次。原检查 rtol=1e-5、atol=1e-7 保持不变。verify_analytic_gradient 实际比较的是解析路径与 autograd 路径生成的**归一化、加权扰动 delta**，不是未归一化原始梯度；上述 max_abs 是该扰动张量差异。

| 检查 | 本次真实状态 |
|---|---|
| 合成数据梯度/扰动参考检查 | 已先通过，由未改代码的控制流证明；具体 max error 未持久化 |
| C0 | 完成一批，有限分类头梯度、一次 disposable update、分类头确实变化、backbone/BN逐位不变的检查均先于进入 C1 通过 |
| C0/C1 输入与初始化配对 | 比较已通过，由到达 C1 后续门禁的控制流证明；具体哈希字符串未持久化 |
| C1 解析 vs autograd 参考 | FAIL；max_abs=5.364418029785156e-7 |
| C1 非零扰动独立断言 | 未到达，不能报告通过 |
| C1 分类头 optimizer update | 未到达，等价性异常发生在其 optimizer 构造之前 |
| C1 post-update backbone/BN 检查 | 未到达 |
| C2 | 未执行 |
| 三组全部通过 | 否 |
| 成功 smoke_report.json | 未生成，不伪造成功状态 |
| Smoke 调用次数 | 1 |
| 9组正式训练 | 0 |

表中的 C0 和配对结论来自已执行的强制检查及异常位置；原工具只在整个 Smoke 成功后写结果，因此不存在这些局部步骤的独立成功报告，不能补造实际哈希、误差或扰动幅度。

本次没有保存实际 analytic/reference 张量、相对误差或超差元素，**新梯度等价性故障的根因仍未确定**。最大差异较小不能证明它仅为无害舍入。没有放宽容差、替换公式或更改精度以求通过。

证据：[Smoke 原始日志](smoke.log)、[实际执行与守卫](smoke_execution.json)、[失败摘要](SMOKE_FAILURE_SUMMARY.json)。当前不具备正式训练准入条件。

## 5. 既有 GAP 修复状态

此前对同一 [128,64,8,8] FP32 特征证明：原 avg_pool2d 路径与 Clean logits 逐位一致；adaptive 路径 GAP 最大绝对误差1.9073486328125e-6，logits 最大绝对误差2.6226043701171875e-6，导致原门禁失败。统一 Clean-compatible avg_pool2d + view 后，同一特征上的两项误差均为0。

本次该 helper AST 未改，完整1,980张验证再次通过原等价性断言及40.050505%复现。早先 GAP 证据和失败日志全部保留在 ../validation_20261106_02/，本次新的 Smoke 故障不改变已经完成的 GAP 根因结论。

## 6. TEST、历史资产和 Git

四个本轮运行进程（完整修复前对照、修复后对照、prepare、Smoke）的守卫均记录：TEST 文件打开尝试0、CIFAR100(train=False)0、blocked attempts0、official_TEST_used=False。未读取官方TEST图像或标签。

原五个数组、61个此前诊断文件、140个历史未跟踪文件及38个受保护输入均核验未变。没有生成正式训练目录、正式 checkpoint 或 completion marker。C0 的一次可丢弃更新未写回原 checkpoint。

[POSTRUN_PROTECTION.json](POSTRUN_PROTECTION.json) 和 [VALIDATION_STATUS.json](VALIDATION_STATUS.json) 给出完整核验状态。代码 diff 见 [GIT_DIFF.patch](GIT_DIFF.patch)，最终 Git 状态和新增文件列表见交付目录中的对应文件。改动仅限本实验目录，没有切换分支；未提交、未推送，等待用户审查。

## 7. 后续边界

本轮严格停止在这一次 Smoke 的失败处。后续如获新的诊断授权，最小工作是只读复现 C1 原实际批次，保存解析/autograd 扰动张量、未归一化梯度、归一化范数、相对误差和超差元素，区分公式与数值实现问题；不做 optimizer 更新，不重跑九组训练，不利用 TEST，也不预先放宽容差。本轮未执行该额外诊断。
"""
save("CCSAM_PREPARATION_SMOKE_FINAL_REPORT.md",report)
print(json.dumps(dict(report=str(OUT/"CCSAM_PREPARATION_SMOKE_FINAL_REPORT.md"),
    sampler_parity="PASS",preparation="PASS",smoke="FAIL",smoke_invocations=1,formal_runs=0),ensure_ascii=False))
