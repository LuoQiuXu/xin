# Fish16 六条件单 Epoch Smoke：失败后停止

日期：2026-11-01（Asia/Shanghai）。**结论：FAILED_STOPPED，仅 A0 完整完成。**

分支：`experiment/fish16-six-condition-smoke-20261101`；执行 HEAD：`8cbfa69bd194083053917bb0cef615572abdc182`。本报告的执行 SHA 指实际运行源码，发布提交 SHA 见 Git 历史。开始时工作区干净，HEAD 与 origin 一致。

## 协议与前置检查

完整读取并按 CODEX_RUN_FISH16_SIX_CONDITION_SMOKE.txt 执行。仅一次 micro-batch=16、accumulation=1、effective batch=16、seed42 运行，未安装或调整软件包。
语法检查7/7；AMP恢复测试12/12；既有模型/数据契约10/10；数值诊断测试5/5，所有退出码为0。完整控制台日志与退出码保存在外部预检目录。

Python：`D:\app\minicondaaa\envs\xiaoju\python.exe`，版本 3.12.12；PyTorch 2.14.1+cu132；CUDA build 13.2；cuDNN 92400；GPU NVIDIA GeForce RTX 5070 Laptop GPU。

使用锁定的Fish16划分与官方19c8e357初始化。训练224×224、FP16 autocast/GradScaler、4 workers、SGD momentum0.9；Stage-I首warmup LR0.002、wd1e-4；Stage-II计划首LR0.01、wd5e-4，未执行完整Stage-II Epoch。未缩放90/10原日程到1 Epoch。
H2T rho0.25、PI-H2TF paper_theoretical_bounds/ratio_detach=True、原PIF公式和初始化均保留。Stage-II原计划冻结extractor和BN；四组初始单批测试已验证冻结，未将其称为完整Epoch验证。
cudnn.deterministic=True，benchmark=False，cudnn.allow_tf32=False，cuda.matmul.allow_tf32=False。

## 真实执行状态

`ACTUAL_EXECUTION_STATUS.json`：FAILED_STOPPED；completed_conditions=[A0]；stage1_completed=[A0]；stage2_completed=[]；current_condition=B0。
四组真实Stage-II单批测试A1/A2/B1/B2均PASS，classifier有更新且extractor/BN未变；这些一次性更新均丢弃。只有A0存在完整result.json；B0无result.json，其余四组Epoch未开始；SMOKE_SUMMARY.json不存在。预设Epoch预算不是完成证据。

| 条件 | 完整Epoch状态 | 样本数 | 尝试步数 | 实际更新 | AMP跳步 | Epoch均值loss | 训练秒 | peak allocated MiB |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| A0 | 完成1 Epoch | 16269 | 1017 | 1017 | 0 | 1.016161536 | 113.090 | 995.446 |
| B0 | Stage-I失败，未完成 | 未记录 | 未记录 | 未记录 | 未记录 | 未记录 | 未记录 | 未记录 |
| A1 | 完整Epoch未开始 | 0 | 0 | 0 | 0 | — | — | — |
| A2 | 完整Epoch未开始 | 0 | 0 | 0 | 0 | — | — | — |
| B1 | 完整Epoch未开始 | 0 | 0 | 0 | 0 | — | — | — |
| B2 | 完整Epoch未开始 | 0 | 0 | 0 | 0 | — | — | — |

上表步数只统计完整Epoch路径，排除已丢弃的单批测试。B0未知计数不能填0；A1/A2/B1/B2的0来自完整Epoch未进入。loss为result.json保存的样本加权Epoch均值，单独最后一批loss未持久化。train_seconds不包含validation；peak是runner从Epoch开始重置后、保存result时的峰值。

| 条件 | Validation Overall % | Macro % | Head图像加权 % | Medium图像加权 % | Tail图像加权 % | Worst-class % |
|---|---:|---:|---:|---:|---:|---:|
| A0 | 95.570793 | 69.888085 | 97.173004 | 93.040291 | 22.000000 | 0.000000 |
| B0 | — | — | — | — | — | — |
| A1 | — | — | — | — | — | — |
| A2 | — | — | — | — | — | — |
| B1 | — | — | — | — | — | — |
| B2 | — | — | — | — | — | — |

A0 Head/Medium/Tail类宏平均：98.780814% / 88.181817% / 16.923077%。仅validation，5396张；其余条件没有validation结果。

本次runner到失败用时 153.044 秒；含Python启动及退出的进程wall time为 155.822 秒。A0 checkpoint SHA256为 `a4989aef49cfeb51e8bcb53dc43781fbf4733face96c3898e78834929fd57f43`，权重仅保留本地。

## 失败证据与未知项

`GradientContractError: Non-finite gradients do not match the single audited PIF b overflow.`
调用位置：smoke.py:214 -> amp_recovery.py:208。现有门禁拒绝了不满足唯一已审计正Inf模式的非有限梯度，进程exit code=1。未修改代码、未放宽检查、未手动修补梯度。
这不是CUDA OOM，不满足micro-batch=8/accumulation=2的唯一重试条件，因此没有_02运行，也没有继续剩余条件。FAILURE.json与ACTUAL_EXECUTION_STATUS.json均已保留。
B0 optimizer_step_events.json、input_controls.json和result.json在失败前均未持久化，因此确切失败batch、参数名/坐标、NaN/Inf数量、实际更新/跳步次数、当前scale及范围未确定。不能把本次错误认定为历史batch6事件，也不能断言此前没有发生过该事件；65536->32768和后续真实更新均未在本轮证实。源码在调用门禁前检查logits/loss有限，但失败批次的具体数值没有保存在输出中。
历史独立数值诊断保持原样。本轮不新增重放、不尝试不同scale、不改PIF/融合公式、初始化、采样或其他超参数。当前B0事件的具体数值根因未确定，交由科学审查。

## 配对、科学限制和保护

CPU契约通过；四组真实单批测试通过。完整Stage-I A0/B0增强和MixUp逐批配对未完成验证；完整Stage-II recipient/donor配对、初始化一致性和BN冻结也未运行验证，不能标为PASS。
未得到A2 vs A1或B2 vs B1的完整比较；A0高低频validation指标仅说明此次smoke的诊断输出，不构成方法性能、跨域退化或算法优劣结论。
复核了24个受保护输入/源文件的SHA256、大小和mtime，均未变化；原始H2T/PI-H2T源码哈希和全部7个执行任务源码哈希一致，旧CPU契约日志和源码哈希记录保持原样。

## 产物

外部运行目录：`D:\h2t_cross_domain_results\fish16_six_condition_smoke_seed42_20261101_01`。
预检与原始日志目录：`D:\h2t_cross_domain_results\fish16_smoke_preflight_20261101_01`。
本Git目录只含SMOKE_RESULT.json、SMOKE_REPORT.md、OUTPUT_SHA256.csv。索引覆盖外部run_manifest、实际状态、FAILURE与唯一存在的A0/result.json，共4项。图像、数据manifest、逐样本标识/图像哈希、采样数组、input_controls、donor_pair_audit、checkpoint与原始日志均未上传。

**未读取或评价TEST图像，未启动正式90+10 Epoch训练，未运行seed43/44。任务已停止，等待科学分析；本报告不授权任何后续训练、修补或重试。**
