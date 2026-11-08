# CC-SAM C3 正式训练执行报告

## 固定第 30 轮的配对结果

三个 Stage II seed 共享 Clean Stage I seed 42、epoch 184 的冻结表征与同一 TRAIN holdout。验证集含 1,980 张图像，类 1–99 各 20 张。下表为三组均值 ± 样本标准差，单位为 %。

| 条件 | Overall | Many | MediumN | Few |
|---|---:|---:|---:|---:|
| C0 | 46.869 ± 0.175 | 64.804 ± 0.340 | 48.619 ± 0.675 | 24.500 ± 0.289 |
| C1 | 46.347 ± 0.029 | 58.431 ± 0.306 | 48.571 ± 0.247 | 30.056 ± 0.192 |
| C2 | 46.902 ± 0.105 | 63.235 ± 0.441 | 49.048 ± 0.918 | 25.889 ± 0.419 |
| C3 | 46.296 ± 0.117 | 57.892 ± 0.370 | 48.619 ± 0.218 | 30.444 ± 0.347 |

Many 按原 TRAIN 类计数 >100，Few 按 <20，MediumN 覆盖 20–100。配对差值按 seed 顺序列出，单位为百分点。

| 对照 | 指标 | 123 | 124 | 125 | 均值 ± 样本标准差 |
|---|---|---:|---:|---:|---:|
| C3−C1 | Overall | -0.152 | +0.000 | +0.000 | -0.051 ± 0.087 |
| C3−C1 | Many | -0.588 | -0.294 | -0.735 | -0.539 ± 0.225 |
| C3−C1 | MediumN | +0.143 | -0.143 | +0.143 | +0.048 ± 0.165 |
| C3−C1 | Few | +0.000 | +0.500 | +0.667 | +0.389 ± 0.347 |
| C3−C2 | Overall | -0.859 | -0.505 | -0.455 | -0.606 ± 0.220 |
| C3−C2 | Many | -5.735 | -4.559 | -5.735 | -5.343 ± 0.679 |
| C3−C2 | MediumN | -0.857 | -0.857 | +0.429 | -0.429 ± 0.742 |
| C3−C2 | Few | +4.667 | +4.500 | +4.500 | +4.556 ± 0.096 |
| C3−C0 | Overall | -0.909 | -0.404 | -0.404 | -0.572 ± 0.292 |
| C3−C0 | Many | -7.500 | -6.176 | -7.059 | -6.912 ± 0.674 |
| C3−C0 | MediumN | -0.286 | -0.286 | +0.571 | +0.000 ± 0.495 |
| C3−C0 | Few | +5.833 | +6.000 | +6.000 | +5.944 ± 0.096 |
| C1−C2 | Overall | -0.707 | -0.505 | -0.455 | -0.556 ± 0.134 |
| C1−C2 | Many | -5.147 | -4.265 | -5.000 | -4.804 ± 0.473 |
| C1−C2 | MediumN | -1.000 | -0.714 | +0.286 | -0.476 ± 0.675 |
| C1−C2 | Few | +4.667 | +4.000 | +3.833 | +4.167 ± 0.441 |

### 机制判断

C3 只将 Many 样本的输入扰动设为零。Many 样本仍参与两项交叉熵，所有类别继续更新同一个分类器。配对差值包含训练轨迹随干预产生的变化。

- **Many。** C3 相对 C1 在三个 seed 均下降，平均低 0.539 pp。相对 C2 的 Many 差距从 4.804 pp 扩大到 5.343 pp。
- **Few 收益保留。** C3−C2 为 +4.556 pp，3/3 个 seed 为正。原 C1−C2 为 +4.167 pp，保留比例为 109.3%。
- **Overall。** C3 相对 C1 下降 0.051 pp，相对 C2 下降 0.606 pp，相对 C0 下降 0.572 pp。
- **剩余 Many 差异。** C3−C2 为 -5.343 pp。该差异出现在 Many 的直接扰动为零的条件下，与 MediumN/Few 扰动经共享 Softmax 分类器间接影响 Many 决策边界的解释一致。当前设计估计的是净差异，具体边界移动和训练轨迹的贡献仍未分解。
- **类别覆盖。** class 0 的 TRAIN 计数为 500，三个采样计划均包含该类，属于 C3 的 Many 屏蔽范围。该类在 holdout 中有 0 张图像，Many 结果覆盖其余 34 个有验证样本的 Many 类。
- **证据范围。** 结果描述固定表征和开发用 TRAIN holdout 上的配对变化。三个 seed 是 Stage II 的随机性重复。语义丰富度解释、独立 TEST 性能、统计显著性和方法创新均超出这些证据的范围。

### Few 类诊断

按三 seed 平均准确率筛选 C1>C2 且 C3<C1，共 1 个 Few 类。每类的 20 张验证图像在三个 seed 间复用，每张图像对应单 seed 准确率 5 pp。逐类差值用于定位变化。

| 类 ID | TRAIN 计数 | C1−C2 (pp) | C3−C1 (pp) | C3−C2 (pp) |
|---:|---:|---:|---:|---:|
| 71 | 18 | +1.667 | -1.667 | +0.000 |

完整逐 seed 准确率、八项指标的逐 seed 差值及 99 个有验证样本类别的结果见 [结果表](c3_final_epoch_metrics.csv)、[配对表](c3_paired_deltas.csv) 和 [逐类表](c3_per_class.csv)。

### 兼容性附表

旧 Head、OfficialMedium、Tail 口径，均值 ± 样本标准差，单位为 %。

| 条件 | Head | OfficialMedium | Tail |
|---|---:|---:|---:|
| C0 | 63.762 ± 0.297 | 48.667 ± 0.595 | 24.310 ± 0.299 |
| C1 | 57.619 ± 0.297 | 48.857 ± 0.247 | 29.713 ± 0.199 |
| C2 | 62.286 ± 0.429 | 49.048 ± 0.918 | 25.747 ± 0.434 |
| C3 | 57.095 ± 0.360 | 48.905 ± 0.218 | 30.115 ± 0.359 |

## 执行与门禁

正式训练使用 NVIDIA GeForce RTX 5070 Laptop GPU。CIFAR-100-LT IF=100 的封存 TRAIN 含 10,847 张图像。按原协议使用 64-D GAP、FP32、batch size 64 和恒定学习率 0.01 的 SGD，momentum 为 0.9，weight decay 为 0.0002。

所有命令从实验目录执行，解释器为 `D:\app\minicondaaa\envs\xiaoju\python.exe`。时间为本机 UTC+08:00。

| 阶段 | 解释器参数 | 开始 | 结束 | 退出码 |
|---|---|---|---|---:|
| syntax | `-B -m py_compile train_ccsam.py train_c3.py analyze_c3.py test_c3_semantics.py` | 2026-11-08T20:55:47+08:00 | 2026-11-08T20:55:47+08:00 | 0 |
| unit_tests | `-B -m unittest -v test_c3_semantics` | 2026-11-08T20:55:47+08:00 | 2026-11-08T20:55:53+08:00 | 0 |
| partial_batch_cpu | `-B -c 〈31 样本重复标签与 FP32 容差验证〉` | 2026-11-08T20:55:53+08:00 | 2026-11-08T20:55:58+08:00 | 0 |
| preflight | `-B -u train_c3.py --preflight` | 2026-11-08T20:55:58+08:00 | 2026-11-08T20:56:04+08:00 | 0 |
| smoke | `-B -u train_c3.py --smoke-only` | 2026-11-08T20:57:28+08:00 | 2026-11-08T20:57:37+08:00 | 0 |
| post_smoke_seal | `-B -c 〈Smoke 后输入与原始结果复核〉` | 2026-11-08T20:57:37+08:00 | 2026-11-08T20:57:43+08:00 | 0 |
| formal | `-B -u train_c3.py --formal` | 2026-11-08T20:57:43+08:00 | 2026-11-08T21:18:15+08:00 | 0 |
| formal_verification | `-B -c 〈实际训练状态、最终权重、样本数及封存 SHA 验证〉` | 2026-11-08T21:18:26+08:00 | 2026-11-08T21:18:34+08:00 | 0 |
| analysis | `-B -u analyze_c3.py` | 2026-11-08T21:18:34+08:00 | 2026-11-08T21:18:40+08:00 | 0 |

语法检查和 6 项 CPU 单元测试通过。附加 31 样本 CPU 验证使用重复标签，沿用 rtol=1e-5 和 atol=1e-7。Preflight 通过后执行一次真实 Smoke，覆盖三个 seed 各两个连续增强 RNG batch 和一次丢弃的分类器更新。六个 GAP SHA 与原 C0 轨迹匹配，Many 扰动精确为零，同一分类器状态下非 Many 扰动逐位等于 C1。全 Many 的 loss 等于 C2，全非 Many 的 loss 等于 C1。

正式训练实际完成 3 个 seed、90 个 epoch、15,300 次优化器更新和 976,230 个采样位置。每个 epoch 的实写记录均包含 170 个训练批次，最后一批为 31 张。

| Seed | 实际 Epoch | Optimizer steps | 采样位置 | 与 C0 匹配批次 | Class 0 采样位置 | 末轮平均扰动 L2 |
|---:|---:|---:|---:|---:|---:|---:|
| 123 | 30 | 5,100 | 325,410 | 5,100 | 3,254 | 0.972159 |
| 124 | 30 | 5,100 | 325,410 | 5,100 | 3,254 | 0.966387 |
| 125 | 30 | 5,100 | 325,410 | 5,100 | 3,254 | 0.971283 |

分析验证了 C0/C1/C2/C3 的初始分类器与 Backbone 哈希以及完整批次轨迹逐项一致。轨迹涵盖样本与标签、增强图像及 GAP 特征 SHA。全部 90 轮通过 Backbone 和 BN 缓冲区的逐位不变检查。三个最终分类器均为有限 FP32，且参数已相对共同初始化更新。

Stage I checkpoint SHA256 为 `7ced9d07f71335357c08623573f038acd0dfdcdeadb55d95d57b332a14f83a88`。

原九组封存清单及 52 项文件逐项 SHA256 一致，16 项封存输入和 140 项历史未跟踪资产经复核保持一致。受保护文件变更数为 **0**。Preflight、Smoke 和正式训练的官方 CIFAR-100 TEST guard 均为 0 次 TEST 文件打开尝试和 0 次 CIFAR100(train=False) 尝试。固定第 30 轮分析读取已保存的 holdout 评价与配对轨迹。

[完整命令和终端日志](C3_EXECUTION.log) · [Preflight](preflight_report.json) · [Smoke](smoke_report.json) · [完成标记](formal_completed.json) · [正式训练复核](C3_FORMAL_VERIFICATION.json) · [初始封存输入](C3_INPUT_SEAL.json)

## 本机权重与轨迹

下列六个文件保留在本机。SHA256 及大小也登记在 [产物清单](ARTIFACT_SHA256.csv)。

| Seed | 本机绝对路径 | SHA256 |
|---:|---|---|
| 123 | `D:\xiangmuxunlian\xin\H2T_repro_audit\ccsam_stage2_controlled_cifar100_20261106\results\seed_123\C3_NO_MANY_PERTURB\final_classifier.pth` | 045bc61ff1c9d633c4b958a0d29639a290ccb11bbf1101a137a5e1a2a65f0a69 |
| 123 | `D:\xiangmuxunlian\xin\H2T_repro_audit\ccsam_stage2_controlled_cifar100_20261106\results\seed_123\C3_NO_MANY_PERTURB\batch_trace.csv` | e422f9c106cf90399831df7e9d6c6377b8d98e1d677d8ba929d120d55bf4dcdc |
| 124 | `D:\xiangmuxunlian\xin\H2T_repro_audit\ccsam_stage2_controlled_cifar100_20261106\results\seed_124\C3_NO_MANY_PERTURB\final_classifier.pth` | 6d805acf29aeb89389083b074492aa1cec09c87169b672470113c8e89626c16c |
| 124 | `D:\xiangmuxunlian\xin\H2T_repro_audit\ccsam_stage2_controlled_cifar100_20261106\results\seed_124\C3_NO_MANY_PERTURB\batch_trace.csv` | ffa484e36bcb88d1e1c889a49a34c315602de38f105e67ee24a11049d55e44e9 |
| 125 | `D:\xiangmuxunlian\xin\H2T_repro_audit\ccsam_stage2_controlled_cifar100_20261106\results\seed_125\C3_NO_MANY_PERTURB\final_classifier.pth` | 887bb240a0080b289a4ff09de8f9bb83972ddfd622e1c6bce2b58a2637a116de |
| 125 | `D:\xiangmuxunlian\xin\H2T_repro_audit\ccsam_stage2_controlled_cifar100_20261106\results\seed_125\C3_NO_MANY_PERTURB\batch_trace.csv` | abbab224efec845ea9297005938a0aeaabb055c8221bb0b11feaee9b38637888 |

## Git 与封存范围

- 工作分支为 `experiment/ccsam-c3-no-many-perturb-20261108`，归档前 HEAD 为 `3ade5e466a33093f7d559b99e106ad4bf893803e`，与同名 GitHub 跟踪分支一致。
- 原结果基线为 `f916957ea4a7b90cf82dc5839eb193e044d881dd`，基线到归档前 HEAD 的 diff 检查通过。
- 归档前 git diff --check 返回 0，git status 中跟踪文件无变更。其余未跟踪项为 10 个已封存历史目录和本轮独立产物，历史目录逐文件验证。
- 本轮归档报告、状态、指标和核查证据。归档属性保留 SHA256 清单所记录的文件原字节。

分支相对原结果基线的源码及协议变更见下列清单。生产文件的改动详见 [源码差异](C3_PRODUCTION_DIFF.patch)。

- 新增 `H2T_repro_audit/ccsam_stage2_controlled_cifar100_20261106/CODEX_C3_NO_MANY_PERTURB_20261108.txt`
- 新增 `H2T_repro_audit/ccsam_stage2_controlled_cifar100_20261106/analyze_c3.py`
- 新增 `H2T_repro_audit/ccsam_stage2_controlled_cifar100_20261106/c3_selective/C3_PROTOCOL.md`
- 新增 `H2T_repro_audit/ccsam_stage2_controlled_cifar100_20261106/test_c3_semantics.py`
- 新增 `H2T_repro_audit/ccsam_stage2_controlled_cifar100_20261106/train_c3.py`
- 修改 `H2T_repro_audit/ccsam_stage2_controlled_cifar100_20261106/train_ccsam.py`

本次归档文件路径相对实验目录。

- `c3_selective/.gitattributes`
- `c3_selective/ARTIFACT_SHA256.csv`
- `c3_selective/C3_EXECUTION.log`
- `c3_selective/C3_EXECUTION_REPORT.md`
- `c3_selective/C3_FORMAL_VERIFICATION.json`
- `c3_selective/C3_INPUT_SEAL.json`
- `c3_selective/C3_PRODUCTION_DIFF.patch`
- `c3_selective/C3_RESULTS.md`
- `c3_selective/c3_final_epoch_metrics.csv`
- `c3_selective/c3_paired_deltas.csv`
- `c3_selective/c3_per_class.csv`
- `c3_selective/c3_summary.json`
- `c3_selective/formal_completed.json`
- `c3_selective/preflight_report.json`
- `c3_selective/smoke_report.json`
- `results/seed_123/C3_NO_MANY_PERTURB/metrics.csv`
- `results/seed_123/C3_NO_MANY_PERTURB/status.json`
- `results/seed_124/C3_NO_MANY_PERTURB/metrics.csv`
- `results/seed_124/C3_NO_MANY_PERTURB/status.json`
- `results/seed_125/C3_NO_MANY_PERTURB/metrics.csv`
- `results/seed_125/C3_NO_MANY_PERTURB/status.json`
