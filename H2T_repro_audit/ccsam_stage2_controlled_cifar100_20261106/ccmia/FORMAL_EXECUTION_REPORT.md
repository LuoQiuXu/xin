# CC-MIA Formal 实际执行与验证

三个种子各完成固定 30 Epoch，实际配对批次数为 15,300，两臂实际分类头更新次数合计为 30,600。Formal 与 Analysis 退出码均为 0。

分支为 `experiment/ccsam-cross-class-margin-influence-audit-20261117`，执行提交为 `74be7db15297a092171730daba833a0a07f724ec`。正式运行从 2026-11-18T16:41:04+08:00 开始，于 2026-11-18T16:59:55+08:00 完成。

启动采用已通过的独立验证目录 diagnostics/rounding_ledger_validation_20261118_02。源码指纹匹配后直接进入 Formal，所有实验配置沿用已批准版本。报告中的 CPU 13/13、Preflight 和 Smoke 状态来自该版本封存的验证证据。

| Seed | 完成 Epoch | 实际配对批次 | 实际分类头更新 |
|---|---|---|---|
| 123 | 30 | 5100 | 10200 |
| 124 | 30 | 5100 | 10200 |
| 125 | 30 | 5100 | 10200 |

固定第 30 轮的 TRAIN-derived holdout 准确率如下，单位为百分比。

| Seed | C2 Overall | C3 Overall | C2 Many | C3 Many | C2 Few | C3 Few |
|---|---|---|---|---|---|---|
| 123 | 47.0202 | 46.1616 | 63.2353 | 57.5000 | 25.5000 | 30.1667 |
| 124 | 46.8687 | 46.3636 | 62.7941 | 58.2353 | 25.8333 | 30.3333 |
| 125 | 46.8182 | 46.3636 | 63.6765 | 57.9412 | 26.3333 | 30.8333 |

C3 − C2 的三个种子均值和样本标准差如下，准确率差值单位为百分点。

| 指标 | 均值 | 样本标准差 |
|---|---|---|
| Overall | -0.606061 | 0.220146 |
| Many | -5.343137 | 0.679236 |
| MediumN | -0.428571 | 0.742307 |
| Few | 4.555556 | 0.096225 |
| 固定 Many margin | -0.540255 | 0.009334 |

三个种子中的直接扰动分量、实际固定 margin 变化及 Many 准确率变化均为负向，满足原协议的 SUPPORTIVE 判据。参数路径分解保留原定义，并独立记录写回舍入。

| 末轮 margin 分量 | 三种子均值 |
|---|---|
| 实际 C3 − C2 | -0.5402551563 |
| 直接扰动 | -19.92306309 |
| 状态反馈 | 19.34303008 |
| 权重衰减 | 0.03977768314 |
| 写回舍入 | 7.474344598e-08 |
| 重构残差 | 9.548594121e-08 |

状态反馈在这组有序路径分解中抵消了大部分直接负向 margin 分量。最负的跨目标平均来源类为 40、61、84。最负的来源—目标 Many 组合如下。

| 来源类 | 目标 Many 类 | 三种子平均累积直接 margin 分量 | 三个种子方向一致 |
|---|---|---|---|
| 61 | 10 | -29.56486911 | True |
| 71 | 23 | -29.49517859 | True |
| 81 | 13 | -24.14339821 | True |
| 86 | 16 | -22.48811710 | True |
| 95 | 30 | -20.76063228 | True |

完整记录核验覆盖每个种子的八个 CSV、两个末轮 checkpoint 和 status.json。每个种子的行数如下。

| 文件 | 行数 |
|---|---|
| batch_trace.csv | 5100 |
| fixed_competitors.csv | 680 |
| margin_decomposition.csv | 30 |
| margin_metrics.csv | 62 |
| per_class_margin.csv | 2108 |
| source_margin_influence.csv | 66300 |
| source_target_group_gradient.csv | 5850 |
| step_influence.csv | 5100 |

每个种子的 5,100 条 trace 与其完成的 C0 参考逐行相同。所有表格数值字段均为有限值，65 个来源类及 34 个 Many margin 目标类的定义核验通过，90 个 epoch 的来源累积贡献与直接分量加性、margin 分解求和均通过独立核验。

全程最大参数重构相对误差为 7.33427946e-06，最大动量缓冲重构相对误差为 1.328435682e-05，原相对门槛为 0.005。全部完成步骤通过绝对 2e-6 门禁。逐步 CSV 记录相对误差，全程精确最大参数绝对误差未单独保存。

90 轮中固定 margin 重构最大绝对残差为 9.031792527e-06，这一量的单位为 logit margin。

backbone 和 BN 的 bitwise 检查通过，官方 TEST 文件打开和 CIFAR100(train=False) 两项计数均为 0。数据为 1,980 张 TRAIN-derived holdout，class 0 缺席。本实验提供三种子配对机制证据，结论限定于该数据划分和有序路径分解，独立测试泛化、统计显著性与分解顺序无关的因果归因仍超出本轮证据范围。

全部逐种子 CSV 和汇总报告纳入本轮提交。六个分类头 checkpoint 与完整控制台日志保留本地，SHA256 和文本换行规范化后的 SHA256 见 ARTIFACT_SHA256.csv。原始 SHA256 对应实际本地字节。formal_completed.json 和 analysis/summary.json 按父目录属性保留原始 CRLF 字节，其余文本如被 Git 规范化为 LF，可用 sha256_lf_normalized 核验。

运行状态与完整指纹见 RUN_VALIDATION.json，科学结果见 CCMIA_RESULTS.md 和 analysis/ 下的配对、来源类别及来源—目标结果表。

提交检查识别到上述两个字节保留 JSON 的 CRLF。新增 ccmia/.gitattributes 为这两个文件声明 cr-at-eol，同时保留默认尾随空格、空白行和空格先于制表符检查。实验数据文件保留原始字节，Git 检查日志保存在 diagnostics/formal_git_whitespace_20261118_01.log。
