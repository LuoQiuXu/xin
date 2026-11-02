# B0 AMP 恢复验证与单 Epoch Stage-I 报告

日期：2026-11-02（Asia/Shanghai）。**真实CUDA负Inf恢复验证通过；B0完整Stage-I单Epoch完成。仅到此停止，等待科学分析。**

分支：`experiment/fish16-six-condition-smoke-20261101`；执行HEAD：`d47cbd7293402fd02634c176d7e16db288b936b9`，与同步后的origin一致，起始工作区干净。当前报告的执行SHA指实际运行源码，报告发布commit见Git历史。

## 按顺序完成的三层证据

1. CPU：3个文件分别py_compile通过；AMP恢复12/12（含正负Inf）、模型/数据契约10/10、统计诊断5/5，所有命令exit0。原始源码、manifest与官方19c8e357权重SHA校验通过。
2. 快照：只使用2026-11-01的Batch13失败快照，一次真实CUDA重放，精确梯度核验通过后才调用修正后的guard。
3. 训练：快照验证全部通过后，另起全新进程执行smoke.py --micro-batch 16 --b0-stage1-only；一次完整Epoch及validation，未复用一次性恢复副本。外部文件mtime顺序及启动前断言均符合该顺序。

环境：Python 3.12.12（`D:\app\minicondaaa\envs\xiaoju\python.exe`）；PyTorch 2.14.1+cu132；CUDA build 13.2；cuDNN 92400；NVIDIA GeForce RTX 5070 Laptop GPU。

## Batch13 真实GradScaler恢复

沿用已有取证重放流程，严格载入post-forward/pre-step模型、失败前optimizer及unscale后scaler状态，恢复CPU/CUDA RNG；同一mixed input、MixUp targets/lambda、FP16 autocast和保存scale32768。没有数据集访问或scale/精度对照。
162个参数张量无缺失梯度、0 NaN、总共1个Inf；唯一异常是extractor.pif.fuse.weight[0,1,0]的负Inf，其余梯度有限。loss、logits及PIF梯度与保存值逐位一致，没有差异元素。
原生guard结果为recoverable_pif_b_overflow_skip：scale32768->16384，真实optimizer.step计数0；全部模型参数SHA不变，optimizer state逐项相同，原快照SHA不变。该副本被丢弃，没有用于后续训练。train前向会正常更新副本BN buffers，不声称buffers未改变。
原快照SHA256：`428be9c2db422ed1fcb2e5a1a440c803764a6fd8729914c1401ee16be0bb4b65`。

## B0完整Stage-I Epoch真实状态

`ACTUAL_EXECUTION_STATUS.json`及`B0_STAGE1_ONLY_SUMMARY.json`均为COMPLETE_B0_STAGE1_ONE_EPOCH_ONLY；completed_conditions=[B0]、stage1_completed=[B0]、stage2_completed=[]。B0/result.json与两者内容一致，进程exit0。没有用预设预算充当完成证据。
锁定条件：seed42、原ResNet50/PIF公式与初始化、官方19c8e357预训练、224输入、原sampler/MixUp/transforms；SGD LR0.002、momentum0.9、wd1e-4；micro/effective=16、4 workers、FP16/原生GradScaler。未手动降低scale、裁剪/替换梯度、调整学习率或重试。

| 项目 | 实际值 |
|---|---|
| 训练样本 / optimizer窗口 | 16269 / 1017 |
| 真实optimizer.step / AMP跳步 | 1015 / 2 |
| 最大连续跳步 | 1（上限2；总跳步上限3） |
| 初始 / 最终scale | 65536.0 / 16384.0 |
| Epoch样本加权train loss | 1.015388660212 |
| 训练秒 / 进程wall秒 | 128.016 / 147.291 |
| 图像/秒 / peak allocated MiB | 127.085 / 1040.354 |

train_seconds不包含validation；peak是runner在Epoch开始重置后、保存result时的峰值。loss包含所有已处理前向样本（包括原生跳步的两个窗口），单独最后一批loss未持久化。

| 全Epoch batch | Inf坐标/数量 | 本轮event直接记录的符号 | 对应已有符号证据 | 原生scale变化 | 更新计数 | 下一批真实更新 |
|---:|---|---|---|---|---|---|
| 6 | PIF b [0,1,0] / 1，无NaN | 未持久化 | 旧取证中+Inf | 65536->32768 | 6->6 | batch7，32768 |
| 13 | PIF b [0,1,0] / 1，无NaN | 未持久化 | Batch13快照及本轮精确重放为-Inf | 32768->16384 | 12->12 | batch14，16384 |

每次跳步都由native GradScaler完成，参数SHA在前后完全相同；下一窗口均计数+1并使用已降低scale。全1017个事件逐项核验：1015次真实更新、2次跳步，合计准确，未跳过数据批次。
**符号记录限制：**现有StepOutcome只持久化Inf总数和坐标，没有正负计数。本表明确将本轮直接日志与已有参考符号分列，不能把参考符号伪装为本轮直接落盘。两个事件的跳步前参数SHA均与对应参考状态一致，batch6/13的图像视图、MixUp RNG和lambda控制也精确匹配；这是对应关系证据，不是新增梯度符号日志。未为补齐符号而修改runner或追加计算。

## Validation，非TEST

| 指标 | 百分比 |
|---|---:|
| Overall | 97.498146 |
| Macro | 73.356071 |
| Worst-class | 0.000000 |
| Head / Medium / Tail，图像加权 | 98.964760 / 92.673996 / 37.000000 |
| Head / Medium / Tail，类宏平均 | 98.867939 / 87.636363 / 28.461539 |

仅5396张validation图像。PIF初值a/b=[-0.027100861072540283, -0.604386568069458]，末值a/b=[-0.6486831307411194, -0.41493043303489685]。只保存末轮Epoch1，未用TEST选模。checkpoint SHA256：`231295f0c7e9a7c7621f80ab698637bf67cef1c489eb4fbebf275f866745c817`，权重留本地。

## 结论、限制与产物

本轮直接证明：保存的负Inf签名可由修正后的原生GradScaler安全跳步/回退；B0一个完整Epoch在既有约束下完成。仍未定位最早产生负Inf的内部操作，不将恢复成功解释成数值根因已经证实。
本轮不证明完整训练稳定性、六条件完成、Stage-II有效性、跨域泛化或TEST性能。直接逐事件符号日志的缺口如上保留，不自行修补或增加实验。
验证外部根目录（OUTPUT_SHA256 root=validation）：`D:\h2t_cross_domain_results\fish16_b0_amp_recovery_validation_20261102_01`。
旧输入根目录（root=input_forensic）：`D:\h2t_cross_domain_results\fish16_b0_gradient_forensic_seed42_20261101_01`。
完整预检/重放/训练日志、梯度统计、scale/step事件、快照及checkpoint仅在外部目录。Git只含B0_AMP_RECOVERY_RESULT.json、B0_AMP_RECOVERY_REPORT.md、OUTPUT_SHA256.csv；索引覆盖必要CPU/重放/训练结果及权重摘要，不上传任何raw tensor、样本身份、图像、原始日志或数据manifest。
保护核验：52个旧资产/输入的SHA、大小和mtime均未变；原H2T/PI-H2T和执行任务源码哈希均一致，旧取证与六条件报告未覆盖。

**未执行A0/A1/A2/B1/B2、六组smoke、Stage-II、TEST、正式90+10或seed43/44；没有重试或额外实验。停止等待科学分析。**
