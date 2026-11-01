# B0 梯度取证报告

日期：2026-11-01（Asia/Shanghai）。**诊断完成：捕获并精确重放 batch=13 的 PIF b 负 Inf。runner 的真实状态仍是 FAILED_STOPPED。**

分支：`experiment/fish16-six-condition-smoke-20261101`；执行 HEAD：`794eabbb55167b616d4562a0d2b7c8e62a006f34`。工作区开始时干净，HEAD 与 origin 一致。

## 范围与检查

完整执行 CODEX_B0_GRADIENT_FORENSIC.txt。仅运行 smoke.py --forensic-b0-only --max-batches 100 --micro-batch 16；没有六组重跑、A0/A1/A2/B1/B2、validation 或 TEST。
3/3文件语法检查，12/12 AMP测试、10/10契约测试、5/5数值诊断测试全部通过，所有退出码0。源码核对确认：post-forward/backward/unscale之后、任何失败窗口step/update之前保存JSON和快照，再重新抛出异常。
环境：Python 3.12.12（`D:\app\minicondaaa\envs\xiaoju\python.exe`），PyTorch 2.14.1+cu132，CUDA build 13.2，cuDNN 92400，NVIDIA GeForce RTX 5070 Laptop GPU。

使用既有锁定manifest、seed42采样与相同增强、官方19c8e357权重、原PIF初始化/公式，SGD LR0.002、momentum0.9、wd1e-4，micro/effective=16，FP16 autocast。未手工修改AMP scale、精度或其他超参数；未运行diagnostics/diagnose_b0.py。

## 本轮实际拒绝窗口

| 项目 | 实际值 |
|---|---|
| batch index（从0计） | 13，即第14批 |
| 失败批次前完整处理样本 | 208（13批） |
| 此前optimizer窗口 | 13：12次实际更新，1次原生AMP跳步 |
| 失败窗口样本 | 16 |
| unscale前/后scale | 32768 / 32768 |
| 失败窗口实际optimizer.step | 未执行；计数保持12 |
| 失败窗口GradScaler.step / update | 均未调用 |
| loss / logits有限 | 2.419346094131470 / 两者均有限 |

`GradientContractError: Non-finite gradients do not match the single audited PIF b overflow.` 已在退出前保存完整梯度证据和模型/优化器/GradScaler/输入快照。

## 真实梯度证据

162个优化器参数张量没有任何缺失梯度（missing=[]）；其余161个参数张量的梯度全部有限。唯一非有限参数如下；坐标未截断。

| 参数 | 参数/梯度shape | 梯度dtype | NaN | +Inf | -Inf | 全部非有限坐标 |
|---|---|---|---:|---:|---:|---|
| extractor.pif.fuse.weight | [1,2,1] / [1,2,1] | torch.float32 | 0 | 0 | 1 | [0,1,0]，即b |

PIF参数 a=-0.023714775219559669，b=-0.63910478353500366。PIF梯度张量保存在外部快照中；其有限元素仅1个，min=max=max_abs=L2=0.03009033203125，非有限元素为b坐标的负Inf。Git仅保留统计、坐标与模型参数标量，不上传原始梯度张量。

这不是历史唯一获准的正Inf模式：本轮拒绝事件的符号为负、batch为13、scale为32768。因此原狭窄门禁正确地在step/update前停止，没有对这个事件调用原生GradScaler跳步/回退。
此前batch6确实记录了一个获准正Inf事件：原生GradScaler保持参数不变、实际更新计数6->6、scale65536->32768；batch7随后在32768做了真实更新。这个先前事件与batch13拒绝事件必须区分，不能因为batch6成功恢复而放宽对负Inf的检查。

## 恰好一次只读反向重放

严格加载 model_state_after_forward_before_optimizer_step、optimizer_state_before_failed_step、scaler_state_after_unscale_before_update；恢复已保存CPU/CUDA RNG，在train模式下复用mixed_input、两组MixUp targets与lambda。同一FP16 autocast、同一保存scale32768；只调用一次backward和一次unscale。没有数据集访问，没有scale/精度对照。
重放得到相同负Inf模式后，先核对其必走拒绝分支，再调用原guard复核异常；它在内部step/update之前再次抛出同一GradientContractError。未对有限或获准模式进入可能更新的guard路径。

| 核验 | 结果 |
|---|---|
| backward次数 | 1 |
| loss逐位一致 / 绝对差 | True / 0 |
| logits逐位一致 / 最大绝对差 | True / 0 |
| 已保存PIF梯度逐位一致 | True |
| 完整缺失/非有限摘要与guard diagnostics | 全部相同，差异字段=[] |
| 异常类型及message | 相同 |
| optimizer.step / scaler.step / scaler.update调用 | 0 / 0 / 0 |
| 参数、optimizer state、scaler公开state | 未改变 |

快照是失败前向之后的状态。train模式重放时BN仍使用当前批次统计；重放副本的running buffers会再正常更新一次，因此不声称全部BN buffer未变。原快照未改写，重放副本没有用于继续训练。

## 已证实、未确定与决策

已证实的停止原因：本轮B0窗口出现唯一PIF b负Inf，不匹配仅允许正Inf的既有契约；此状态和原异常可从保存快照精确复现。
尚未确定：最早产生负Inf的具体内部运算及更底层数值机制。本轮没有获准的低scale或FP32对照，不能宣称这些对照会通过，也不能把此前正Inf的数值归因直接套用到本次负Inf。
此前六条件运行没有保存B0批次/梯度，故不能把本轮batch13/负Inf回填为旧运行的已证事实。只有本轮事件和快照重放有直接证据。没有任何方法性能或准确率比较结论。
**决策：STOP_FOR_SCIENTIFIC_REVIEW；不修复，不追加实验。**

## 产物与保护

外部取证目录：`D:\h2t_cross_domain_results\fish16_b0_gradient_forensic_seed42_20261101_01`。
预检/重放脚本及完整日志目录：`D:\h2t_cross_domain_results\fish16_b0_forensic_preflight_20261101_01`。
B0/GRADIENT_FAILURE.json和B0/GRADIENT_FAILURE_SNAPSHOT.pt包含失败窗口全部本地证据；READONLY_REPLAY_RESULT.json和READONLY_REPLAY_PIF_GRADIENT.pt保存一次重放证据。外部JSON和快照的大小、SHA256见本目录OUTPUT_SHA256.csv。
复核38个受保护旧文件/输入的SHA256、大小和mtime，全部一致；本次执行源码和原H2T/PI-H2T源码保持原样，旧六条件结果与报告未覆盖。
本目录只提交B0_FORENSIC_RESULT.json、B0_FORENSIC_REPORT.md、OUTPUT_SHA256.csv。原始图像、样本身份、批次图像哈希、manifest、完整梯度、模型/optimizer/scaler大张量、原始日志均不上传。
未运行六组smoke、其他条件、validation、TEST、正式90+10训练或seed43/44。取证和一次重放完成后停止。
