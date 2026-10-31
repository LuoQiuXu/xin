# PIF_GRAD_DIAG_REPORT

日期：2026-10-31（Asia/Shanghai）。

## 已证实的根因

**在受控 B0 重放中，AMP loss scale=65536 使 PIF Conv1d 的共享系数 b 的 FP16 权重梯度溢出为 +Inf。** 主干与分类头梯度均有限且无缺失。当前 smoke 在 GradScaler 正常 skip/update 之前执行严格梯度断言，因而停止。不是前向 NaN，也没有证据支持将其称为 PI-H2T 的算法性能退化。

首次复现失败：seed=42，epoch=0（第1个smoke epoch），batch=6（第7批），此前仅6次成功SGD更新。真实复现仍执行原 smoke.finite_gradients 门禁，失败批次没有进行SGD更新；未绕过或放宽门禁。

历史 B0 原日志没有 batch 编号，因此不把本次 batch6 写回旧记录。此次从相同初始化重放，并逐批严格核对旧同流增强图像 SHA、MixUp RNG SHA 和 lambda，全部匹配；源码、版本和采样顺序也受锁定。

## 锁定条件与复现

分支 experiment/fish16-h2t-pih2t-cross-domain-20261031，起始工作区干净，HEAD为4916ba98855910659433c6f50ac596ca0fb1d30f。原19c8e357权重、tensor副本、原始源码及既有输出保持只读。

manifest SHA：`62d9a2e9fffaaca22ef0c99c0ce4134b4d0e1ecc308333a7cb57b10cf95f0f30`；原权重 SHA：`19c8e3572231adff6824a2da93fd67b5986919a2e65f8b6007eab4edee220097`。

B0 batch/effective=16、AMP、4 workers；首epoch warmup LR0.002、SGD momentum0.9、wd1e-4、MixUp alpha1。所有增强直接复用原 data.Views；Stage-I顺序从旧npz读取并与seed42生成值精确比较。cudnn deterministic=True、benchmark=False、两项TF32=False，均与旧smoke一致。没有重跑A0或任何Stage-II epoch。

初始PIF a/b=[-0.027100861072540283, -0.604386568069458]；失败前a/b=[-0.02449430152773857, -0.6112035512924194]；失败批次 MixUp lambda=0.814341567785799，loss=1.709420561790466。

所有参数逐名记录了None、有限性、NaN/Inf数、最大绝对值、L2、dtype，并分别汇总PIF、classifier、stem及layer1–4。该批所有梯度路径存在；只在PIF的一个系数上发现1个Inf、0个NaN。

## 同一快照、同一批次的数值对照

每个条件都创建新的原模型，严格载入同一失败前模型/BN与optimizer状态，恢复RNG，并复用已保存的实际mixed input和targets；不沿用上一个条件的更新。对照不执行SGD更新，只有独立GradScaler行为副本调用正常step/update来验证跳步。

| 条件 | scale | 严格梯度门禁 | 未缩放 da | 未缩放 db | Conv1d实际权重梯度dtype |
|---|---:|---|---:|---:|---|
| amp_original_scale | 65536.0 | FAIL | -0.0006914138793945312 | +Inf | torch.float16 |
| amp_32768 | 32768.0 | PASS | -0.0006914138793945312 | 1.2802734375 | torch.float16 |
| amp_16384 | 16384.0 | PASS | -0.0006914138793945312 | 1.2802734375 | torch.float16 |
| amp_8192 | 8192.0 | PASS | -0.0006914138793945312 | 1.2802734375 | torch.float16 |
| amp_4096 | 4096.0 | PASS | -0.0006914138793945312 | 1.2802734375 | torch.float16 |
| fp32 | 无AMP | PASS | -0.0007348607759922743 | 1.2799643278121948 | torch.float32 |

全部AMP对照的输入、PIF feature、logit、loss SHA完全一致；仅反向loss scale不同。FP32输入SHA一致，forward数值因精度不同而变化，未将近似相等当作精确重放。

有限AMP对照给出 db=1.2802734375，四个scale的未缩放PIF梯度逐值相同。乘65536后应为 **83904.0**，超过本地torch.finfo(float16).max=**65504.0**；scale32768下实际cast权重梯度为41952且有限。

叶子参数仍为FP32，但实际Conv1d反向保存的权重和其梯度为FP16。钩子已在这个cast权重梯度处捕获+Inf，早于回传到FP32参数和unscale；unscale无法把已溢出的Inf变回有限数。输入、layer1–4输出、PIF输出、logits和loss全部有限。各层梯度不是缺失路径，也没有在此批发现主干梯度溢出。

### 正常GradScaler行为（独立一次性副本）

相同原scale65536下正常调用scaler.step/update：optimizer实际调用次数=0、全部参数不变、scale从65536降到32768。unscale前后scale均为65536，下降发生在update。训练forward本身会更新BN；已另核对step/update没有改变其forward后buffer。本副本随后丢弃，没有用于实际smoke或后续训练。

## 失败批次及范围

完整16个sample ID、track ID、class/label、MixUp伙伴及lambda见 FAILING_SAMPLES.csv。逐张核验原图SHA且全部来自train。原始批次和混合批次张量范围、各阶段features/logits/loss范围见对应forward.json与batch_trace.json。

| 张量 | dtype | shape | 最小值 | 最大值 | 有限 |
|---|---|---|---:|---:|---|
| input | torch.float32 | [16, 3, 224, 224] | -2.1179039478302 | 2.640000104904175 | True |
| PIF_input | torch.float16 | [16, 2048, 7, 7] | 0.0 | 14.125 | True |
| PIF | torch.float16 | [16, 2048, 7, 7] | -8.9609375 | 0.0238494873046875 | True |
| pooled_features | torch.float16 | [16, 2048] | -1.966796875 | 0.011932373046875 | True |
| classifier | torch.float16 | [16, 16] | -0.6318359375 | 2.314453125 | True |
| loss | torch.float32 | [] | 1.7094205617904663 | 1.7094205617904663 | True |

本批数据本身有限，同批FP32及较低scale均产生有限梯度，故不能归因为坏图、无效标签或一个在FP32下也不适定的数据批次。但触发幅度会依赖数据、状态与初始化；本轮没有证明所有其他批次都安全。

## PIF论文与本地初始化核对

[作者 arXiv v2](https://arxiv.org/html/2506.00625v2) §III-C/3.3，归档PDF第4–5页Eq.(3)(4)(6)：通道均值、a*(F-mean)+b*F、池化前融合、1×1的2输入1输出Conv1D无bias两参数均可核实，本地逐项一致。完整来源指纹、页图和TeX记录见paper_evidence及PAPER_FORMULA_INITIALIZATION_AUDIT.json。

作者Algorithm1仅泛指随机初始化模型；可用正文/附录TeX未给出PIF a/b的精确初值或分布。当前PyTorch Conv1d默认Kaiming uniform(a=sqrt(5))、fan_in=2，对应±1/sqrt(2)范围；这是本地实现选择，不能称为已核实作者初始化。seed42下实际初值已经记录。诊断未改变PIF初始化、公式、架构或bias。

AMP和GradScaler、Fish16 ImageNet预训练ResNet50与本轮预算也属于本地受控适配。该结果只确定当前实现/环境下的数值机制，不证明论文方法缺陷、作者实现有同样错误或跨域精度退化。

## 测试、产物与保护

新增CPU统计测试5/5、既有契约测试10/10、诊断脚本语法检查通过。诊断只顺序复现B0的0–6批，随后执行6个同批对照及1个正常GradScaler副本。失败前snapshot、原始模型/BN/optimizer/RNG、mixed输入均保存在exact_failed_batch_snapshot.pt；它不是新的训练模型。

关键文件：ROOT_CAUSE_EVIDENCE.json、DIAGNOSIS_RESULTS.json、FAILING_BATCH.json、FAILING_SAMPLES.csv、SCALE_COMPARISON.csv、ALL_PARAMETER_GRADIENTS.csv、GRADIENT_GROUP_SUMMARY.csv、batch_log.csv；每批/每条件目录有forward、backward、全部参数梯度和PIF实际梯度张量。所有报告、数据和snapshot在本新外部目录。最终旧文件保护检查见FINAL_VALIDATION.json及OUTPUT_SHA256.csv。

原smoke.py、models.py、data.py、PIF/H2T源码、初始化与原梯度门禁保持不变。没有Stage-II epoch、TEST模型评价、seed43/44或正式训练；没有覆盖旧FAILURE、报告、checkpoint或日志，未提交或推送。

## 剩余问题及最小后续验证

本次失败的数值根因已有充分证据，不需新实验才能认定。尚未验证：完整B0 epoch或后续批次的数值稳定性、其他初始化/软件环境是否同样触发、作者a/b精确初始化、以及任何方法精度结论；原历史运行的batch编号也无直接日志证明。

科学复核后，最小后续验证是先明确AMP非有限梯度处理规范，再仅对B0做有完整scale/skip/梯度日志的短smoke验证。该规范不能通过本报告自动变更；现有实际smoke有限性门禁仍保持原样。本轮到此停止，等待科学审查。
