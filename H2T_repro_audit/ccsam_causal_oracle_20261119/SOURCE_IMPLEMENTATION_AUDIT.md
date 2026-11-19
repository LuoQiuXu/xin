# CC-MIA 来源与实现审计

本轮分支：`experiment/ccmia-causal-oracle-20261119`。
起始 HEAD：`398200c263fff2c6130bd1d12428b559af5c4a68`。
已完成父实验 commit：`63ae0ec84ff6b6579d682aec230811effc7a0ae7`。
以下结论来自本地实际源码、完成标记、CSV schema、权重和 SHA256，
不来自历史实验的预设 epoch 预算。

## 复用证据

- `source_audit.json` 记录 90 项复用文件指纹、45 项历史产物校验和实际
  CSV 列名、行数。10 项父实验代码/协议 raw SHA 与父完成记录一致。
  历史文本仅在其原有清单明确给出 LF 标准化 SHA 时接受纯换行转换；
  新代码与权重仍使用实际文件 SHA，不推断缺失指纹。
- 父 `formal_completed.json` 及三个真实 `status.json` 均完成 30 轮、
  每种子 5,100 个配对 batch；本地六个 C2/C3 分类头 checkpoint 均存在，
  SHA、epoch、seed 和 TEST-free 来源通过核验。
- Clean Stage-I：seed 42，epoch 184，ResNet32，冻结 64-D GAP。
  原权重 SHA256：
  `7ced9d07f71335357c08623573f038acd0dfdcdeadb55d95d57b332a14f83a88`。
  不重新训练或替换。
- 原 TRAIN 为 10,847 张，holdout 为 1,980 张；原始 raw-ID 集合无交叠。
  三个种子的全部 30 轮采样计划，共 976,230 个索引，逐项匹配固定的
  官方连续采样器。具体统计见 `sampling_audit.json`。

## 实际调用链和数值策略

父 `train_ccsam.compute_arm_loss` 实现 C2 加权/非加权 clean CE 混合，
以及 C3 加权 adversarial CE 与 clean CE 混合。C3 调用原有
`analytic_classwise_feature_delta`，再调用 `mask_many_feature_delta`，
只将原 TRAIN 频次大于 100 的输入扰动置零。分类头仍正常更新。
本轮直接复用这些函数，不复制或修改原公式。

父 `ccmia_core.decompose_perturbation_gradient_by_source` 输出
`[source, classifier_row, feature_dim]` 和 `[source, classifier_row]`。
source 是被扰动输入的真实类别，target 是接收梯度的分类器行。
本轮只对预先选定的 source/Many-row direct 项施加零掩码；原 clean
autograd 梯度在各组自身当前状态下完整保留。它是 optimizer-gradient
干预，不声称等价于某个新的端到端损失。

实际训练为 FP32，无 AMP。SGD 仍为 lr=0.01、momentum=0.9、coupled
weight_decay=0.0002、Nesterov=False；batch=64，每轮 170 batch，末批31。
alpha 使用原函数，绝不按结果调节。

逐批检查未掩码解析重构与 C3 autograd，保留父 atol=2e-5、rtol=2e-4。
参考重放继续使用父 SGD 影响分解与独立 FP32 写回舍入账本，保留
abs=2e-6、rel=5e-3。FP64 仅用于父归因累积及统计，不改变模型精度。
每个实际 optimizer 的参数及 momentum 写回均独立 bitwise replay。

## 隔离与预先声明的实现选择

父 `prepare_context` / `load_prepared` 会加载完整 holdout，因此本轮不调用
它们。`HoldoutVault` 使用无缓冲的 NPY 行级读取，只允许固定 Calibration
行；直接整文件加载、mmap、绕过分区的行读取均由守卫拒绝。

拆分使用原有 cache allocation 的每类20行次序和原始 TRAIN ID，先验证
Calibration 的每类10张；Evaluation 的实际标签、特征和完整缓存 SHA
均推迟到三种子掩码全部封存之后验证。任一标签/次序不一致即停止。
`split_manifest.json` 明确保留实际行位置、原始 ID 和各自哈希。

固定竞争类别是初始分类头在 Calibration 上每个 Many 类的最大类均值
non-Many logit 对应类别。该类级映射封存后直接用于 Evaluation，
不根据 Evaluation 特征另选竞争类别。

单行干预不会移除 non-Many 竞争行梯度，所以排序使用目标行的 direct
margin 贡献，而非将未干预竞争行的变化也算入该 pair。原 full-source
margin 投影同时作为独立诊断列保留。完整30轮评分范围、频次分层、
tie-break、K=221 及 Oracle Selection 固定保留K的目标均已提前写入协议。

## 已执行的实现验证

7 个新增 Python 文件语法检查通过；本轮 CPU 测试23/23、父 CC-MIA
CPU 测试13/13通过。包含 FP64 数学、FP32 误差、非方形维度、缺失类、
行置换、K=0/K=all、精确频次匹配、泄漏拒绝、篡改拒绝与配对 Bootstrap。
CPU 原始日志保留于本轮 `logs/`，通过收据在
`validation/CPU_VALIDATION.json`。

Preflight 通过；三种子真实首批 baseline probe 通过。完整参考重放、
共同封存、生产 Smoke 与正式训练的状态分别由后续实际完成 JSON 证明，
本文件不将这些阶段的计划视为已完成证据。

## 证据边界

Calibration / Evaluation 均为 TRAIN-derived development 数据。父研究
已查看完整原 holdout，且 Clean Stage-I 当时在该完整 holdout 上选模；
新隔离只能约束本轮评分和掩码数据流，不能消除历史选择影响。
class 0 缺失；Many 指标只覆盖原35个 Many 类中的34个。
置信区间条件于三个固定训练种子及封存掩码，不表示独立 TEST 泛化或
总体统计显著性。Gate 准入留待人工综合审查。
