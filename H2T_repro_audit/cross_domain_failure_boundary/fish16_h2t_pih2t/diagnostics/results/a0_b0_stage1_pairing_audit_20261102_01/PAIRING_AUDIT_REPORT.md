# Fish16 A0/B0 Stage-I 配对审计与 Stage-II 静态准入检查

**总体结论：AUDIT_PASS_WITH_DOCUMENTED_OPTIMIZER_STEP_ASYMMETRY。** 数据、增强输入、MixUp过程证据、公共初始化与checkpoint完整性通过；实际更新次数差异必须保留。本轮不授予任何GPU实验权限。

日期：2026-11-02（Asia/Shanghai）。本轮仅CPU读取既有文件、Git源码审查及checkpoint strict-load；没有模型前向/反向、optimizer更新、图像读取、CUDA初始化、Stage-II或TEST评价。

分支：`experiment/fish16-six-condition-smoke-20261101`；审计起始HEAD与同步origin：`de82e100e200de575380feba540a62ff9e94eca6`，工作区干净。
A0实际执行：`8cbfa69bd194083053917bb0cef615572abdc182`；B0实际执行：`d47cbd7293402fd02634c176d7e16db288b936b9`。后续报告提交A0=`52c8bf00d55966d0fa7d02fe52f788736083a02c`、B0=`de82e100e200de575380feba540a62ff9e94eca6`不当作执行源码SHA。

## 独立判定

| 项目 | 判定 |
|---|---|
| STAGE1_SAMPLING_PAIRING | PASS |
| STAGE1_RECIPIENT_VIEW_PAIRING | PASS |
| STAGE1_MIXUP_RNG_PAIRING | PASS |
| STAGE1_INITIALIZATION_COMPATIBILITY | PASS |
| SOURCE_EQUIVALENCE | PASS |
| CHECKPOINT_INTEGRITY | PASS |
| STAGE2_SAMPLING_READINESS | PASS |
| OPTIMIZER_UPDATE_MATCHING | DOCUMENTED_DIFFERENCE |

## 既有产物与训练数据

A0目录：`D:\h2t_cross_domain_results\fish16_six_condition_smoke_seed42_20261101_01`。
B0目录：`D:\h2t_cross_domain_results\fish16_b0_amp_recovery_validation_20261102_01\b0_stage1_one_epoch`。
全部指定文件存在；A0根状态仍为FAILED_STOPPED且只列A0完成，B0为COMPLETE_B0_STAGE1_ONE_EPOCH_ONLY且只列B0完成。审计不将过去的六组smoke失败状态改成完成。
锁定manifest元数据27133图、16类；train16269图/5208轨迹，validation5396图/1736轨迹，索引及轨迹交集均0。三个数组均落在相应train支持上，Stage-I恰为train全排列；recipients/donors仅从train抽样，recipient覆盖16类。类别频次重算的Head/Medium/Tail分组与两份manifest完全一致。
Fish16 manifest SHA256：`62d9a2e9fffaaca22ef0c99c0ce4134b4d0e1ecc308333a7cb57b10cf95f0f30`。
官方19c8e357权重SHA256：`19c8e3572231adff6824a2da93fd67b5986919a2e65f8b6007eab4edee220097`。
seed42、历史运行环境、micro/effective=16、AMP、4 workers、90/10日程分母、数值flags、train/validation增强配置逐项相同。ColorJitter(.4)保留原亮度增强含义。只读取manifest元数据，没有打开任何图像。

## 逐元素采样与逐Batch输入配对

| 数组 | A0/B0 shape | dtype | 元素比较 | payload SHA256（两边相同） |
|---|---|---|---|---|
| stage1 | [16269] | int64 | 16269/16269一致 | `bb49e02a1d4d626d7e661b1c0d876c236aac39f37eaba36ee59f3dea02a76c40` |
| recipients | [16269] | int64 | 16269/16269一致 | `49b04634d5799f096e0105749b480d7dca456d97f506d218f66f913a97a0e29f` |
| donors | [16269] | int64 | 16269/16269一致 | `f18013f638137b5d50ed1d05242bef60df99be8cc50ab41ced4df71756f4ec26` |

数组SHA是C连续payload字节的SHA256，shape/dtype分别核对；NPZ容器SHA也相同。donor_pair_audit两份16269行逐字段一致，且与对应采样数组、manifest类标签及轨迹关系全部一致；未仅凭文件大小判断。

| batch字段 | 精确一致记录数 |
|---|---:|
| batch | 1017/1017 |
| samples | 1017/1017 |
| recipient_sha256 | 1017/1017 |
| mixup_rng_sha256 | 1017/1017 |
| mixup_lambda | 1017/1017 |
| donor_sha256 | 1017/1017 |

两边均为batch0..1016、前1016批各16样本、末批13样本，总数16269；Stage-I donor_sha256全部null。mixup_lambda用float64字节精确核对，没有allclose。input_controls完整文件SHA也相同；没有首个不一致batch或字段。
recipient_sha256对应进入MixUp前的增强张量；mixup_rng_sha256是torch.get_rng_state()的CPU Torch状态摘要，不能说成全部NumPy/CUDA RNG状态已比较。结合相同样本/标签顺序、相同batch大小、相同CPU randperm源码与运行环境，以及每批相同lambda，可以建立MixUp操作的确定性一致证据。**未保存的permutation和mixed tensor没有被直接比较。**

## 初始化与执行源码

CPU构造两个初始模型，公共backbone的318个state tensor逐值一致；classifier张量一致。共同初始classifier SHA256：`2f78989d043cc7d5b63abf8a18e0d778c63ff64ab7850b0bd71d1e002766c434`，与A0/B0 result记录相同。
B0仅额外增加已有PIF的两个共享可学习标量，其初值与B0记录一致；公共backbone/head的初始化RNG通过原fork_rng隔离。不能要求整个extractor_initial_sha256相同，因为B0的extractor额外包含PIF。此处只核对本地已锁定初始化，不新增作者初始化已核实的主张。
models.py、data.py、preflight.py在两个执行提交完全一致；8项上游原始源码一致。smoke.py正常Stage-I核心25/25项AST检查通过，包括seed、优化器/参数组/日程、数据视图、MixUp/forward/loss与缩放backward。具体源码行号、CRLF记录哈希解释及差异分类见SOURCE_DIFF_AUDIT.md。
恢复门禁的符号范围、异常快照、B0-only/forensic入口及状态记录确有差异；新增只读target context复制不消耗随机数、不改变训练张量。SOURCE_EQUIVALENCE=PASS限定于上述正常科学计算核心及明确记录的差异，不声称整个程序或优化器轨迹逐字/逐值等价。

## Checkpoint完整性

| 条件 | SHA256 | CPU strict-load | model state项 |
|---|---|---|---:|
| A0 | `a4989aef49cfeb51e8bcb53dc43781fbf4733face96c3898e78834929fd57f43` | PASS，无missing/unexpected keys | 320 |
| B0 | `231295f0c7e9a7c7621f80ab698637bf67cef1c489eb4fbebf275f866745c817` | PASS，无missing/unexpected keys | 321 |

两份均用torch.load(map_location="cpu", weights_only=True)，smoke_only=True、condition正确、seed42、epoch1；全部浮点state有限，模型状态完整且extractor终态hash与result一致；optimizer参数组、最终scaler及validation记录也一致。没有CUDA forward或额外训练。完整细项见CHECKPOINT_AUDIT.json。

## 更新次数：配对中保留的非对称

| 指标 | A0 | B0 |
|---|---:|---:|
| 训练样本 | 16269 | 16269 |
| 尝试窗口 | 1017 | 1017 |
| 真实optimizer.step | 1017 | 1015 |
| AMP跳步 | 0 | 2 |

B0 batch6 scale65536->32768、batch13 scale32768->16384；两次窗口都没有真实参数更新、参数SHA不变，下一batch7/14均在降低后的scale做了真实更新。计数逐事件核对，未补做两次优化步骤，未把1015写成1017。
本Epoch事件日志没有直接保存Inf正负号，既有正Inf取证和Batch13负Inf精确重放只能作为参考证据。本轮没有重放或补记符号。**数据/随机性配对通过，不等于实际更新轨迹匹配，也不允许把步数差异忽略为无影响。**

## Stage-II准入范围：静态准备，不是运行授权

| 条件 | 共同初始化来源 | 方法 |
|---|---|---|
| A1 / A2 | 同一A0 checkpoint、同一训练后classifier | cRT / H2T |
| B1 / B2 | 同一B0 checkpoint、同一训练后classifier | cRT / PI-H2TF |

四组可共享已保存的recipient数组，A2/B2共享donor数组。已有路由整段AST一致；strict-load包含classifier。源码freeze()使extractor参数requires_grad=False并eval以冻结BN，classifier requires_grad=True/train，Stage-II optimizer仅含classifier，extractor前向在no_grad内而classifier损失反向在外。这里只证明源码允许该配对/冻结/更新路径，没有执行Stage-II。
当前CLI没有独立复用这两个已完成checkpoint的Stage-II入口；默认六组路径会先重训Stage-I。因此STAGE2_SAMPLING_READINESS=PASS只表示采样、checkpoint及静态配对协议基础成立，不是可直接启动默认runner或其他GPU任务的结论。未来执行入口和授权仍未在本轮建立。

## 既有Validation诊断指标

| 指标（%） | A0 | B0 |
|---|---:|---:|
| Overall | 95.570793 | 97.498146 |
| Macro | 69.888085 | 73.356071 |
| Tail image-weighted | 22.000000 | 37.000000 |
| Tail class-macro | 16.923077 | 28.461539 |
| Worst-class | 0.000000 | 0.000000 |

这些是一个Epoch的validation诊断值，不是TEST性能。本轮没有重算预测、显著性检验或因果归因；不能宣称PIF有效提升、PI-H2T优于H2T、跨域泛化成立或尾类问题解决。PIF架构因素与两次AMP跳步都必须保留在解释边界中。

## 审计产物与停止状态

外部目录：`D:\h2t_cross_domain_results\fish16_a0_b0_pairing_audit_seed42_20261102_01`。
PAIRING_AUDIT_RESULT.json为聚合证据，SOURCE_DIFF_AUDIT.md给出执行源码/静态路径审查，CHECKPOINT_AUDIT.json给出CPU完整性证据，FILE_SHA256.csv索引输入和审计文件的大小/SHA。完整只读工具、原始diff和工作记录仅在外部目录。
Git只新增五份精简审计文件，并只更新README顶部CURRENT EXECUTION STATUS；历史协议正文按字节保留，旧报告和原始代码不变。不上传采样数组、逐样本记录、图像、manifest内容、checkpoint或梯度快照。
**当前没有新的GPU实验授权。审计通过仅提供继续科学分析的数据/初始化基础，保留优化器步数非对称与Stage-II独立入口缺口。完成提交后停止等待科学分析。**
