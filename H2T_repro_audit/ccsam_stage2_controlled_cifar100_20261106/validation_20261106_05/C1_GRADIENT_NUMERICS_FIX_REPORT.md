# C1 解析梯度数值路径修复与一次性 Smoke 报告

**结论：只读候选诊断完成；F 通过全部 510 个真实批次和 10 类压力测试后，才修改生产函数；生产回归、py_compile 和唯一一次 Smoke 均通过。Formal training still not started。**

工作分支 `experiment/ccsam-gradient-numerics-fix-20261106`，HEAD `6dd45e70040b5fd14a6d563a9fa19d0e52ac4753`；基线提交 `e63e190edd72b9ca62b763639f190bbc07b81ede`。修复分支原已存在于远端，本轮仅建立其对应本地跟踪分支，没有创建其他研究分支。基线分支引用、历史文件、checkpoint、prepared 和协议均保持不变。本轮未 commit、push 或 merge。

## 1. 根因及证据边界

已证实独立 Softmax 与 CE 内部 LogSoftmax 的 FP32 数值差异会进入线性分类器的特征梯度，并被归一化放大。原 seed123/epoch0/batch0 精确重现 `max_abs=5.364418029785156e-07`；原生产 A 在 510 个真实批次中有 216 个批次、683 个最终 delta 元素超限。全部输入和阶段张量有限。

完整批次零填充 norm 与 compact norm 另有可分离的舍入差异。实际 Autograd 对 `losses[mask].mean()` 的 logits 梯度已逐类捕获，随后通过 Linear backward 得到特征梯度；首个重复类批次为 seed123/batch1，捕获形状 `[51,64,100]`。F 的缩放 logits 梯度与该直接捕获在全部 510 批次数值和字节均一致。

必须如实区分：本协议的真实采样批次最大 n_c=2。单独移动计数缩放的 E 与 A 的真实最终结果相同，**没有证据把计数缩放顺序另认定为这 510 批次失败的独立根因**。C 也在这 510 批次通过原 allclose。选择 F 是按任务优先候选对齐实际 mean-backward/Linear 和完整 norm 数值路径，得到全部真实最终 delta 字节一致；不是宣称 C 在已测批次失败。

非二次幂计数压力测试明确显示 F 仍有小的运算顺序差：n_c=3/5/7 时 logits 梯度最大差 `2.9802322387695312e-08`，最终 delta 最大差 `4.470348358154297e-08`，最大 normalized_error `0.03865184040132525`，全部通过原门禁；未为追求这些场景的 byte_equal 进一步改数学操作。

## 2. 固定输入、RNG 与运行环境

解释器 `D:\app\minicondaaa\envs\xiaoju\python.exe`；PyTorch `2.14.1+cu132`、CUDA `13.2`、cuDNN `92400`、GPU `NVIDIA GeForce RTX 5070 Laptop GPU`。deterministic=True，benchmark=False，cudnn.allow_tf32=True，cuda.matmul.allow_tf32=False，FP32，AMP=False。

Clean checkpoint seed42/epoch184 SHA256：`7ced9d07f71335357c08623573f038acd0dfdcdeadb55d95d57b332a14f83a88`。原 TRAIN 与 TRAIN-holdout split、10,847/1,980 样本及三个 prepared 采样计划不变。直接读取官方 TRAIN pickle 与 split，prepared 的标签和像素逐项一致；从未读取官方 TEST。

seed123/batch0 的以下哈希与第四轮完全一致（含图像及分类头的额外核验）：

- indices：`5c07ee3aab6c32bf948c53bc2a552e137a14ef7a5e7170e19b776eaebbf7056f`
- labels：`7d50452453966659686dd84d989c81faa2bd76488828fd0c3ca8f4499f20aca8`
- augmented images：`856f4274c898407ec6c07c60de61902b9b86bd81f403ec077cb099596d9e20c1`
- GAP：`35340f51e7494d3da8cbe1787250a352815ae32f887485649b2eb236a0e2bfee`
- classifier state：`69c0bb9e681774576b8b69843ab571ca020a31907225a572dd7dd9bc71910cc2`

实际只读 sweep 每个 seed 调用 seed_everything 一次并只初始化一次 augmentation generator，seed 为 seed+300009。逐批记录 generator 前后状态哈希，连续性独立复核通过。每个 seed 的 170 批次包含 10,847 个**采样出现次数**；因官方重复采样，不等于遍历全部不重复图像（seed123/124/125 分别 5,811/5,812/5,815 个不同 TRAIN 索引）。模型均为同一封存初始化，无 optimizer，无参数/BN 更新，classifier 参数 .grad 始终 None。

## 3. 完整 30 Epoch 结构审计

3×30×170 = 15,300 批次；976,230 个位置与锁定官方连续 ClassAwareSampler 逐项一致，0 差异；每轮最后一批为 31。全局 100 类均非空，0 越界，0 标签/索引不一致。未用单轮类计数差≤1 代替官方序列核验。

第一 epoch 重复类组数：seed123=774、seed124=750、seed125=748，合计 2,272 组。含 n_c=2 的实际批次共 305 个；n_c=1 分组覆盖 510 批次；真实 n_c=3、4+ 没有观测，不能写成真实测试通过。它们由下述独立 synthetic 压力测试补充。

## 4. 只读候选矩阵

A=原生产；B=直接调用未改动 autograd_reference_delta 的唯一 oracle；C=仅 LogSoftmax.exp；D=仅完整 norm；E=仅 count 预缩放；F=LogSoftmax.exp+count 预缩放+完整 norm。第五轮 E 的命名与第四轮组合候选 E 不同，见 CANDIDATE_MATRIX.json。

| 候选 | 通过批次 | 失败批次 | delta 超限元素 | 最大绝对误差 | 最大 normalized_error | 最终 delta 全部 byte_equal |
| --- | --- | --- | --- | --- | --- | --- |
| A | 294 | 216 | 683 | 3.6805868148803711e-06 | 9.867700337416947 | False |
| B | 510 | 0 | 0 | 0 | 0 | True |
| C | 510 | 0 | 0 | 1.7881393432617188e-07 | 0.023260568413672123 | False |
| D | 293 | 217 | 684 | 3.6805868148803711e-06 | 9.8687388240868508 | False |
| E | 294 | 216 | 683 | 3.6805868148803711e-06 | 9.867700337416947 | False |
| F | 510 | 0 | 0 | 0 | 0 | True |

原门禁始终为 rtol=1e-5、atol=1e-7；normalized_error=abs(error)/(1e-7+1e-5*abs(reference))；max_relative 的分母为 max(abs(reference),1e-12)。B 参考的逐层仪器化最终 delta 对直接调用 B 精确核对；A 对原生产函数精确核对。记录 probability、logits gradient、raw feature gradient、class-averaged full feature gradient、class norm、direction 和 final delta 七阶段。

真实 F 各阶段最大数值误差均为 0；完整类梯度 byte_equal=False 的差异位于零元素的正负符号，三个持久化真实快照已逐比特确认，其范数、方向和最终 delta 均字节一致。没有针对符号零增加数学改动。每类 n_c、weight、candidate norm 和 oracle norm 见 CLASS_NORMS_REAL.csv；七阶段所有逐批最坏位置、值、容差、NaN/Inf、违规计数见 WORST_CASES.csv。

G/H 为“F 失败后”的条件候选。F 已通过全部规定对照，因此 G/H 明确记为 NOT_RUN_F_PASSED_ALL_REQUIRED_CASES，没有伪造统计，也没有额外参数搜索。

## 5. 重复类别压力测试

这些是独立、确定性 synthetic 对照，未替代任何真实数据批次。每个案例均对照同一直接 Autograd oracle，使用独立 seed=2026110600+case_index，原始输入和全部中间张量本地保存。

| 案例 | B | 最大 n_c | F PASS | 超限元素 | 最大绝对误差 | 最大 normalized_error | delta byte_equal |
| --- | --- | --- | --- | --- | --- | --- | --- |
| unique_64 | 64 | 1 | True | 0 | 0 | 0 | True |
| one_duplicate_2 | 64 | 2 | True | 0 | 0 | 0 | True |
| several_duplicate_2 | 64 | 2 | True | 0 | 0 | 0 | True |
| 32_classes_x2 | 64 | 2 | True | 0 | 0 | 0 | True |
| 16_classes_x4 | 64 | 4 | True | 0 | 0 | 0 | True |
| single_class_x64 | 64 | 64 | True | 0 | 0 | 0 | True |
| observed_modal_duplicate | 64 | 2 | True | 0 | 0 | 0 | True |
| tail_31 | 31 | 1 | True | 0 | 0 | 0 | True |
| one_triplicate_3 | 64 | 3 | True | 0 | 4.4703483581542969e-08 | 0.038651840401325251 | False |
| nonpower_counts_5_7 | 64 | 7 | True | 0 | 4.4703483581542969e-08 | 0.035761483592611658 | False |

## 6. 实际生产修复及回归

只修改 train_ccsam.py 的 analytic_classwise_feature_delta()：

1. 独立 softmax 对齐为 F.log_softmax(...).exp()。
2. 先按 counts[labels] 缩放 logits 梯度，再与 W 做乘积。
3. 类梯度放回完整 [B,64] 形状后求 norm，再取本类位置生成 delta。
4. 加入非空 shape、dtype/device、标签范围和 count/broadcast 断言。

函数签名、六个 stats 字段、ADVER_REG、类别权重、乘除顺序、finite 检查及 norm<=1e-12 的零梯度语义保留。原 Oracle、verify_analytic_gradient 容差、compute_arm_loss、loss/alpha/SGD/超参数、C0/C1/C2 均未改。AST 独立核验确认生产模块其余内容相同。PRODUCTION_DIFF.patch 保存实际 diff。

修改前 SHA256 `decf55ddb84d13bb08fee1ef3bd6e4b523e1feffae0998ef555d8a3b53007689`；修改后 SHA256 `7e3e26752e2b350618081e164bf0e5249294c72a651311c0b4db5f6feed1f3e6`。

修复后从真实捕获重放 seed123/batch0、含重复类的 batch1、31 样本 batch169 三个生产函数回归，全通过原门禁并等于已准入 F 的输出。对当前相关 9 个 Python 文件执行 py_compile 通过；本 CPU 报告脚本另通过 py_compile。

## 7. 唯一一次 Smoke

原 main 以 `train_ccsam.py --smoke-only` 参数执行一次，2026-11-06 22:44:28+08:00 开始，22:44:36+08:00 结束，exit=0。未改 Smoke 实现或检查项。为遵守“全部新报告写本目录”，只在调用进程中把 write_json 的原 smoke_report.json 输出位置映射到本目录并排他创建，报告 payload 原样保存；输入 ROOT 和所有计算路径不变，原 TEST guard 由原 main 在数据加载前安装。原实验根目录没有新增 smoke_report.json 或 results/。

| arm | loss | mean delta L2 | classifier updated | backbone/BN bitwise unchanged |
| --- | --- | --- | --- | --- |
| C0_BALANCED_CE | 0.7871572375297546 | 0.0 | True | True |
| C1_CCSAM_FEATURE_GRAD | 0.8877630829811096 | 1.1679407358169556 | True | True |
| C2_WEIGHTED_CE_NO_PERTURB | 0.8018354177474976 | 0.0 | True | True |

原 synthetic 梯度检查和真实批次梯度检查最大误差均 0；C1 非零有限扰动，C0/C2 零扰动，三组配对输入/特征/初始化哈希一致。仅执行既有 Smoke 三个 disposable classifier SGD 更新，全部丢弃；没有写 checkpoint。原 smoke_report.json 与 POST_SMOKE_VERIFICATION.json 均为真实 PASS。

## 8. 文件保护、执行记录与限制

1147 个受保护文件逐文件 SHA256 前后一致，变更清单为空；包括原 140 个历史未跟踪资产、历史 validation_01..04、全部 prepared、原 sampler、数据 split、TRAIN pickle 和 Clean checkpoint。基线分支引用仍为 e63e190；当前 HEAD 未变；staged diff 为空，tracked diff 仅 train_ccsam.py。git diff --check 通过。TEST 打开尝试和 train=False 尝试在所有独立数据进程均为 0。

执行前快照第一次因新工具未把历史相对路径 resolve 成绝对路径而失败；完整栈保留于 snapshot.log/execution JSON。只修复 validation_05/preflight.py，使用独立 snapshot_pathfix 日志成功完成保护快照后才进入数据/模型审计。该失败不是数值候选失败，未发生 Smoke 重试。EXECUTION_LOG.txt 保留全部八个 phase 的命令、时间、返回码和完整 stdout/stderr。

本轮数值范围为固定 Clean Stage I 初始分类头、三个 seed 的 epoch0 全部真实批次与十类 synthetic 输入，不能声称验证了分类头训练更新后的 30 epoch，也不能保证所有未来 head/logits、GPU 或 PyTorch 版本逐位相同。现有 epoch0 real 最大 n_c=2；非二次幂计数的 F 残差通过容差但没有字节一致。没有用全局 max_abs 或“误差很小”代替逐元素门禁。

所有新结果、日志、CSV、JSON 均排他创建；NPZ 本地保留，.gitignore 忽略，每个快照旁记录 SHA256、shape、dtype 和逐张量哈希。ARTIFACT_SHA256.csv 索引全部本轮产物及修改后的生产文件。生产更改待用户审阅；本轮未提交/推送，未开始任何正式训练。

## 9. 交付状态

- 只读诊断：完成。
- 生产修复：已实施，仅指定函数。
- 原始门禁：通过，容差未改。
- 510 个真实 Batch：全部通过，最终 delta 字节一致。
- 重复类别压力测试：10/10 通过。
- Smoke：唯一一次 PASS。
- formal_training_started：false。

要求的报告、CANDIDATE_MATRIX.json、REAL_BATCH_SWEEP.csv、BATCH_STRUCTURE_30EPOCH.csv、SYNTHETIC_DUPLICATE_STRESS.csv、WORST_CASES.csv、EXECUTION_LOG.txt、PROTECTION_REPORT.json、GIT_STATUS_FINAL.txt、ARTIFACT_SHA256.csv、PRODUCTION_DIFF.patch、smoke_report.json 均在本目录。完成后停止，等待用户科学审查。
