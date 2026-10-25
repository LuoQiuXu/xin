# H2T 历史特征重放哈希不一致：静态诊断与持久化修复

**根因状态：未证实。** 已确认历史和当前重放代码存在数值配置及执行路径差异，但本轮没有执行模型推理，不能将其中任一差异认定为哈希不一致的已证实原因。

本轮在 `experiment/fusion-semantic-audit-20261025` 工作区开展。仅运行合成 CPU 单元测试、语法/AST/文件哈希静态检查和环境元数据查询；没有重跑 preflight 或完整审计，没有真实数据模型前向、训练、官方 CIFAR-100 TEST 访问、分支切换、提交或推送。原有其他实验目录的未跟踪文件保持原状。

## 1. 已知失败与缺失观察

原失败位置为 seed124 / epoch10 / batch0，批大小128。该批 `indices_sha256`、`images_sha256` 已通过；`features_sha256` 不匹配。

历史 `lowvs_norm_direction/results/seed124/LowVS-Full/batch_signatures.csv` 中该行的预期值为：

```text
7d111aeb5a0dc30cf76af609ffb251fc4152c0c38f80baaa01ae0e3862eafbb5
```

原失败进程没有保存实际哈希、特征张量或其元数据。实际哈希、实际 shape/dtype、有限性结果与数值差异统计在 `STATIC_DIAGNOSIS.json` 中均明确记为 `null`，不得把下面的源码推导当作失败现场测量。既有审计产物中没有可恢复这些观察的特征文件；预期哈希不能反推出实际特征。本轮遵守“只运行单元测试和静态检查”，没有为补齐实际值重新推理。

## 2. 历史哈希的生成方式

调用链为 `lowvs_norm_direction/run_experiment.py::train` → `ravs_v1/run_ravs_v1.py::signatures` → 锁定 `common.py::array_hash`。

1. 对增强后的接收者、供体分别执行冻结的 `DataParallel(ResNet32)` 前向；模型为 `eval()`，参数不求梯度，前向位于 `torch.no_grad()` 中。
2. ResNet 特征提取器返回 layer3 后的完整特征图，没有经过分类头或 GAP。按模型结构与 batch=128 推导，两者均应为 `[128,64,8,8]`；输入由 transforms/stack 得到 `[128,3,32,32]`。
3. checkpoint 的全部浮点模型状态为 `torch.float32`。正常未启用混合精度的源码路径应产生 float32 特征，但原失败现场 dtype 没有被独立记录。
4. 按接收者、供体顺序分别使用 `tensor.detach().cpu().numpy()`。哈希前没有 `.float()`、`.astype()`、池化、舍入或归一化。
5. 对每个数组依次加入 `str(array.shape).encode()`、`str(array.dtype).encode()`、`array.tobytes()`；`tobytes()` 默认使用逻辑 C 顺序。两个数组共同生成一个 SHA256。此平台 float32 的 NumPy dtype 字符串为 `float32`，字节序表示为 `<f4`。

当前审计采用相同的转换和哈希顺序。单元测试提取锁定历史函数的 AST，在合成数组（包括非连续数组和不同 dtype）上核验哈希契约一致；这不是对真实失败批次的重放证明。

## 3. 环境和推理路径证据

| 项目 | 历史签名运行 | 失败记录/当前代码 | 证据结论 |
|---|---|---|---|
| PyTorch、CUDA、GPU | 2.14.1+cu132、13.2、RTX 5070 Laptop GPU | 原失败记录中三者相同 | 没有版本/GPU改变的证据 |
| cuDNN、torchvision、NumPy | 92400、0.29.1+cu132、2.5.2 | 新的只读元数据进程相同；原失败记录未保存这三项 | 不能把当前查询当作过去进程的完整环境快照 |
| Stage-I checkpoint | seed42 / epoch184；SHA256 `7ced9d07f71335357c08623573f038acd0dfdcdeadb55d95d57b332a14f83a88` | 原预检与当前静态校验一致；严格加载 `state_dict_model` | 没有 checkpoint 来源变化的证据 |
| 模型与推理模式 | 同一锁定 ResNet32，冻结/eval/no_grad | 当前源码相同 | 未发现训练模式或分类头被错误用于特征哈希 |
| 索引、增强输入 | 历史完整数组哈希 | 原失败记录确认两个字段通过 | 相应形状、dtype、逻辑数组字节的哈希一致 |
| cuDNN benchmark | False | 源码显式 False | 未发现差异 |
| cuDNN deterministic | 历史 `config.json` 为 True；`fresh→configure_compute` 显式设置 | 重放调用 `common.seed_all(seed, False)` | **已证实代码配置不一致；未证实其因果作用** |
| cuDNN/matmul TF32 | 历史二者均 False，并由源码显式设置 | 审计未显式设置；原失败现场未记录；新进程默认分别 True/False | **缺少历史配置复用已证实；失败时实际 TF32 状态未确定** |
| warmup 与前向历史 | 两次 warmup，随后 epoch1–9 也有前向 | 没有 warmup，只在 epoch10 提取特征 | **路径差异已证实；其对数值/算法选择的影响未确定** |
| 驱动及内核选择 | 没有足够的失败现场信息可逐项对应 | 本轮未做内核/参数试验 | 未确定 |

`run_experiment.py`、`norm_policy.py` 的当前内容与历史 config 记录的源码哈希一致；`run_ravs_v1.py`、`run_safeguards.py` 与历史 manifest 一致。锁定 common、模型源码与数据来源校验亦通过，因此上述源码差异有可核查的历史来源支持。

本轮没有修改 deterministic/TF32、增加 warmup、改变推理精度或扫描参数。不能据此声称已经修复真实重放不一致，也不能用 `allclose` 代替精确哈希。

## 4. 修改内容

- `audit_fusion_semantics.py`：保留原字段顺序与逐项字符串相等门禁，将门禁调用提取到 `verify_replay_signatures`；首次不匹配时在新的 `diagnostics/replay_mismatch_*/replay_mismatch.json` 保存实际/预期哈希、截至首个失败项的检查、两张特征图的 shape/dtype/stride/有限性计数、模型模式、数值设置和 checkpoint 来源，执行 flush/fsync 后抛错。没有历史特征数值时不伪造数值差异。
- 每次诊断创建独立目录，不覆盖旧记录；使用单次目录创建，权限错误立即失败，避免 Windows 标准临时目录函数的权限重试。诊断写入失败时仍严格停止。
- `test_fusion_semantic_audit.py`：新增8项合成 CPU 测试，验证哈希契约、持久化顺序、近似相等仍拒绝、非有限值记录、独立输出、成功无诊断副作用、首失败顺序与写失败停止。测试临时目录限定在本实验 `diagnostics/` 内，完成后清理。
- `CODEX_FUSION_SEMANTIC_AUDIT.txt`：分支名修正为本实验分支，并说明独立诊断输出及禁止复用旧源码哈希。
- `PROTOCOL.md`：补充失败证据持久化规则；原科学问题、严格门禁、数据限制均不变。

## 5. 验证结果

指定解释器 `D:\app\minicondaaa\envs\xiaoju\python.exe` 下最终 **13/13 单元测试通过，退出码0**；测试自身耗时0.043秒，完整命令耗时2.821秒。最终 Python 编译与静态检查通过，`git diff --check` 通过。

AST 对照确认33个原有函数未变，重放函数除接入诊断门禁外，原推理、随机数、加载与数组转换语句均未变；哈希算法未变。93项受保护历史源码、输入及原审计结果的前后哈希一致。

最初测试遇到沙箱系统临时目录权限问题并中断，定位栈指向 Python tempfile 的 Windows PermissionError 重试路径；生产诊断改为单次目录创建后，系统临时目录仍导致5项测试错误，测试临时目录改到项目可写目录后全部通过。完整尝试和最终日志分别保存在 `TEST_ATTEMPTS.json`、`unit_tests.log`，这些尝试均为合成测试，与真实失败批次无关。

`STATIC_CHECKS.json` 是首次实现的静态快照；当前最终代码应以 `STATIC_CHECKS_FINAL.json` 的源码哈希与检查为准。

## 6. 尚未解决

实际失败哈希、特征内容及有限性仍不可得；没有历史对应特征值可相减，数值差异统计未生成。配置差异尚未经过受控因果验证。真实失败路径的持久化仅经合成测试验证，没有真实重放验证。

修改审计源码后，旧 `results/preflight.json` 中的审计源码哈希必然不再匹配。该历史文件和其他原产物保持原样；不得修改其中的哈希或删除结果来通过门禁。本轮没有执行任何新的 preflight/run，后续真实重放必须另有明确授权及独立的输出安排。

独立证据文件：`STATIC_DIAGNOSIS.json`、`STATIC_CHECKS_FINAL.json`、`TEST_ATTEMPTS.json`、`unit_tests.log`。原失败记录仍位于原 `results/`，未被改写。
