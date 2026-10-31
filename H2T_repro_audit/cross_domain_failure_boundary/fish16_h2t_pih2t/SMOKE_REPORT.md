# Fish16 adapter / smoke report

日期：2026-10-31（Asia/Shanghai）。分支：`experiment/fish16-h2t-pih2t-cross-domain-20261031`。

**结论：权重适配和 CPU 测试通过；完整 GPU smoke 未通过，已停止。正式训练未启动。**

## 用户确认的范围

本轮采用方案1：原始 ResNet_FE 架构及源码指定 19c8e357 ImageNet 权重，只在新实验目录增加适配器。最新指令仅授权 CPU 测试和真实数据 smoke，暂不执行正式90/10 epoch实验。原 H2T/PI-H2T 源码、协议、历史结果和原始数据不修改，不提交或推送。

## 权重与模型核验

- 官方URL：`https://download.pytorch.org/models/resnet50-19c8e357.pth`。
- 文件102,502,400字节；SHA256：`19c8e3572231adff6824a2da93fd67b5986919a2e65f8b6007eab4edee220097`。
- 独立缓存：`D:\h2t_cross_domain_results\fish16_weight_cache_20261031_01`；未覆盖、未使用已缓存11ad3fa6。
- 原生 factory 的 KeyError 已保全于此前 preflight.json。新适配器仅用原 factory 分配原始 ResNet_FE，再在推理前严格加载全部 backbone 张量。
- 原文件267键；仅移除1000类FC的fc.weight/fc.bias。265个backbone参数和浮点BN统计逐项同形且加载前后数值完全相等。
- 旧tar格式不支持weights_only=True；仅在固定官方完整SHA匹配后使用兼容读取。53个缺失项全部是现代BN整数计数器，按PyTorch严格兼容规则置零，没有补填学习参数或浮点统计。
- 新tensor-only副本318键，strict=True通过；之后均weights_only=True读取；副本SHA：`c1bc33a75ef4d4dd607aa1cb0635ed42d547265a41ad0467081276e5009a7b3e`。
- CPU逐张量确认Plain/PIF backbone及分类头初始化相同；PIF额外只有2个共享参数，原公式和默认初始化保留。

## 测试与数据门禁

| 项目 | 真实结果 |
|---|---|
| 工作区起始状态 | 目标分支、干净；HEAD 61d2e42200d3d46b77f74c0553d63fe5721f74ed |
| 固定数据 | 27,133张、16类；train16269/5208轨迹，val5396/1736，test5468/1736 |
| 图像路径与SHA | 全部27,133张再次核验，缺图和哈希不匹配为0 |
| 轨迹/组/精确哈希跨split | 0；manifest未重新生成或修改 |
| CPU契约测试 | 10/10 PASS，完整复跑两次，无跳过 |
| 语法检查 | 5个新Python文件全部PASS |
| 原始来源文件 | 8个文件SHA前后一致，未改动 |
| 真实Stage-II单batch | A1、A2、B1、B2全部PASS，均[16,2048,7,7]、FP16 |
| 冻结与分类头 | 4个单batch均extractor+BN完全不变，分类头梯度有限且更新 |
| H2T/PI单batch | 512个共享替换通道；PI比例detach=True |

CPU测试覆盖官方权重与键名、共同初始化、真实ResNet feature map、PIF公式/两参数/置换性质、原H2T替换语义、PI recipient/donor mass及detach/FC梯度、Stage-II参数与BN冻结、采样仅train及复现、真实recipient增强匹配、路径越界/mask/排除类别拒绝。H2T的CPU测试仅模拟.cuda索引传送，真实GPU单batch使用未修改原函数。

## 一epoch smoke 实际执行

共同配置：seed42、micro/effective batch16、AMP、4 workers。Stage-I首warmup LR0.002、wd1e-4；Stage-II首LR0.01、wd5e-4；momentum0.9。只用validation，无TEST模型前向或准确率。

| 条件 | 单batch | 完整smoke epoch |
|---|---|
| A0 Plain Stage-I | 原始骨干CPU与后续真实训练路径通过 | 完成1epoch |
| B0 PIF Stage-I | PIF公式/梯度CPU通过，冻结PIF GPU单batch通过 | 梯度门禁失败，epoch未完成 |
| A1 Plain-cRT | PASS | 未运行，遵守全局停止 |
| A2 H2T | PASS | 未运行，遵守全局停止 |
| B1 PIF-cRT | PASS | 未运行，遵守全局停止 |
| B2 PI-H2TF | PASS | 未运行，遵守全局停止 |

A0：处理16269张训练图像、1017次SGD更新，全部梯度检查通过；train loss=1.016162；训练耗时75.253s、吞吐216.191图像/s、峰值PyTorch allocated显存995.446MiB（不是整机显存占用）。

仅作smoke功能记录的A0 validation：Overall=95.570793%，Macro=69.888085%，Tail image=22.000000%，Tail class-macro=16.923077%。这不是正式实验准确率或迁移结论。

## 失败事实与未确定项

PIF Stage-I在首个100-batch进度打印之前，前向logits/loss有限性门禁已通过，随后在backward与unscale之后触发：

```text
AssertionError: Nonfinite/missing gradients
```

现有断言同时覆盖“没有梯度”和“存在非有限梯度”；未保存失败批次编号、逐参数梯度或当时的AMP scale。因此具体失败batch和实际梯度分布均未确定，不能直接认定为AMP溢出、PIF数学缺陷或H2T/PI-H2T跨域失效。未观察到OOM异常，没有启用batch8回退。

遵守停止条件：保留FAILURE.json、原日志、A0 checkpoint及所有已产生文件；不继续其他条件，不改AMP/融合公式/PIF初值/学习率，不通过跳过条件或降低梯度门禁放行。下一步需先补足失败批次的数值诊断，才能决定是否重跑smoke；本轮未执行额外诊断训练。

初始run_manifest.json中stage1_epochs_executed/stage2_epochs_executed是预先写入的目标smoke预算，不能用作完成证据。原记录不覆盖；真实完成状态以新增ACTUAL_EXECUTION_STATUS.json为准。没有SMOKE_SUMMARY.json成功标记，不能报告六条件epoch全部通过。

## 产物

- 代码与本报告：`D:\xiangmuxunlian\xin\H2T_repro_audit\cross_domain_failure_boundary\fish16_h2t_pih2t`。
- 原生加载问题及全量数据核验：`D:\h2t_cross_domain_results\fish16_preflight_seed42_20261031_01\preflight.json`。
- 官方权重、转换和265张量相等证据：`D:\h2t_cross_domain_results\fish16_weight_cache_20261031_01\ADAPTER_PROVENANCE.json`。
- CPU日志、原始源码指纹和smoke总日志：`D:\h2t_cross_domain_results\fish16_validation_seed42_20261031_01`。
- 本轮GPU单batch、A0 checkpoint、采样与配对记录、失败现场：`D:\h2t_cross_domain_results\fish16_smoke_seed42_20261031_01`。
- A0 checkpoint SHA256：`a4989aef49cfeb51e8bcb53dc43781fbf4733face96c3898e78834929fd57f43`。

本轮仅证明加载适配、CPU数学与冻结契约、四个真实Stage-II单batch可运行，以及Plain的一epoch路径可运行。完整smoke门禁未通过，不能启动正式seed42，也不能形成方法退化候选。
