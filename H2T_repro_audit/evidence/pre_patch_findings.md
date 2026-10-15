# 修补前发现与处理边界

记录时间：2026-10-09，Asia/Shanghai。正式训练未启动。

## 已复现的启动错误

1. `official/models/resnet_cifar.py:132` 把 `feature_tuning=False` 传给 `nn.Conv2d`，实际运行 `resnet32_fe()` 得到 `TypeError: Conv2d.__init__() got an unexpected keyword argument 'feature_tuning'`。
2. 同一个构造函数第 137 行引用 `feature_tuning`，但函数签名未定义它。即使只修复第 1 项，仍会得到 NameError。
3. 当前 pytorch 环境缺少 `pathlib2`，`utils/util.py` 导入时会失败。Python 3 的标准库 `pathlib.Path` 可替代此旧版回移库。

影响：两个训练阶段和评估程序都不能直接使用此官方 ResNet-32 构造函数；H2T 通道替换函数本身未涉及这些错误。

建议与本次处理：只在独立 `runtime` 副本中为 `ResNet_fe.__init__` 补充默认关闭的 `feature_tuning=False` 参数，并从 Conv2d 调用中移除该参数；替换旧版 pathlib2 导入。保留官方构造顺序、初始化、正常前向输出、未启用的 PIF 分支与原始 H2T 实现。所有差异输出至 `compatibility.diff`，`official`、`H2T`、`H2T_official_clean` 不编辑。

## 已发现但不改变的行为

- Stage II YAML 使用 `H2T: 0.25`，论文最佳比例为 0.3；本次采用仓库配置 0.25，记录差异。
- Stage I 的默认调度来自 `utils/logger.py`，为 multistep，并有 5 epoch warmup；Stage II 是余弦调度，实际分类器初始 LR 为 `0.2 * 0.1 = 0.02`。
- Stage II `torch.no_grad()` 与只包含 classifier 的 SGD 冻结 backbone 参数；`shift_bn=True` 保持 backbone.train()，BN running_mean、running_var、num_batches_tracked 会更新。这是原始实现行为，保留并实测。
- Stage II 缺失 checkpoint 时会记录日志后继续训练随机 backbone。运行审计与提供命令时必须检查 checkpoint 存在；不改原始训练核心。
- H2T 使用原地通道赋值以及默认设备 `.cuda()`。本机单卡 CUDA:0 的官方 no_grad 路径可用；保留原始随机数流与赋值方式。
- 官方 eval 文件实际拼写为 `eval-modifed.py`，README 的 `eval-modified.py`、`eval.py` 与省略 `config/cifar100` 的路径不可直接运行。独立记录并给出真实命令。
- PyTorch 2.6+ 的 torch.load 默认 weights_only=True 可能不兼容带 NumPy scalar ECE 的官方 checkpoint；先用真实短测 checkpoint 验证，确认阻断后才添加独立加载兼容补丁。

后续验证：用本次 Stage I 的真实 1 epoch checkpoint 调用默认 `torch.load`，已确认 `_pickle.UnpicklingError`，拒绝 `numpy._core.multiarray.scalar`；详见 `checkpoint_load_probe.json`。处理建议与采用的独立兼容补丁为：训练和评估入口显式传入 `weights_only=False`，只加载本次自己生成且已记录 SHA256 的 checkpoint，保留官方 checkpoint 字段和数值。此修改不改变网络、采样、损失、梯度或 SGD。

## 来源限制

git clone 的网络连接失败，GitHub connector 也不可用。使用工作区已有 GitHub ZIP，ZIP comment 内含 `e0c849235d9655fca31cd9fa8a5f38f92a77c7e8`，原样提取并生成逐文件 SHA256 清单。通过可访问的官方网页对照关键代码与 YAML；不宣称已成功克隆或确认最新远端 HEAD。
