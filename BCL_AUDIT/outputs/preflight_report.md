# BCL CIFAR-100-LT GPU Preflight

**READY FOR FULL PILOT**

分支 `research/bcl-cifar100lt-audit-20261122`，HEAD `6cf5f8ac06a701fb5daa6af8c745495f1bae47d9`，本地与 GitHub 分支头核对一致。CPU pytest、静态数据审计及两组实测依序执行。

## 软件测试

`python -B -m pytest BCL_AUDIT/cifar100lt/tests -q -p no:cacheprovider`

全部 8 项通过。pytest 退出码 0，输出为 `8 passed in 13.80s`。缓存插件仅为本地执行关闭。

## CIFAR-100-LT IF=100 数据和拓扑

本地数据根目录 `D:\xiangmuxunlian\xin\H2T\data\cifar100`。从原始均衡 TRAIN 的 50,000 张图生成 IF=100 指数不平衡划分，使用 seed 0 和 `int(500 * 0.01 ** (class_id / 99.0))`，完整类别计数和索引逐元素复核通过。首尾类别为 500／5 张，共 10,847 张，源索引全部唯一。 本地数据审计文件（未提交）：`data_audit/data_audit_summary.json`；固定索引：`data_audit/cifar100lt_if100_train_indices_seed0.npy`

100 个 fine class 每类只对应一个 coarse superclass。20 个 superclass 各有 5 个 fine class。按 >100／20–100／<20 分组，Many／Medium／Few 各为 35／35／30 类。1,050 个 Many–Few 对逐项核对通过，其中同 superclass 35 对、跨 superclass 1015 对。同 superclass 对分布在 13/20 个 superclass，覆盖 21/30 个 Few 类。 本地审计 CSV（未提交）：`data_audit/class_counts.csv`、`data_audit/superclass_counts.csv`、`data_audit/head_tail_pair_topology.csv`

| 数据文件 | SHA256 |
| --- | --- |
| CIFAR-100 TRAIN | `735e79b04f092ca3d2e6d07f368c0a7d70d48c48d28865950cc24454cf45129b` |
| CIFAR-100 meta | `a5d4786345c961390f865e93b434dbd5c6904ce880667e0cb888c97d449f28b9` |
| IF=100 seed 0 源索引 | `be73a9441289dbadec4190c8c30a1bfc551434f17b33f6dc0ceea7fb77340ac7` |

当前 torchvision 默认会对 TRAIN 和 TEST 都进行归档完整性检查。本轮使用临时 TRAIN-only 包装，保留 torchvision 的 TRAIN／meta 校验和训练样本加载，禁用下载、TEST 构造和 TEST 文件打开。静态审计和 GPU 探针各加载一次 TRAIN，两处 TEST 构造与文件读取尝试均为 0。静态审计出现一条文档字符串 Windows 路径转义的 `SyntaxWarning`，进程退出码为 0。 本地数据门禁回执（未提交）：`preflight_20261122_01/data_validation.json`

## CUDA 环境与固定设置

Python 3.12.12（`D:\app\minicondaaa\envs\xiaoju\python.exe`），PyTorch 2.14.1+cu132，torchvision 0.29.1+cu132，CUDA runtime 13.2，cuDNN 92400。GPU NVIDIA GeForce RTX 5070 Laptop GPU（`cuda:0`）。CUDA 初始化后可用显存 6.835 GiB（7,338,983,424 B），设备可见总显存 7.960 GiB（8,546,484,224 B）。

使用原训练器的真实 `DataLoader`，`num_workers=0`，训练 RNG seed 42。一次采样得到的 256 张 TRAIN 图分别做 v1、v2、v3 独立增强，两组共享同一批张量与同一初始权重。每视图 `[256,3,32,32]`，拼接后 backbone 输入 `[768,3,32,32]`，单次训练态前向、反向和 SGD 更新。FP32，AMP、TF32 和梯度累积关闭。SGD 首轮 warmup LR 0.03（基准 0.15），momentum 0.9，weight decay 5e-4。

数据加载、解码与增强耗时 0.408604 s，该批涵盖 68 个 fine class。配对三视图与标签 SHA256 `ff2ad26e03868a454c14ff3b6b160e0a0eeb9f8b74615b4df89c9dac9dac99c2`。

## 完整训练单步实测

| 组别 | 状态 | 实际 SGD 更新 | 单步耗时 | 峰值 allocated | 峰值 reserved |
| --- | --- | ---: | ---: | ---: | ---: |
| LC-only | passed | 1 | 0.780012 s | 1.883 GiB（2,021,376,000 B） | 2.238 GiB（2,403,336,192 B） |
| BCL | passed | 1 | 0.471897 s | 1.884 GiB（2,022,882,304 B） | 2.264 GiB（2,430,599,168 B） |

单步耗时从主机到 CUDA 的输入传输起计，包含三视图拼接、前向、loss、反向、SGD 与 CUDA 同步。表中峰值以每组清缓存并重置显存峰值后的 PyTorch 记录为准。GPU 驱动报告的显存余量与 PyTorch allocator 统计采用各自的计数口径。

- **LC-only** 开始时 allocated／reserved 为 0.000 GiB（0 B）／0.000 GiB（0 B）。参数更新已逐张量核对，backbone 93、classifier 2。总 loss 14.6432438，LC 7.3216219，BCL 0.0000000。
- **BCL** 开始时 allocated／reserved 为 0.063 GiB（67,108,864 B）／0.086 GiB（92,274,688 B）。参数更新已逐张量核对，backbone 93、classifier 2、sample_mlp 6、prototype_mlp 6。总 loss 18.1177711，LC 7.3216219，BCL 5.7908788。

LC-only 用 v1 logits 的 LC 损失。BCL 对 768 个 backbone 特征执行样本投影，使用 v2/v3 的 embedding 及由分类器权重生成的 100 个 prototype 计算对比损失。两组已验证梯度和参数有限性，SGD 实际各更新一次。

## 异常与停止状态

GPU 预检进程退出码 0。两组均未发生 OOM，异常记录为空。

未启动 200 Epoch 正式训练，未保存 checkpoint，未运行 DINOv2 或 TEST 评估。审计后本地 TRAIN 与 meta 的 SHA256 仍匹配。Git 对已跟踪文件及暂存区的差异为空，本轮产物保存在本地忽略的 `BCL_AUDIT/outputs/`。 本地 GPU 原始回执（未提交）：`preflight_20261122_01/gpu_preflight.json`；临时预检脚本：`preflight_20261122_01/bcl_preflight_once.py`

数据验证回执 SHA256 `f8f0f95c94342975d9e37b57a889d2d7913578033d0fb71703e4c4cf046c27e2`。GPU 回执 SHA256 `5447471faa4b7b134832fa2a27b5926f73242688d880c384a41fae54fe8acca3`。


## Post-preflight source revalidation note

This report records the GPU-capacity probe performed against source commit
`6cf5f8ac06a701fb5daa6af8c745495f1bae47d9`. Subsequent review found that the
committed training/topology/probe paths still used stock
`torchvision.datasets.CIFAR100(train=True)`, whose torchvision 0.29.1
integrity check also inspects `test_list`. The current branch has since been
hardened with a permanent TRAIN-only loader and a regression test.

Therefore this report remains valid as historical evidence that the 3-view
FP32 batch fits the RTX 5070 Laptop 8 GB GPU, but its READY status does not
authorize the full pilot against the updated source until
`CODEX_PREFLIGHT_ONLY.txt` is rerun and a fresh report is generated against
the current executable source.
