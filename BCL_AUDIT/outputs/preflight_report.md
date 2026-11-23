# BCL CIFAR-100-LT GPU Preflight

**READY FOR FULL PILOT**

分支 `research/bcl-cifar100lt-audit-20261122`。当前源码 HEAD 与本轮实际测试的 `source_commit` 均为 `dc150d1023510c1ec40841fdc133fb8ba106774c`，执行前与 GitHub 分支头一致。两次 GPU 实测发生在全部 pytest 和新版静态审计通过后。

## 复用路径

- `data_audit_output_dir` = `D:\xiangmuxunlian\xin\BCL_AUDIT\outputs\data_audit_recheck_20261122_01`
- `indices_file` = `D:\xiangmuxunlian\xin\BCL_AUDIT\outputs\data_audit_recheck_20261122_01\cifar100lt_if100_train_indices_seed0.npy`
- `indices_sha256` = `be73a9441289dbadec4190c8c30a1bfc551434f17b33f6dc0ceea7fb77340ac7`

## 软件测试与 TEST 文件边界

全部 **10/10** 项通过。pytest 实际输出 `10 passed, 1 warning in 3.13s`。使用 `python -B -m pytest BCL_AUDIT/cifar100lt/tests -q -rA -p no:cacheprovider` 执行。静态审计脚本的 Windows 路径示例产生一条文档字符串 `SyntaxWarning`，测试和静态审计均以退出码 0 完成。

```text
PASSED test_cifar100_train_only_integrity_never_checks_test_files
PASSED test_reuse_static_audit_loads_without_rewriting
PASSED test_if100_count_vector_and_indices
PASSED test_resnet32_and_bcl_forward_shapes
PASSED test_bcl_loss_is_finite_and_differentiable_with_all_prototypes
PASSED test_gradient_routes_match_the_declared_computation_graph
PASSED test_lc_only_and_bcl_share_three_view_dataset_contract
PASSED test_batchnorm_audit_captures_backbone_and_bcl_sample_projection_views
PASSED test_head_tail_did_is_paired_and_direction_specific
PASSED test_bcl_loss_matches_released_official_balscl_formula
```

新增完整性测试确认 `CIFAR100TrainOnly._check_integrity()` 只访问 `train_list`。实际 DataLoader 使用已提交的 `BCL_AUDIT.cifar100lt.data.CIFAR100TrainOnly`，`train=True` 和 `download=False`。GPU 进程通过 Python 打开事件记录 TRAIN 2 次、meta 2 次、TEST **0 次**。事件观察未替换任何 loader。

## 原始 TRAIN/meta 静态审计

使用源码自带 `scripts/audit_cifar100lt_topology.py` 的原始 TRAIN/meta 入口。数据根目录 `D:\xiangmuxunlian\xin\H2T\data\cifar100`。入口校验官方 TRAIN 和 meta 的 MD5 后直接读取这两个 pickle。原始 TRAIN 含 50,000 张图，每 fine class 500 张。静态输出记录 `test_split_loaded=false` 和 `dino_similarity_computed=false`。

IF=100 使用 `int(500 * 0.01 ** (class_id / 99.0))` 与 split seed 0。首尾计数 500／5，总计 **10,847** 张。100 类计数、原始索引去重与逐元素顺序均通过复核。本轮索引 SHA256 与旧 seed 0 划分一致。 [静态审计摘要](data_audit_recheck_20261122_01/data_audit_summary.json) · [复核回执](preflight_recheck_20261122_01/data_validation.json)

100 个 fine class 各映射到唯一 superclass。20 个 superclass 各含 5 类。Many／Medium／Few 为 35／35／30 类。完整表含 1,050 个 Many–Few 对，其中同 superclass 35 个，跨 superclass 1,015 个。同 superclass 对覆盖 13/20 个 superclass、21/30 个 Few 类。 [逐类计数](data_audit_recheck_20261122_01/class_counts.csv) · [Superclass 表](data_audit_recheck_20261122_01/superclass_counts.csv) · [Many–Few 对表](data_audit_recheck_20261122_01/head_tail_pair_topology.csv)

## CUDA 环境与实测结果

Python 3.12.12（`D:\app\minicondaaa\envs\xiaoju\python.exe`），PyTorch 2.14.1+cu132，torchvision 0.29.1+cu132，CUDA 13.2，cuDNN 92400。设备 NVIDIA GeForce RTX 5070 Laptop GPU（`cuda:0`），初始化后驱动可用显存 6.835 GiB（7,338,983,424 B），设备可见总量 7.960 GiB（8,546,484,224 B）。

原训练器 DataLoader 以 `num_workers=0` 从审计后的 IF=100 索引取一个真实批次。两组共享同一批增强视图及同一组初始权重。每视图 `[256,3,32,32]`，三视图沿 batch 拼接成为 FP32 `[768,3,32,32]`。AMP、TF32 和梯度累积关闭。SGD 首轮 LR 0.03（基准 0.15），momentum 0.9，weight decay 5e-4。

训练批次含 68 个 fine class。配对批次 SHA256 `ff2ad26e03868a454c14ff3b6b160e0a0eeb9f8b74615b4df89c9dac9dac99c2`，共享初始状态 SHA256 `cae88d1a2b9238b30d6512da052032f07739bb45c21800b4ee01a3f6228e88ca`。DataLoader 获取及三视图增强耗时 0.412492 s。

| 组别 | 状态 | 实际 SGD 更新 | 单步耗时 | 峰值 allocated | 峰值 reserved |
| --- | --- | ---: | ---: | ---: | ---: |
| LC-only | passed | 1 | 0.530132 s | 1.883 GiB（2,021,376,000 B） | 2.238 GiB（2,403,336,192 B） |
| BCL | passed | 1 | 0.404214 s | 1.884 GiB（2,022,882,304 B） | 2.264 GiB（2,430,599,168 B） |

单步计时包含主机到 CUDA 传输、三视图拼接、前向、loss、反向、SGD 和 CUDA 同步。两组开始前清理梯度及可释放的缓存，并重置显存峰值。

- LC-only 起始 allocated／reserved 为 0.000 GiB（0 B）／0.000 GiB（0 B）。已变化参数张量 backbone 93、classifier 2。 LC loss 7.3216219，BCL loss 0.0000000，总 loss 14.6432438。
- BCL 起始 allocated／reserved 为 0.063 GiB（67,108,864 B）／0.086 GiB（92,274,688 B）。已变化参数张量 backbone 93、classifier 2、sample_mlp 6、prototype_mlp 6。 LC loss 7.3216219，BCL loss 5.7908788，总 loss 18.1177711。

LC-only 将 v1 分类 logits 输入 LC。BCL 对三视图 backbone 特征投影，取 v2/v3 的 embedding 与分类器导出的 100 个 prototype，执行 LC 和 BCL 联合目标。通过的两组均验证梯度与参数有限性，以及恰好一次 SGD 参数更新。

## 异常与保留范围

CUDA 预检退出码 0。OOM 0 次，本轮无执行异常。

本轮未运行 DINOv2、200 Epoch 训练或 TEST 评估，未写 checkpoint。新结果独立保存。原数据审计与首轮预检的 8 个文件逐一复核，SHA256 保持一致。报告此前的版本可在 Git 历史中查阅。 [原始 GPU 回执](preflight_recheck_20261122_01/gpu_preflight.json) · [本地预检脚本](preflight_recheck_20261122_01/gpu_preflight_recheck.py)

数据验证回执 SHA256 `105cd73ac67dc5fb50fa0e5d2b9c1af47f3b7aceefb1aa75aeec2c23a35e711e`。GPU 回执 SHA256 `71e6aefdeb58f09b0931ca10c0fe8d37c22fd298d5dd0ae00e01001bca0da5e3`。

## Current-source revalidation — cdcc085 (2026-11-23)

**READY FOR FULL PILOT**

`tested_source_commit` = `cdcc085c207297d3d6d382b526bcadf03ac19b04`。仓库 `LuoQiuXu/xin`，
分支 `research/bcl-cifar100lt-audit-20261122`。本地同步前 HEAD 为 `fefab371855f44225003958dcbb9cd38905430a5`。
工作区的 13 个受跟踪 Python 文件逐一与测试提交核验一致。
本报告和两份 JSON 由后续独立的证据提交保存。此前的 READY 记录继续作为各自源码版本的历史证据。

### 回归测试

`python -B -m pytest BCL_AUDIT/cifar100lt/tests -q -rA -p no:cacheprovider`
退出码 0。实际结果 `36 passed, 1 warning in 6.92s`，进程耗时 8.410 s。
拓扑脚本第 9 行的既有路径示例警告为 `invalid escape sequence '\d'`。完整输出保存于新目录下的本地
`pytest.log`，逐项通过记录与日志 SHA256 见数据验证回执。

### IF=100 静态数据

- `data_root` = `D:\xiangmuxunlian\xin\H2T\data\cifar100`
- `data_audit_output_dir` = `D:\xiangmuxunlian\xin\BCL_AUDIT\outputs\data_audit_recheck_20261122_01`
- `indices_file` = `D:\xiangmuxunlian\xin\BCL_AUDIT\outputs\data_audit_recheck_20261122_01\cifar100lt_if100_train_indices_seed0.npy`
- `indices_sha256` = `be73a9441289dbadec4190c8c30a1bfc551434f17b33f6dc0ceea7fb77340ac7`

原 TRAIN 50,000 张，100 个 fine class 每类 500 张。固定 IF=100 索引含 10,847 张，
实际 loader 的逐类频次与静态表一致，首尾 500／5。Many／Medium／Few 为 35／35／30 类，
20 个 superclass 每个包含 5 类，Many–Few 对为同 superclass 35、跨 superclass 1,015。
五个原文件在测试与显存预检前后的 SHA256 逐项相同，四个 CSV/JSON 的 Git blob 与
`083d515` 一致。

| 原文件 | 预检前后相同的 SHA256 |
| --- | --- |
| `class_counts.csv` | `1d9c441e1035f52d29b6a538b196d6c064b0eef8cec45043566a9591542c873c` |
| `superclass_counts.csv` | `32f30202190090ce6810d5e646ed67c975dbd359ef9b2d9cb6d468612016a13c` |
| `head_tail_pair_topology.csv` | `56faf5af86db93063c6949055de1847002ac24a63e3469dbb810dd058d8caee9` |
| `data_audit_summary.json` | `2eaa3b9e3bf84494b95340cc7ad5c09be6c5d37ab06720de3888d9c99187d6ff` |
| `cifar100lt_if100_train_indices_seed0.npy` | `be73a9441289dbadec4190c8c30a1bfc551434f17b33f6dc0ceea7fb77340ac7` |

### 配对数据流与初始状态

两套真实 `make_train_loader` 使用原 `CIFAR100TrainOnly`，`train=True`、`download=False`、
`num_workers=0`。分别用 epoch seed 4200000 重置全局 RNG 与 DataLoader RNG。
每套读取 43 个批次、10,847 个唯一 TRAIN 原始索引，索引顺序逐项完全相同并覆盖固定划分。
遍历只核对数据，GPU 更新均使用同一个经配对验证的首批。

| 配对核验项 | 两组相同的 SHA256 |
| --- | --- |
| 完整 TRAIN 源索引顺序 | `4900ae2f520eec1547f6f213a970e78a3a43341fb4304f5b9308777524aa615b` |
| 首批源索引 | `8fe5d09256557fc70e84f23286ee00fc914a5d0b622fd8daa022571025f9416f` |
| 首批三视图字节 | `68ca753a0506e00aa62b6d86ea44d906cd32c847ad3314667e35355c9c4f9133` |
| 首批三视图、标签和源索引合并 | `59158b5332a0b03f6b18ee091c81847a94ef1b367c72c287eb95e07e0585ba21` |
| 共享初始 state_dict | `11c52fe9e2b082b0c36ef025f19ae2d963dbc19c456e0d5aa17aac14fffdf440` |
| 本地 `init_seed42.pth` 文件 | `6a86b7fe21062c079970856f08489c844543942ae260291c87e518572a3e2421` |

逐视图以及首批标签的指纹也逐项一致，完整记录见 GPU 回执。两个组通过
`load_initial_model_state()` 独立加载同一文件，加载后的完整状态均通过 SHA256 核对。
Python 文件打开记录为 TRAIN 4 次、meta 4 次、官方 TEST
**0 次**，`test_split_loaded=false`。

### RTX 5070 Laptop 单步实测

Python 3.12.12（`D:\app\minicondaaa\envs\xiaoju\python.exe`），PyTorch 2.14.1+cu132，
torchvision 0.29.1+cu132，CUDA 13.2，cuDNN 92400。
设备 NVIDIA GeForce RTX 5070 Laptop GPU（`cuda:0`），开始时空闲显存
6.835 GiB（7,338,983,424 B），设备可见总量 7.960 GiB（8,546,484,224 B）。

两组均以 batch=256 的三视图拼接执行一次 backbone 前向，输入 `[768, 3, 32, 32]`。
使用 FP32 和 SGD（LR 0.15、momentum 0.9、weight decay 5e-4），AMP、TF32 和梯度累积关闭。
LC-only 用 v1 logits 和 `2.0*LC`。BCL 用 v1 logits、v2/v3 embeddings 和分类器派生 prototypes，
目标为 `2.0*LC+0.6*BCL`。梯度与更新后参数均有限。

| 组别 | 单步 | 真实 SGD 更新 | 总 loss | GPU 单步耗时 | 峰值 allocated | 峰值 reserved |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| LC-only | 通过 | 1 | 13.77327156 | 0.744806 s | 1.883 GiB（2,021,376,000 B） | 2.238 GiB（2,403,336,192 B） |
| BCL | 通过 | 1 | 17.19762230 | 0.515965 s | 1.884 GiB（2,022,882,304 B） | 2.264 GiB（2,430,599,168 B） |

计时包括数据传输、拼接、前向、loss、反向、SGD 和 CUDA 同步。
GPU 进程退出码 0，无 OOM 或执行异常。
DINOv2 未启动，两组均为 **0/200 Epoch**，未运行 TEST 评估或离线分析。
本轮仅为预检保留初始权重文件，无训练 checkpoint。

[静态数据和测试核验](preflight_recheck_20261122_03/data_validation.json) ·
[配对与 GPU 回执](preflight_recheck_20261122_03/gpu_preflight.json)

数据回执 SHA256 `f94404a031a6bdadd149e67f159808c3e0fd1fef408634f5ebf643eb84814607`。GPU 回执 SHA256 `b47c9a775a9ede7bb625e0a1b379953c73e22b2951ec856e29dc9086376e4d87`。
