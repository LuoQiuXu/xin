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
