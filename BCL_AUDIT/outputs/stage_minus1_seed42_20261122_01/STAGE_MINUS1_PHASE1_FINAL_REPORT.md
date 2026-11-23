# BCL CIFAR-100-LT Stage -1 Phase 1 最终报告

**Phase 1 完成，等待人工验收。**

## Git 与预检源码

仓库 `LuoQiuXu/xin`，分支 `research/bcl-cifar100lt-audit-20261122`。开始时 GitHub 与本地 HEAD 均为 `be216f9cf537d86ab471319c9fb02832378259b8`。结束时本地 HEAD 为 `be216f9cf537d86ab471319c9fb02832378259b8`。预检实际测试 `cdcc085c207297d3d6d382b526bcadf03ac19b04`，本轮开跑时逐文件核对 13 个受跟踪 Python 文件。结束时 Python 源码复核通过。原暂存文件及其工作区字节均已保留。

```text
## research/bcl-cifar100lt-audit-20261122...origin/research/bcl-cifar100lt-audit-20261122
M  BCL_AUDIT/.gitignore
A  BCL_AUDIT/outputs/preflight_recheck_20261122_02/data_validation.json
A  BCL_AUDIT/outputs/preflight_recheck_20261122_02/gpu_preflight.json
M  BCL_AUDIT/outputs/preflight_report.md
?? H2T_repro_audit/ccsam_causal_oracle_20261119/
?? H2T_repro_audit/ccsam_stage2_controlled_cifar100_20261106/
```

## 回归测试

`36 passed, 1 warning in 3.63s`。退出码 0，进程耗时 4.643 s。完整输出 `D:\xiangmuxunlian\xin\BCL_AUDIT\outputs\stage_minus1_seed42_20261122_01\pytest.log`。

## 数据与 DINOv2

数据根目录 `D:\xiangmuxunlian\xin\H2T\data\cifar100`，固定索引 `D:\xiangmuxunlian\xin\BCL_AUDIT\outputs\data_audit_recheck_20261122_01\cifar100lt_if100_train_indices_seed0.npy`。索引 SHA256 为 `be73a9441289dbadec4190c8c30a1bfc551434f17b33f6dc0ceea7fb77340ac7`。原静态文件预检前后 SHA256 复核通过。

| 原始审计文件 | 开跑时 SHA256 | 结束时 SHA256 |
| --- | --- | --- |
| class_counts.csv | `1d9c441e1035f52d29b6a538b196d6c064b0eef8cec45043566a9591542c873c` | `1d9c441e1035f52d29b6a538b196d6c064b0eef8cec45043566a9591542c873c` |
| superclass_counts.csv | `32f30202190090ce6810d5e646ed67c975dbd359ef9b2d9cb6d468612016a13c` | `32f30202190090ce6810d5e646ed67c975dbd359ef9b2d9cb6d468612016a13c` |
| head_tail_pair_topology.csv | `56faf5af86db93063c6949055de1847002ac24a63e3469dbb810dd058d8caee9` | `56faf5af86db93063c6949055de1847002ac24a63e3469dbb810dd058d8caee9` |
| data_audit_summary.json | `2eaa3b9e3bf84494b95340cc7ad5c09be6c5d37ab06720de3888d9c99187d6ff` | `2eaa3b9e3bf84494b95340cc7ad5c09be6c5d37ab06720de3888d9c99187d6ff` |
| cifar100lt_if100_train_indices_seed0.npy | `be73a9441289dbadec4190c8c30a1bfc551434f17b33f6dc0ceea7fb77340ac7` | `be73a9441289dbadec4190c8c30a1bfc551434f17b33f6dc0ceea7fb77340ac7` |

官方 `dinov2_vits14`，batch 16，Top-5。模型来自 `facebookresearch/dinov2 via torch.hub`，使用原始平衡 TRAIN 与 meta。原始 TRAIN 50,000 张，每类 500 张，固定 LT TRAIN 含 10,847 张。提取耗时 167.88 s，峰值 allocated 0.199 GiB（213,856,256 B）。提取阶段 TRAIN、meta 和 TEST 打开次数依次为 2、2、0。

本地权重 `D:\xiangmuxunlian\xin\BCL_AUDIT\outputs\stage_minus1_seed42_20261122_01\torch\hub\checkpoints\dinov2_vits14_pretrain.pth`，SHA256 `b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9`。类均值和相似度矩阵形状分别为 `[100, 384]` 与 `[100, 100]`。

100 类对称图有 333 条边，其中 141 条同 superclass。度数最小值、中位数、最大值为 **5／6／17**，均值 6.66。有向 Top-5 中同 superclass 边占 241/500（48.20%），均匀随机邻居期望 4.0404%，实测为期望的 11.9295 倍。Many–Few 共 1,050 对，同／跨 superclass 35／1015，其中 62 对属于 DINO 对称 Top-5（同／跨 23／39）。

| 类别频次组 | 类数 | 度数最小／中位／最大 | 同 superclass 有向边／总边 |
| --- | ---: | --- | ---: |
| many | 35 | 5／5／12 | 87/175 |
| medium | 35 | 5／6／17 | 84/175 |
| few | 30 | 5／6／17 | 70/150 |

| Superclass | 同 superclass 有向边／总边 | 平均图度数 |
| --- | ---: | ---: |
| aquatic_mammals | 12/25 | 8.60 |
| fish | 13/25 | 6.80 |
| flowers | 15/25 | 5.20 |
| food_containers | 14/25 | 6.60 |
| fruit_and_vegetables | 12/25 | 5.80 |
| household_electrical_devices | 13/25 | 6.80 |
| household_furniture | 18/25 | 5.40 |
| insects | 12/25 | 6.60 |
| large_carnivores | 12/25 | 6.20 |
| large_man-made_outdoor_things | 8/25 | 6.60 |
| large_natural_outdoor_scenes | 13/25 | 7.20 |
| large_omnivores_and_herbivores | 9/25 | 5.80 |
| medium_mammals | 8/25 | 6.00 |
| non-insect_invertebrates | 8/25 | 8.40 |
| people | 20/25 | 5.20 |
| reptiles | 7/25 | 6.80 |
| small_mammals | 17/25 | 8.80 |
| trees | 16/25 | 8.60 |
| vehicles_1 | 8/25 | 6.00 |
| vehicles_2 | 6/25 | 5.80 |

六个 DINO 文件逐项通过复核。

- `D:\xiangmuxunlian\xin\BCL_AUDIT\outputs\data_audit_recheck_20261122_01\DINO_CLASS_MEAN_FEATURES.npy`，SHA256 `18b0afd32a37449c6e33079feedc74b71583f199237f4da177796daebbf0e989`
- `D:\xiangmuxunlian\xin\BCL_AUDIT\outputs\data_audit_recheck_20261122_01\DINO_CLASS_SIMILARITY.npy`，SHA256 `1c12f28fda137e57b61a829a7b785887e75459dcc2ece6d410dc705ec03c4c9f`
- `D:\xiangmuxunlian\xin\BCL_AUDIT\outputs\data_audit_recheck_20261122_01\DINO_CLASS_TOPOLOGY.csv`，SHA256 `1fc0b65539557f92fc65031e1d467d7275712342bef06776669826afd0a0a47e`
- `D:\xiangmuxunlian\xin\BCL_AUDIT\outputs\data_audit_recheck_20261122_01\head_tail_pair_topology_dino.csv`，SHA256 `0c09b9838e56137e38c8b9a652a09a57205922f4fcbae0a4a756c92fef832c8a`
- `D:\xiangmuxunlian\xin\BCL_AUDIT\outputs\data_audit_recheck_20261122_01\data_audit_dino_summary.json`，SHA256 `e50ddb1dec840598f2db5c2be997c22ba7b0de85c3c66125191e7800f854470e`
- `D:\xiangmuxunlian\xin\BCL_AUDIT\outputs\data_audit_recheck_20261122_01\dino_metadata.json`，SHA256 `4daa31fa255a305211ee3426353b3af47807d9f9c50301b3540b03bb8eb8cedf`

## 训练与最终 TEST

固定配置为 ResNet-32、seed 42、IF=100、FP32、batch 256、三视图合计 backbone batch 768。使用 SGD、基准 LR 0.15、momentum 0.9、weight decay 5e-4，5 epochs warmup，完成 160 和 180 epochs 后各乘 0.1。LC-only 使用 v1 分类损失 `2.0*LC`。Original BCL 使用 v1 分类、v2/v3 样本投影与分类器派生 prototype，损失 `2.0*LC+0.6*BCL`。训练配置关闭 AMP、TF32 和梯度累积。

环境为 Python 3.12.12、PyTorch 2.14.1+cu132、torchvision 0.29.1+cu132、CUDA 13.2、cuDNN 92400、NVIDIA GeForce RTX 5070 Laptop GPU。开跑时 CUDA 空闲显存 6.835 GiB（7,338,983,424 B）。

| 组别 | 完成 epochs | 训练时间 | GPU 峰值 allocated | Overall | Many | Medium | Few |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LC_only | 200/200 | 4778.13 s | 1.884 GiB（2,023,082,496 B） | 47.8300% | 62.5714% | 49.2000% | 29.0333% |
| BCL | 200/200 | 5714.43 s | 1.887 GiB（2,025,941,504 B） | 50.7600% | 66.5429% | 51.7429% | 31.2000% |

首个 TEST 打开前已验证 200/200 个配对 epoch 和两组各五个 checkpoint。放行前 TEST 打开 0 次。实际 TEST 打开总计 4 次。最终指标均来自 epoch 200，Many/Medium/Few 为组内逐类宏平均。
共同初始权重 `D:\xiangmuxunlian\xin\BCL_AUDIT\outputs\stage_minus1_seed42_20261122_01\init_seed42.pth`，文件 SHA256 `f24891d70e8d1fcbe378214d7610cc04a1a9ca270a5e9030f6a04725158bf106`，完整模型状态 SHA256 `11c52fe9e2b082b0c36ef025f19ae2d963dbc19c456e0d5aa17aac14fffdf440`。

## 配对数据流与 BatchNorm

配对校验比较 200 个 epoch，差异 0 项。每轮 LC-only 与 BCL 分别覆盖 10,847 个唯一且来自固定集合的 LT TRAIN 源索引，完整顺序和首批三视图 SHA256 一致。两个 backbone 均接收三视图拼接。31 个 backbone BatchNorm2d 在每个里程碑都有逐视图记录，仅 BCL 的 sample projection BatchNorm1d 被执行。

| 组别 | 审计 epochs | 末轮 backbone BN 批次计数 | 末轮 sample projection BN 批次计数 |
| --- | --- | ---: | ---: |
| LC_only | 40, 100, 160, 180, 200 | 8600 | 0 |
| BCL | 40, 100, 160, 180, 200 | 8600 | 8600 |

## Checkpoint、日志和结果文件

正式运行目录 `D:\xiangmuxunlian\xin\BCL_AUDIT\outputs\stage_minus1_seed42_20261122_01`。

### LC_only

- `D:\xiangmuxunlian\xin\BCL_AUDIT\outputs\stage_minus1_seed42_20261122_01\LC_only_seed42\checkpoints\epoch_040.pth`，SHA256 `4c48b4e706ee665db212f9052cf07146cf0ff6e2c9261961e00e6acd69d44487`
- `D:\xiangmuxunlian\xin\BCL_AUDIT\outputs\stage_minus1_seed42_20261122_01\LC_only_seed42\checkpoints\epoch_100.pth`，SHA256 `63af82d45df6636aaa128a53ed655158a36d63b2ae6d3310ce9e5cc4178def35`
- `D:\xiangmuxunlian\xin\BCL_AUDIT\outputs\stage_minus1_seed42_20261122_01\LC_only_seed42\checkpoints\epoch_160_pre_decay.pth`，SHA256 `7517f6c8648e80d4ce4db4a33f465351200f21d882c8dc80a77f3e0914a70ef7`
- `D:\xiangmuxunlian\xin\BCL_AUDIT\outputs\stage_minus1_seed42_20261122_01\LC_only_seed42\checkpoints\epoch_180_pre_decay.pth`，SHA256 `7ce22ab9634231b47d36bb3591b8da8c9076f0e8ca0cfe44c56fee8a965134a6`
- `D:\xiangmuxunlian\xin\BCL_AUDIT\outputs\stage_minus1_seed42_20261122_01\LC_only_seed42\checkpoints\epoch_200.pth`，SHA256 `583f50f8fb5cc3c8885566164eebf71a7144f9b8a1d259c907ca989c858e3167`

- `D:\xiangmuxunlian\xin\BCL_AUDIT\outputs\stage_minus1_seed42_20261122_01\LC_only_seed42\history.jsonl`
- `D:\xiangmuxunlian\xin\BCL_AUDIT\outputs\stage_minus1_seed42_20261122_01\LC_only_seed42\training_summary.json`
- `D:\xiangmuxunlian\xin\BCL_AUDIT\outputs\stage_minus1_seed42_20261122_01\LC_only_seed42\batchnorm_audit.jsonl`
- `D:\xiangmuxunlian\xin\BCL_AUDIT\outputs\stage_minus1_seed42_20261122_01\LC_only_seed42\final_test_metrics.json`
- `D:\xiangmuxunlian\xin\BCL_AUDIT\outputs\stage_minus1_seed42_20261122_01\LC_only_seed42\final_test_predictions.npz`
- `D:\xiangmuxunlian\xin\BCL_AUDIT\outputs\stage_minus1_seed42_20261122_01\LC_only_seed42\final_test_confusion_matrix.npy`

### BCL

- `D:\xiangmuxunlian\xin\BCL_AUDIT\outputs\stage_minus1_seed42_20261122_01\BCL_seed42\checkpoints\epoch_040.pth`，SHA256 `df1b15ac32b836e2171313a9e19b0cb5649a5b8c77bedb3ebd7addc78e75dd30`
- `D:\xiangmuxunlian\xin\BCL_AUDIT\outputs\stage_minus1_seed42_20261122_01\BCL_seed42\checkpoints\epoch_100.pth`，SHA256 `b570af338c2d6e7956818167bc7450c3e551996f3c2a033557d3cadaaa2e618c`
- `D:\xiangmuxunlian\xin\BCL_AUDIT\outputs\stage_minus1_seed42_20261122_01\BCL_seed42\checkpoints\epoch_160_pre_decay.pth`，SHA256 `8d44bf1f5911fffd9a0b80b82d534a65121ea1e6a6e76351b99ad2e8a95e6841`
- `D:\xiangmuxunlian\xin\BCL_AUDIT\outputs\stage_minus1_seed42_20261122_01\BCL_seed42\checkpoints\epoch_180_pre_decay.pth`，SHA256 `7ff4bc4b8538b90dc15e20ddb98a08e776bac47b8e38b0452c38bba56b612d80`
- `D:\xiangmuxunlian\xin\BCL_AUDIT\outputs\stage_minus1_seed42_20261122_01\BCL_seed42\checkpoints\epoch_200.pth`，SHA256 `11193e2df710373d8c11eb1daad61e2ab57210d7b858a5324be961765795a17e`

- `D:\xiangmuxunlian\xin\BCL_AUDIT\outputs\stage_minus1_seed42_20261122_01\BCL_seed42\history.jsonl`
- `D:\xiangmuxunlian\xin\BCL_AUDIT\outputs\stage_minus1_seed42_20261122_01\BCL_seed42\training_summary.json`
- `D:\xiangmuxunlian\xin\BCL_AUDIT\outputs\stage_minus1_seed42_20261122_01\BCL_seed42\batchnorm_audit.jsonl`
- `D:\xiangmuxunlian\xin\BCL_AUDIT\outputs\stage_minus1_seed42_20261122_01\BCL_seed42\final_test_metrics.json`
- `D:\xiangmuxunlian\xin\BCL_AUDIT\outputs\stage_minus1_seed42_20261122_01\BCL_seed42\final_test_predictions.npz`
- `D:\xiangmuxunlian\xin\BCL_AUDIT\outputs\stage_minus1_seed42_20261122_01\BCL_seed42\final_test_confusion_matrix.npy`

- `D:\xiangmuxunlian\xin\BCL_AUDIT\outputs\stage_minus1_seed42_20261122_01\dino.log`
- `D:\xiangmuxunlian\xin\BCL_AUDIT\outputs\stage_minus1_seed42_20261122_01\dino.stderr.log`
- `D:\xiangmuxunlian\xin\BCL_AUDIT\outputs\stage_minus1_seed42_20261122_01\dino_validation.json`
- `D:\xiangmuxunlian\xin\BCL_AUDIT\outputs\stage_minus1_seed42_20261122_01\training.log`
- `D:\xiangmuxunlian\xin\BCL_AUDIT\outputs\stage_minus1_seed42_20261122_01\training.stderr.log`
- `D:\xiangmuxunlian\xin\BCL_AUDIT\outputs\stage_minus1_seed42_20261122_01\formal_training.log`
- `D:\xiangmuxunlian\xin\BCL_AUDIT\outputs\stage_minus1_seed42_20261122_01\formal_training.stderr.log`
- `D:\xiangmuxunlian\xin\BCL_AUDIT\outputs\stage_minus1_seed42_20261122_01\formal_training_process.json`
- `D:\xiangmuxunlian\xin\BCL_AUDIT\outputs\stage_minus1_seed42_20261122_01\training_execution.json`
- `D:\xiangmuxunlian\xin\BCL_AUDIT\outputs\stage_minus1_seed42_20261122_01\test_open_gate.json`
- `D:\xiangmuxunlian\xin\BCL_AUDIT\outputs\stage_minus1_seed42_20261122_01\paired_data_stream_audit.json`
- `D:\xiangmuxunlian\xin\BCL_AUDIT\outputs\stage_minus1_seed42_20261122_01\paired_pilot_summary.json`

## 警告、异常与停止边界

第一次后台启动在训练入口前退出，原空日志及启动记录已保留。后续使用相同训练配置启动，仅更换了后台启动方式。

- D:\xiangmuxunlian\xin\BCL_AUDIT\scripts\audit_cifar100lt_topology.py:9: SyntaxWarning: invalid escape sequence '\d'
- D:\xiangmuxunlian\xin\BCL_AUDIT\outputs\stage_minus1_seed42_20261122_01\torch\hub\facebookresearch_dinov2_main\dinov2\layers\swiglu_ffn.py:51: UserWarning: xFormers is not available (SwiGLU)
- warnings.warn("xFormers is not available (SwiGLU)")
- D:\xiangmuxunlian\xin\BCL_AUDIT\outputs\stage_minus1_seed42_20261122_01\torch\hub\facebookresearch_dinov2_main\dinov2\layers\attention.py:33: UserWarning: xFormers is not available (Attention)
- warnings.warn("xFormers is not available (Attention)")
- D:\xiangmuxunlian\xin\BCL_AUDIT\outputs\stage_minus1_seed42_20261122_01\torch\hub\facebookresearch_dinov2_main\dinov2\layers\block.py:40: UserWarning: xFormers is not available (Block)
- warnings.warn("xFormers is not available (Block)")

正式训练和两组最终 TEST 已完成。Phase 2、离线 analyzer 和 Oracle 均未执行，等待人工验收。
