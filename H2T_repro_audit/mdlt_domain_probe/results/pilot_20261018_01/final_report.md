# OfficeHome-MLT H2T 供体域条件机制试验

## 命题和协议

本轮检验相同供体类别和 batch 共享通道掩码下，供体域条件是否改变固定特征 Stage II 的结果。
四个域均进入 train/val/test；不是 unseen-domain OOD。使用 ImageNet V1 预训练 ResNet-18、65 logits。
Stage I 为 5001 updates，BN running statistics 固定而 affine 可训练；仅按 val domain-class macro 选模，平局取早。
Stage II 使用 eval transform 的固定 GAP 特征，整个 backbone/BN 冻结；这是机制 pilot，区别于随机图像增强的原始 H2T Stage II。
128/512 完整通道按 batch 共享替换，rho=0.25；NumPy 独立 RNG 不复刻原 torch.randperm 路径。
12 个 epoch10 模型先保存并锁定，之后统一打开 test；Stage II 不选择 best。

## 固定 epoch10 最终 TEST 结果（%）

| Seed | Arm | Overall | Domain-class macro | Many | Medium | Few |
|---|---|---:|---:|---:|---:|---:|
| 42 | DR-NoFusion | 75.3462 | 75.3462 | 85.3165 | 75.6757 | 50.0000 |
| 42 | H2T-Global | 75.2308 | 75.2308 | 82.9114 | 76.2162 | 52.4242 |
| 42 | H2T-DomainMatched | 75.6923 | 75.6923 | 84.1772 | 76.4865 | 51.8182 |
| 42 | H2T-CrossDomain | 75.5769 | 75.5769 | 84.0506 | 76.5541 | 50.9091 |
| 43 | DR-NoFusion | 75.5000 | 75.5000 | 85.3165 | 76.4189 | 47.8788 |
| 43 | H2T-Global | 75.9231 | 75.9231 | 84.0506 | 76.7568 | 52.7273 |
| 43 | H2T-DomainMatched | 76.0769 | 76.0769 | 84.5570 | 76.6892 | 53.0303 |
| 43 | H2T-CrossDomain | 76.5000 | 76.5000 | 84.9367 | 77.1622 | 53.3333 |
| 44 | DR-NoFusion | 75.2692 | 75.2692 | 84.6835 | 76.2162 | 48.4848 |
| 44 | H2T-Global | 76.2308 | 76.2308 | 84.8101 | 77.1622 | 51.5152 |
| 44 | H2T-DomainMatched | 76.6923 | 76.6923 | 85.0633 | 77.7703 | 51.8182 |
| 44 | H2T-CrossDomain | 76.1923 | 76.1923 | 84.8101 | 76.6892 | 53.3333 |

### 三 seed 等权均值

| Arm | Overall | Domain-class macro | Many | Medium | Few |
|---|---:|---:|---:|---:|---:|
| DR-NoFusion | 75.3718 | 75.3718 | 85.1055 | 76.1036 | 48.7879 |
| H2T-Global | 75.7949 | 75.7949 | 83.9241 | 76.7117 | 52.2222 |
| H2T-DomainMatched | 76.1538 | 76.1538 | 84.5992 | 76.9820 | 52.2222 |
| H2T-CrossDomain | 76.0897 | 76.0897 | 84.5992 | 76.8018 | 52.5253 |

## 预注册判读

- seed 42：Global−DR Few=+2.4242 pp；Cross−Matched Few=-0.9091 pp，macro=-0.1154 pp。
- seed 43：Global−DR Few=+4.8485 pp；Cross−Matched Few=+0.3030 pp，macro=+0.4231 pp。
- seed 44：Global−DR Few=+3.0303 pp；Cross−Matched Few=+1.5152 pp，macro=-0.5000 pp。
- Global 在三个 seed 的 Few 上方向一致为正；样本只有三个训练重复，不能宣称统计显著或普适收益。
- 2/3 个 seed 满足 Cross 的 Few 或 macro 低于 Matched；在本次配对条件下支持供体域不匹配可能构成风险的初步线索。仍不是新模块有效的证据。

## 配对控制、退回与局限

每 seed 四组共享同一 Stage I、初始 FC、receiver、donor class、batch 顺序和掩码；actual training control hashes 完全一致。
C/D 仅在相应 domain-class 池为空时退回 B 的同一供体；逐条退回见各 seed donor_fallbacks.csv。
DR 的 donor 仅为配对计划中的虚拟供体，不执行融合；诊断中的域/类别分布应按该角色解释。
Many>=60、Medium20..59、Few<20 按训练 domain-class cell 定义，不能与 CIFAR shot 或论文 CIFAR TEST 比较。
260 个 cell、四域准确率、最差 cell 全部并列项和测试支持数见对应 CSV/JSON；不会只挑一个有利 cell。
三 seed 重复使用相同测试图像，不能视作 7800 个独立训练重复；本轮无调参、无新增损失或模块。
不根据即时 margin/测试结果追加实验。到此停止，后续研究需另行确认。

## 文件与审计证据

- config.json / environment.json / dataset_audit.json：配置、来源、环境与数据检查。
- seed42/43/44/stage1_epochs.csv：全部 51 次 val 检查，stage1_summary.json：选择、BN、梯度与耗时。
- seed42/43/44/stage2_epochs.csv：每组十 epoch，固定第十 epoch；output 中保留本地计划、特征、模型和预测。
- final_checkpoints.lock.json / test_access.json：12 模型 SHA256 与 test 首次访问时间。
- epoch10_summary.csv / epoch10_mean.csv / per_domain.csv / domain_class_metrics.csv：全部主指标。
- paired_changes.csv：逐图 gain/loss/net；net 与相应准确率差由程序核对。
- donor_diagnostics.csv / state_audit.json：供体、配对控制与 BB/BN 状态。
- execution.log：完整标准输出及异常栈。源码 SHA256 见 config.json。


## 补充核验与明确科学结论

### 数据来源和统计口径

实际图像目录为 `D:/xiangmuxunlian/xin/data/OfficeHomeDataset_10072016`。用户的 `data/OfficeHome.txt` 与 `protocol/OfficeHome.csv` 字节一致，SHA256 为 `0a6c4e7800f629b8c6b67ba71dd238a83d4f513ac77694ab58796e528985a5b5`。图像目录未重命名，CSV 以显式路径读取，未更改 split。

逐文件 stat 核查 15588/15588 路径存在，无重复路径、解析路径别名或跨 split 路径重叠。train/val/test=11688/1300/2600。Many/Medium/Few=79/148/33 个 domain-class 单元，test 支持数分别为790/1480/330；每 cell 10 张。2 个 train=0 单元保留在 Few 指标中。

### Stage I 选模与运行预算

| Seed | 最早最佳 update | 最佳 val macro (%) | 成功更新数 | AMP 溢出重试数 | Stage I 总耗时(s) |
|---|---:|---:|---:|---:|---:|
| 42 | 4300 | 72.0769 | 5001 | 4 | 299.21 |
| 43 | 3900 | 71.1538 | 5001 | 4 | 291.71 |
| 44 | 5000 | 71.6923 | 5001 | 4 | 291.46 |

每 seed 恰好51次 val 检查（100..5000及5001），重新从日志核验了最早最佳 update。三 seed 合计15003次成功 Stage I 更新；12组 Stage II 每组10 epochs、920次更新、每epoch11688次接收抽样，每类179或180次。每组分类器训练耗时约1.27–1.40秒（不包含特征提取、配对计划和 Stage I）。

正式训练、缓存和最终测试循环共1061.02秒（约17.68分钟）。PyTorch allocator 的 GPU allocated 峰值429.81 MiB、reserved峰值660.00 MiB；不等同于整机或驱动报告显存。

ImageNet 预训练权重实际复用 `D:/torch_cache/hub/checkpoints/resnet18-f37072fd.pth`，SHA256=`f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`。没有随机骨干替代。

### 配对、BN 与真实供体域干预

每组 FC 的33345个参数发生更新；backbone状态、BN的 running_mean/running_var/num_batches_tracked 在 Stage II 逐位不变。所有 BN counter 为0。`classifier_state_changes.csv`记录 FC 相对相应 Stage I 的 L2 参数变化；`../../verification/final_artifact_verification.json`复核了已保存权重、独立初始 FC、源文件哈希和实际批次控制哈希。

| Seed | Matched fallback/116880 | fallback率 (%) | Global跨域率 (%) | Matched跨域率 (%) | Cross跨域率 (%) |
|---|---:|---:|---:|---:|---:|
| 42 | 290 | 0.2481 | 72.7883 | 0.2481 | 100.0000 |
| 43 | 280 | 0.2396 | 72.8662 | 0.2396 | 100.0000 |
| 44 | 271 | 0.2319 | 72.9851 | 0.2319 | 100.0000 |

Cross组没有退回。逐条验证了退回当且仅当对应类别-域候选池为空，且退回到该位置的 Global 供体；没有更改供体类别。DR 的供体只存在于虚拟配对计划，实际融合次数为0。

### CrossDomain 相对 DomainMatched 的逐图主对比

| Seed | 组 | gain | loss | net | 差值(pp) |
|---|---|---:|---:|---:|---:|
| 42 | Overall | 41 | 44 | -3 | -0.1154 |
| 42 | Few | 5 | 8 | -3 | -0.9091 |
| 43 | Overall | 54 | 43 | 11 | 0.4231 |
| 43 | Few | 8 | 7 | 1 | 0.3030 |
| 44 | Overall | 55 | 68 | -13 | -0.5000 |
| 44 | Few | 15 | 10 | 5 | 1.5152 |

Overall 每张图像相当于0.03846pp；Few每张相当于0.30303pp。不能把 seed43 的Few +0.30303pp（净1张）夸大为稳定收益。三 seed重复同一批测试图像，不把7800张当作独立训练重复。

Global−DR 的Few分别为 +2.4242/+4.8485/+3.0303pp，均值+3.4343pp；但Many均值下降1.1814pp，Overall均值只增加0.4231pp，存在分组权衡。

Cross−Matched 的macro分别为−0.1154/+0.4231/−0.5000pp，均值−0.0641pp；Few为−0.9091/+0.3030/+1.5152pp，均值反而+0.3030pp。**宏指标达到预注册的2/3 seed下降这一宽松方向门槛，但Few只在1/3 seed下降，变化方向混杂，未证明跨域供体会稳定损害低样本单元。** Matched−Global 的Few均值为0，无法宣称同域供体带来额外Few收益。

域条件干预确实改变了供体域分布、分类器和逐图预测；当前证据不足以支持一个稳定、可定位的跨域负迁移机制，也不足以支持新增域兼容模块。本轮到此停止，不启动新模块、其他seed或参数搜索。

C/D 在改变域条件时也改变了类内实例身份、候选池规模和重复率；控制供体类别与掩码并不等于仅改变了抽象的域风格因果变量。当前结论限定于本次固定GAP特征、三个骨干seed和官方split。

### 四域准确率（三 seed 等权均值，%）

| Arm | Art | Clipart | Product | Real World |
|---|---:|---:|---:|---:|
| DR-NoFusion | 62.1538 | 77.5897 | 85.4359 | 76.3077 |
| H2T-Global | 63.1795 | 76.4103 | 85.4359 | 78.1538 |
| H2T-DomainMatched | 63.3333 | 77.2308 | 85.5385 | 78.5128 |
| H2T-CrossDomain | 63.5385 | 77.3333 | 85.5897 | 77.8974 |

逐 seed 四域明细见 `per_domain.csv`；260 cell × 12模型的全部结果见 `domain_class_metrics.csv`。

### 最差 cell（保留所有并列项）

| Seed | Arm | 最低准确率(%) | 域/class_id（train数；test数） |
|---|---|---:|---|
| 42 | DR-NoFusion | 0.0 | Art/33 (5; 10) |
| 42 | H2T-Global | 0.0 | Art/33 (5; 10) |
| 42 | H2T-DomainMatched | 0.0 | Art/33 (5; 10) |
| 42 | H2T-CrossDomain | 0.0 | Art/33 (5; 10) |
| 43 | DR-NoFusion | 10.0 | Art/43 (11; 10); Real World/33 (8; 10); Real World/52 (51; 10) |
| 43 | H2T-Global | 20.0 | Art/33 (5; 10); Art/43 (11; 10); Art/50 (5; 10); Real World/43 (44; 10); Real World/52 (51; 10) |
| 43 | H2T-DomainMatched | 10.0 | Art/33 (5; 10); Real World/43 (44; 10) |
| 43 | H2T-CrossDomain | 10.0 | Art/33 (5; 10) |
| 44 | DR-NoFusion | 0.0 | Art/33 (5; 10) |
| 44 | H2T-Global | 0.0 | Art/33 (5; 10) |
| 44 | H2T-DomainMatched | 0.0 | Art/33 (5; 10) |
| 44 | H2T-CrossDomain | 0.0 | Art/33 (5; 10) |

每 cell 只有10张测试图像，最差cell的0%、10%等估计不精确，不能据此新增策略或筛选结果。

### 测试隔离、异常与交付

12个最终模型在测试前均已保存并进行 SHA256 封存；离线再次核对了所有文件hash和mtime。最终测试阶段于 **2026-10-18 21:39:26.739（Asia/Shanghai）** 开始，UTC时间见 `test_access.json`。每个骨干只在此阶段提取test特征，每个固定分类器评估一次；之后的核验只读取保存的预测，没有再次运行图像评估。

编译、10项CPU测试、真实smoke和完整pilot均成功，无训练异常退出。smoke发生1次、每个完整Stage I发生4次默认GradScaler溢出重试；每次复用原增强batch，成功后才计更新，未静默少跑训练。两次额外只读核验命令因自动审批超时未启动；随后用沙箱内只读核验成功，不影响训练产物。

完整stdout保留在本目录 `execution.log`，另存可提交的 `execution_stdout.txt`；编译/CPU日志在 `../../verification/`。根目录新增文件清单与复现命令见 README；实际命令参数见各run `config.json`的argv。官方H2T和历史实验文件不在本轮改动范围内，模型/特征/图像不进入本地代码提交。
