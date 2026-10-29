# H2T 在真实人工错标长尾数据中的失效场景筛查

本分支仅测试 **原始 H2T 与同源 No-Fusion 在真实人类标注错误叠加长尾分布时的最终泛化收益差异**；不修改原始H2T，不做新donor/mask/loss模块，不使用冻结margin代替训练效用。

**文献出处**：H2T（AAAI 2024）https://ojs.aaai.org/index.php/AAAI/article/view/29262 ；CIFAR-100N（ICLR 2022）https://github.com/UCSC-REAL/cifar-10-100n ，使用官方人工fine noisy_label。全数据约40.2%噪声率，必须在IF100子集重新计算实际比例。

## 数据准备

先在本地将：
- 原始Python CIFAR-100 TRAIN五万张的fine labels → `raw_clean_labels.npy`
- CIFAR-100N官方 `CIFAR-100_human.pt`经可信安全加载/转换为纯整数 `human_labels.npz`，键 `clean_label`/`noisy_label`
- 已审计IF100的raw TRAIN ID和不相交的TRAIN holdout → `split.npz`，键 `train_ids`/`val_ids`

运行：

```powershell
python H2T_repro_audit/noisy_lt_failure_screen/prepare_protocol.py --original-labels raw_clean_labels.npy --human-labels human_labels.npz --split split.npz --output local_noisy_protocol
```

先检查 `DATA_AUDIT.json`（注意观测标签类频变化/空类），确认完整50k标签逐项对齐，才能开始训练。干净/noisy训练集**图像ID完全相同**，noisy训练和采样不得读取干净GT；清洁TRAIN holdout只用于固定epoch评估。

## 成对训练

Codex的完整执行说明在 `CODEX_CIFAR100N_LT_FAILURE_SCREEN.txt`。

- StageI seed42从相同初始化各训一组200 epochs：clean/noisy，固定最后epoch。
- StageII seeds124/125每个Regime分别训DR(NoFusion)及原H2T，各10epochs，StageI来源/FC初始参数/receiver/donor/BN前向协议配对；共8次StageII。
- 主要指标：在同一干净TRAIN holdout上固定最后epoch的Overall、Many、Medium、Few及每类准确率，`Interaction=(H2T−DR)_noisy−(H2T−DR)_clean`。
- 当noisy观测标签类缺失导致原始class balanced sampler无法原样运行时必须停止，不能oracle补充。
- 这是**有真实人工噪声的场景整体实验**，标签噪声也会改变observed类频和取样，不允许断言是单一的监督目标因果机制。

## 最终结果表

生成CSV，字段恰为

`stage1_seed,stage2_seed,regime,method,group,accuracy_pct,n,stage1_sha,val_ids_sha,epoch`

regime=clean/noisy，method=no_fusion/h2t，group=Overall/Many/Medium/Few，epoch必须是10。

```powershell
python H2T_repro_audit/noisy_lt_failure_screen/compare_outcomes.py --csv results_input.csv --output results/new_run
python -m unittest discover -s H2T_repro_audit/noisy_lt_failure_screen -p "test_*.py" -v
```

严格门槛：两个StageII seed中Few的noise interaction都不大于−2pp，且noisy H2T Few均低于noisy DR，才标记为“有待新StageI seed独立确认的候选失效场景”。不满足就停止这条线，不追加噪声强度或模型改进。

本次提交的是可运行的数据协议审计、配对结果汇总和最小测试；**完整训练适配由Codex读取TXT后按现有原始训练源码接入，尚未在本轮执行真实训练**。
