# H2T 科研主线去重与下一步决策门禁（2026-10-28）

**这不是新方法实验；本轮绝不训练模型。** 目标是从已有十余项实验中找到科学上仍成立、未被现有方法覆盖、且可以小成本证伪的单一问题；如果找不到则明确 STOP，而不是重新设计 RiskOff/LowVS/Donor 规则。

## 已知事实与核心风险

- 已完成原始 H2T CIFAR100-LT IF100 Stage I/II 复现；原作者式 StageII fixed epoch 10 H2T 47.87%，DR 47.00%（官方 TEST 历史已使用，禁止继续用它择优）。
- 固定 BN、配对输入下 H2T 47.25%，DR 46.49%，是特征替换对分类器轨迹的条件性效应，不等于已证明为什么有效。
- LowVS 范数匹配后差异小且跨种子变号；GU/RAVS 预注册 NO-GO；IF200 不支持提高 IF 就必然失效。
- 历史的原始 H2T 官方 TEST、反复使用 TRAIN 留出1980张（缺 class0）、不同BN设置与多个 StageI checkpoint，不能混合做新显著性主张。
- PI-H2T 作者公开库的完整训练实现还未证实；历史独立重建的退化不能宣称作者方法失效。
- 某些分支只有预注册脚本而没有结果，或 replay 严格失败；不能将它们计为已经否定或支持假设。

## 本轮三个门禁

G1 **重复性**：是否已做过相同干预、相同结局和相同关键对照？逐个新候选对比 `EVIDENCE_LEDGER.json`，回答“新在哪里、不是换名字”。

G2 **科学证据**：是否存在稳健的现象，且排除了 BN、采样、掩码粒度、扰动范数、分类头校准/预测类比例与验证选择等更简单解释？若没有，不得创造新的失效故事。

G3 **研究新颖性/可证伪性**：原始 H2T、PI-H2T、SEL、FeatRecon、DBG、Mixup选择和近期同题工作有无相同机制？提出一个 **最小、可使假设失败的**反事实及可以独立评估的终点；没有就 STOP。

## 执行

仅需标准 Python，不访问网络、数据集、checkpoint、torch 或 GPU：

```powershell
python -m unittest discover -s H2T_repro_audit/h2t_decision_20261028 -p "test_*.py" -v
python H2T_repro_audit/h2t_decision_20261028/analyze_ledger.py --output H2T_repro_audit/h2t_decision_20261028/results/decision_01
```

只有本地历史证据文件存在，脚本才会建立新结果目录；哈希验证只证明“读到了该文件”，不是论文意义上的科学证明。若历史缺失，请报告缺失并停止，不要编辑字段来让程序通过。Codex 还须逐份阅读历史报告、检查原论文后写出独立决策。

## Codex 的唯一有效交付

- `HISTORICAL_CONTRADICTION_AUDIT.md`：整理假设、结果、存在的混杂、协议差异、停止条件和证据 URL，不重复堆表。
- `MECHANISM_NOVELTY_MATRIX.md`：原始H2T/PI-H2T/SEL/FeatRecon/DBG/近期论文，与每一候选研究问题的具体重叠点、缺失点和来源。
- `ONE_QUESTION_OR_STOP.md`：只允许 GO_ONE_QUESTION / STOP_NO_DEFENSIBLE_GAP / BLOCKED_EVIDENCE 三种结论；GO 需要一句严谨假设、反例、主终点、一个最小实验与无需重训StageI的验证可能性，但本轮不得实际训练。
- `EXECUTION_REPORT.md`：命令、退出码、哈希、改动、报告质量、未训练声明。
- `CODEX_NEXT_EXPERIMENT_DRAFT.txt` **仅在 GO 时**草拟下一轮用户另行授权的最小实验；当前禁止运行。

## 保护

全部历史 Git 分支/训练结果/checkpoint 不改；只将本目录小文件提交到本分支。不得创建另一个分支、PR、合并或清理用户工作树；不得为结论好看而更改 G1/G2/G3。
