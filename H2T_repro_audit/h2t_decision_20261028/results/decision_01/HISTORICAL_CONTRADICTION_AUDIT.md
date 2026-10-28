# H2T 历史机制证据整合与反证审计

日期：2026-10-28；仓库：LuoQiuXu/xin；工作分支：`research/h2t-decision-ledger-20261028`。

**科学结论：STOP_NO_DEFENSIBLE_GAP。14/14 登记记录已覆盖，但“有记录”不等于“14 项均完成实验”。** 本轮读到了各登记分支的固定 commit 中的真实文本产物，并交叉核查指标 CSV、配置、日志声明及实际算子/调用路径；没有训练、模型推理、checkpoint 反序列化或数据集访问。不能把这次文本审查说成重新验证了历史模型参数。

标准本地 `analyze_ledger.py` 因四项材料不在当前工作树而按原规则失败。按任务 TXT 第 3 步，另行用 GitHub 在线 branch heads 和 `git show <commit>:<path>` 追溯来源：14 个登记路径全部存在。保留 [LOCAL_GATE_BLOCKED.json](LOCAL_GATE_BLOCKED.json)，没有编辑登记路径或将标准门禁改写为 PASS。完整 branch、40 位 commit、路径、原始字节 SHA256 和 URL 在 [BRANCH_SOURCE_INDEX.csv](BRANCH_SOURCE_INDEX.csv)；105 份主记录/支持文件的 Git blob 校验见 [SOURCE_MANIFEST.json](SOURCE_MANIFEST.json)，另有 3 份 Office-Home 上下文见 [ADDITIONAL_SOURCES.json](ADDITIONAL_SOURCES.json)。

## 1. 判定口径

- `SUPPORTED`：只在明确标出的模型、输入分布和结局范围内受到支持，不代表机制因果链已闭合。
- `REJECTED_SPECIFIC_RULE`：给定规则没有通过原预设门槛；不等于所有梯度、边界或选择思想均被否定。
- `UNSUPPORTED`：强解释没有建立；同时可以存在受到支持的较弱描述性事实。
- `PROTOCOL_ONLY`：已登记问题/代码，但所查分支没有可核验的结果，既不是正结果也不是负结果。
- `BLOCKED`：完整问题受 parity 或重放问题阻断，不用于科学效应方向判断。

以下百分数是历史准确率，差值单位为百分点（pp）。固定 epoch 10 为主，best 仅作为选择风险说明。表中 `124/125` 始终按这两个 Stage-II seed 的顺序排列。详细分类及协议字段见 [EVIDENCE_REVIEW.csv](EVIDENCE_REVIEW.csv)。

## 2. 14 项记录逐项核对

|记录及固定来源|实际问题、主对照和真实产物|结论与不能跨越的边界|
|---|---|---|
|[H2T_BN][H2T_BN]|配对、BN 全冻结下 H2T 对 DR：Overall 47.25 对 46.49，+0.76；Tail 26.138 对 23.690。两臂 `epochs.csv`、850 批输入校验和冻结状态日志均存在。|**SUPPORTED，条件性分类器训练效应。** 一个 Stage-I seed123/epoch198、一个 Stage-II seed123、官方 TEST 10000 张；不能由此推出 BN 的独立贡献或 Backbone 表征改善。|
|[DONOR_K4][DONOR_K4]|固定 DR epoch10 reference；每个接收 draw 的同一供体类抽 K=4 个实例，掩码相同。20000 组中 5404 组融合判对/判错并存：27.02%；Tail 为 1844/5792=31.8370%。`analysis_summary.json`、采样表和核验日志对应。|**SUPPORTED，仅即时诊断。** 全体只有 6611 张不同接收原图，Tail 286 张；80k candidates 不是 80k 独立样本。以真实标签 margin 排序后的 Low-D 优势不证明训练效用。|
|[MASK][MASK]|Shared 新训练，PerSample/Low-VS 复用旧结果；Shared−PerSample Overall +0.2525/+0.1010，Tail +0.1724/0.0000。`all_three_epoch10.csv`、`pairing_hash.json` 和继续授权记录真实存在。|**UNSUPPORTED，若命题是孤立的掩码粒度因果效应；描述性差异存在。** 每 seed 输入/索引/Gumbel 850/850 一致，但历史 features/GAP/scores 均 0/850，联合 masks 438/850。不得声称是严格单因素完整配对。|
|[LOWVS_NORM][LOWVS_NORM]|同一新面板的四臂，逐样本配平 GAP 扰动 L2。Full Low−U Overall +0.2525/+0.3030，Tail +0.8621/+1.0345；Matched 变为 Overall +0.2525/−0.1515、Tail +0.5172/−0.1724。原表及 `paired_verified.csv`、范数和 W/b hybrid 表齐备。|**UNSUPPORTED：“Low-VS 有稳定且独有的更安全判别方向”。** Matched 两 seed 变号；均值仅 +0.0505 Overall、+0.1724 Tail。不能从 Full−Matched 差计算“多少收益由范数造成”。|
|[GU][GU]|Full/Half/GU/Shuffled 八组真实训练记录。GU−Full Overall −0.8081/−0.6566，Tail −3.9655/−4.1379；相对 Shuffled 也为负。`epoch10_verified.csv`、预设规则和 action histogram 表存在。|**REJECTED_SPECIFIC_RULE。** 以 TRAIN balanced CE 梯度对齐、每 epoch 更新 reference、在 0/.5/1 三档贪心选强度这一规则 NO-GO。不能否定所有增强效用估计；Shuffled 只保持自身 batch 的直方图，跨臂 20/20 epoch 均不相等。|
|[RAVS][RAVS]|receiver 频次与当前 margin 困难度调 Bottom-K 池，替换量仍为 16/64；十组结果。RAVS−LowVS Overall −0.1010/−0.4040，Tail −0.1724/−0.6897；相对 FrequencyOnly 两 seed Overall 均 −0.1010、Tail 均 −0.1724。|**REJECTED_SPECIFIC_RULE。** 两个原门槛都未过。best 的顺序反转不能替代固定终点；不再调 sigmoid、K 或改名风险筛选。|
|[IF200][IF200]|新 Stage-I 固定 epoch200；四臂、两个 Stage-II seed、BN 冻结。Shared 对 DR Overall 42.8283/42.6263 对 41.6162/41.8687，Tail 17.9310/17.9310 对 16.7241/16.8966。`epoch10.csv`、`paired.csv` 和数据核验记录一致。|**UNSUPPORTED：“提高 IF 本身就令 H2T 失效”。** 此面板 H2T 式融合的 Overall/Tail 两 seed 都正向；不等于所有 IF 都有效，也不是 IF100→200 的同协议因果效应。作者 IF200 `shift_bn=True`，此面板固定 BN。|
|[BOUNDARY][BOUNDARY]|冻结 Stage-I head，重放 epoch1，每 seed 10847 draws。TailEligible n=3064/2986、各 286 张不同原图；单位扰动方向 q 的 Low−U 均值 −0.037316/−0.040377，matched CE +0.175334/+0.175349。`group_summary.csv` 和重放核验存在。|**UNSUPPORTED：“即时 margin 可指认最终有益融合”；几何现象本身 SUPPORTED。** 只测 TRAIN 增强输入，不训练；更负 margin 可表示挑战，也可表示标签冲突。Head 同样存在，不能称 Tail 特异机制。|
|[CLEAN_CE][CLEAN_CE]|A=LowVS，B=加 Tail clean CE，C=加 Tail top-competitor softplus。首要 C−B Tail −0.5172/−1.0345，Overall +0.2020/+0.1515；尾类净正确数 −3/−6。六组表、配对表和损失源码一致。|**REJECTED_SPECIFIC_RULE。** 特定竞争监督未优于普通 clean CE。B−A Tail 两 seed均 +1.8966，但 Overall −0.6061/−0.6566，是代价交换。相同 λ=.5 不等于相同梯度预算。|
|[OFFICE_RISK][OFFICE_RISK]|固定三套 epoch10 H2T-Global heads；每张 VAL 图 4 candidates、3 条 donor domain 条件。1300 VAL ×4×3 arms×3 seeds=46800 替换。方向分数 AUROC 0.82053–0.83973；Cross−Matched ΔCE +0.048448 nats。summary、预测指标 CSV 和实现一致。|**SUPPORTED，仅冻结分类器下即时 CE 风险排序。** 相同 receiver/donor class/mask，不同 donor identity；domain 效应未被完全隔离。bootstrap 区间只条件于三套已训练模型，不能外推未来训练 seed；此 PASS 不是本轮 GO。|
|[PARITY][PARITY]|旧研究决策 `STOP_AND_REPAIR_PARITY`。历史 25 参数 PIF 与论文 2/3 参数不同；旧 PIF A/B/C 的 B−C 为 +.01/−.14/+.22，未过全 seed gate。对照原始表和 prior-evidence 文档。|**BLOCKED，作者方法归因受 parity 阻断。** 旧下降是真实本地现象，不是已发表方法缺陷；参数数目不一致也没有被证实就是下降根因。10/27 后续见第 5 节。|
|[TARGET][TARGET]|三臂已预注册：hard recipient、.75 receiver+.25 donor target、冻结 cRT 首次 crossing competitor target。该 scope 只有 5 个协议/策略/测试文件，**results 文件 0**。|**PROTOCOL_ONLY。** 未验证结果，参考 cRT 来源还有硬门禁；CLEAN_CE 的辅助损失负结果不能替代此软目标问题的结果。也不能将已有预注册问题重命名为新问题。|
|[RECEIVER][RECEIVER]|IF200 TailOff−逐 batch 精确计数 RandomOffMatched 的四臂问题已登记。该 scope 只有 5 个协议/runner/测试文件，**results 文件 0**。|**PROTOCOL_ONLY。** 既不支持 TailOff，也不否定它；已问过同一反事实。剂量匹配只控制全局总次数，不能自动控制 class-conditional SGD 效应。|
|[SEMANTIC][SEMANTIC]|`FAILED_STOP.json`：seed124/epoch10/batch0，indices/images 通过，features SHA 不匹配，原实际哈希为 null。后续主脚本持久化修复和静态诊断存在，但登记分支没有完整成功语义结果。|**BLOCKED，完整审计仍无有效结局。** 不把失败视为语义风险不存在/存在。另发现本地受控单批次日志，须单独标注来源及有限结论，见第 5 节。|

## 3. 协议不能合并为一条“性能进展曲线”

|证据家族|Stage I 与模型|数据和统计单位|Stage II/推理与选择边界|
|---|---|---|---|
|原复现 / BN / K4 donor|ResNet32；seed123 best epoch198；SHA `0cf3e67a87f2a8c45f4470fadc06c3d7c47796c691797d21ed6c2fa8c6b7033e`|IF100 TRAIN 10847；BN 结局来自官方 TEST 10000；K4 只用 TRAIN，重复 BS draws|BN 对照仅一个 Stage-II seed123；donor 用 fixed-BN DR epoch10 head SHA `1b054d8b45753c25dedd9897e3052e5f33c73b0c38b47acaa277c0f61c50283d`，不更新模型|
|MASK / NORM / GU / RAVS / BOUNDARY / CLEAN_CE|Clean ResNet32 seed42、开发验证选 epoch184；SHA `7ced9d07f71335357c08623573f038acd0dfdcdeadb55d95d57b332a14f83a88`|IF100 TRAIN 10847；反复使用的 TRAIN holdout 1980，class1–99 各20，class0 缺失；BOUNDARY 仅 TRAIN draws|大多数是同一 Stage I 的 Stage-II seeds124/125，10×85 更新，SGD 有效初始 LR .02、momentum .9、wd .0005；31 BN eval，只有 6500 FC 参数；不等于多个独立 Stage-I 复现|
|IF200 / RECEIVER 协议|新 ResNet32 seed42、固定 epoch200；SHA `5abef38180bf85cd5c60c38bae08efe23173bdf4775fe44bc38299458448548f`|IF200 TRAIN 9502，最少2/最多500；仍用相同 1980 开发验证|IF200 已完成 2×4 臂、10×75 更新，wd .0002，其余主要 FC 优化沿用作者 IF200 设置；RECEIVER 只有协议，不能加到已完成组数|
|Office-Home pilot / risk|ImageNet 预训练 ResNet18，65 类，GAP 512 维；seed42/43/44 三套不同 Stage-I SHA 见原 lock|4 domains；TRAIN 11688、VAL 1300（每 domain-class 5张）、TEST metadata 2600。risk 只读 train/val 缓存；pilot 曾记录锁定12个终点后单次 TEST|Stage I Adam 5e−5、5001 updates、VAL 选模；Stage II frozen cached GAP、SGD .01、10轮。risk 固定 H2T-Global head、没有更新，不能与 CIFAR 准确率直接比较|
|PI 旧重建 / 10/27 新重建|旧25参数 PIF；新2参数 PIF为新训 seed123 固定 epoch200，SHA `64f53e45a1b7f73f494f9c88374607e714b58dc6eb55118fa31154c4389f93ef`|历史均已使用官方 CIFAR TEST；旧 CE/PIF 两 setting 还涉及不同 shortcut/checkpoint/selection|旧多个 Stage-II seeds 共用各自一个 Stage I；新仅 Stage-II seed42，一对配对臂，BN update。不能和固定 BN LowVS 合并归因|

Head/OfficialMedium/Tail 的 ID 划分是 [0,36)/[36,71)/[71,100)。1980 集有效 Head/Medium/Tail 分母是 700/700/580；这里 Overall 等于 **99 类** macro，不是 100 类 macro。频次 Many/Medium/Few 另由训练样本数定义：IF100 有效分母 680/700/600；IF200 为 580/620/780，不能拿 Few 代替 Tail。官方 TEST 对应 ID 三组分母为 3600/3500/2900。Office-Home 还有 domain-class 分组，均不得混算。

BS/IS 描述的是抽样概率，不是类别硬过滤：原 H2T 的 receiver/donor 均可来自100类，receiver 并非只限 Tail，donor 也并非必为 Head。冻结特征的 Stage II 只改变分类头；在该设置中说“Tail Backbone 表示被改善”超出了实际训练路径。

## 4. 看似矛盾的证据，实际上检验了什么

1. **“H2T 都是 BN 收益”不成立，但 BN 独立贡献也未识别。** 原作者式运行 epoch10 H2T 47.87、DR 47.00（+.87）；fixed-BN 配对仍 +.76。两个面板同时改变 BN、通道 RNG 与数据流耦合方式，缺少“BN update + 同样隔离 RNG”的交叉臂。因此不能把 .87−.76=.11pp 当作 BN 中介量；也不能说 BN 在任意 PI 融合设置下无关。原 TEST 已经用于历史查看和 best 选择，不是新终检。

2. **K4 donor 的多样即时响应不等于有一个低风险供体策略会提高训练精度。** K4 在同一类内、同一 mask 改 donor instance，说明实例差异确实影响冻结头的结果；BOUNDARY 进一步显示范数配平后仍有方向差。二者都没有更新分类器。真实标签参与 margin/CE，Low-D 判对优势部分来自排序定义本身；“更难”与“更有益”不能互换。

3. **MASK 的历史重放失败必须与后续严格面板分开。** MASK 的继续授权保存了失配，没有产生严格一因素证据；NORM、GU、RAVS 等各自的新面板有自己的输入与特征匹配日志。不能用后者追溯升级旧 MASK 的证据，也不能由新面板成功推定所有旧表已精确重放。新 Shared 的 Gumbel 第一行平铺与作者 `torch.randperm` 也不是逐随机数同程序。

4. **“Full 正、Matched 不稳”与“单位方向更扰边界”兼容。** q 衡量固定 Stage-I head 的局部投影；epoch10 准确率来自更新后的 W,b。GAP 范数匹配会同时收缩不同通道位置，完整 feature-map L2 并未匹配。不能用静态线性 margin 分解来分摊最终准确率收益；也不能因为范数配平后的方向差仍在，就宣布它能带来训练收益。

5. **W/b hybrid 没有排除广义偏置/校准解释。** NORM 的四项 Tail 比较只换 b 都是 0pp，只换 W 可复现相应 Tail 变化。这排除了这几个保存终点中“直接换 b 就解释 Tail 差”的狭义说法；W 同时包含类尺度、范数、方向和训练轨迹，仍可承载偏置。也没有证明 b 在联合优化中无作用。不能把“变化在 W”写成“已证实新增语义”。已有 [bias-vs-discrimination 协议][BIAS] 只有提案，不能作为新颖问题或已验证反证。

6. **GU NO-GO 与 Office risk PASS 没有证据冲突。** Office 分数是输入空间一阶量 `∇f CE(Wf,y)·δ`，结局也是同一个 W 下的 CE 差；GU 比较候选参数梯度与 TRAIN balanced reference gradient，结局为多步 SGD 后开发验证精度。模型、数据、特征层、reference、时间尺度均不同。Office 的自有筛选门槛 PASS 不能代替本轮 G2/G3；GU 的失败也不能反证 Office 已保存的 AUROC。

7. **GU 的 Shuffled 不是严格的跨臂剂量相等。** 两臂根据各自演化的 head 重新选 action；shuffle 只保持本臂选择前后的 batch 直方图。原 CSV 中两个 seed 共20个 epoch 的跨臂直方图无一完全相同。故“GU 对 Shuffled 负”否定的是整条给定规则，不是已完全隔离的分配身份效应。

8. **IF200 不能支持“终于找到了 H2T 失败压力”。** Shared−DR Tail 净正确数为 +7/+6，Overall 为 +24/+15。少量反复受损类别只有每类20张，不能后验拿 class72/77/79/87 定义干预。IF100 和 IF200 的数据、Stage-I 训练、checkpoint 选择及 wd 不同；不能跨面板声称 IF 是唯一因果因素，也不能继续扫 IF 寻找失败点。

9. **CLEAN_CE 的 Tail 增益有真实 Head/Overall 代价，且不等于 TARGET。** B 比 A 的 Tail 净增11张/seed，同时 Overall 净减12/13张；减少 Tail→Head 还包括“从错误 Head 转到另一个错误类”。C−B 的 Tail 负结果否定该 softplus 辅助项，未检验 TARGET 的单次软目标 CE；两者不能互相冒充成功或失败。

10. **开发集重复使用限制所有很小的差值。** 1980 总体每张 0.050505pp，580 Tail 每张 0.172414pp，每类每张 5pp。NORM Matched 两 seed Overall 净 +5/−3、Tail +3/−1；RAVS−FrequencyOnly 每 seed Overall −2、Tail −1。重复的同一模型、同一 holdout、相同 Full 端点是可复算性证据，不是新的独立泛化重复。没有合并 seed×图像作显著性或把 best 改成主终点。

## 5. 两项必须保留的后续事实

### PI-H2T：两参数重建仍退化，但作者 parity 仍未建立

10/27 [独立重建报告][STRICT] 实际完成 200+10+10 轮：Stage I 39.91%，no_fusion 45.53%，H2TF 36.50%，配对差 −9.03pp。此次重新加总已保存的100行逐类计数，得到4553/3650正确、796 wins/1699 losses，数字一致；10轮输入流 SHA 日志也全部相同。它排除了“历史所有退化都只能用25参数错配解释”这个过强说法。

但 **它仍不排除 W-reference、detach、BN、初始化、shortcut、训练配置和 checkpoint 选择的作者协议差异**。10/27 [作者来源审计][AUTHOR] 的结论为 `BLOCKED_NEEDS_AUTHOR_CONFIRMATION`；完整 PI 训练代码未核实，公开片段与活动调用路径不一致。此次 GitHub heads 复查仍只有 `main@e0c849235d9655fca31cd9fa8a5f38f92a77c7e8`。作者论文 r≈.65 是 receiver 权重，本地约 .77208 是 donor mass，不能混同，更不能将 .65 当调参目标。根因仍 **UNDETERMINED**，不称作者方法失效。

### SEMANTIC：本地单批次 TF32 诊断不能冒充完整审计恢复

在受保护的本地未跟踪历史目录中读到 `single_batch_compute_20261025_01` 的已有 JSON/报告；没有运行它，也没有复制或暂存历史文件。其来源未出现在登记 commit 或本地可达 Git 历史，故 [LOCAL_SUPPLEMENT.json](LOCAL_SUPPLEMENT.json) 明确标记 `LOCAL_UNTRACKED_READ_ONLY`，没有虚构 GitHub URL。

这四组日志在同一 seed124/epoch10/batch0、相同两次 warmup 下记录：A/B（cuDNN TF32=True）为 `e5dc4ee26962c028fdb137d398a4ae67f5886e918cde2203a60c3b106e1e034d`；C/D（False）为期望 `7d111aeb5a0dc30cf76af609ffb251fc4152c0c38f80baaa01ae0e3862eafbb5`，deterministic 开关不改变该批输出。记录均为有限 float32 [128,64,8,8]。这支持 **该受控批次里 TF32 足以造成/消除差异**，不是“原因全无证据”。

原失败进程实际哈希仍未保存，且没有 warmup；C/D 只覆盖一个 batch，没有完整语义统计。不能反填原 null、断言原进程已测得 A 哈希、宣称所有 batch 恢复、或据此解释 MASK 的全部历史失配。登记完整审计维持 `BLOCKED`；其科学结局不是负结果。本轮未放宽任何 SHA 门禁。

## 6. 去重决定

停止将 LowVS 方向、RAVS 困难度、GU 强度、RiskOff、供体判别风险及 TailOff/target 旧提案改名后重新立项。仍未知的内容是：即时扰动如何转化为多步分类器收益、冻结特征收益可由多少校准解释、作者 PI 完整协议是什么。这些未知没有自动通过 G1/G2/G3。最终门禁和后续边界在 [ONE_QUESTION_OR_STOP.md](ONE_QUESTION_OR_STOP.md)，文献重叠及不等价之处在 [MECHANISM_NOVELTY_MATRIX.md](MECHANISM_NOVELTY_MATRIX.md)。

[H2T_BN]: https://github.com/LuoQiuXu/xin/blob/d52f8c52a836b75717e2ad949bcd458ef4860c75/H2T_repro_audit/bn_audit_runs/fixed_bn_20261011_2255/H2T_BN_AUDIT.md
[DONOR_K4]: https://github.com/LuoQiuXu/xin/blob/d52f8c52a836b75717e2ad949bcd458ef4860c75/H2T_repro_audit/donor_audit_runs/donor_perturbation_20261012_phase1_seed123/H2T_DONOR_PERTURBATION_AUDIT.md
[MASK]: https://github.com/LuoQiuXu/xin/blob/e288c98852ffd02628fb86f40d70ad1f0073d17e/H2T_repro_audit/mask_granularity_iteration/results/MASK_GRANULARITY_RESULTS.md
[LOWVS_NORM]: https://github.com/LuoQiuXu/xin/blob/bef8adc8a47f5714bddd896963c15aceeeb5eecd/H2T_repro_audit/lowvs_norm_direction/results/LOWVS_NORM_DIRECTION_SCIENTIFIC_REPORT.md
[GU]: https://github.com/LuoQiuXu/xin/blob/6247394a36d95bd98c796d042b51774911e125f2/H2T_repro_audit/gu_lowvs/results/GU_LOWVS_SCIENTIFIC_REPORT.md
[RAVS]: https://github.com/LuoQiuXu/xin/blob/82c94109ad3073b2e459ef5186c601f29a67b81d/H2T_repro_audit/ravs_v1/results/RAVS_V1_SCIENTIFIC_REPORT.md
[IF200]: https://github.com/LuoQiuXu/xin/blob/fcfdb3e47ccba553c2ae07a1456e8e48fadf96e0/H2T_repro_audit/if200_stress/results/IF200_STRESS_SCIENTIFIC_REPORT.md
[BOUNDARY]: https://github.com/LuoQiuXu/xin/blob/cc627c1dfbdd5983feae20d323c84133fc01a822/H2T_repro_audit/lowvs_boundary_direction/results/BOUNDARY_DIRECTION_SCIENTIFIC_REPORT.md
[CLEAN_CE]: https://github.com/LuoQiuXu/xin/blob/f7387942489a2db23ed59b4097b828bb8d5c2e81/H2T_repro_audit/lowvs_disc_preserve/results/LOWVS_DISC_SCIENTIFIC_REPORT.md
[OFFICE_RISK]: https://github.com/LuoQiuXu/xin/blob/cce7ad6b9b1447c0b5b5452b409079f939520efc/H2T_repro_audit/mdlt_domain_probe/results/transfer_risk_probe_20261019_01/report.md
[PARITY]: https://github.com/LuoQiuXu/xin/blob/d52f8c52a836b75717e2ad949bcd458ef4860c75/H2T_repro_audit/research_gap_20261026/results/evidence_20261026_01/ONE_QUESTION_DECISION.md
[TARGET]: https://github.com/LuoQiuXu/xin/blob/2a2f8e4b91d80794f0edb27e868b917429abd48c/H2T_repro_audit/lowvs_target_semantics/PROTOCOL.md
[RECEIVER]: https://github.com/LuoQiuXu/xin/blob/50da70fec95a565b3f585b624ed2cc8490b34a61/H2T_repro_audit/receiver_selectivity/README.md
[SEMANTIC]: https://github.com/LuoQiuXu/xin/blob/592655371c52354b3a3d02983a7fc96f8ba74855/H2T_repro_audit/fusion_semantic_audit/results/FAILED_STOP.json
[STRICT]: https://github.com/LuoQiuXu/xin/blob/049f91158fca3de754f337f8e3ac0d41d7c5af7a/H2T_repro_audit/pi_h2t_strict_20261027/runs/pih2t_if100_seed123_42_v1/SCIENTIFIC_RESULTS.md
[AUTHOR]: https://github.com/LuoQiuXu/xin/blob/c30a36cf3d14415690d0834ab91c9f27aee52385/H2T_repro_audit/pi_h2t_author_audit_20261027/results/source_gate_01/STRICT_REPRODUCTION_DECISION.md
[BIAS]: https://github.com/LuoQiuXu/xin/blob/9273b9d7c8083349a7599ef23a7f591dc448f7b9/H2T_repro_audit/h2t_bias_discrimination_20261028/README.md
