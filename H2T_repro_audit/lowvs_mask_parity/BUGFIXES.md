# 执行前代码审查与工程补丁

日期：2026-10-21；基线提交：edfa6cc5aa5a33b741fa5fd58837500dbd69f931。
所有补丁在首次 unit / preflight / train 之前完成，不根据准确率选择修补。

## 阻塞协议完整执行的问题与修补

1. 原 preflight 仅做数组/文件检查，无真实GPU forward/fusion、冻结状态或新模型重放证据。新增官方TRAIN源文件已知SHA校验、作者seed0 LT有序索引重放、全量LT像素/标签及TRAIN留出ID不重叠核验；对两个seed新加载两次StageI，验证实际前向、CE有限、16通道整图替换及13字段重复一致。preflight不调用backward或optimizer.step。
2. 原入口未保存通过记录/锁定环境，train可以绕过preflight。新增unit.json/preflight.json、源文件与环境锁；每臂首批须与本次preflight一致，并按预注册8臂顺序执行。没有对历史不同环境的特征哈希提出匹配要求，也没有豁免本轮新对照的任何匹配。
3. 原错误仅在终端打印。新增逐命令UTF8完整console、FAILED_STOP.json及配对实际/期望字段记录；已有输出或失败标记拒绝重跑。新增运行期写保护与官方TEST多路径保护。旧目录只读。
4. 原记录缺有效配置、环境、源码、分类器变化、BN哈希、显存、逐epoch耗时/CE及fallback计数。补齐证据；best/final仍按原规则，final保存logits逐位重载核验。只更新原6500参数FC，不增加辅助损失。
5. 原paired报告遗漏Few和per-class delta，未输出掩码实际差异、组成与曲线。补齐这些预注册输出，并检查保存预测==logits.argmax及指标一致性；所有主表仅来自8个本轮新模型。
6. 原非GO标签为NOT_GO_OR_INCONCLUSIVE。报告将未通过原联合门槛明确标记为NO_GO_FOR_PREREGISTERED_SCREEN，方向分歧和代价单列保留不确定性；不把未通过筛选解释成证明方法普遍无效。

## 不改变的科学定义

mask_policy.py、test_mask_policy.py、PROTOCOL.md、原任务文件均不修改。Shared/Individual的评分聚合、Bottom48、16个完整通道、rho0.25、fallback和基础Gumbel不变。原BS/IS、增强顺序与seed+300009、StageI checkpoint/统计/split/plan、SGD/CE/cosine、10epoch/850步/seed124与125不变。
cuDNN/TF32等实际设置只记录与锁定，未主动更改；没有用调整CUDA设置豁免失败的配对哈希。
新增日志/哈希/统计/画图不抽取额外训练随机数；选中掩码仍由原choose_masks和原online_masks产生。

## Diff及证据

完整差异见 ENGINEERING_FIX.patch；修补前原始入口保存在 results/source_snapshot/run_parity_before.py（本地，不提交重复源码快照）。辅助审计文件 audit_support.py 只负责核验与报告。
只有本次目录新增/修改；不触碰旧源、报告或checkpoint。没有运行失败后重跑或更换seed的隐式恢复路径。

## 发布范围

只提交本次小型源码、协议/修补说明、console、配置、epochs和compact CSV/JSON/MD及小型曲线图。850行逐batch签名、模型、actual_masks和源图像/特征缓存保留本地；完整配对字段的850/850计数保存在pairing_verification.csv，原始文件路径和SHA可复查。

## 核心定义文件SHA256（执行前）

- `mask_policy.py`: `2bfeaec984668c672de156eab5b713a68b72ec5360cf1601c5dcede657143c36`
- `test_mask_policy.py`: `3797451ffee9e8d0dfea1b19034001d9351bf79b7c61c02e22df8ee50202ebac`
- `PROTOCOL.md`: `b16a73013dc456e045a1f30553c5cf0cfbf2f2a9ff9c8a6f3b7996a0c32d27df`
- `CODEX_LOWVS_MASK_PARITY.txt`: `c0db942b60fb50eeab2463842db72a10fa9f8850fe1fee329ead5d5cff39996f`
