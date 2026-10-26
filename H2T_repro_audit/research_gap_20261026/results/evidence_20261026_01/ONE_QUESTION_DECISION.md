# ONE_QUESTION_DECISION

**判断：STOP_AND_REPAIR_PARITY。** 不是GO；也不是已经证明一切机制问题均被解决。当前最强证据是一个**可定位的复现一致性缺口**，尚不是已确认的发表方法缺陷或新研究空白。

|门槛|状态|依据与限制|
|---|---|---|
|G1 作者机制已验证|PASS（核心公式）|已打开PI v2 p4–5 Eq3/6/7/8与脚注2，原H2T公式/阶段及三篇相关工作全文；W-reference、detach、BN不在核心公式核验之外偷填为确定|
|G2 parity建立或缺口精确隔离|PARTIAL / NOT ESTABLISHED|历史PIF三层25参数vs论文单层2/3明确；review两参数版本与历史训练版本分离。初始化、BN、W-reference、checkpoint作者归属仍未定，完整parity未建立|
|G3 问题可重复且不由实现不一致解释|FAIL|PIF同一StageI checkpoint三StageII seeds存在低准确率，但B>C严格gate失败；单seed endpoint落差仍可能依赖PIF参数化/协议。没有排除复现偏差的反事实|
|G4 后续工作尚未解决该具体问题|NOT ESTABLISHED|PI已处理固定表示/随机通道/手调ratio，且承认过度融合及理论交互；FeatRecon研究支持域，SEL邻类/可靠性，DBG生成边界。没有能跨越parity缺口的一个新颖具体命题|
|G5 单因素、8GB可证伪验证可预注册|WITHHELD|当前首先需要文档/作者配置/代码身份核对；多个未知项同时变化时，训练对比不能被称为单因素因果检验。本轮无显存实测或训练资源承诺|

唯一保留的待澄清问题（**复现问题，不称新颖性已证实**）：

> 历史PIF+H2TF的StageII退化，在严格恢复论文PIF参数化、明确ratio所用W及BN/训练/checkpoint协议后，是否仍成立？

它可被“parity对齐后退化消失”反驳；但这句话不是本轮安排新训练的授权，也不把未对齐时的34.82%视作作者方法的反例。当前只确认参数化不同，**没有证实其就是退化根因**。

本轮不选择TailOff、MarginProtect、boundary-r重命名或新的donor/attention机制，也不同时保留其他随机漏洞。作者原文开放问题不自动意味着我们有新的可发表解决；未主张期刊级新颖性、总体统计显著性或实证提升。

## parity修复所需的证据（不是新实验配方）

先核对作者发布版本/配置与PIF确切层数、bias和初值；明确Eq8 W是当前还是冻结、r是否detach、BN buffers状态、StageI/II实际LR倍率及checkpoint选择；将这些字段和候选checkpoint SHA绑定。若只能得到本地独立重建，应继续标UNVERIFIED，不得改历史输出或强行校准r到.65。任何未来训练须另行授权，且不得以多个未决字段同时改动后准确率上升就宣称发现新机制。

由于G2/G3/G4未满足，没有制定NEXT_EXPERIMENT的arms、阈值或搜索范围；`NEXT_EXPERIMENT_PREREG.md`只记录未注册状态。
