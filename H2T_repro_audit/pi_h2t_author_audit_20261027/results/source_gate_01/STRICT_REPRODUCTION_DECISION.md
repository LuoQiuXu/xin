# 严格复现科学门禁结论

**最终决策：BLOCKED_NEEDS_AUTHOR_CONFIRMATION。当前不具备声称“严格复现作者PI-H2T”的条件。本轮作者来源审计已完成，没有GPU训练，后续训练门禁保持关闭。**

## 支持结论的第一手证据

1. arXiv v2 p1的源码链接可确认指向Keke921/H2T；但固定[main@e0c849235d9655fca31cd9fa8a5f38f92a77c7e8](https://github.com/Keke921/H2T/tree/e0c849235d9655fca31cd9fa8a5f38f92a77c7e8)的README仍自述AAAI2024。已实时核查全部公开heads/tags、releases、完整58项树及79条可达提交，没有据仓库名称判定实现。
2. 公开代码确实包含PIF/H2Tv2片段，不能说完全没有PI相关代码。PIF片段有25参数、非法Conv2d参数和丢弃out_hat的问题；仅CPU构造便复现TypeError。StageII实际调用H2T通道替换，H2Tv2未被接入。具体[文件/行号与调用链](AUTHOR_REPO_INVENTORY.md)均可核证。
3. 论文恢复了主要数学路径，但初始化、Eq8 W引用/梯度、BN buffers、作者精确数据与训练seed、完整StageII配置及StageI checkpoint来源未唯一恢复。36项[协议矩阵](PI_PAPER_PROTOCOL_MATRIX.md)逐行区分论文、公开原H2T/片段和本地重建。
4. 实得arXiv v2全文与在线SHA一致，附录和TeX包均检查；正式IEEE页面/PDF未取得可读正文，期刊与v2的一致性及独立supplement仍UNVERIFIED。

因此不选择AUTHOR_PI_CODE_VERIFIED。虽然论文足以支撑有明确缺口的独立重建，但关键项恰是本轮严格来源门禁要解决的内容，现有公开运行入口也不能构造，故本轮选择更严格的BLOCKED，而不是以“可以重建”代替作者协议已明确。

## 至少需要作者确认的关键项

|字段|论文能确认的部分|作者训练实现状态|
|---|---|---|
|PIF|通道均值、Eq3/6、主结果+2参数|完整活动单层代码、初值和bias：UNVERIFIED；公开25参数片段不匹配|
|H2TF|Eq7接收者权重；Eq8/脚注2理论边界归一化|完整调用、dis语义与论文实现：UNVERIFIED|
|W|StageII分类头会更新|r引用冻结StageI还是当前StageII：UNVERIFIED|
|ratio梯度|Eq8依赖f和w_y|detach/no_grad：UNVERIFIED|
|BN|表示参数冻结|train/eval、running buffers、双分支流量：UNVERIFIED|
|配置|CNN ResNet32、200+10轮、batch128、SGD .9及StageI基本LR|Option、bias、seed、warmup/wd/StageII具体LR：PARTIAL或UNVERIFIED|
|checkpoint|v2 Fig9标best49.22、last49.04|StageI42.19对应epoch/文件及所有模型SHA：UNVERIFIED|
|正式版本|DOI与v2关联已确认|VoR全文及补充附件一致性：UNVERIFIED|

## 本地结果的归属和科学边界

历史独立重建的StageI39.91%、H2TF fixed-last36.50%、matched-no_fusion45.53%来自既有run报告；本轮没有重新评估。作者v2的42.19%、best49.22%、last49.04%分别来自PDF p10 Table6和p12 Fig9(c/d)。36.50与49.04相差−12.54pp只是不同未完全对齐协议的描述，不能定为作者方法失效，也不是对某个W/BN配置的因果验证。公开代码构造问题与我们的历史独立重建不是同一程序，不能直接用它解释旧运行退化。

禁止以本地准确率或r≈.65择优“补全”协议；不启动未执行的reference-w实验分支，不设计新模块或新训练臂。本轮唯一允许的后续是取得作者可核实release/commit、完整CIFAR配置与checkpoint来源，或澄清答复，然后另行做静态与CPU一致性复核。AUTHOR_CLARIFICATION_DRAFT.md已拟好但没有发送，未创建issue/PR或联系任何人。
