# PI-H2T 论文—代码一致性审计

状态对象是**具体字段**，不是整套复现的通行证。MATCH表示已读论文明确规则与对应静态实现相符；MISMATCH表示有可定位的实现/参数化差异；UNVERIFIED表示论文/作者配置/运行归属不足。论文公式确认不等于历史模型达到论文协议。最终Stage-II异常状态：**REPRODUCTION_UNRESOLVED**。

## 代码谱系

1. `H2T_official_clean/H2T-main/`：前次component audit冻结的AAAI H2T路径；仍以通道替换为主证据，未改动。
2. `PI_H2T_review/H2T-main/review/upstream_provenance.json`：记录2026-09-30 codeload ZIP，commit comment `e0c849235d9655fca31cd9fa8a5f38f92a77c7e8`，archive SHA256 `3f16f6759cfba1a5fbaa77b3d4a3bd5c43597b829298fd078c2459e12a79adcc`。本轮验证该记录文件hash；没有重新取得ZIP或核对GitHub该commit字节，不能把记录自证为当前官方完整PI发布。
3. `review/pi_h2t_vs_official_main.diff:7-34`保留当时上游草稿非法Conv2d参数、未定义feature_tuning、三层Conv1d和计算out_hat却return out的差异证据。独立审核包删掉草稿，另外写单层PIF、独立入口和冻结BN。其自述明确为**复现候选**，不是作者原码。主来源URL为 https://github.com/Keke921/H2T ，本轮网络旁证请求未完成，未据其假装确认归属。
4. `H2T/`是实际历史训练使用的本地改写分支：`resnet32_pif_fe`→PIF-capable ResNet32/Option-A→三层Conv1d→活动返回out_hat。训练时保存的`H2T/saved/cifar100_imb001_stage1_ce_mixup_pif_202608082320/codes/models/resnet_cifar.py:139-170`也含三层与有效返回。**因此历史PIF确实有活动实现证据，但不是审核包那份两参数PIF**。六个conv1d.* checkpoint键只证明三层结构存在，不证明等价作者参数化。

## 关键代数与因果边界

历史三层无激活，令输入x=[F,F-mean]，则y=W3 W2 W1 x+W3 W2 b1+W3 b2+b3。可以折叠成c_F F+c_R(F-mean)+c_0；故不能声称它必然是更复杂的非线性融合，或25参数本身证明论文方程完全不可表示。论文p5允许单层含bias的3参数版本，函数族可能相同；但是**分解参数化、默认初始化、bias、SGD/weight decay作用和随机数顺序不同**，训练轨迹/初始有效系数没有证明相同。拼接顺序也可以通过系数交换等价，单独把它叫“公式写反”同样不严谨。

`STATIC_PARITY_CHECKS.json`从AST数得历史25、review2；标量边界cos=-1/0/1时r=1/.5/0、lambda=0/.5/1。只是源码与代数核验，不是从真实checkpoint提取a,b,c，不是模型前向，不是因果实验。未对ratio调参、交换权重、换loss或写入原代码。

历史模型和头使用strict state loading（ratio-only:260-264）；旧路径中的`新`与当前仓库`xin`不同，当前候选checkpoint文件位置、大小及SHA见`CHECKPOINT_FILE_METADATA.json`，仅原始字节哈希。没有历史SHA作逐字节同一性证明，也没有反序列化权重；作者checkpoint身份仍未确定。review loader:21只接受pif.fuse.weight，历史为conv1d.*，两套模型不能静默互换。

原论文p16 StageI算法line5的F/F_fuse和p6 Eq10/b项记号不一致保留为文稿歧义；活动PIF依正文Eq3/6及随后推理说明核对，不能用伪代码笔误“证明”应返回未融合F。reference W、detach、BN、初始化的未定项也不能靠本地任选约定写成MATCH。

## 逐字段状态表

|字段|论文规则/位置|历史H2T/代码|状态|独立review代码|状态|限定|
|---|---|---|---|---|---|---|
|PIF channel mean|PI v2 PDF p4 Eq4-5|H2T/models/resnet_cifar.py:173|MATCH|PI_H2T_review/H2T-main/models/resnet_cifar.py:172|MATCH|该均值不等于把整个输出特征变成置换不变；保留原F项|
|PIF residual equation|PI v2 p4 Eq3,6; p5 Sec3.3|H2T/models/resnet_cifar.py:144|MISMATCH|PI_H2T_review/H2T-main/models/resnet_cifar.py:167|MATCH|历史是2→4→2→1三层；无激活可约成有效仿射混合，不能从结构差异直接推断性能根因|
|PIF parameterization/bias|PI v2 p5 Sec3.3; p12 Table10|H2T/models/resnet_cifar.py:145|MISMATCH|PI_H2T_review/H2T-main/models/resnet_cifar.py:167|MATCH|历史18个weight+7个bias=25；单层审核包2；后者没有被用于历史PIF结果|
|PIF concat order|PI v2 p4 Eq6|H2T/models/resnet_cifar.py:174|MISMATCH|PI_H2T_review/H2T-main/models/resnet_cifar.py:173|MATCH|历史[F,F-mean]为顺序差异，可交换有效系数重参数化；不是单独数学错误或精度因果证据|
|PIF tensor dimensions|PI v2 p4-5 Eq3,6|H2T/models/resnet_cifar.py:172|MATCH|PI_H2T_review/H2T-main/models/resnet_cifar.py:175|MATCH|CIFAR静态[B,64,8,8]→[B*64,2,64]→原形状；本轮未测真实张量|
|PIF active return|PI v2 p4 Eq3; p16 Algorithm1|H2T/models/resnet_cifar.py:180|MATCH|PI_H2T_review/H2T-main/models/resnet_cifar.py:184|MATCH|历史训练快照codes/models/resnet_cifar.py:170也return out_hat；不能把上游草稿return out误算为该历史运行未启用PIF|
|PIF initialization|PI v2 p4-5 Sec3.3; p16 Algorithm1|H2T/models/resnet_cifar.py:150|UNVERIFIED|PI_H2T_review/H2T-main/models/resnet_cifar.py:165|UNVERIFIED|历史_weights_init只处理Conv2d/Linear，Conv1d默认初始化；审核包也默认Conv1d；均未恢复作者初值|
|H2TF location|PI v2 p2 extension; p16 Algorithm1 lines12-14|H2T/train_pi_h2t_ratio_only_stage2.py:533|MATCH|PI_H2T_review/H2T-main/pi_h2t.py:29|MATCH|p5 Eq7后文字仍称再pooling；按p2与算法明确流程解释，不因文字不一致擅改原文|
|H2TF receiver/donor polarity|PI v2 p5 Eq7|H2T/train_pi_h2t_ratio_only_stage2.py:608|MATCH|PI_H2T_review/H2T-main/pi_h2t.py:38|MATCH|本地lambda是donor mass，r=1-lambda；未发现写反|
|Ratio cosine/theoretical bounds|PI v2 p5 Eq8 footnote2|H2T/train_pi_h2t_ratio_only_stage2.py:292|MATCH|PI_H2T_review/H2T-main/pi_h2t.py:19|MATCH|数学推出r=(1-cos)/2，lambda=(1+cos)/2；无batch min/max，不把相似度当距离|
|Ratio clipping|PI v2 p5 Eq8|H2T/train_pi_h2t_ratio_only_stage2.py:298|MATCH|PI_H2T_review/H2T-main/pi_h2t.py:22|MATCH|native A只是数值范围clip；扩展D/E另有边界clip，历史A/B/C未选择D/E|
|Ratio reference classifier|PI v2 p5 Eq8; p16 Algorithm1|H2T/train_pi_h2t_ratio_only_stage2.py:750|UNVERIFIED|PI_H2T_review/H2T-main/train_pi_h2t.py:37|UNVERIFIED|历史固定Stage-I W；审核包随当前Stage-II W更新；论文未明确stop-gradient/reference约定，不能称二者同时已验证官方等价|
|Ratio gradient/detach|PI v2 p5 Eq8; p16 Algorithm1|H2T/train_pi_h2t_ratio_only_stage2.py:532|UNVERIFIED|PI_H2T_review/H2T-main/pi_h2t.py:25|UNVERIFIED|历史ratio在no_grad且W冻结；审核包默认detach=True；这不是论文逐项声明|
|Supervision labels/loss|PI v2 p5 Sec3.4; p7 Sec4.2|H2T/train_pi_h2t_ratio_only_stage2.py:627|MATCH|PI_H2T_review/H2T-main/train_pi_h2t.py:40|MATCH|donor-label只供审计；没有按lambda混合标签|
|BS/IS sampling|PI v2 p5 Eq9|H2T/train_boundary_h2t_stage2.py:176|MATCH|PI_H2T_review/H2T-main/train_pi_h2t.py:99|MATCH|H2T dataset:98保留原索引、后续接收者重采样；不是head-only/tail-only，也不是多候选选择|
|CNN backbone and head|PI v2 p7 Sec4.2; p4 notation|H2T/train_pi_h2t_ratio_only_stage2.py:240|UNVERIFIED|PI_H2T_review/H2T-main/models/resnet_cifar.py:178|UNVERIFIED|PIF历史Option-A；CE外部checkpoint可能Option-B，代码按state判定；论文未给全部shortcut/bias细节。审核包普通FC带bias|
|Stage I optimizer/MixUp|PI v2 p7 Sec4.2|H2T/config/cifar100/cifar100_imb001_stage1_ce_mixup_pif.yaml:37|MATCH|PI_H2T_review/H2T-main/train_pi_h2t.py:89|MATCH|Match仅限明确列出的配方；wd、warmup、增强全细节不能仅凭正文同名认定全协议匹配|
|Stage II optimizer/schedule|PI v2 p7 Sec4.2|H2T/train_pi_h2t_ratio_only_stage2.py:753|UNVERIFIED|PI_H2T_review/H2T-main/train_pi_h2t.py:97|UNVERIFIED|历史及审核包都继承.1×.2首轮.02和cosine；论文正文没有确认这套倍率/wd的完整作者配置|
|Backbone gradient freeze|PI v2 p16 Algorithm1 line15|H2T/train_pi_h2t_ratio_only_stage2.py:264|MATCH|PI_H2T_review/H2T-main/pi_h2t.py:42|MATCH|requires_grad/no_grad/optimizer边界一致，不由此推断BN统计固定|
|BN running buffers|PI v2 p6 and p16 Algorithm1|H2T/train_pi_h2t_ratio_only_stage2.py:477|UNVERIFIED|PI_H2T_review/H2T-main/pi_h2t.py:44|UNVERIFIED|历史默认train更新BN；T1额外fixed；审核包eval固定；两者是已证实的本地差异，但谁匹配作者尚未确定|
|Checkpoint lineage/selection|PI v2 p12 Fig9-10|H2T/outputs/pi_h2t_ratio_only_pif_stage1seed123_seed42/experiment.json:2|UNVERIFIED|PI_H2T_review/H2T-main/train_pi_h2t.py:16|UNVERIFIED|历史epoch164/40.27；审核包要求pif.fuse.weight，历史conv1d.*，严格loader拒绝互换；只核验当前文件hash，未反序列化|
|Inference no H2TF|PI v2 p6; p16 AppendixD|H2T/train_pi_h2t_ratio_only_stage2.py:430|MATCH|PI_H2T_review/H2T-main/train_pi_h2t.py:104|MATCH|当前只读审计，没有跑evaluate或读取TEST|
|Accuracy/evaluation protocol|PI v2 p8 Table2; p10 Table6; p12 Fig9|H2T/outputs/pi_h2t_ratio_only_pif_stage1seed123_seed42/summary.csv:2|UNVERIFIED|PI_H2T_review/H2T-main/PI_H2T_REVIEW_ZH.md:1|UNVERIFIED|本地40.27→last34.82是该实现/协议结果；原审核包未证明论文精度，数值差异不是随机误差显著性或作者缺陷证据|

字段对应的原文规则详见同名CSV的paper_rule列。
