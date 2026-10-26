# H2T / PI-H2T 深度机制研究报告

2026-10-26；分支`research/h2t-pih2t-gap-20261026`；审计基准HEAD `de2198a85117a46972d294fb4ae09951b3dbb018`。**结论：STOP_AND_REPAIR_PARITY；历史StageII异常归类REPRODUCTION_UNRESOLVED。** 已确认本地参数化差异，尚未证实性能下降根因，不将复现异常称为已发表PI-H2T缺陷。

## 六阶段完成情况

1. 先执行既有5项单元测试；默认temp路径被沙箱拒绝，改为本新结果目录内的临时fixture后5项全过，测试逻辑未改。既有verify_evidence成功核对10个来源锚点，生成EVIDENCE_INDEX/SOURCE_MANIFEST。锚点通过只证明原记录存在，不证明科学命题为真。
2. 取得五篇指定主论文全部PDF并锁定版本/SHA，逐页定位机制正文/附录，11张关键页图像复核；H2T额外用arXiv附录。作者主张、输入输出、阶段、假设、配方、对照和自述限制见`AUTHOR_MECHANISM_CROSSWALK.md`及页账本。
3. 沿本地实际工厂/forward/ratio/reference/optimizer/loader/evaluate/checkpoint调用追踪，输出`PI_PAPER_CODE_PARITY.md/.csv`，不把独立review包当作历史训练或作者官方代码。
4. 只读40份核心源/元数据/历史结果，重算18行A/B/C和endpoint差，见`PRIOR_EXPERIMENT_EVIDENCE.md`、`HISTORICAL_CONTRASTS.csv`及`HISTORICAL_RECALC.json`。没有合并不同checkpoint/不同评估协议。
5. 明确G1–G5；G3失败且完整parity未建立。不设计新模块，不注册新训练，只保留一个复现问题。
6. 完整性、引用、AST/JSON/CSV和提交范围检查见`EXECUTION_REPORT.md`及最终JSON；仅发布本轮目录中的小体积报告、CSV/JSON与审计脚本，PDF/渲染/全文抽取留本地忽略。已获用户授权提交推送同名分支，不PR/merge。

## 最强证据链

论文PI v2 p4–5规定单层Conv1d(2,1,1)，主实现2参数无bias（可选含bias3参数）。产生历史PIF实验的`H2T/models/resnet_cifar.py:144-147`及训练保存代码却是带bias的2→4→2→1，共**25参数**；独立`PI_H2T_review/`才是2参数实现，但不是那些历史结果的训练来源。三层间无非线性，可以代数折叠成有效仿射混合，故这个发现证明了**训练参数化不一致**，不证明函数表达能力必然不同或它就是准确率下降的原因。

Eq7/8及脚注2核验后，本地比例方向本身是匹配的：r=(1-cos)/2为接收者保留量，lambda=(1+cos)/2为供体；不能将r和lambda混同，不能通过交换二者贴合论文图中r≈.65。历史r≈.23、lambda≈.77与论文分布不同，是诊断线索，不足以定位原因。ratio的W是否用当前head、detach、BN、初值与作者checkpoint身份仍不完全可确认。

在同一历史PIF checkpoint、StageII seed42中，T2无融合45.69、T3所实现native融合34.82、T4尺度匹配37.26。两次骨干前向及BN输入对齐使这个局部对照有解释力：损失主要伴随所实现融合出现，尺度缩小只恢复2.44点。但它不能反驳论文，也不能排除BN与融合交互或未对齐参数化的作用。

PIF A/B/C三种子B−C为+.01/−.14/+.22，严格gate失败；CE三个正差来自另一无PIF checkpoint，不能补足完整PI证据。boundary-r负结果同样保留。论文PI已经承认过度融合语义噪声及PIF—校准理论交互；FeatRecon、SEL、DBG分别覆盖表示支持域、邻类可靠性与生成边界问题，泛称“融合有害/供体不盲目/几何重要”不足以建立新颖机制问题。

## 保留的一个问题与当前停止点

唯一待澄清项是：**严格恢复论文PIF与明确全流程协议后，历史StageII退化是否仍存在？** 当前是parity修复问题，不能包装为已确认研究空白。没有任何新模块、随机机制、训练arms或参数搜索。只有在未来parity证据和可重复因果问题同时成立时，才有资格讨论一个具体可证伪研究命题；本报告没有越过这道门槛。

## 阅读导航

- `AUTHOR_MECHANISM_CROSSWALK.md`：作者证据与我们的解释分栏，五篇完整机制比较。
- `PI_PAPER_CODE_PARITY.md/.csv`：23个字段逐项MATCH/MISMATCH/UNVERIFIED。
- `PRIOR_EXPERIMENT_EVIDENCE.md`：历史控制、局限、负结果和测试集使用。
- `ONE_QUESTION_DECISION.md`：五门槛与明确停止判断。
- `EXECUTION_REPORT.md`：命令退出码、源文件与论文hash、保护及发布验证。

本轮没有GPU调用、模型前向、训练、checkpoint反序列化或官方TEST读取。历史test使用只通过源码与已有报告识别；原始H2T、PI review与全部历史实验资产均未修改。
