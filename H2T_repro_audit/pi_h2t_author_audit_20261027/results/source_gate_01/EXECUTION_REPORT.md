# 执行与完整性报告

日期2026-10-27；唯一仓库LuoQiuXu/xin；分支audit/pi-h2t-author-source-gate-20261027；开始HEAD=363e7c4954163990a048862c989ff32a618dedc8。启动时已在用户要求的分支，无需额外checkout，未新建/切换到其他分支。初始无tracked/staged改动或冲突；只有三个已知历史未跟踪目录，保持不动。

## 顺序、命令与退出码

先完整读取本轮TXT、README、SOURCE_EVIDENCE及工具/测试；做第一手来源核查；执行CPU测试、语法和本地来源门禁；再完成原文逐页/关键页视觉核查、调用链和三方矩阵；最后汇总门禁、完整性、显式提交推送。所有Python工作使用D:/app/minicondaaa/envs/xiaoju/python.exe，未改变解释器/环境。

|命令/检查|实际结果|
|---|---|
|-B -X utf8 -m unittest discover -s H2T_repro_audit/pi_h2t_author_audit_20261027 -p test_*.py -v|首次exit1，9项在TemporaryDirectory夹具创建时均PermissionError；未改断言；同一命令正常临时目录权限重跑exit0，9/9，0.011s|
|-B -X utf8 -m py_compile .../run_source_gate.py .../test_source_gate.py|exit0，2文件语法通过|
|-B -X utf8 .../run_source_gate.py --output H2T_repro_audit/pi_h2t_author_audit_20261027/results/source_gate_01|新路径创建成功，exit0，11条LOCAL_TEXT_ANCHOR_ONLY通过；原脚本仍明确不验证远端作者PI、不授权训练|
|作者41个文本Git blob身份 + 33个Python AST|全部对象SHA匹配、AST通过；utils/Unzip.py有3条invalid escape SyntaxWarning，未执行该工具|
|公开ResNet_fe CPU构造探针|成功完成诊断检查；捕获预期TypeError（feature_tuning非法关键字），不将此错误写成模型可运行通过|
|arXiv v2全文/TeX与关键页|PDF17页全文读取，pp4/5/7/10/12/16渲染视觉核查；PDF在线SHA与旧缓存一致；TeX包无训练文件|
|IEEE正式VoR/supplement|网页HTTP202空体，标准PDF HTTP418；无法做完整期刊-v2一致性确认，保持UNVERIFIED|
|历史保护|1644个本轮以外tracked小文件原始SHA聚合、26个tracked大文件元数据、7个受保护目录2221文件元数据均与本轮前一致|

暂存格式检查发现8份本轮新建证据文件的EOF多余空行，提交前仅整理文件末尾换行，未改变JSON值或门禁；SHA索引随之重算。完整测试真实输出、首轮权限错误和重试理由在CHECK_RESULTS.json；没有隐藏首轮失败，没有硬编码/放松来源验证。临时PDF读取命令曾因fitz未安装退出，随后同一指定解释器只读加载捆绑pypdf；一次临时读取命令括号笔误已修正。都未修改审计工具/作者训练代码或产生任何实验指标。旧web接口批量检索未完成，改用可用检索连接器/第一手HTTP；记录了访问失败，不以失败代替来源不存在。

关键文件实际SHA见AUDIT_SOURCE_HASHES.json、PUBLIC_SOURCE_STATIC_AUDIT.json、LOCAL_ANCHORS.csv；保护前后聚合见HISTORICAL_PROTECTION_BEFORE/AFTER.json。原始字节SHA与Git blob身份分开记录：Windows core.autocrlf可改变文本行尾，Git内容比较时显式归一化CRLF；模型/论文原始字节SHA不作文本规范化。

## 主要访问记录

- Keke921/H2T branches、releases、完整固定Git tree、79条commit和相关两个patch：GitHub连接器只读取得；tags端点不支持，改用git ls-remote --heads --tags核查。
- arXiv abs/html/pdf/src固定v2、作者publications页、原H2T项目页；GitHub作者仓库检索25项及PI-H2T名称查询。只以作者/原论文/登记元数据支持技术结论，不从第三方摘要恢复配置。
- Crossref主登记提供DOI和IEEE document/11601054，完整HTTP状态、SHA、UTC访问时间见PRIMARY_ACCESS_RECEIPTS.json。IEEE正式页空体/标准PDF拒绝；没有伪造期刊PDF页码、SHA或supplement内容。
- PDF标题/版本/页数/SHA与各附录的可读范围见PAPER_VERSION_AND_SUPPLEMENT_AUDIT.md。并未把AAAI原H2T的README、checkpoint或原始参数变成“已验证PI协议”。

## 历史与执行边界

H2T_official_clean/、H2T/、PI_H2T_review/、pi_h2t_strict_20261027/及三个历史目录均未编辑；没有覆写旧结果或运行其训练入口。元数据保护不是所有历史大文件内容SHA，这一边界已明确写入保护JSON。没有反序列化checkpoint、读取CIFAR图像/预测或执行模型评估；只有一个CPU构造诊断，没有forward/backward/optimizer步骤，没有CUDA查询或GPU训练、参数搜索、准确率拟合、新机制或新实验。

全部新增产物在本轮results/source_gate_01/。primary_cache只存公开来源数据/TeX包及PNG，均留本地并列SHA；只显式暂存本轮小体积MD/CSV/JSON与该结果目录的局部.gitignore，不提交原PDF/TeX包、模型或数据，不执行git clean/reset --hard/add -A，不PR/merge/force push。最终提交SHA由提交后答复给出，报告不自指虚构未来SHA。

## 最终结果

审计完成；科学门禁=BLOCKED_NEEDS_AUTHOR_CONFIRMATION，strict_author_reproduction_ready=False。9项测试、11条锚点通过仅说明这些检查的范围成立，不恢复作者未给出的关键协议。没有发送AUTHOR_CLARIFICATION_DRAFT，没有联系作者。本轮无GPU训练。
