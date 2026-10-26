# 执行与验证报告

日期：2026-10-26（Asia/Shanghai）；判断：**STOP_AND_REPAIR_PARITY**；异常：**REPRODUCTION_UNRESOLVED**。

## 分支、授权和目录

仓库`D:/xiangmuxunlian/xin`，分支`research/h2t-pih2t-gap-20261026`，审计基准HEAD `de2198a85117a46972d294fb4ae09951b3dbb018`。开始前已完整读取任务清单和README、PAPER_SOURCE_REGISTRY、LOCAL_EVIDENCE。初始Git状态只有：

```text
?? H2T_repro_audit/fusion_semantic_audit/
?? H2T_repro_audit/lowvs_mask_parity/
?? H2T_repro_audit/mdlt_domain_probe/
```

按任务文件dirty-stop要求首次停止，随后用户明确允许保留并忽略这三项；恢复时重新检查，未发现其他修改、暂存内容、冲突或已跟踪变更。全过程不切分支，不stash/reset/clean，不移动、删除、覆盖或修改这三项目录；不将它们纳入提交。

新结果目录（原先不存在，由既有verifier创建）：`D:/xiangmuxunlian/xin/H2T_repro_audit/research_gap_20261026/results/evidence_20261026_01`。本轮新增脚本、报告、缓存、日志均仅在此目录。没有修改原任务工具、测试、清单、原始H2T、独立PI审核代码或旧实验资产。

## 解释器与命令

统一使用`D:/app/minicondaaa/envs/xiaoju/python.exe`，Python 3.12.12；`-B`不生成pyc，文本打印用`-X utf8`。没有包安装或环境切换。论文解析复用已存在的bundled纯Python pypdf，PDF渲染为CPU Poppler；没有导入torch/torchvision或训练入口，没有模型/特征前向、反向、优化、GPU查询或GPU训练。

|命令/操作|退出码|结果|
|---|---:|---|
|`git status --porcelain=v1`、`git branch --show-current`、`git rev-parse HEAD`、`git ls-files --unmerged`、`git diff --name-only`、`git diff --cached --name-only`|0|只存在获准排除的3项；无冲突及已跟踪修改|
|`& 'D:\app\minicondaaa\envs\xiaoju\python.exe' -B -m unittest discover -s H2T_repro_audit/research_gap_20261026 -p 'test_*.py' -v`，首次|1|5项fixture创建均因沙箱默认TEMP不可写而ERROR，尚未触发断言；不是已定位的测试逻辑失败|
|`& 'D:\app\minicondaaa\envs\xiaoju\python.exe' -B -X utf8 H2T_repro_audit/research_gap_20261026/verify_evidence.py --output H2T_repro_audit/research_gap_20261026/results/evidence_20261026_01`|0|PASS 10 local evidence anchors；没有缺失锚点或硬编码放行|
|指定解释器执行stdlib unittest discovery，同一目录/同一pattern，仅`tempfile.tempdir`设为本结果下`test_scratch`|0|5 tests，OK；未改测试及核验器。结束时仅删除已空的本轮test_scratch|
|指定解释器`-B -X utf8 .../collect_primary_sources.py`（公开PDF下载、pypdf提取）|0|5篇主论文+H2T补充版本全部取得；只写本轮目录|
|浏览工具获取arXiv版本/出版信息、主论文PDF|工具无shell退出码|PI为v2，2026-07-18；CVF浏览器代理PDF 403，直接urllib请求成功，已用SHA锁定|
|可选GitHub pinned raw/arXiv HTML旁证请求|终止等待|响应长期未完成，主动停止只读请求；未将其作为已核验官方代码身份或新论文证据|
|CPU `pdftoppm`逐页渲染并`view_image`复核|0|11页，PI p4/5/12/16，H2T p3，FeatRecon p7/8，SEL p3/4，DBG p3/4|
|`& 'D:\app\minicondaaa\envs\xiaoju\python.exe' -B -X utf8 .../inspect_local_evidence.py`|0|40个核心文件，18行历史结果，PIF参数25/2；CE gate=True，PIF gate=False|
|`& 'D:\app\minicondaaa\envs\xiaoju\python.exe' -B -X utf8 .../author_study_reports.py`|0|23字段parity表、93页来源账本、研究/机制/历史/判断报告|
|额外只读YAML/损失/默认值核查|0|另4个来源及SHA见ADDITIONAL_SOURCE_MANIFEST|
|最终AST/JSON/CSV/源码引用/保护范围核查|0（通过记录见FINAL_VALIDATION.json）|验证核心字段、原始报告锚点、所有登记source hash及保护范围|
|`git diff --check`及暂存后`git diff --cached --check`|0（通过后提交）|逐项暂存本轮小文件，不使用git add -A|

初始测试错误的完整关键输出、恢复后的5项测试名称和verifier输出保存在`EXECUTION_LOG.json`。源码阅读用rg/按行读取/AST，仅以原文定位；`rg --files -g AGENTS.md`无命中退出1（没有新增仓库AGENTS文件），不解释成科学证据缺失。所有代码检查均是静态，未执行PI smoke（它可能调用GPU），也没有“为确认而跑一遍”训练/evaluate。

## 论文实际访问与证据级别

`PAPER_ACCESS.json`含每份PDF的确切URL、字节数、页数、SHA；`PAPER_PAGE_LEDGER.csv`93行覆盖五篇主文及H2T额外版本，逐页区分正文/证明/参考文献、文本摘要和视觉核查状态。阅读证明用于核对假设和范围，没有宣称独立证明全部定理。全文提取和原PDF保留于本地，不提交全篇转录或大体积资产。

|来源|本轮版本/页数|SHA-256|
|---|---|---|
|H2T|AAAI2024 primary，9页|38ec38129c3dfa7d9d42d10dc1e9018ffd25bdca01ea5772c74c57f2234ccc0a|
|PI-H2T|arXiv2506.00625v2，2026-07-18，17页|b68e80b98b31ad2a317f6462cab93e59f15ac4f456b5bbd1d23386339f3d5dc8|
|FeatRecon|ICLR2025 Conference PDF，33页含附录|2452227af12940e9c3b115c381e669493ec8252865b1d2a0c71e4247fed22783|
|SEL|ICCV2025 CVF accepted PDF，11页|60cde7cf1aee220b51965ddd68720c7a70f6dc16b1421810585147a0220aa688|
|DBG|CVPR2026 CVF accepted PDF，10页|e5e2503be6d9745f7d199d86e25b43c1d99566ba1f0350a9a1a7833333cf04ad|
|H2T补充|arXiv2306.06963v3，13页|74a71f43104b3d4e19be54ceb75a85d7e0eb4e64e478e4ffca021ca4bfeb9153|

Poppler对SEL/DBG报告Symbol/ArialUnicode替代字体警告；对应11张机制页已目视检查，所引用公式与脚注可辨。PI图9/10轴上的r与约.65均经p12图像确认，未将提取乱码当作可读数字。出版页面仅辅助确认版本/来源，具体机制取自完整PDF。

## 已核对文件与小体积结果

40份核心源/历史报告及元数据逐文件列于`SOURCE_REVIEW_MANIFEST.json`，另4份配置/default/loss文件见`ADDITIONAL_SOURCE_MANIFEST.json`，附路径、行数与SHA。包括原component报告、PI review及上游记录/diff、H2T活动PIF和训练时归档代码、StageI/II调用者、采样、优化与评估、全部指定历史报告及各seed原summary/experiment。不是仅搜关键词或只读摘要。

主要交付：`RESEARCH_REPORT.md`、`AUTHOR_MECHANISM_CROSSWALK.md`、`PI_PAPER_CODE_PARITY.md/.csv`、`PRIOR_EXPERIMENT_EVIDENCE.md`、`ONE_QUESTION_DECISION.md`、本报告。`NEXT_EXPERIMENT_PREREG.md`明确WITHHELD，没有新实验arms/阈值/训练命令。自动证据`EVIDENCE_INDEX.csv`、`SOURCE_MANIFEST.json`保持核验器原始输出，未改锚点或替换来源。

`STATIC_PARITY_CHECKS.json`从实际AST计数PIF参数和层，附r/lambda端点代数及仿射链约化；没有随机张量/合成batch运行或真实checkpoint系数测量。`HISTORICAL_CONTRASTS.csv`、`HISTORICAL_RECALC.json`只重算既有数字，无显著性检验。`CHECKPOINT_FILE_METADATA.json`只对当前候选weight文件原始字节算SHA，没有反序列化；旧绝对路径含“新”且历史缺原SHA，不能宣称与作者checkpoint逐字节匹配。

## 保护、数据与未确定项

初始只读阶段建立1493个已跟踪非数据文本文件的SHA基线（排除data目录），并记录3个未跟踪历史资产目录共264个文件的大小/mtime元数据；本地基线在被忽略paper_cache内。最终重算与Git检查结果见`FINAL_VALIDATION.json`，不把元数据对照写成历史资产全内容哈希验证。没有改写旧报告中的“official/缺陷”等原词，只在新报告限制其科学解释。

**官方TEST本轮未触及。** 没有加载任何CIFAR数据、历史预测NPZ或模型。历史测试/选模使用情况仅从已有报告与源码识别：ratio-only逐轮evaluate；boundary在训练结束evaluate；StageI历史best与多轮研究已使用test，不能把固定last当作独立未使用测试集。论文DBG的test-feature指标也没有在本地执行。

尚未确定：作者精确PIF初值/部分发布实现、ratio使用当前或冻结W及detach、BN buffers策略、全套训练细节/作者checkpoint身份；PIF参数化差异是否造成退化的因果关系；parity修复后是否仍有可重复且有新颖性的机制问题。因此只给STOP_AND_REPAIR_PARITY，不宣称发表缺陷、期刊级新颖性或改进效果。

## 提交推送

用户已要求提交、推送至当前同名GitHub分支。本轮仅逐项暂存本结果目录的报告、CSV、JSON、审计脚本及.gitignore；不提交PDF、渲染、整篇抽取、checkpoint、原数据或三个历史目录。发布前检查Git分支与暂存清单，正常commit+push，不改写历史，不PR/merge；最终commit SHA与远端一致性在会话最终答复中报告，本文保留审计基准HEAD以避免提交自引用。
