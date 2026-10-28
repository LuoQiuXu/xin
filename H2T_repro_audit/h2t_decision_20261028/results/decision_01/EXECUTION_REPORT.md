# H2T Decision Ledger 执行与验证报告

日期：2026-10-28。仓库 `LuoQiuXu/xin`；唯一工作分支 `research/h2t-decision-ledger-20261028`。起始 HEAD `832e9747d992e2d67b32325a4e54d917e00c0325`，确认祖先 `research/h2t-pih2t-gap-20261026@d52f8c52a836b75717e2ad949bcd458ef4860c75`。没有切换/新建分支、stash、reset、clean、PR 或 merge。

**交付的科学决策为 STOP_NO_DEFENSIBLE_GAP。** 这是一轮只读历史证据与论文机制整合，没有训练、参数搜索、新模块、GPU 前向、Torch 导入、checkpoint 反序列化或 CIFAR/Office 数据数组访问。数值工作只处理已保存的文本 CSV/JSON，不重新生成任何历史签名或预测。

## 执行顺序与真实退出状态

所有 Python 命令均使用 `D:\app\minicondaaa\envs\xiaoju\python.exe`，本轮未安装包、未更改环境。下表的 python 均指该绝对路径。

|顺序|命令或操作|真实状态|
|---|---|---|
|1|`git branch --show-current`、`git status --porcelain`、`git diff`、`git merge-base --is-ancestor d52f8c52... HEAD`|分支正确；已跟踪/暂存修改为0，无冲突；祖先检查0|
|2|完整读取 TXT、README、EVIDENCE_LEDGER、analyze_ledger.py、test_analyze_ledger.py|已读；不修改原登记和分析器|
|3|`python -B -X utf8 -m unittest discover -s H2T_repro_audit/h2t_decision_20261028 -p test_*.py -v`|**6/6 PASS，exit0**，测试体0.009s；在允许标准临时目录访问的执行环境中运行，未改测试|
|4|`python -B -X utf8 -m py_compile H2T_repro_audit/h2t_decision_20261028/analyze_ledger.py H2T_repro_audit/h2t_decision_20261028/test_analyze_ledger.py`|PASS，exit0|
|5|`python -B -X utf8 .../analyze_ledger.py --output .../results/decision_01`|**FAIL CLOSED**。首次 shell 报1；仅为捕获 native 状态原命令复核，Python 确认 **exit2**，均因缺 Office risk 登记路径；该工具未生成结果目录|
|6|GitHub branches GET；固定 SHA 后 `git show SHA:path`、`git rev-parse SHA:path`、`git ls-tree`|成功；14登记路径均在其指定分支，4项不在当前工作树；没有 checkout/fetch/pull|
|7|`python -B -X utf8 .../decision_01/build_evidence_index.py`|exit0；14记录、105文本来源，逐项 SHA256 与 Git blob 校验；JSON/CSV/AST 静态解析成功；输出拒绝覆盖|
|8|只读核对 Office pilot 的 dataset_audit/status/test_access，另读已有本地单批次日志|exit0；3个 Git 文本上下文来源、9个本地文本来源独立标注；未打开其 NPZ/模型/图像|
|9|stdlib 复算已保存表格：`FIXED_ENDPOINTS.csv`、`RECONCILED_CONTRASTS.csv`、`NUMERICAL_RECONCILIATION.json`|exit0；50固定终点行、32对照、114项差值/配对计数核对，0差异；容差1e−8 pp仅用于十进制表格算术，**与特征 SHA 门禁无关**|
|10|论文缓存提取、公开原论文检索及六张机制页渲染/目视核对|最终exit0；8份PDF原始SHA入表；没有生成/编辑PDF。新下载两篇 ICLR PDF，其他只读旧缓存；补查 cRT/Logit Adjustment 网页PDF|

详细初始命令原始输出在 [EXECUTION_COMMANDS.json](EXECUTION_COMMANDS.json)。其余关键步骤、可复核失败与访问情况在 [REVIEW_VALIDATION.json](REVIEW_VALIDATION.json)、[LITERATURE_ACCESS.json](LITERATURE_ACCESS.json)。没有把工具成功、文件存在或哈希一致当作科学证明。

### 保留的异常

- 标准工具的本地历史文件缺失是真失败。四项为 OFFICE_RISK、TARGET、RECEIVER、SEMANTIC；均在各自分支 commit 中定位成功。按 TXT 第3步继续的来源清单独立保存，原门禁维持 FAIL，不产生伪造的 `EVIDENCE_CHECK.csv` 或修改原 `EVIDENCE_LEDGER.json`。
- 首次新论文 urllib 下载受沙箱 DNS 限制 exit1；改用已获授权的联网只读执行后成功 exit0，不安装包或切换解释器。
- 一次只读 AST 查询误将 RECEIVER README 送进解析器，exit1；修正查询只选择 .py 后 exit0。不是历史源码 SyntaxError，也没有运行任何历史程序。
- Poppler 对 SEL/DBG 的 Symbol/ArialUnicode 发出字体提示，渲染进程均exit0；已检查对应页面的公式与脚注，结合文本提取核对。其余机制页正常。缓存/PNG 均留本轮 `primary_cache/`，不提交。
- IEEE 正式正文与独立 supplement 等价性未取得；部分 CVF 当前入口403；采用已保存、原SHA一致的指定论文正文，并明确版本。CurrMix 只见官方索引摘要，其完整机制 UNVERIFIED，不用于决定性去重。
- 作者 GitHub 当前 main 与前轮审计 SHA 一致，releases GET 返回空表；tags 端点被连接器标为不支持，公开未认证 API 补查又触发403限流（exit1），故本次 tags 状态 UNVERIFIED，没有继续重试或声称已重新穷尽所有发布入口。

## 核查深度与限制

14份主记录已逐项阅读，支持文件覆盖每项的结果/失败/协议及相关实际调用；[CODE_PATH_EVIDENCE.csv](CODE_PATH_EVIDENCE.csv)给出固定 commit、真实文件行号与核查命题，不仅依据关键词。

本轮能重新核实的是 Git 文本身份、表格计数/算术、算子与调用关系，以及日志/配置所记载的模型来源。历史冻结参数、数据隔离及采样配对的全量数值真实性仍依赖原始日志和已有核验；没有重新加载checkpoint、重算 logits 或重新进行 GPU replay，不能声称这些已在本轮独立运行验证。

本地 TF32 单批次报告不在登记分支 commit；仅记录文件原SHA和日志内容，不把本地未发布资产冒充 GitHub 证据。目标批次已恢复匹配不等于完整 SEMANTIC 科学结果恢复。新两参数 PIF 重建是历史独立重建，不升级成作者方法已验证。

## 工作区与历史保护

起始四个历史未跟踪目录为 `fusion_semantic_audit/`、`lowvs_mask_parity/`、`mdlt_domain_probe/`、`pi_h2t_author_audit_20261027/`（均在 H2T_repro_audit 下）。第四个是前轮作者审计的既有缓存。它们全部保留，不删除、清理、覆盖、移动、修改或加入暂存。当前新增文件只在本结果目录。

[HISTORICAL_PROTECTION_BEFORE.json](HISTORICAL_PROTECTION_BEFORE.json)记录保护基线；最终复核见 [HISTORICAL_PROTECTION_AFTER.json](HISTORICAL_PROTECTION_AFTER.json)：1599个非本轮已跟踪小文件的原始字节摘要、26个大文件元数据摘要、4855个历史文件 size/mtime 摘要前后一致。后两者是元数据核查，**不是大文件内容已全部重新哈希**。没有更改原始 H2T、历史代码/实验结果/checkpoint，未操作历史分支引用。

## 本轮产物与提交范围

全部小产物位于 `H2T_repro_audit/h2t_decision_20261028/results/decision_01/`；只显式暂存下列文件类别，不使用 `git add -A`。

|用途|文件|
|---|---|
|四份科学/执行报告|[HISTORICAL_CONTRADICTION_AUDIT.md](HISTORICAL_CONTRADICTION_AUDIT.md)、[MECHANISM_NOVELTY_MATRIX.md](MECHANISM_NOVELTY_MATRIX.md)、[ONE_QUESTION_OR_STOP.md](ONE_QUESTION_OR_STOP.md)、本文件|
|14项逐分支与分类|[BRANCH_SOURCE_INDEX.csv](BRANCH_SOURCE_INDEX.csv)、[BRANCH_HEADS.json](BRANCH_HEADS.json)、[EVIDENCE_REVIEW.csv](EVIDENCE_REVIEW.csv)|
|小体积可审查证据|[SOURCE_MANIFEST.json](SOURCE_MANIFEST.json)、[ADDITIONAL_SOURCES.json](ADDITIONAL_SOURCES.json)、[CODE_PATH_EVIDENCE.csv](CODE_PATH_EVIDENCE.csv)、[FIXED_ENDPOINTS.csv](FIXED_ENDPOINTS.csv)、[RECONCILED_CONTRASTS.csv](RECONCILED_CONTRASTS.csv)、[NUMERICAL_RECONCILIATION.json](NUMERICAL_RECONCILIATION.json)、[LOCAL_SUPPLEMENT.json](LOCAL_SUPPLEMENT.json)|
|文献与验证|[PAPER_FILES.json](PAPER_FILES.json)、[LITERATURE_ACCESS.json](LITERATURE_ACCESS.json)、[LOCAL_GATE_BLOCKED.json](LOCAL_GATE_BLOCKED.json)、[EXECUTION_COMMANDS.json](EXECUTION_COMMANDS.json)、[REVIEW_VALIDATION.json](REVIEW_VALIDATION.json)、前/后保护清单|
|来源采集器与本轮缓存排除|[build_evidence_index.py](build_evidence_index.py)、`.gitignore`（仅排除本结果目录 `primary_cache/`）|

不存在下一实验草案；没有数据、权重、完整历史结果、论文PDF或PNG进入提交。最终 Git diff/文件链接/结构/范围检查状态在 REVIEW_VALIDATION；最终 commit SHA 与远端同分支一致性在提交后的用户交付消息中给出，避免把自身 commit SHA 写入其内容造成循环。
