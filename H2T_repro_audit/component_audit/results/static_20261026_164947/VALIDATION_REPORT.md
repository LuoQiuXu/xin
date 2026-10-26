# H2T Component Audit 验证报告

日期：2026-10-26。结果目录：`D:/xiangmuxunlian/xin/H2T_repro_audit/component_audit/results/static_20261026_164947`。

## Git基准与工作区

- 仓库根：`D:/xiangmuxunlian/xin`。
- 审计分支：`audit/h2t-component-map-20261026`，符合任务指定；本轮未切分支。
- 审计基准HEAD：`9a89df6dbe9f89c80dfd8792a7e06137333d2233`。此SHA记录审计所依据的源版本；用户后续授权的产物提交会生成后续commit，不改变审计基准。
- 初始`git status --short`如下；这些目录在任务开始前已存在，没有纳入本轮提交范围：

```text
?? H2T_repro_audit/fusion_semantic_audit/
?? H2T_repro_audit/lowvs_mask_parity/
?? H2T_repro_audit/mdlt_domain_probe/
```

结束检查确认新增/修改路径均在`H2T_repro_audit/component_audit/`。冻结目录`H2T_official_clean/H2T-main/`的git status和相对HEAD的diff均为空；工作区整体并非干净，因为保留了上述既有未跟踪内容。

## 解释器和执行范围

指定解释器：`D:/app/minicondaaa/envs/xiaoju/python.exe`，实测Python 3.12.12（Anaconda，MSC v.1929，64bit）。所有Python审计命令使用此解释器，`-B`禁写bytecode；需打印论文Unicode文本的命令另用`-X utf8`。没有安装包、切换解释器或修改环境。

单元测试、静态AST/文本解析、CSV/JSON/Markdown校验、文件原始字节哈希和CPU文档工具是本轮执行范围。未导入原始训练模块或torch，未构建模型，未调用CUDA、查询GPU、进行模型前向/反向/训练/参数更新，也未读取CIFAR训练/官方TEST数据。原目录中的checkpoint/历史NPZ等仅作为完整性快照的原始字节参与哈希，未反序列化或分析。

## 命令、退出码与测试结果

以下命令从仓库根执行；完整关键stdout/stderr保存在`EXECUTION_LOG.json`。大量源码查阅使用`rg`、`Get-Content`以及指定解释器的stdlib逐行/AST检查，均为只读，退出码0；没有通过执行原始入口来验证调用链。

|操作/命令|退出码|结果|
|---|---:|---|
|`git rev-parse --show-toplevel` / `git branch --show-current` / `git rev-parse HEAD` / `git status --short`|0|确认仓库、分支、基准SHA和初始脏状态|
|`& 'D:\app\minicondaaa\envs\xiaoju\python.exe' -B -m unittest discover -s H2T_repro_audit/component_audit -p 'test_*.py' -v`，首次|1|4项中1项失败：扫描器按关键词过滤定义，遗漏`train_step`调用所有者|
|同一单元测试命令，修复后|0|5项全部通过，含新增泛名`Plain.forward`作用域/空类别回归测试|
|`& 'D:\app\minicondaaa\envs\xiaoju\python.exe' -B H2T_repro_audit/component_audit/audit_components.py`|0|新建本结果目录，75个源文件；无AST parse error|
|指定解释器stdlib `urllib.request`获取原论文v3，并新建PDF/来源JSON|0|5,368,004字节；SHA-256见下文|
|尝试`pdftotext`抽取论文|1|环境无该可执行文件，未据此中止或捏造内容；改用pypdf|
|指定解释器加载bundled pypdf抽取论文并打印，首次|1|13页文本保存成功，随后GBK stdout无法打印数学Unicode字符；不是PDF读取失败|
|指定解释器`-B -X utf8`提取本地slides/poster及复核论文文本|0|UTF-8读取成功；slides12页，poster1页|
|CPU `pdftoppm`渲染论文3–7、10页，随后逐页图像检查|0|关键公式/表格可辨；字体警告见限制|
|`& 'D:\app\minicondaaa\envs\xiaoju\python.exe' -B -X utf8 H2T_repro_audit/component_audit/write_component_reports.py`|0|42行人工证据条目、16列CSV、15列Markdown表、完整主报告、配置解析|
|指定解释器`ast.parse`检查新增/修改的component_audit工具|0|4个Python文件均可解析，无H2T代码执行|
|`& 'D:\app\minicondaaa\envs\xiaoju\python.exe' -B -X utf8 H2T_repro_audit/component_audit/verify_component_outputs.py --run-dir H2T_repro_audit/component_audit/results/static_20261026_164947`|0|79处路径/行号检查、42行一致性、配置值、完整树哈希与Git范围全部通过|
|`git diff --check`|0|无空白错误|
|`git diff --cached --check`，首次暂存后|1|新提取的paper/slides/poster TXT含PDF提取产生的行尾空格；仅清理本轮派生TXT的行尾空格，原PDF和按页JSON保持原样|
|`git diff --cached --check`，清理并重新暂存后|0|最终提交前检查通过|
|`git status --short -- H2T_official_clean/H2T-main/`|0|空输出|
|`git diff HEAD -- H2T_official_clean/H2T-main/`|0|空输出|

单元测试首次失败根因：`ComponentVisitor._record_symbol`把没有命中组件关键词的函数定义直接丢弃，导致泛名函数不能保留调用归属上下文。修复只删除该过滤，并保留类别为空的语义；新增测试检查`Plain.forward`的scope为`Plain`、categories为空。并未把“所有定义”改称“所有活跃组件”。原有四项测试和新项共5项通过，最终输出`Ran 5 tests ... OK`。

## 产物与机器检查

主要交付文件：

- `COMPONENT_AUDIT_REPORT.md`：调用流程、配置/默认值/CLI边界、7个专门机制讨论、42行×15列组件表、已有消融和未决问题。
- `COMPONENT_EVIDENCE_MATRIX.csv`：UTF-8 BOM，42行，字段顺序严格为任务指定16列；removal_effect枚举无越界。
- `VALIDATION_REPORT.md`：本文件。

辅助证据：

- 扫描器原始输出`source_manifest.json`、`static_inventory.json`、`STATIC_COMPONENT_MAP.md`。
- `REVIEWED_COMPONENTS.json`、`COMPONENT_TABLE.md`、`CONFIG_RESOLUTION.json`和`EVIDENCE_VALIDATION.json`。
- `SOURCE_REFERENCE_CHECKS.json`逐条保存79个定义/调用引用的路径、行号及该行源文本；矩阵与人工条目逐字段一致，主报告组件表42行/15列。
- `FROZEN_TREE_BEFORE.json`、`FROZEN_TREE_AFTER.json`、`FINAL_INTEGRITY.json`及`EXECUTION_LOG.json`。
- `paper_evidence/`中的原论文PDF、来源、按页文本、slides/poster提取文本以及6张已检查的论文页PNG。

配置复核验证了Stage1的默认sampler/MixUp/multistep及Stage2的balance receiver、long-tailed donor、rho=.25、shift_bn=True、cosine .2倍率；另核对两阶段CE默认、头中尾列表。合并解析只恢复静态字面量，约定opts=[]；没有伪称捕获实际CLI或运行环境。手工推导的64通道替换16及参数组首轮LR .02均与解析结果一致。

## 源码保护与哈希

`source_manifest.json`记录75个文本源文件的路径、字节数及逐文件SHA-256，生成器版本与分支HEAD也在文件中。扫描得到579个定义、887个相关调用、33个context、0个parse error。最终逐文件重算75份SHA均匹配。

更广的冻结树快照覆盖269个文件、224,482,146字节，包括已有源码、文档、二进制权重、pyc和旧输出；只做原始字节哈希。前后对照：新增0、删除0、大小或SHA变化0。`FINAL_INTEGRITY.json`同时保存空的frozen git status/diff以及仅component_audit新增变化的检查。源目录基线在初始只读查阅/静态扫描后生成；在此前没有对冻结目录进行写入，开始和结束Git检查也均为空。

本轮允许范围内的工具文件变更是：修复`audit_components.py`、扩展`test_audit_components.py`，新增人工报告生成器`write_component_reports.py`及只读验证器`verify_component_outputs.py`。所有结果新建在本时间戳目录；未改动或覆盖任何旧实验结果，也未修改模型、方法、损失或训练逻辑。

## 解析警告、工具边界与论文访问

静态扫描退出码为0，但Python AST解析原有`utils/Unzip.py:53,54,58`的Windows字符串发出3条`SyntaxWarning: invalid escape sequence '\i'`；原文件未修。没有SyntaxError/parse error，警告不来自选定训练调用链。

扫描器是定位器：它不处理完整YAML语法/任意列表，不解析真实CLI、动态eval或任意调用图，也不证明模块激活或移除的精度效应；注意力关键词候选可能来自替代架构或本地辅助脚本。本轮保留所有定义仅用于scope定位。人工逐一追踪了要求的核心文件、配置loader、全部16个模型Python文件和两份loss；本轮没有把辅助实验脚本的结果作为原论文消融。

原论文可访问：[arXiv v3](https://arxiv.org/pdf/2306.06963v3)，13页，SHA-256 `74a71f43104b3d4e19be54ceb75a85d7e0eb4e64e478e4ffca021ca4bfeb9153`。表5、图6、附录Listing1/表7及相关方法/配置页已按物理页码核对，6个渲染页面已目视检查。Poppler对Symbol/ArialUnicode替代字体发出警告；所引用关键公式和数字仍清晰可辨，并与提取文本交叉核对。其他页面文本读取不等同于逐页视觉检查。

本地slides（12页）SHA-256 `9905889eaed993b56b71ac47206cddf7de46e6056ccf93b265b9b103eb6dc43e`，poster（1页）SHA-256 `58228a6275ca61186327d4a118aed9ad22e56b0237e26cdc18ca2de1ccfab53b`；slides p.3/poster p.1辅助支持双分支和接收者标签口径。未使用PI-H2T。pypdf来自已存在的bundled纯Python库路径，仍由指定解释器运行，无安装操作。

BN/no_grad接口语义另对照PyTorch官方文档；文档版本2.14不代表本轮安装环境版本，本轮没有导入torch以探测它。论文Eq.(1)掩码极性、真实运行CLI/checkpoint、BN独立贡献及可选接口动态行为仍未确定，报告未将推导写成实验结论。

## 发布授权与结束状态

任务文件原要求在后续明确授权前不提交/推送。用户随后明确要求“将本轮的产物本地提交并推送到github”，因此仅对本次component_audit工具和结果进行正常本地提交并推送现有分支`audit/h2t-component-map-20261026`至`origin`。不会将既有其他实验目录打包提交，不切换分支或改写历史。发布后的commit SHA和远端核验在会话交付中报告；本报告保留发布前的审计基准HEAD和完整验证证据，避免自引用commit SHA。

无训练、无GPU实验、无参数搜索或下一组实验设计。本轮发现的原源码/文档问题全部只记录，未修改冻结源。
