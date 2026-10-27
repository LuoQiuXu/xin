# PI-H2T 作者实现来源核查 — strict reproduction entry gate

**当前阶段：只读核查。禁止 GPU 训练、模型调参、融合比例扫描及新消融。**

## 本轮目的

此前 CIFAR100-LT IF100/ResNet32 的独立重建完成了 Stage I 200 + Stage II 10 + 10 epochs，但本地 Stage II H2TF Last 为 **36.50%**，作者 PI-H2T v2 报告 **49.04% Last / 49.22% Best**。这些数值不能直接证明作者 PI-H2T 失效。历史实验中 PIF、BN、`W` 引用等实现有未核实的作者语义，部分只从 AAAI 2024 H2T 代码继承。

此前创建但**没有运行**的 `repro/pi-h2t-reference-w-parity-20261027` 只是猜测 W-reference 的实验分支，**不要执行**。本核查分支从已完成的 `repro/pi-h2t-cifar100-paper-parity-20261027` 创建，绝不继承该未执行实验。

## 当前已得到的第一手线索（仍需由 Codex 再核查）

1. PI-H2T 期刊页面/摘要把 `https://github.com/Keke921/H2T` 列为 Source Code，DOI `10.1109/TPAMI.2026.3711911`。论文 arXiv `https://arxiv.org/abs/2506.00625`，核查公式以实际取得的 PDF 修订版本为准。
2. 作者该仓库 `main`，此前检查 SHA `e0c849235d9655fca31cd9fa8a5f38f92a77c7e8`，完整 Git tree 58 条目，README 明确写的是 AAAI 2024 H2T；核心是 `methods.py` 中的 H2T。当前可见只有 `main` 分支，未见 PI-H2T/PIF/H2TF 专用训练文件。
3. **只代表指定仓库指定快照未找到完整 PI 代码，不构成全球不存在 PI 代码的断言**。还应核查论文补充材料、版本、仓库历史/相关公开项目和作者更正。
4. `H2T_official_clean/H2T-main` 是原始 H2T 代码存档。历史的 `H2T/` 与 `PI_H2T_review/` 是本地独立复现/审计候选，不可当作作者 PI 官方代码。

## 使用方法（只读，Python 标准库）

在 GitHub 对应分支的仓库根目录执行：

```bash
python -m unittest discover -s H2T_repro_audit/pi_h2t_author_audit_20261027 -p "test_*.py" -v
python H2T_repro_audit/pi_h2t_author_audit_20261027/run_source_gate.py --output H2T_repro_audit/pi_h2t_author_audit_20261027/results/source_gate_01
```

程序读取 `SOURCE_EVIDENCE.json` 及历史论文—代码审计的原文位置和哈希，写入三个小文件：

- `AUTHOR_SOURCE_GATE.md`：已能证实的范围与未解决的关键问题；
- `LOCAL_ANCHORS.csv`：每条本地引用所在位置与来源文件 SHA256；
- `GATE_STATUS.json`：明确阻止把“本地文本锚点通过”升级为“官方 PI-H2T 完整复现已验证”。

代码**不调用网络、不导入任何训练模块、不读取模型/官方 CIFAR TEST、不访问 CUDA**。完整作者源核查需要 Codex 用 GitHub/API、论文 PDF 及补充材料完成，不能靠源文字串匹配解决。

## Codex 必须生成的科学报告

1. `AUTHOR_REPO_INVENTORY.md`：按官方仓库 URL、commit/branch/tag（能访问的范围）、所有含 PIF/H2TF 功能的文件、准确路径:行号，判断是否可跑 PI-H2T 两阶段，不得仅从 repo 名称推断。
2. `PI_PAPER_PROTOCOL_MATRIX.md`：PIF、H2TF、ratio `W`、detach、BN、数据和种子、网络、优化器调度、checkpoint Best/Last、评价设置的论文证据—作者代码—本地实现三方核查表。**每条必须附可查的主证据**；查不到就写 UNVERIFIED。
3. `STRICT_REPRODUCTION_DECISION.md`：分三种结论：
   - **AUTHOR_PI_CODE_VERIFIED**：完整 PI 官方实现找到了，给出精确源码与运行入口，可另行按作者配置训练；
   - **PAPER_RECONSTRUCTION_WITH_EXPLICIT_GAPS**：作者无完整已验证代码，但论文足以恢复主要路径，严格列出残余未知；
   - **BLOCKED_NEEDS_AUTHOR_CONFIRMATION**：关键实现不可唯一恢复，应先向作者询问，而不是对 TEST 反复试错。
4. `EXECUTION_REPORT.md`：真实测试退出码、检查的 URL/提交/文件/PDF 版本、历史资产不变、未训练声明、阻塞项。
5. 如结论为 BLOCKED，写 `AUTHOR_CLARIFICATION_DRAFT.md`，优先询问最会影响复现的 3–5 个精确问题，不发送邮件、不编造联系方式。

## 不做什么

- 不再跑冻结 W / 当前 W 两臂实验，也不调 H2TF 的 r 或 donor mass 去凑作者论文图。
- 不在旧的 `H2T/`、`H2T_official_clean/`、`PI_H2T_review/` 中修改一行训练代码。
- 不对已经多次观察的官方 TEST 进行模型选择。
- 不“为了看起来完整”设计 CI、自动下载数据、重建整套训练框架。
- 不删除用户的历史未跟踪目录、文件、checkpoint 或旧分支，不使用 `git clean`、`git reset --hard`、`git add -A`。

交付的价值是获得一个**作者方法能否严格复现的可审计结论**。根据此结论再决定下一个训练任务，而不是继续做猜测性科学实验。
