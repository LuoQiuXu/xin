"""Render new Markdown reports from persisted, independently checked evidence."""
import json
from analyze_measurements import REPORT, ROOT, EVIDENCE, read_csv


def write(name, lines):
    with (REPORT / name).open("x", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


summary = json.loads((REPORT / "AUDIT_EVIDENCE_SUMMARY.json").read_text(encoding="utf-8"))
isic, fish = summary["isic2019"], summary["fish4knowledge"]
mapping = json.loads((ROOT / "fish4knowledge/class_mapping_audit.json").read_text(encoding="utf-8"))
proofs = json.loads((EVIDENCE / "ISIC_CONFLICT_BYTE_VERIFICATION.json").read_text(encoding="utf-8"))
classes = read_csv(REPORT / "FISH_CLASS_COUNTS.csv")

lines = [
    "# CROSS_DOMAIN_DATA_AUDIT_REPORT", "", "日期：2026-10-31（Asia/Shanghai）。",
    "分支：`experiment/h2t-cross-domain-data-gate-20261030`；HEAD 保持 `e65f9a9d42f42fd5cbbd7b0841f77e9fe1981fe7`。", "",
    "## 最终科学判断", "", "| 数据集 | 真实读取/类别 | 数据门禁 | 下一阶段资格 |", "|---|---:|---|---|",
    "| ISIC 2019 | 25,331 / 8 | **NO_GO_DATA_INTEGRITY** | 冲突标签，未生成 split；不得进入八类确认性训练 |",
    "| Fish4Knowledge | 27,370 / 23 发布类别 | **PASS_LIMITED_SCOPE** | 仅 16 类、27,133 图像、8,680 轨迹的受限探索 |", "",
    "**下一轮优先候选为 Fish4Knowledge 的 16 类受限范围；ISIC 暂停。** 本轮不训练、不设计新模块、不改标签或门槛。数据入口问题不能解释为 H2T / PI-H2T 方法失效。", "",
    "## 环境、执行和保护边界", "",
    r"解释器：`D:\app\minicondaaa\envs\xiaoju\python.exe`；Python 3.12.12；Pillow 12.3.0；Windows-11-10.0.26200-SP0。审计入口不导入 torch。",
    "按附件顺序完成 CPU 测试、语法检查、ISIC 审计、Fish 审计和独立分析。", "",
    "最初沙箱执行 11 项测试，6 项因临时目录 WinError 5 报错，已定位权限原因并保留记录；无跳过。具备必要文件系统访问权限后，新增映射门禁并完整复跑 **15/15 PASS**；加入逐图测量持久化和进度日志后再次 **15/15 PASS**。相关 Python 文件均通过语法检查。", "",
    "上轮根目录两份报告与 audit_20261031_01 外部结果原样保留；新报告写入当前 results/formal_data_audit_20261031_01 子目录。未修改原始模型、数据、标签、mask、历史权重或结果；不切分支、不清理、不提交或推送。保护验证见 VALIDATION_REPORT.md。", "",
    "## 数据与接口确认", "",
    "三个 CSV 均已补齐，绝对路径见本轮 DATASET_DIRECTORY_INSPECTION.md。整个 data 树有 95,667 个文件、4 个 CSV，无遍历错误或目录链接。Fish 图像已解压，直接使用 fishRecognition_GT/fish_image；mask_image 的 27,370 张掩码未参与分类审计，没有重复解压或下载。", "",
    "ISIC GroundTruth 有 25,331 行，表头为 image,MEL,NV,BCC,AK,BKL,DF,VASC,SCC,UNK；全部为精确 0/1 one-hot 且 UNK=0。Metadata 同为 25,331 行，含 image,age_approx,anatom_site_general,lesion_id,sex。ID 与真实图像一一对应。所有图像 verify 后重新打开 load 并计算文件 SHA256，无损坏、缺图或标签解析错误。", "",
    "## ISIC：完整图像核验后的严格停止", "",
    "### 逐类真实计数", "",
    "已知病灶数指提供了不同 lesion_id 的数量；缺失样本只是协议允许的 image-ID 回退，不伪造真实病灶独立性。", "",
    "| 类别 | 实测/期望图像 | 不同已知 lesion_id | 缺失 lesion_id 图像 | 缺失率 | 原始组（含回退） |",
    "|---|---:|---:|---:|---:|---:|",
]
for c in isic["class_counts"]:
    lines.append(f"| {c['label']} | {c['actual_images']} / {c['official_expected_images']} | {c['distinct_known_lesion_ids']} | {c['missing_lesion_id_images']} | {c['missing_lesion_id_pct']:.4f}% | {c['protocol_base_groups_with_fallback']} |")
lines += [
    "", f"已知 lesion_id 共 {isic['known_distinct_lesion_ids']:,} 个；缺失 {isic['missing_lesion_id_images']:,} 张（{isic['missing_lesion_id_pct']:.6f}%）。原始组 {isic['raw_protocol_groups_with_fallback']:,}，合并后形成 {isic['diagnostic_merged_components']:,} 个诊断连通分量，其中 1 个含不同标签。混合分量会在逐类诊断统计中重复出现，不能将逐类分量数直接求和当作总独立组数。", "",
    "50 组精确重复图像均为两张一组，即 50 张额外重复、25,281 个不同哈希；其中 11 组跨原始 base group。原始文件全部保留。", "",
    "### 已证实的冲突", "",
    "| 图像 A（标签 / lesion_id） | 图像 B（标签 / lesion_id） | 字节数 | SHA256 前 16 位 | 字节比较 |",
    "|---|---|---:|---|---|",
]
for proof in proofs:
    a, b = proof["members"]
    lines.append(f"| {a['image_id']}（{a['label']} / {a['base_group']}） | {b['image_id']}（{b['label']} / {b['base_group']}） | {proof['file_bytes']} | `{proof['sha256'][:16]}` | 相等 |")
lines += [
    "", "四张图像连接 lesion:BCN_0003560（MEL）与 lesion:BCN_0000237（NV）。独立分析重新读磁盘，重新计算两对文件的完整 SHA 并逐字节比较，全部确认一致；完整哈希与路径位于仓库外 ISIC_CONFLICT_BYTE_VERIFICATION.json。没有根据期望值伪造哈希。", "",
    "**已证实：本地相同图像字节对应不同诊断标签。仍未确定：冲突源于官方发布、本地包/CSV 版本组合还是其他数据来源问题。** 本轮未对远端完整发布作逐字节同一性验证，不能据此断言官方标注或论文有缺陷。", "",
    "### 分组、泄漏和八类资格", "",
    "70/15/15 与每类 train≥5、val≥3、test≥5 门槛保持原值。merge_leakage_groups 在分配 split 前拒绝混合标签组，exit code=2、状态 NO_GO_DATA_INTEGRITY。没有创建任何 ISIC train/val/test 分配；泄漏检查是 **NOT_EVALUATED_NO_SPLITS_CREATED**，不能写作“0 泄漏通过”。", "",
    "已知病灶 ID 数量达到预声明计数前提，但完整性先失败，八类 primary 未获认证。未删除冲突样本/类别、改标签、拆开重复组或降低门槛。缺失 lesion_id 的 8.227074% 样本另有真实病灶独立性未验证限制。", "",
    "原失败 audit_summary.json 保留真实错误和未知计数 null；两个 manifest 和 class_counts.csv 仅有表头，表示未完成分组。本轮未覆盖它们。实际完成的图像数量与逐类统计已另存为 ISIC_INTEGRITY_DIAGNOSIS.json、image_integrity_inventory.csv 和本报告的 ISIC_CLASS_COUNTS.csv。空表不代表实际读取数为 0，也不能训练。", "",
    "## Fish4Knowledge：有限类别范围通过", "",
    "### 独立映射核验", "",
    "class_id.csv 的真实表头为 file_name,cluster_id，共 27,370 行。前 20 行及样例见 CLASS_ID_MAPPING_EVIDENCE.md。官方页面将其定义为 filename→cluster ID 反查表，类别通过本地文件夹和官方代表性类别表逐项核对。cluster ID 在此作为发布类别标识，不能据此证明生物学簇纯度；本轮不进行额外生物分类学推断。", "",
    f"逐条同时比较 tracking ID、fish ID 和类别，共 {mapping['n_matched_rows']:,}/{mapping['n_mapping_rows']:,} 完整匹配，{mapping['matches_requiring_leading_zero_normalization']:,} 条需要前导零长度规范化。只在核验中去掉明确 fish_ 前缀并比较十进制 ID；原始文件名、tracking token 和 split group 不改。无缺失、多余、重复或错类，fish ID 恰好 1…27,370。错轨迹、错类、缺行和重复的拒绝逻辑均经测试。", "",
    "### 全 23 类统计", "",
    "每个 primary split 单元为“图像 / 轨迹组”；全部 23 类始终保留在全量清单。", "",
    "| ID | 发布类别 | 图像 | 原始/合并轨迹 | train | val | test | Primary |",
    "|---:|---|---:|---:|---:|---:|---:|---|",
]
for c in classes:
    eligible = c["primary_eligible"] == "True"
    cells = [f"{c[f'primary_{s}_images']} / {c[f'primary_{s}_groups']}" if eligible else "—" for s in ("train", "val", "test")]
    state = "保留" if eligible else f"排除：{c['all_independent_groups']}<15 轨迹"
    lines.append(f"| {c['class_id']} | {c['label']} | {c['all_images']} | {c['raw_lesion_or_tracking_ids']} / {c['all_independent_groups']} | {' | '.join(cells)} | {state} |")
lines += ["", "所有逐类图像数、原始轨迹数均与协议及官方表一致：总量 27,370 图像、23 发布类别、8,725 条轨迹。", "",
          "### 各集合支持", "", "| 范围 | train 图像/组 | val 图像/组 | test 图像/组 | 总计图像/组 |", "|---|---:|---:|---:|---:|"]
for scope, name, total in (("all", "23 类描述性", (27370, 8725)), ("primary", "16 类 primary", (27133, 8680))):
    cells = [f"{fish['splits'][scope][s]['images']} / {fish['splits'][scope][s]['groups']}" for s in ("train", "val", "test")]
    lines.append(f"| {name} | {' | '.join(cells)} | {total[0]} / {total[1]} |")
lines += [
    "", "固定 seed=42，60/20/20 的轨迹级划分，16 类均满足 9/3/3 的最低组数，超过至少 12 类要求。ID17–23 因各自仅 8、5、6、6、8、6、6 条轨迹被排除出 primary；237 张图像、45 条轨迹仍在全量清单。", "",
    "310 对完全重复图像、27,060 个不同哈希；重复均在同一原始轨迹内，未删除。原始 tracking ID、合并 group_id、相同图像 SHA 在 all 和 primary 中的跨 split 计数均为 **0**。独立脚本重读清单验证计数、组数、标签、最低支持及 CSV 全部映射，均通过。本轮两份新 manifest 与上轮清单逐字节一致，旧结果未覆盖。", "",
    "Head/Medium/Tail 是 primary 训练类频三分位（6/5/5 类），训练 IF=253.172414，不是 CIFAR 绝对阈值。极少类测试可能仅 3 条轨迹；未来结论限于这 16 类，不能推广到全部 23 类。", "",
    "## 状态定义与下一阶段", "", "| 状态 | 本轮解释 |", "|---|---|",
    "| PASS | 完整范围协议通过；本轮未出现 |",
    "| PASS_LIMITED_SCOPE | Fish 的 16 类范围通过 |",
    "| BLOCKED_MISSING_DATA | 必需输入缺失；本轮 CSV 补齐后未触发 |",
    "| NO_GO_DATA_INTEGRITY | ISIC 的跨标签精确重复冲突 |",
    "| NO_GO_INDEPENDENT_GROUP_SUPPORT | primary 类数不足的状态；Fish 7 类被排除但其余 16 类达到总量门槛 |", "",
    "当前仅建议下一轮为 Fish 16 类制定受限探索性训练协议；不具备直接开展两个完整领域确认性训练的条件。ISIC 暂停，不能自行修复标签后放行。骨干、两阶段适配及预算需在下一轮先行固定，本轮没有训练或新模块。", "",
    "## 产物与来源", "",
    f"- `{ROOT / 'isic2019'}`：五个标准失败诊断文件、真实逐图测量、input_statistics 和新增 ISIC_INTEGRITY_DIAGNOSIS.json。",
    f"- `{ROOT / 'fish4knowledge'}`：五个标准完整文件、真实测量及 class_mapping_audit.json。",
    f"- `{EVIDENCE}`：测试日志、初始权限失败、输入/代码 SHA、目录检查、冲突字节证明、重复明细、最终验证及输出索引。",
    "- 本报告目录：两数据集逐类汇总 CSV、AUDIT_EVIDENCE_SUMMARY.json、映射证据、目录报告、VALIDATION_REPORT.md 和只读分析代码。", "",
    "[Fish4Knowledge 官方页](https://homepages.inf.ed.ac.uk/rbf/Fish4Knowledge/GROUNDTRUTH/RECOG/) 支持发布计数、轨迹定义和反查表含义；本轮只核查网页，无数据集下载。© 2013 Robert Fisher。引用 Boom, Huang, He, Fisher, Supporting Ground-Truth annotation of image datasets using clustering, ICPR 2012。", "",
    "[ISIC 官方数据页](https://challenge.isic-archive.com/data/) 提供发布总量。原始 ATTRIBUTION.txt、LICENSE.txt（CC-BY-NC，来源包括 BCN_20000/HAM10000/MSK）保持原样。本地 CSV 记录 SHA，未声称远端完整发布字节验证通过。",
]
write("CROSS_DOMAIN_DATA_AUDIT_REPORT.md", lines)

lines = ["# CLASS_ID_MAPPING_EVIDENCE", "", r"文件：`D:\xiangmuxunlian\xin\data\class_id.csv`。",
         "真实表头 file_name,cluster_id；27,370 行。保留前 20 条数据记录的原始字符串：", "",
         "| file_name | cluster_id |", "|---|---:|"]
for row in mapping["first_20_data_rows"]:
    lines.append(f"| `{row['file_name']}` | {row['cluster_id']} |")
lines += ["", "## 对齐样例", "", "| CSV file_name | 实际 image_id | cluster_id |", "|---|---|---:|"]
for row in mapping["examples"]:
    lines.append(f"| `{row['csv_file_name']}` | `{row['actual_image_id']}` | {row['cluster_id']} |")
lines += ["", f"完整匹配 {mapping['n_matched_rows']} 条；{mapping['matches_requiring_leading_zero_normalization']} 条需前导零数值等价比较。只在比较中去前缀并数值归一化，原始 tracking 与 split 不变。",
          "", "官方表述及逐条文件夹一致性支持将 cluster 解释为发布类别，不据此证明生物学物种纯度。完整核验、问题列表和输入 SHA 位于仓库外 fish4knowledge/class_mapping_audit.json。"]
write("CLASS_ID_MAPPING_EVIDENCE.md", lines)

lines = ["# VALIDATION_REPORT", "", "日期：2026-10-31（Asia/Shanghai）。", "", "| 检查 | 结果 |", "|---|---|",
         "| 分支/HEAD | 目标分支及 HEAD 保持不变 |", "| 初始工作区 | 仅上轮六项已知修改，无未知冲突 |",
         "| 首次测试 | 11 项，6 项沙箱临时目录权限错误；INITIAL_CPU_FAILURE.json 保留原因 |",
         "| 完整复跑 | 15/15 PASS，无跳过；修复持久化后再跑 15/15 PASS |",
         "| 语法检查 | 审计、测试、目录工具及本轮分析/报告工具 PASS |",
         "| 解码回归 | 截断 JPEG 的实际 load 拒绝路径 PASS |",
         "| 映射负例 | 错 tracking、错类别、缺行、重复均拒绝，能触发严格完整性门禁 |",
         "| ISIC 解码/数量 | 25,331 张、8 类；逐类匹配，无解码失败 |",
         "| ISIC 完整性 | FAIL：1 混合标签分量，4 图像；2 对直接字节比较相等 |",
         "| ISIC 划分/泄漏 | 未创建 split；NOT_EVALUATED，不伪报 0 泄漏 |",
         "| Fish 数量 | 27,370 / 23 类 / 8,725 轨迹，逐类一致 |",
         "| Fish 类别映射 | 27,370 条双 ID 和类别全部匹配；独立脚本再次核验 |",
         "| Fish primary | 16 类，每类9/3/3均满足，7类排除原因完整 |",
         "| Fish split 泄漏 | 原始轨迹/合并组/精确哈希，all/primary 均 0 |",
         "| Fish 旧清单复核 | 新旧两份 manifest 均字节相同；旧文件未写入 |",
         "| 历史/数据保护 | 文件大小/mtime 与指定小文件 SHA；最终见 FINAL_VALIDATION.json |",
         "| Git diff --check | 最终结果记录于 FINAL_VALIDATION.json |",
         "| GPU/训练/下载 | 0；无 Stage I、Stage II 或 PI-H2T 调用 |",
         "| 标签/图像修改 | 0，不解压已有图像，不改 mask、门槛或 split 规则 |",
         "| 提交/推送/切分支 | 0 |", "",
         "调用路径：main → run_audit → load_isic/load_f4k → record_image → inspect_image(verify+load) 与 sha256_file → merge_leakage_groups → allocate_group_splits → verify_no_split_leakage → 写清单。ISIC 在 merge 混合标签检查停止；Fish 执行完整路径及 class_id 门禁。", "",
         "独立分析只重建诊断连通分量，不调用 split 分配；真实逐图测量在严格合并前已持久化，因此停止后无需重跑图像审计即可保全证据。失败原记录保留，实际数量以新增诊断补全。", "",
         "首次报告生成尝试发生 Windows 路径转义 SyntaxError，解析期未写任何报告；改用独立报告脚本正确处理路径后生成新报告。", "",
         "保护检查比较既有文件大小/mtime_ns，及上轮两报告、协议、三个 CSV 的 SHA；不将其称为全部历史文件内容校验。清单、图像哈希和详细重复记录保留仓库外。", "",
         "尚未确定：ISIC 本地冲突的上游来源、2,084 张缺失 ID 的真实病灶独立性、任何模型性能及 H2T/PI-H2T 跨域效应。"]
write("VALIDATION_REPORT.md", lines)
print("REPORTS_CREATED")
