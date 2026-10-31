# CROSS_DOMAIN_DATA_AUDIT_REPORT

日期：2026-10-31（Asia/Shanghai）。
分支：`experiment/h2t-cross-domain-data-gate-20261030`；HEAD 保持 `e65f9a9d42f42fd5cbbd7b0841f77e9fe1981fe7`。

## 最终科学判断

| 数据集 | 真实读取/类别 | 数据门禁 | 下一阶段资格 |
|---|---:|---|---|
| ISIC 2019 | 25,331 / 8 | **NO_GO_DATA_INTEGRITY** | 冲突标签，未生成 split；不得进入八类确认性训练 |
| Fish4Knowledge | 27,370 / 23 发布类别 | **PASS_LIMITED_SCOPE** | 仅 16 类、27,133 图像、8,680 轨迹的受限探索 |

**下一轮优先候选为 Fish4Knowledge 的 16 类受限范围；ISIC 暂停。** 本轮不训练、不设计新模块、不改标签或门槛。数据入口问题不能解释为 H2T / PI-H2T 方法失效。

## 环境、执行和保护边界

解释器：`D:\app\minicondaaa\envs\xiaoju\python.exe`；Python 3.12.12；Pillow 12.3.0；Windows-11-10.0.26200-SP0。审计入口不导入 torch。
按附件顺序完成 CPU 测试、语法检查、ISIC 审计、Fish 审计和独立分析。

最初沙箱执行 11 项测试，6 项因临时目录 WinError 5 报错，已定位权限原因并保留记录；无跳过。具备必要文件系统访问权限后，新增映射门禁并完整复跑 **15/15 PASS**；加入逐图测量持久化和进度日志后再次 **15/15 PASS**。相关 Python 文件均通过语法检查。

上轮根目录两份报告与 audit_20261031_01 外部结果原样保留；新报告写入当前 results/formal_data_audit_20261031_01 子目录。未修改原始模型、数据、标签、mask、历史权重或结果；不切分支、不清理、不提交或推送。保护验证见 VALIDATION_REPORT.md。

## 数据与接口确认

三个 CSV 均已补齐，绝对路径见本轮 DATASET_DIRECTORY_INSPECTION.md。整个 data 树有 95,667 个文件、4 个 CSV，无遍历错误或目录链接。Fish 图像已解压，直接使用 fishRecognition_GT/fish_image；mask_image 的 27,370 张掩码未参与分类审计，没有重复解压或下载。

ISIC GroundTruth 有 25,331 行，表头为 image,MEL,NV,BCC,AK,BKL,DF,VASC,SCC,UNK；全部为精确 0/1 one-hot 且 UNK=0。Metadata 同为 25,331 行，含 image,age_approx,anatom_site_general,lesion_id,sex。ID 与真实图像一一对应。所有图像 verify 后重新打开 load 并计算文件 SHA256，无损坏、缺图或标签解析错误。

## ISIC：完整图像核验后的严格停止

### 逐类真实计数

已知病灶数指提供了不同 lesion_id 的数量；缺失样本只是协议允许的 image-ID 回退，不伪造真实病灶独立性。

| 类别 | 实测/期望图像 | 不同已知 lesion_id | 缺失 lesion_id 图像 | 缺失率 | 原始组（含回退） |
|---|---:|---:|---:|---:|---:|
| MEL | 4522 / 4522 | 1339 | 337 | 7.4525% | 1676 |
| NV | 12875 / 12875 | 7095 | 1549 | 12.0311% | 8644 |
| BCC | 3323 / 3323 | 1310 | 0 | 0.0000% | 1310 |
| AK | 867 / 867 | 324 | 0 | 0.0000% | 324 |
| BKL | 2624 / 2624 | 1269 | 198 | 7.5457% | 1467 |
| DF | 239 / 239 | 113 | 0 | 0.0000% | 113 |
| VASC | 253 / 253 | 135 | 0 | 0.0000% | 135 |
| SCC | 628 / 628 | 262 | 0 | 0.0000% | 262 |

已知 lesion_id 共 11,847 个；缺失 2,084 张（8.227074%）。原始组 13,931，合并后形成 13,927 个诊断连通分量，其中 1 个含不同标签。混合分量会在逐类诊断统计中重复出现，不能将逐类分量数直接求和当作总独立组数。

50 组精确重复图像均为两张一组，即 50 张额外重复、25,281 个不同哈希；其中 11 组跨原始 base group。原始文件全部保留。

### 已证实的冲突

| 图像 A（标签 / lesion_id） | 图像 B（标签 / lesion_id） | 字节数 | SHA256 前 16 位 | 字节比较 |
|---|---|---:|---|---|
| ISIC_0067502（MEL / lesion:BCN_0003560） | ISIC_0067980（NV / lesion:BCN_0000237） | 593611 | `33d39335eecc56d3` | 相等 |
| ISIC_0069013（NV / lesion:BCN_0000237） | ISIC_0071017（MEL / lesion:BCN_0003560） | 523816 | `d541d12ddc9cc778` | 相等 |

四张图像连接 lesion:BCN_0003560（MEL）与 lesion:BCN_0000237（NV）。独立分析重新读磁盘，重新计算两对文件的完整 SHA 并逐字节比较，全部确认一致；完整哈希与路径位于仓库外 ISIC_CONFLICT_BYTE_VERIFICATION.json。没有根据期望值伪造哈希。

**已证实：本地相同图像字节对应不同诊断标签。仍未确定：冲突源于官方发布、本地包/CSV 版本组合还是其他数据来源问题。** 本轮未对远端完整发布作逐字节同一性验证，不能据此断言官方标注或论文有缺陷。

### 分组、泄漏和八类资格

70/15/15 与每类 train≥5、val≥3、test≥5 门槛保持原值。merge_leakage_groups 在分配 split 前拒绝混合标签组，exit code=2、状态 NO_GO_DATA_INTEGRITY。没有创建任何 ISIC train/val/test 分配；泄漏检查是 **NOT_EVALUATED_NO_SPLITS_CREATED**，不能写作“0 泄漏通过”。

已知病灶 ID 数量达到预声明计数前提，但完整性先失败，八类 primary 未获认证。未删除冲突样本/类别、改标签、拆开重复组或降低门槛。缺失 lesion_id 的 8.227074% 样本另有真实病灶独立性未验证限制。

原失败 audit_summary.json 保留真实错误和未知计数 null；两个 manifest 和 class_counts.csv 仅有表头，表示未完成分组。本轮未覆盖它们。实际完成的图像数量与逐类统计已另存为 ISIC_INTEGRITY_DIAGNOSIS.json、image_integrity_inventory.csv 和本报告的 ISIC_CLASS_COUNTS.csv。空表不代表实际读取数为 0，也不能训练。

## Fish4Knowledge：有限类别范围通过

### 独立映射核验

class_id.csv 的真实表头为 file_name,cluster_id，共 27,370 行。前 20 行及样例见 CLASS_ID_MAPPING_EVIDENCE.md。官方页面将其定义为 filename→cluster ID 反查表，类别通过本地文件夹和官方代表性类别表逐项核对。cluster ID 在此作为发布类别标识，不能据此证明生物学簇纯度；本轮不进行额外生物分类学推断。

逐条同时比较 tracking ID、fish ID 和类别，共 27,370/27,370 完整匹配，27,137 条需要前导零长度规范化。只在核验中去掉明确 fish_ 前缀并比较十进制 ID；原始文件名、tracking token 和 split group 不改。无缺失、多余、重复或错类，fish ID 恰好 1…27,370。错轨迹、错类、缺行和重复的拒绝逻辑均经测试。

### 全 23 类统计

每个 primary split 单元为“图像 / 轨迹组”；全部 23 类始终保留在全量清单。

| ID | 发布类别 | 图像 | 原始/合并轨迹 | train | val | test | Primary |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | Dascyllus reticulatus | 12112 | 4240 / 4240 | 7342 / 2544 | 2365 / 848 | 2405 / 848 | 保留 |
| 2 | Plectroglyphidodon dickii | 2683 | 1225 / 1225 | 1588 / 735 | 547 / 245 | 548 / 245 | 保留 |
| 3 | Chromis chrysura | 3593 | 1175 / 1175 | 2114 / 705 | 687 / 235 | 792 / 235 | 保留 |
| 4 | Amphiprion clarkii | 4049 | 1021 / 1021 | 2513 / 613 | 825 / 204 | 711 / 204 | 保留 |
| 5 | Chaetodon lunulatus | 2534 | 536 / 536 | 1469 / 322 | 547 / 107 | 518 / 107 | 保留 |
| 6 | Chaetodon trifascialis | 190 | 79 / 79 | 101 / 47 | 26 / 16 | 63 / 16 | 保留 |
| 7 | Myripristis kuntee | 450 | 71 / 71 | 171 / 43 | 125 / 14 | 154 / 14 | 保留 |
| 8 | Acanthurus nigrofuscus | 218 | 71 / 71 | 144 / 43 | 33 / 14 | 41 / 14 | 保留 |
| 9 | Hemigymnus fasciatus | 241 | 58 / 58 | 144 / 34 | 46 / 12 | 51 / 12 | 保留 |
| 10 | Neoniphon sammara | 299 | 53 / 53 | 205 / 31 | 52 / 11 | 42 / 11 | 保留 |
| 11 | Abudefduf vaigiensis | 98 | 42 / 42 | 59 / 26 | 14 / 8 | 25 / 8 | 保留 |
| 12 | Canthigaster valentini | 147 | 28 / 28 | 64 / 16 | 26 / 6 | 57 / 6 | 保留 |
| 13 | Pomacentrus moluccensis | 181 | 27 / 27 | 133 / 17 | 39 / 5 | 9 / 5 | 保留 |
| 14 | Zebrasoma scopas | 90 | 23 / 23 | 48 / 13 | 25 / 5 | 17 / 5 | 保留 |
| 15 | Hemigymnus melapterus | 42 | 16 / 16 | 29 / 10 | 9 / 3 | 4 / 3 | 保留 |
| 16 | Lutjanus fulvus | 206 | 15 / 15 | 145 / 9 | 30 / 3 | 31 / 3 | 保留 |
| 17 | Scolopsis bilineata | 49 | 8 / 8 | — | — | — | 排除：8<15 轨迹 |
| 18 | Scaridae | 56 | 5 / 5 | — | — | — | 排除：5<15 轨迹 |
| 19 | Pempheris vanicolensis | 29 | 6 / 6 | — | — | — | 排除：6<15 轨迹 |
| 20 | Zanclus cornutus | 21 | 6 / 6 | — | — | — | 排除：6<15 轨迹 |
| 21 | Neoglyphidodon nigroris | 16 | 8 / 8 | — | — | — | 排除：8<15 轨迹 |
| 22 | Balistapus undulatus | 41 | 6 / 6 | — | — | — | 排除：6<15 轨迹 |
| 23 | Siganus fuscescens | 25 | 6 / 6 | — | — | — | 排除：6<15 轨迹 |

所有逐类图像数、原始轨迹数均与协议及官方表一致：总量 27,370 图像、23 发布类别、8,725 条轨迹。

### 各集合支持

| 范围 | train 图像/组 | val 图像/组 | test 图像/组 | 总计图像/组 |
|---|---:|---:|---:|---:|
| 23 类描述性 | 16408 / 5235 | 5443 / 1745 | 5519 / 1745 | 27370 / 8725 |
| 16 类 primary | 16269 / 5208 | 5396 / 1736 | 5468 / 1736 | 27133 / 8680 |

固定 seed=42，60/20/20 的轨迹级划分，16 类均满足 9/3/3 的最低组数，超过至少 12 类要求。ID17–23 因各自仅 8、5、6、6、8、6、6 条轨迹被排除出 primary；237 张图像、45 条轨迹仍在全量清单。

310 对完全重复图像、27,060 个不同哈希；重复均在同一原始轨迹内，未删除。原始 tracking ID、合并 group_id、相同图像 SHA 在 all 和 primary 中的跨 split 计数均为 **0**。独立脚本重读清单验证计数、组数、标签、最低支持及 CSV 全部映射，均通过。本轮两份新 manifest 与上轮清单逐字节一致，旧结果未覆盖。

Head/Medium/Tail 是 primary 训练类频三分位（6/5/5 类），训练 IF=253.172414，不是 CIFAR 绝对阈值。极少类测试可能仅 3 条轨迹；未来结论限于这 16 类，不能推广到全部 23 类。

## 状态定义与下一阶段

| 状态 | 本轮解释 |
|---|---|
| PASS | 完整范围协议通过；本轮未出现 |
| PASS_LIMITED_SCOPE | Fish 的 16 类范围通过 |
| BLOCKED_MISSING_DATA | 必需输入缺失；本轮 CSV 补齐后未触发 |
| NO_GO_DATA_INTEGRITY | ISIC 的跨标签精确重复冲突 |
| NO_GO_INDEPENDENT_GROUP_SUPPORT | primary 类数不足的状态；Fish 7 类被排除但其余 16 类达到总量门槛 |

当前仅建议下一轮为 Fish 16 类制定受限探索性训练协议；不具备直接开展两个完整领域确认性训练的条件。ISIC 暂停，不能自行修复标签后放行。骨干、两阶段适配及预算需在下一轮先行固定，本轮没有训练或新模块。

## 产物与来源

- `D:\h2t_cross_domain_audit\isic2019`：五个标准失败诊断文件、真实逐图测量、input_statistics 和新增 ISIC_INTEGRITY_DIAGNOSIS.json。
- `D:\h2t_cross_domain_audit\fish4knowledge`：五个标准完整文件、真实测量及 class_mapping_audit.json。
- `D:\h2t_cross_domain_audit\formal_data_audit_20261031_01`：测试日志、初始权限失败、输入/代码 SHA、目录检查、冲突字节证明、重复明细、最终验证及输出索引。
- 本报告目录：两数据集逐类汇总 CSV、AUDIT_EVIDENCE_SUMMARY.json、映射证据、目录报告、VALIDATION_REPORT.md 和只读分析代码。

[Fish4Knowledge 官方页](https://homepages.inf.ed.ac.uk/rbf/Fish4Knowledge/GROUNDTRUTH/RECOG/) 支持发布计数、轨迹定义和反查表含义；本轮只核查网页，无数据集下载。© 2013 Robert Fisher。引用 Boom, Huang, He, Fisher, Supporting Ground-Truth annotation of image datasets using clustering, ICPR 2012。

[ISIC 官方数据页](https://challenge.isic-archive.com/data/) 提供发布总量。原始 ATTRIBUTION.txt、LICENSE.txt（CC-BY-NC，来源包括 BCN_20000/HAM10000/MSK）保持原样。本地 CSV 记录 SHA，未声称远端完整发布字节验证通过。
