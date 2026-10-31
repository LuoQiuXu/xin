# CROSS_DOMAIN_DATA_AUDIT_REPORT

日期：2026-10-31（Asia/Shanghai）。分支：`experiment/h2t-cross-domain-data-gate-20261030`。起始 HEAD：`e65f9a9d42f42fd5cbbd7b0841f77e9fe1981fe7`。

## 执行结论

- Fish4Knowledge：**PASS_LIMITED_SCOPE**。完整 23 类 27,370 张图像、8,725 条原始轨迹均通过发布数量检查及实际像素解码；16 类满足预先锁定的 primary 门槛。
- ISIC 2019：**NO-GO（缺少必需元数据与标签）**，完整数据审计受阻。程序诊断状态为 `NO_GO_AUDIT_ERROR`，缺失输入安全停止，exit code=2；不是图像损坏、八类支持不足或方法失败的证据。
- 本轮 0 次 GPU 训练、0 次 Stage I/II、未下载数据、未改原始模型或数据、未提交或推送；没有切换分支。用户已明确确认本轮按 TXT 只做 CPU 数据审计。

## 环境与门禁

Python：`D:\app\minicondaaa\envs\xiaoju\python.exe`；Python 3.12.12；Pillow 12.3.0；Windows-11-10.0.26200-SP0。审计入口不导入 torch。

原有 CPU 契约测试 **5/5 PASS**；修复后最终 **11/11 PASS**。审计脚本与测试脚本 py_compile 通过；新增目录检查工具亦纳入最终语法检查。所有新增原始清单、图像 SHA 和诊断数据位于仓库外：

`D:\h2t_cross_domain_audit\audit_20261031_01`

## 本地目录复查与 ISIC 阻断

对 `D:\xiangmuxunlian\xin\data\` 递归检查全部 95,664 个文件条目，包括隐藏项；无遍历错误或目录链接。通过 os.walk 与 rg --hidden --no-ignore 两种方式核对。只找到一个 CSV：

`D:\xiangmuxunlian\xin\data\OfficeHomeDataset_10072016\ImageInfo.csv`（2,673,777 字节）。

以下两个文件在整个 data 树中均未发现，不能提供已存在的绝对路径：

- `ISIC_2019_Training_Metadata.csv`
- `ISIC_2019_Training_GroundTruth.csv`

真实 JPEG 根目录：`D:\xiangmuxunlian\xin\data\ISIC_2019_Training_Input\ISIC_2019_Training_Input\`，包含 25,331 个 .jpg 和 2 个 .txt（ATTRIBUTION、LICENSE）。图像 ZIP 的目录表亦不包含 CSV。图像文件数等于官方总数，但本轮没有将这等同于标签、病灶组数、可解码性或无泄漏通过。

目录全表与各类文件名样例见 `DATASET_DIRECTORY_INSPECTION.md`。ISIC 每类实际图像数、lesion_id 缺失量、独立病灶组数、逐图 SHA/损坏/重复检查及 train/val/test 支持均为 **UNVERIFIED**；未分配真实 split，不能确认 8 类 primary 门槛。预声明期望值如下，绝不当作实测：

| 类别 | 协议期望图像数 | 本地实际标签/组数 |
|---|---:|---|
| MEL | 4522 | UNVERIFIED |
| NV | 12875 | UNVERIFIED |
| BCC | 3323 | UNVERIFIED |
| AK | 867 | UNVERIFIED |
| BKL | 2624 | UNVERIFIED |
| DF | 239 | UNVERIFIED |
| VASC | 253 | UNVERIFIED |
| SCC | 628 | UNVERIFIED |

ISIC 输出的两个 manifest 及 class_counts 为只有表头的空文件，表示审计未完成；JSON 未测计数为 null，不能解释为观察到 0 张图像或 0 个类别。缺失 lesion_id 时原协议允许 image_id_fallback 并单独计数，该回退不提供真实病灶独立性保证，也不伪造 lesion_id。

## Fish4Knowledge 真实审计

只使用 `D:\xiangmuxunlian\xin\data\fishRecognition_GT\fish_image\`。23 个 fish_01…fish_23 子目录共 27,370 个 PNG；同级 mask_image 有 23 个 mask 目录及另 27,370 个 PNG，均没有混入分类审计。

每个文件按最后一个下划线拆分原始 tracking 部分与 fish ID，例如 `fish_000000009598_05281.png` → tracking=`fish_000000009598`、fish ID=5281；保留原 tracking token，未猜测或重建新轨迹。数值 fish ID 恰好覆盖 1…27,370 且每个只出现一次。

全部 27,370 张分类图像经过 Image.verify + 重新打开并 load 的真实解码，记录宽高与文件字节 SHA256。无无效记录/损坏图像/无法解析 ID；所有 23 类图像数及原始轨迹数与官方发布表一致。原始轨迹数用于发布版本比较，重复图像合并后的组数用于独立性门槛，两个口径分开保存。

共 27,060 个不同图像 SHA；310 组重复 SHA，每组两张，即 310 张重复额外图像。重复均在同一原始轨迹内：合并前后仍为 8,725 组，没有删除或覆盖图像。无跨标签组冲突。

### 全 23 类计数与 primary 范围

每个 primary split 单元为“图像数 / 独立组数”。全部 23 类仍在 manifest_all 和 class_counts；排除只影响 primary manifest。

| ID | 类别 | 全部图像 | 原始/合并组 | Primary train | Primary val | Primary test | 状态 |
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
| 17 | Scolopsis bilineata | 49 | 8 / 8 | — | — | — | 排除：仅 8 组，需至少 15 |
| 18 | Scaridae | 56 | 5 / 5 | — | — | — | 排除：仅 5 组，需至少 15 |
| 19 | Pempheris vanicolensis | 29 | 6 / 6 | — | — | — | 排除：仅 6 组，需至少 15 |
| 20 | Zanclus cornutus | 21 | 6 / 6 | — | — | — | 排除：仅 6 组，需至少 15 |
| 21 | Neoglyphidodon nigroris | 16 | 8 / 8 | — | — | — | 排除：仅 8 组，需至少 15 |
| 22 | Balistapus undulatus | 41 | 6 / 6 | — | — | — | 排除：仅 6 组，需至少 15 |
| 23 | Siganus fuscescens | 25 | 6 / 6 | — | — | — | 排除：仅 6 组，需至少 15 |

16 个 primary 类均满足 train≥9、val≥3、test≥3 条独立轨迹；达到协议至少 12 类要求。被排除的 7 类为 ID17–23，共 237 张图像、45 条轨迹，全部保留在原始数据和全量清单。不得将未来 16 类结果称作完整 23 类结果。

### 各集合实际支持

按 seed=42、每类分组 60/20/20（固定取整与门槛）划分，不按图片随机分割。

| 范围 | train 图像/组 | val 图像/组 | test 图像/组 | 总图像/组 |
|---|---:|---:|---:|---:|
| 全 23 类描述性 | 16408 / 5235 | 5443 / 1745 | 5519 / 1745 | 27370 / 8725 |
| 16 类 primary | 16269 / 5208 | 5396 / 1736 | 5468 / 1736 | 27133 / 8680 |

脚本检查及独立读取两个 manifest 复核：原始 tracking、合并 group_id、完全相同图像 SHA 在 all/primary 中的跨 split 泄漏均为 **0**；同一合并组只有一个标签。核对了每类 split 计数、全部 fish ID、全量/primary 行关系及每个 primary 类最低组数。

Primary 训练图像频次 IF=253.172414；按训练类频排序划 Head/Medium/Tail 三分位（6/5/5 类），不采用 CIFAR Many/Medium/Few 的绝对阈值。

## 审计工具修复与验证边界

1. 合成截断 JPEG 证明旧 Image.verify 单独调用会返回尺寸，但实际像素 load 抛 OSError；现增加真实解码，新增回归测试。
2. ISIC 严格检查 0/1 one-hot、UNK=0、重复/空 CSV ID、lesion_id 列、缺失元数据行和未匹配图像，避免把无效标签当作合法类别。
3. Fish ID 按数值验证全局唯一与官方范围；原始轨迹计数和重复合并后的组数分开，未修改 tracking token、标签、split 比例或最低组门槛。
4. class_counts 保留缺失的预期类别；输出目录必须位于仓库及源图像目录之外且全新，文件排他创建，防止覆盖旧结果。失败时先保存明确的错误、未知计数和空表头，再以非零退出。
5. 最终 11 项测试覆盖以上关键门禁；没有跳过失败测试。额外独立清单核查记录在 VALIDATION.json。

对 81,942 个受保护历史/数据文件逐一复核大小与 mtime_ns，观察到修改/删除 0；这是元数据一致性检查，不声称对全部历史文件重新做了内容 SHA 审计。Git 差异仅限本任务目录，未改 H2T、PI_H2T_review、此前 noisy 两阶段工具或实验资产。

## 交付与下一阶段资格

仓库外 fish4knowledge 子目录的 audit_summary.json、audit_issues.json、class_counts.csv、manifest_all.csv、manifest_primary.csv 为完整结果。isic2019 对应五文件是明确失败诊断，不能拿空清单训练。其他 CPU 测试日志、目录 JSON、RUN_MANIFEST、VALIDATION、SOURCE_REFERENCES、FISH_SPLIT_SUMMARY 及文件索引均在仓库外根目录。

Fish4Knowledge 仅取得 **16 类受限探索性数据入口资格**；ISIC 暂不建议进入训练，需先提供缺失 CSV 并用全新结果目录完成完整审计。尚需下一轮固定共同大图像骨干及 H2T/PI-H2T 两阶段实现和训练预算，本轮没有这些训练步骤。数据门禁通过或失败均不构成 H2T/PI-H2T 性能结论。

## 官方来源与署名

发布数量、文件名与轨迹含义核对：[Fish4Knowledge 官方数据页](https://homepages.inf.ed.ac.uk/rbf/Fish4Knowledge/GROUNDTRUTH/RECOG/)。© 2013 Robert Fisher；保留来源和许可说明。引用 Boom, Huang, He, Fisher, Supporting Ground-Truth annotation of image datasets using clustering, ICPR 2012。

ISIC 2019 的总量、训练标签与 common lesion identifier 元数据类型见 [ISIC 官方数据页](https://challenge.isic-archive.com/data/)。本地 ATTRIBUTION.txt、LICENSE.txt 保持原样；CC-BY-NC，来源包括 BCN_20000、HAM10000、MSK。本轮没有下载数据集。
