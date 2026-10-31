# VALIDATION_REPORT

日期：2026-10-31（Asia/Shanghai）。

| 检查 | 结果 |
|---|---|
| 分支/HEAD | 目标分支及 HEAD 保持不变 |
| 初始工作区 | 仅上轮六项已知修改，无未知冲突 |
| 首次测试 | 11 项，6 项沙箱临时目录权限错误；INITIAL_CPU_FAILURE.json 保留原因 |
| 完整复跑 | 15/15 PASS，无跳过；修复持久化后再跑 15/15 PASS |
| 语法检查 | 审计、测试、目录工具及本轮分析/报告工具 PASS |
| 解码回归 | 截断 JPEG 的实际 load 拒绝路径 PASS |
| 映射负例 | 错 tracking、错类别、缺行、重复均拒绝，能触发严格完整性门禁 |
| ISIC 解码/数量 | 25,331 张、8 类；逐类匹配，无解码失败 |
| ISIC 完整性 | FAIL：1 混合标签分量，4 图像；2 对直接字节比较相等 |
| ISIC 划分/泄漏 | 未创建 split；NOT_EVALUATED，不伪报 0 泄漏 |
| Fish 数量 | 27,370 / 23 类 / 8,725 轨迹，逐类一致 |
| Fish 类别映射 | 27,370 条双 ID 和类别全部匹配；独立脚本再次核验 |
| Fish primary | 16 类，每类9/3/3均满足，7类排除原因完整 |
| Fish split 泄漏 | 原始轨迹/合并组/精确哈希，all/primary 均 0 |
| Fish 旧清单复核 | 新旧两份 manifest 均字节相同；旧文件未写入 |
| 历史/数据保护 | 文件大小/mtime 与指定小文件 SHA；最终见 FINAL_VALIDATION.json |
| Git diff --check | 最终结果记录于 FINAL_VALIDATION.json |
| GPU/训练/下载 | 0；无 Stage I、Stage II 或 PI-H2T 调用 |
| 标签/图像修改 | 0，不解压已有图像，不改 mask、门槛或 split 规则 |
| 提交/推送/切分支 | 0 |

调用路径：main → run_audit → load_isic/load_f4k → record_image → inspect_image(verify+load) 与 sha256_file → merge_leakage_groups → allocate_group_splits → verify_no_split_leakage → 写清单。ISIC 在 merge 混合标签检查停止；Fish 执行完整路径及 class_id 门禁。

独立分析只重建诊断连通分量，不调用 split 分配；真实逐图测量在严格合并前已持久化，因此停止后无需重跑图像审计即可保全证据。失败原记录保留，实际数量以新增诊断补全。

首次报告生成尝试发生 Windows 路径转义 SyntaxError，解析期未写任何报告；改用独立报告脚本正确处理路径后生成新报告。

保护检查比较既有文件大小/mtime_ns，及上轮两报告、协议、三个 CSV 的 SHA；不将其称为全部历史文件内容校验。清单、图像哈希和详细重复记录保留仓库外。

尚未确定：ISIC 本地冲突的上游来源、2,084 张缺失 ID 的真实病灶独立性、任何模型性能及 H2T/PI-H2T 跨域效应。
