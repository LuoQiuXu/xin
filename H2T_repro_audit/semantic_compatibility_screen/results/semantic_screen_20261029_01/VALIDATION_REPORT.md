# Validation report — 2026-10-29

本轮完成了真实特征前向与原始`screen.py`分析；合成测试不作为科学证据。目标分支始终为`experiment/semantic-compatibility-screen-20261029`，起始HEAD `c2c1b4e92b54e764baab27ebe3fbace614bad071`，起始已跟踪/暂存变更0。七个既有未跟踪历史资产目录只读保留；没有执行切分支、git clean、reset --hard、历史数据删除或搬移。

## 实际检查与证据

|检查|结果|证据|
|---|---|---|
|官方标签/数据来源|TRAIN SHA/MD5、meta MD5通过；100类fine→coarse唯一，20个coarse各5类|`COVERAGE_SUMMARY.json`、`FINE_CLASS_COVERAGE.csv`|
|LT成员|seed0重建10,847 raw IDs；逐项等于历史Clean split和训练标签/像素|`coverage.py`实际断言及`COVERAGE_GUARD.json`|
|覆盖可行性|Few305图30类，Many8824图35类；合格210图21类，9类同coarse Many为0；优先300中205合格|305行逐图CSV、300行固定抽样CSV、20行superclass CSV|
|范围确认|用户明确选择方案2，限定210图21类；未自行拓展频组或事后挑类|`SCOPE_RESOLUTION.json`，在真实特征前向前记录|
|Stage-I合法性|Clean seed42 epoch184按TRAIN holdout选best，无TEST选模；SHA符合旧协议|`CHECKPOINT_AVAILABILITY.json`，CPU strict加载|
|checkpoint安全加载|map_location=cpu、weights_only=True，闭合NumPy数值类型白名单，无非受限checkpoint pickle回退|`checkpoint.py`；不恢复optimizer、不调用训练器|
|候选事前固定|210受体，每个32同coarse+32异coarse Many；raw-ID局部RNG；无放回|`PAIR_PLAN.json`、本地`pair_plan.npz`|
|CPU测试|最终9项契约通过；原始screen合成已知零对照/错误数据拒绝通过|`CPU_TEST_RESULTS_FEATURE_PIPELINE.json`及最终完整测试记录|
|小真实pilot|8个固定受体的498张唯一图像，真实GPU特征/原始头一致性通过|`PILOT_VALIDATION.json`；不以pilot正确率筛样本|
|全量真实特征|其余5995张+原样复用498张，共6493张；float32(6493,64)，全部有限|`FEATURE_EXTRACTION.json`及本地bank|
|架构/状态|原模型类strict加载，Option-A无PIF；eval/frozen、无Dropout/反向/optimizer，前后BN及参数hash相同|pilot/full两个JSON及对应guard|
|特征/头输入一致|原始头GPU输出与显式池化后线性输出逐位相等；FP64 NumPy几何为原FP32值精确cast|输入NPZ、runtime诊断；未声称历史特征hash重放|
|原始screen执行|原文件字节未改；默认4距离层；固定mass.25；13440候选，4138匹配、210不同受体|`screen_output/summary.json`和`matched_pairs.csv`|
|同受体控制|候选内rec_feat逐字节相同，原图/raw ID固定；same/diff mass逐个严格.25|输入及`ARTIFACT_VALIDATION.json`|
|距离审计|每个受体/每个匹配的残差已保存；中位.4298%、p90 3.1027%、最大23.9048%|`analysis_v2/RECIPIENT_ANALYSIS.csv`、`MATCH_DISTANCE_AUDIT.csv`|
|预设敏感性|既有匹配相对gap≤5%，3906对仍210受体；不新增供体/比例/特征|`ANALYSIS_SUMMARY.json`；预先写入计划|
|分析复核|从真实输入独立重算原screen匹配结果，最大误差0；再独立核对210受体效应和向量化类聚类区间|`ANALYSIS_SUMMARY.json`、`ARTIFACT_VALIDATION.json`|
|结论门槛|质量项通过；标签一致性效应<5pp、两项95%区间跨0，联合信号门槛不通过|`analysis_v2/SCIENTIFIC_DECISION.json`|
|历史保护|4880个历史文件路径/大小/mtime_ns一致；checkpoint和8个直接来源/协议SHA复核|`ARTIFACT_VALIDATION.json`，不冒充全历史逐字节SHA|
|官方TEST|本轮各阶段文件打开审计中TEST尝试0；未调用旧Dataset包装器|各`*_GUARD.json`；目录元数据stat不是读取TEST内容|
|文件/忽略|数组、缓存、不完整首次导出保留本地并忽略，所有文件SHA/大小可审查|`ARTIFACT_INDEX.csv`；不自哈希|

## 实际执行顺序与命令

工作目录`D:\xiangmuxunlian\xin`，始终使用`D:\app\minicondaaa\envs\xiaoju\python.exe -B -X utf8`，未更换或安装Python环境。脚本目录均为`H2T_repro_audit/semantic_compatibility_screen/`：

1. 完整阅读两份任务文件，以及两份旧donor/PIF否定报告。第三份旧输入偏移报告在本分支不存在，通过`git show 8b00d96b591a5e600048821c0f6915c7c70fc886:H2T_repro_audit/pi_failure_screen/results/FAILURE_SCENARIO_REPORT.md`只读获取，未切分支或恢复历史文件；来源SHA在`PRIOR_EVIDENCE_REVIEW.json`。
2. `coverage.py` → exit0；`checkpoint.py` → exit0，仅CPU strict构建，尚无模型前向。
3. 初始合成测试7项中1项fixture失败；修正fixture后7项通过。候选抽样函数新增测试后9项通过。具体stdout和实际失败原因全部保留。
4. 用户确认方案2，写入`SCOPE_RESOLUTION.json`；锁定`PRE_ANALYSIS_PLAN.json`，未因后续输出修改门槛。
5. `extract_features.py plan` → exit0。
6. `extract_features.py pilot` → exit0。
7. `extract_features.py features` → exit0。
8. 原始screen完整命令（输出目录先通过不存在检查）：

```powershell
& 'D:\app\minicondaaa\envs\xiaoju\python.exe' -B -X utf8 H2T_repro_audit/semantic_compatibility_screen/screen.py --input H2T_repro_audit/semantic_compatibility_screen/results/semantic_screen_20261029_01/primary_cache/pairs.npz --output H2T_repro_audit/semantic_compatibility_screen/results/semantic_screen_20261029_01/screen_output --lambda-fixed 0.25
```

9. `analyze.py`第一次导出exit1：NumPy bool序列化错误。已保留代码和六份输出hash，未将未完成决策文件用作科学结论。
10. 修正为Python bool后用新输出路径重算：

```powershell
& 'D:\app\minicondaaa\envs\xiaoju\python.exe' -B -X utf8 H2T_repro_audit/semantic_compatibility_screen/analyze.py --analysis-output H2T_repro_audit/semantic_compatibility_screen/results/semantic_screen_20261029_01/analysis_v2
```

exit0。未再执行GPU前向或重新匹配；第一次已完成的五份数值CSV/summary与v2逐字节相同，第六份未完成决策JSON原地保留。只修改本任务新增工具的序列化/输出目录，没有修改历史代码或数值门槛。

11. 最终全部CPU测试与语法：

```powershell
& 'D:\app\minicondaaa\envs\xiaoju\python.exe' -B -X utf8 -m unittest discover -s H2T_repro_audit/semantic_compatibility_screen -p 'test_*.py' -v
```

12. 最终只读复核（CPU加载/数组与CSV计算，**不重复模型前向**）：

```powershell
& 'D:\app\minicondaaa\envs\xiaoju\python.exe' -B -X utf8 H2T_repro_audit/semantic_compatibility_screen/verify_outputs.py
git diff --check
```

最终测试/语法及复核输出分别以`FINAL_CHECKS.json`、`ARTIFACT_VALIDATION.json`为准；只有通过后才交付。实际阶段工具回执与失败情况记录在`EXECUTION_RECORD.json`。当前路径已有结果，不能原地重跑覆盖；恢复到新目录的方法见上级[README](../../README.md)。

## 问题定位与仍未验证事项

- 初始fixture只提供了同coarse的Many类，遗漏另一语义臂的合法Many类。实现正确拒绝，修正fixture增加所缺供体类；真实覆盖计数及函数从未因此改变。
- 分析导出失败发生于数值计算完成后，原因是NumPy bool不能直接由标准JSON编码器处理。显式类型转换和新目录修复；未覆盖旧文件，也未借机改数据、比例、距离层数或模型。
- 原始screen的`screen_sufficient_pairs`只计行数，不能单独代表≥30不同受体或科学GO。本轮独立按210不同受体、21类、13superclass及预设效应门槛判断。
- 48.57% clean受体已被旧头误判；训练集几何、近似距离控制、Many类频/norm差异、21类选择偏差和共享供体依赖仍限制解释。未验证作者PI-H2T完整实现、随机mask风险分布、独立验证/TEST泛化、训练收益或新模块。原生PI比例可选项按计划跳过。

最终NO-GO只表示本轮联合推进证据不足，不能改写为语义作用已被严格排除。没有进入Stage-II，也没有设计下一组新机制。
