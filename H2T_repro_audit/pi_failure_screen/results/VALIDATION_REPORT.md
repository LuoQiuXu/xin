# Validation report — 2026-10-28

本轮已完成真实 checkpoint 推理与配对分析；CPU合成PASS不作为实证结论。唯一分支为 `experiment/pi-h2t-failure-scenario-screen-20261028`，未新建或切换到其他分支。起始已跟踪/暂存变更为0；六个既有未跟踪历史资产目录保留，未纳入本次提交。

## 验证结果

|检查|实际结果|证据|
|---|---|---|
|用户提供的CPU合成测试|修改前、扩展分析后均exit0；正确配对PASS，错误配对被拒绝|`EXECUTION_RECORD.json`，原`test_screen.py`字节未改|
|CPU contracts|最终10项全部通过，exit0|`FINAL_CPU_TEST_RESULT.json`保存完整stdout|
|Python语法|本任务全部6个Python文件通过；仅compile字符串，不写历史pycache|`FINAL_STATIC_CHECKS.json`|
|静态差异|`git diff --check` exit0；原TXT、README和原测试未改|`FINAL_STATIC_CHECKS.json`|
|checkpoint可用性|4个Stage-II + 2个Stage-I安全CPU加载；4个eval模型strict=True成功|`CHECKPOINT_METADATA_NUMPY_COMPAT.json`、`CHECKPOINT_AVAILABILITY.json`|
|配对来源|每对共享Stage-I；全部骨干/PIF参数精确等于其Stage-I；最终含BN的完整骨干状态在对内精确相同|186/192状态张量；分类头不同；没有重置BN|
|TRAIN-only数据|重建10,847个LT索引；1,980张固定holdout与训练raw-ID/像素交集均0；像素/标签缓存逐项吻合|`DATA_ATTESTATION.json`|
|视觉审查|固定9图×5条件在推理前检查，保留全部条件，无选择性删图|`VISUAL_AUDIT.json`、`VISUAL_CORRUPTION_AUDIT.png`|
|事前门槛|.5pp、3个净正确数及至少3张负DiD图像，双强度；没有结果后改阈值|门槛SHA同时保存在视觉审查和推理运行JSON|
|真实GPU推理|20/20模型×条件完成，每份(1980,100)有限float32 logits，int64 ids/labels；无pickle|`INFERENCE_VALIDATION.json`，20份本地NPZ|
|eval保护|eval/inference_mode、无Dropout、requires_grad=False；每条件前后全部参数/BN哈希不变|`INFERENCE_VALIDATION.json`|
|输入一致性|所有模型在同一条件的归一化输入SHA一致|`common_normalized_input_hashes`|
|官方TEST / 历史写入|prepare/infer Python文件打开审计均0次尝试|`PREPARE_GUARD.json`、`INFER_GUARD.json`；历史文件stat不是读取TEST内容|
|配对分析|两组均exit0；Overall/Macro/Many/Medium/Few/Worst-Class、逐图转对/转错与DiD齐全|主results和`pi_local_pair/`内CSV及逐图本地NPZ|
|筛查与图|全部16个pair×family×group检查，2个本地Few候选；2张图均已视觉检查|`SCREEN_CANDIDATES.csv`、`SCREEN_DECISION.json`、`GROUPED_*.png`|
|CPU产物复核|独立从20份真实预测重算20行指标及40行DiD；核对6个checkpoint、15个源码/协议文件SHA|`ARTIFACT_VALIDATION.json`；验证器成功后才交付|
|历史保护|4,108个历史文件路径/大小/mtime_ns逐项一致；没有新增、删除或变动|同上；元数据比较不冒充全历史逐文件内容SHA|
|文件索引/大文件|小体积报告/CSV/JSON/科学图提交；20份logits、2份逐图配对NPZ和缓存忽略、仅留本地|`ARTIFACT_INDEX.csv`记录大小和SHA，不自哈希|

CPU10项覆盖raw-ID噪声与batch顺序无关、噪声强度、原图不变、3×3模糊核/通道独立/常量不变、非法像素、浮点标签、重复/浮点ID、非有限或错误shape、排序配对、已知数值的DiD、拒绝覆盖及两个强度/多图门槛。拒绝覆盖测试确认已有CSV字节不变。

## 实际执行命令

工作目录 `D:\xiangmuxunlian\xin`。解释器始终使用用户指定路径；未安装或更换Python环境。以下命令按本次顺序执行，均exit0（最终验证器状态另由其生成的JSON与交付前命令确认）：

```powershell
& 'D:\app\minicondaaa\envs\xiaoju\python.exe' -B -X utf8 H2T_repro_audit/pi_failure_screen/test_screen.py
& 'D:\app\minicondaaa\envs\xiaoju\python.exe' -B -X utf8 -m unittest discover -s H2T_repro_audit/pi_failure_screen -p 'test_*.py' -v
& 'D:\app\minicondaaa\envs\xiaoju\python.exe' -B -X utf8 H2T_repro_audit/pi_failure_screen/inference.py prepare
# 在此处先检查VISUAL_CORRUPTION_AUDIT.png并写入VISUAL_AUDIT.json；不是自动盲目批准。
& 'D:\app\minicondaaa\envs\xiaoju\python.exe' -B -X utf8 H2T_repro_audit/pi_failure_screen/inference.py infer
& 'D:\app\minicondaaa\envs\xiaoju\python.exe' -B -X utf8 H2T_repro_audit/pi_failure_screen/screen.py --manifest H2T_repro_audit/pi_failure_screen/manifest.json --output H2T_repro_audit/pi_failure_screen/results
& 'D:\app\minicondaaa\envs\xiaoju\python.exe' -B -X utf8 H2T_repro_audit/pi_failure_screen/screen.py --manifest H2T_repro_audit/pi_failure_screen/manifest_pi_local.json --output H2T_repro_audit/pi_failure_screen/results/pi_local_pair
& 'D:\app\minicondaaa\envs\xiaoju\python.exe' -B -X utf8 H2T_repro_audit/pi_failure_screen/summarize.py
& 'D:\app\minicondaaa\envs\xiaoju\python.exe' -B -X utf8 -m unittest discover -s H2T_repro_audit/pi_failure_screen -p 'test_*.py' -v
& 'D:\app\minicondaaa\envs\xiaoju\python.exe' -B -X utf8 H2T_repro_audit/pi_failure_screen/verify_outputs.py
git diff --check
```

本次结果已存在，上述输出路径不能原地重跑覆盖。prepare/infer/analysis/summary/validator均有新建模式或明确拒绝覆盖保护；不要删除旧结果来重跑。命令及兼容性过程见 `EXECUTION_RECORD.json`；checkpoint和数据路径由已提交清单锁定。读PDF仍用同一Python解释器，只为pypdf添加现有打包依赖搜索路径，没有改变训练环境。原始数据包装器会同时构建TEST，故实际输入使用原始TRAIN pickle（先验已知SHA）和固定NPZ索引，模型类与ToTensor/Normalize来自保留历史源码。

## 遇到的问题与处理

初始 `torch.load(weights_only=True)` 被旧checkpoint中的NumPy scalar/dtype元数据阻止；第一次受限类型补全仍缺少保存RNG状态使用的UInt32DType。两次均在CPU元数据阶段停止，未执行模型前向。最后仅加入闭合的基础数值NumPy dtype/array/scalar重建白名单，始终保持weights_only=True，六份均成功加载。没有使用weights_only=False回退，也没有保存或转换旧checkpoint。初始错误快照 `CHECKPOINT_METADATA.json` 保留，最终可用性以 `CHECKPOINT_AVAILABILITY.json` 为准；初始兼容性错误不等于最终权重不可用。

本次没有训练、反向传播、优化器更新、额外参数搜索、mask/ratio/loss/donor变体，也没有运行完整历史审计或preflight。读取过旧报告/配置中的历史TEST指标作为来源背景，没有打开官方TEST数据文件。三个历史重点保护目录及其余既有实验资产都未改写，未执行git clean、reset --hard、删除或搬移。

## 仍未验证的事项

作者完整PI-H2T协议/权重一致性、旧T3完整历史Git版本、最终IEEE文本与arXiv v2逐项等价、训练种子层面的独立重复、独立外部holdout、class0行为、完整label preservation及具体输入偏移场景的新颖性。None of these is promoted to PASS. 原型Few优势侵蚀通过的是本轮限定的筛查门槛；最终研究决策仍为NO-GO。
