# Fish16 Seed-42 正式90+10运行记录

**实际状态：FAILED_STOPPED。** 这是正式训练尝试，不是新的Smoke或机制审计。

代码执行提交：`c6805db4b1a806b2a7830a48d131f9d1acfa617e`；分支`experiment/fish16-formal-90x10-20261102`。运行以新的ImageNet初始化启动，未使用任何smoke checkpoint。
CPU语法检查2/2、单元测试34/34全部通过后直接启动。未增加GPU Smoke。

## 实际进度与真实更新

| 条件 | 状态 | 已完成/预算Epoch | 已提交Epoch更新 | 当前未完成Epoch更新 | 总已观测更新 | AMP跳步 | 最近已完成Epoch Validation Overall/Macro |
|---|---|---:|---:|---:|---:|---:|---|
| A0 | FAILED_PARTIAL | 3/90 | 3051 | 976 | 4027 | 0 | 68.8658218383789 / 80.57879281044006 |
| B0 | NOT_STARTED | 0/90 | 0 | 0 | 0 | 0 | None / None |
| A1 | NOT_STARTED | 0/10 | 0 | 0 | 0 | 0 | None / None |
| A2 | NOT_STARTED | 0/10 | 0 | 0 | 0 | 0 | None / None |
| B1 | NOT_STARTED | 0/10 | 0 | 0 | 0 | 0 | None / None |
| B2 | NOT_STARTED | 0/10 | 0 | 0 | 0 | 0 | None / None |

已完成Epoch以实际状态、checkpoint游标及记录确认，未把预算当完成。最近的validation来自最后完整Epoch，不把未完成Epoch的指标补成结果。所有16类accuracy、6/5/5分组、训练loss、LR、时间和内存均见结构化结果及外部逐Epoch文件。

环境：{'python_executable': 'D:\\app\\minicondaaa\\envs\\xiaoju\\python.exe', 'python': '3.12.12', 'torch': '2.14.1+cu132', 'cuda': '13.2', 'cudnn': 92400, 'gpu': 'NVIDIA GeForce RTX 5070 Laptop GPU'}。

## 固定日程、输入与恢复规则

A0/B0预算90，Stage-II四条件预算10；完整LR表在FORMAL_RUN_MANIFEST。Stage-I五轮warmup0.002至0.01后按预先记录公式cosine；Stage-II第一轮0.01。没有从validation调参。
复用未修改的models.py、data.py、amp_recovery.py及上游方法。Stage-II只允许最终formal epoch90初始化，提取器参数及BN冻结。现有smoke checkpoint保留不动。
实际optimizer.step由原生post-hook记录；仅B0的PIF-b单Inf可原生跳步，且下一窗口必须真实更新。NaN、缺失梯度、其他Inf及A0/Stage-II跳步均停止。新formal日志保存PIF Inf正负计数与坐标。未硬编码旧Smoke的跳步次数或批次位置。
Checkpoint保存model、optimizer、scaler、epoch scheduler及Python/NumPy/CPU/CUDA RNG，可恢复至原子提交的Epoch边界；未提交的部分Epoch只可在显式resume后记录重放，不静默重启/跳过完整Epoch。本次没有自动resume或改设置重试。

## 实际Scale增长与失败窗口

| 条件/Epoch/batch | 原生事件 | scale前->后 | 真实SGD计数前->后 |
|---|---|---|---|
| A0/2/982 | optimizer_step | 65536.0->131072.0 | 1999->2000 |
| A0/4/948 | optimizer_step | 131072.0->262144.0 | 3999->4000 |

失败窗口scale在unscale前后均为262144，loss=0.962948381901，logits/loss均有限；SGD调用数4027未变，scaler.step/update均未调用。没有获准AMP跳步或手动回退。梯度异常的第一内部产生操作未在本轮另外诊断。

## 失败与现场

`GradientContractError: Non-finite gradients are not permitted for this condition: classifier.bias`
失败上下文：`{"condition": "A0", "epoch": 4, "batch": 976, "samples_before_window": 15616, "attempted_before_window": 976, "actual_epoch_updates_before_window": 976, "process_optimizer_calls_before_window": 4027, "scale_before_unscale": 262144.0}`。
梯度诊断：`{"missing_parameters": [], "nan_count": 0, "inf_count": 1, "nonfinite_parameters": ["classifier.bias"], "nonfinite_positions": {"classifier.bias": [[0]]}, "nonfinite_details": [{"name": "classifier.bias", "shape": [16], "gradient_dtype": "torch.float32", "nan_count": 0, "positive_inf_count": 0, "negative_inf_count": 1, "inf_count": 1, "inf_positions": [[0]], "inf_positions_truncated": false, "nan_positions": [], "nan_positions_truncated": false}], "scaler_step_called": false, "scaler_update_called": false}`。
失败快照存在=True；记录的真实进程optimizer调用数=4027。原始完整日志、参数/optimizer/scaler/RNG/输入及梯度快照均保留在外部。
遇到此真实错误即停止全部后续条件；没有修改AMP scale、精度、LR、batch、算法或数据来继续。失败步是否调用原生scaler.step/update见诊断布尔字段，不能把门禁拒绝当作已恢复AMP跳步。

## 已提交Epoch的Validation诊断

| 条件 | Overall | Macro | Head/Medium/Tail图像加权 | Head/Medium/Tail类宏平均 | Worst-class |
|---|---:|---:|---|---|---:|
| A0 Epoch3 | 68.865822 | 80.578793 | 67.449730/94.139191/71.000000 | 87.858466/90.000000/62.421977 | 0.000000 |
| B0 | — | — | — | — | — |
| A1 | — | — | — | — | — |
| A2 | — | — | — | — | — |
| B1 | — | — | — | — | — |
| B2 | — | — | — | — | — |

每类accuracy百分比（索引0..15；分组见JSON）：

| 类索引 | A0 | B0 | A1 | A2 | B1 | B2 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 32.600422 | — | — | — | — | — |
| 1 | 99.085922 | — | — | — | — | — |
| 2 | 99.708878 | — | — | — | — | — |
| 3 | 96.121208 | — | — | — | — | — |
| 4 | 99.634369 | — | — | — | — | — |
| 5 | 65.384613 | — | — | — | — | — |
| 6 | 100.000000 | — | — | — | — | — |
| 7 | 66.666672 | — | — | — | — | — |
| 8 | 100.000000 | — | — | — | — | — |
| 9 | 100.000000 | — | — | — | — | — |
| 10 | 78.571426 | — | — | — | — | — |
| 11 | 96.153847 | — | — | — | — | — |
| 12 | 100.000000 | — | — | — | — | — |
| 13 | 72.000000 | — | — | — | — | — |
| 14 | 0.000000 | — | — | — | — | — |
| 15 | 83.333328 | — | — | — | — | — |

## 配对效应

A2-A1：null。
B2-B1：null。
null表示所需条件尚未全部完成，不计算伪配对差值。A2对B2不作单独方法因果比较。一个seed的pilot不建立跨域泛化结论。

## 恢复实现的已知边界

本轮只验证了CPU恢复元数据和RNG往返，没有执行GPU续训。执行版本在恢复时直接比较逐Epoch JSON与checkpoint记录；若记录包含溢出dataclass的tuple字段，JSON会变为list，可能使该恢复分支拒绝语义一致的数据。B0尚未运行，该分支不是本次A0数值错误的原因。本轮未为此改写已执行源码或追加续训，不能把CPU测试通过解释为所有GPU恢复场景均已验证。

## 产物

运行目录：`D:\h2t_cross_domain_results\fish16_formal_seed42_90x10_20261102_01`。
CPU预检目录：`D:\h2t_cross_domain_results\fish16_formal_preflight_20261102_01`。
Git仅发布代码、协议及精简结果/哈希证据；checkpoint、逐批输入控制、原始日志、采样计划及完整梯度现场留本地。
**TEST图像访问/评价=0，seed43/44=0。正式尝试完成或失败后停止等待科学分析，不自行追加实验。**
