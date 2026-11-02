# Fish16 AMP修正后正式训练记录（_02）

**状态：FAILED_STOPPED，90+10未完成。** 旧快照的三Scale重放门禁通过后，已真实启动新的六条件正式训练；随后在A0第66轮遇到新的致命梯度事件并停止。没有自动重试或修复后继续。

分支：`experiment/fish16-amp-scale-stability-20261102`。执行代码提交：`7e2b42a085a5840ab94dc9e8604ce3ff62beb821`。
正式进程exit_code=1，耗时11638.167360600084秒，正式启动次数1。

## 规定检查及原失败快照门禁

规定5个文件与新增重放脚本语法6/6通过；CPU AMP12/12、contract10/10、formal14/14，共36/36；所有命令exit_code=0。精确命令、耗时和SHA见JSON。已提交AMP修正未重复改写：FP16、init_scale65536、growth_interval100000、growth_factor2、backoff_factor0.5、enabled=True；策略纳入协议哈希与manifest。原amp_recovery.py、models.py、data.py、smoke.py及上游算法保持不变，json_normalize恢复修正保留。

旧失败快照SHA256：`e07a48ec6a6cd95cc47b5c7875d6736e6c6f656fa65c05fb62c8505ac192400c`。三组独立进程各反向一次，不读取图像或DataLoader；strict-load同一保存模型和mixed batch，原MixUp CE与原乘除表达式，所有SGD/scaler step/update实际调用数0、参数SHA不变。

| Scale | 全部梯度有限 | NaN/Inf | classifier.bias[0] | logits/loss与保存值 |
|---:|---|---|---|---|
| 65536 | True | 0/0 | -0.271240234375 | 完全相同 |
| 131072 | True | 0/0 | -0.271240234375 | 完全相同 |
| 262144 | False | 0/1 | -Inf | 完全相同 |

65536与131072有限，262144唯一负Inf精确重现旧classifier.bias[0]异常；三组前向最大绝对误差0。因此按协议直接启动_02，不新增Smoke或其他实验。该证据只确定旧快照的Scale依赖反向边界，不保证后续所有批次有限。

## 实际执行状态

| 条件 | 状态 | 完成/预算Epoch | 完整Epoch更新 | 未完成Epoch更新 | 总实际更新 | 总尝试窗口 | AMP跳步 | Validation Overall/Macro/Tail图像 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| A0 | FAILED_PARTIAL | 65/90 | 66105 | 955 | 67060 | 67061 | 0 | 99.53668975830078 / 97.84933471679688 / 97.0 |
| B0 | NOT_STARTED | 0/90 | 0 | 0 | 0 | 0 | 0 | None / None / None |
| A1 | NOT_STARTED | 0/10 | 0 | 0 | 0 | 0 | 0 | None / None / None |
| A2 | NOT_STARTED | 0/10 | 0 | 0 | 0 | 0 | 0 | None / None / None |
| B1 | NOT_STARTED | 0/10 | 0 | 0 | 0 | 0 | 0 | None / None / None |
| B2 | NOT_STARTED | 0/10 | 0 | 0 | 0 | 0 | 0 | None / None / None |

最近Validation来自最后完整的A0 Epoch65，仅使用锁定的5396张validation图像；第66轮未完成，不编造其验证指标。65轮共66105次更新，部分第66轮955次更新，总67060次；再加门禁拒绝的窗口，共67061次尝试。原始状态根节点为FAILED_STOPPED；条件内部RUNNING是被中断的游标，不能视为仍在训练或已完成。所有65轮的训练loss、LR、速度、时间、内存、16类/分组指标均保存在CSV及外部JSON中。

## 新错误：按门禁停止

条件A0；Epoch66；zero-based batch955（第956批）；Scale65536；无缺失梯度。

- `extractor.backbone.conv1.weight`：147个NaN，0个Inf，shape=[64,3,7,7]，梯度dtype=torch.float32。
- `extractor.backbone.bn1.weight[32]`：1个负Inf，shape=[64]，梯度dtype=torch.float32。
- `extractor.backbone.bn1.bias[32]`：1个负Inf，shape=[64]，梯度dtype=torch.float32。
- 合计147个NaN、2个负Inf；这不是获准的B0 PIF-b单Inf异常。
- 失败窗口scaler.step=False、scaler.update=False，真实SGD计数67060未变，AMP原生跳步0。
- 保存的logits有限=True，loss有限=True，loss=0.3762158751487732；unscale前后Scale均65536。
- 完整模型、optimizer、GradScaler、RNG、输入/targets/logits/loss及梯度快照已保存，snapshot_error=null。
- NaN位置在FAILURE.json只列前32个并明确truncated；完整梯度保留在外部快照，没有上传大型张量或样本身份。

新失败快照SHA256：`de6f9b717e2dc334fe9b80492a64ef12945fdf3d12d2970b5abd7dfaaf431a82`。
65536下出现上述非有限梯度已直接观察；第一内部非有限操作、新批次的安全Scale范围及更低Scale是否有限均**未确定**。本轮没有对此新失败追加反向重放或实验。不能将旧A0 classifier-bias事件或历史B0 Conv1d诊断直接认定为新失败的根因。保留严格门禁后立即终止后续五个条件，没有更改Scale、精度、学习率、batch、loss、算法或数据来继续。

## 数值配置与checkpoint

全部65个完整Epoch初末Scale均为65536，部分第66轮同样为65536；自动Scale增长事件0，原生AMP跳步0，没有B0事件（B0未启动）。旧失败位置A0 Epoch4/batch976已在新正式路径中经过，未再次停止；这不代表本次90轮稳定性验证通过。

新的A0从共同官方19c8e357 ImageNet初始化开始，未传--resume，未加载任何_01部分训练checkpoint。B0与Stage-II未启动，没有epoch90最终源checkpoint，不允许将epoch65或best-validation替代。last.pth已确认formal=True、smoke_only=False、epoch_completed65、策略growth_interval100000，record与实际状态一致。best-validation仅用于报告，源策略仍固定最终epoch90。GPU恢复未执行；CPU测试不能证明全部GPU恢复场景。

## 最后完整Epoch65的Validation

所有accuracy单位为%。

| Overall | Macro | Head/Medium/Tail图像加权 | Head/Medium/Tail类宏平均 | Worst-class |
|---:|---:|---|---|---:|
| 99.536689758 | 97.849334717 | 99.701377869/97.435897827/97.000000000 | 99.748963674/95.757575989/97.661538696 | 78.787879944 |

| 类索引 | A0 Epoch65准确率 |
|---:|---:|
| 0 | 99.661727905 |
| 1 | 99.268737793 |
| 2 | 99.563316345 |
| 3 | 100.000000000 |
| 4 | 100.000000000 |
| 5 | 100.000000000 |
| 6 | 100.000000000 |
| 7 | 78.787879944 |
| 8 | 100.000000000 |
| 9 | 100.000000000 |
| 10 | 100.000000000 |
| 11 | 92.307693481 |
| 12 | 100.000000000 |
| 13 | 96.000000000 |
| 14 | 100.000000000 |
| 15 | 100.000000000 |

6/5/5训练频次分组：`{"Head": [0, 3, 2, 1, 4, 9], "Medium": [6, 15, 7, 8, 12], "Tail": [5, 11, 10, 13, 14]}`。
best-validation选择信息（独立于最后完整Epoch）：`{"overall": 99.59229278564453, "epoch": 63}`。

## 配对结果与科学限制

A2-A1=null；B2-B1=null。必要条件没有完成，不能计算配对方法效果，不能据本次运行认定H2T/PI-H2T性能退化或跨域泛化。这里只确认旧快照Scale边界与新运行的真实停止事件。一个seed也不足以形成多种子结论。

## 产物及保护

正式输出：`D:\h2t_cross_domain_results\fish16_formal_seed42_90x10_20261102_02`。
原始快照重放：`D:\h2t_cross_domain_results\fish16_a0_amp_scale_replay_20261102_01`。
CPU预检及保护校验：`D:\h2t_cross_domain_results\fish16_formal_preflight_ampfix_20261102_01`。
旧_01清单及全部20个已索引文件SHA复核一致；历史结果、checkpoint与源文件保持原样。新运行所有索引文件SHA也已复核，checkpoint与原始日志留在仓库外。Git仅发布本轮工具和小体积报告、逐Epoch聚合指标、SHA证据。
环境：`{"python_executable": "D:\\app\\minicondaaa\\envs\\xiaoju\\python.exe", "python": "3.12.12", "torch": "2.14.1+cu132", "cuda": "13.2", "cudnn": 92400, "gpu": "NVIDIA GeForce RTX 5070 Laptop GPU"}`。

**TEST图像访问/评价=0；seed43/44=0；新GPU Smoke=0；新失败后追加GPU/推理/重放=0；自动重试=0。** 本轮已停止等待科学审阅，不自行提出或执行下一组实验。
