# A0失败快照：三Scale无更新重放

**门禁PASS。** 分支`experiment/fish16-amp-scale-stability-20261102`，执行代码`7e2b42a085a5840ab94dc9e8604ce3ff62beb821`。本轮只执行这三个获准的同快照反向重放；每个Scale一次、独立进程，没有额外Smoke、数据审计或参数搜索。

## 已提交修正核对

正式runner保持FP16 autocast、初始Scale65536、growth_factor2、backoff_factor0.5、enabled=True；growth_interval100000超过单条件最大91530窗口，并纳入协议哈希/manifest。原amp_recovery.py、models.py、data.py、smoke.py和上游算法未改。仅原生B0 PIF-b单Inf可跳步并回退；其他异常仍停止。逐Epoch恢复比较已使用json_normalize消除tuple/list表示差异。

语法：规定5个文件及新增重放脚本共6/6通过。CPU：AMP12/12、contract10/10、formal14/14，共36/36；四条检查命令exit_code均为0，精确命令与源码SHA见JSON。CPU检查不等同于GPU续训验证。

## 原始快照与流程

快照：`D:\h2t_cross_domain_results\fish16_formal_seed42_90x10_20261102_01\FAILURE_SNAPSHOT.pt`。
SHA256：`e07a48ec6a6cd95cc47b5c7875d6736e6c6f656fa65c05fb62c8505ac192400c`。
保存上下文为A0 Epoch4、zero-based batch976。先校验SHA再weights_only CPU读取；每组重新构建原FishModel(False)、strict-load同一model，train模式、相同保存mixed_input/target_a/target_b/lambda、原two-target CE及loss*len(targets)/16表达式。相同确定性cuDNN与TF32关闭。原快照、旧实验和初始化权重保持只读。

## 实测结果

| Scale | 全部梯度有限 | NaN | Inf | classifier.bias[0] | Logits/Loss与保存值 | 实际SGD调用 |
|---:|---|---:|---:|---:|---|---:|
| 65536 | True | 0 | 0 | -0.271240234375 | equal=True/True | 0 |
| 131072 | True | 0 | 0 | -0.271240234375 | equal=True/True | 0 |
| 262144 | False | 0 | 1 | -Inf | equal=True/True | 0 |

三组loss均为0.9629483819007874，logits/loss均与保存值逐值完全相同，最大绝对误差0。缺失梯度均为空。262144组唯一异常为classifier.bias[0]负Inf，梯度dtype=torch.float32，参数shape=[16]；无NaN及其他Inf。

每组optimizer.step、scaler.step和scaler.update均由禁止调用桩保护，实际计数全部0；参数SHA前后相同。train forward仅在一次性副本更新BN buffer，不保存回旧文件。所有逐参数梯度统计、分组最大有限绝对梯度及Scaler配置见外部JSON/日志，未上传大型张量或样本身份。

## 结论与边界

同一快照前向完全一致，只有loss scale变化；这支持该批次存在Scale依赖的AMP反向溢出。不能把历史B0 Conv1d内部溢出位置归给本次A0，A0第一个产生Inf的内部操作仍未确定。本结果也不证明之后所有Epoch均有限。

门禁通过后按任务直接启动新的六条件正式实验：`D:\h2t_cross_domain_results\fish16_formal_seed42_90x10_20261102_02`。从共同官方ImageNet初始化重新开始，不从旧部分checkpoint续训；仍保留严格非有限门禁。本文件证明重放通过，不表示90+10已完成；正式进度与结果以_02 ACTUAL_EXECUTION_STATUS及最终报告为准。

TEST图像访问=0；新图像读取=0；DataLoader=0；seed43/44=0。
