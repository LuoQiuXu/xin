# NTC-BCL 训练与最终分类结果

完整原始报告与汇总记录包含协议、先验估计、检查点和最终数值。[科学报告](../NTC_BCL_ROOT_FIX_REPORT.md) · [A/B/C 汇总 JSON](../comparison_summary.json) · [运行配置](../environment.json) · [先验反演](../corrected_prior.json)

## 逐轮训练记录

- B 组使用前向校正 LC 和原始硬标签 BCL。[200 轮 CSV](NTC_LC_HARD_BCL/history.csv) · [200 轮训练日志](NTC_LC_HARD_BCL/training.log)
- C 组使用前向校正 LC。epoch index 0–19 记录实例级 NT-Xent，index 20–199 记录 Soft-BCL。[200 轮 CSV](NTC_FULL_SOFT_BCL/history.csv) · [200 轮训练日志](NTC_FULL_SOFT_BCL/training.log)
- 历史 A 组使用标准 Logit Compensation 与硬标签 BCL。[历史 200 轮 CSV](BCL_sym04/history.csv)
- 原始完整进程日志包含 B、C 训练的全部 400 条 epoch 记录、十份 TRAIN 诊断及末轮评估进度。[完整 formal.log](formal.log)

CSV 保留原始 loss_lc、loss_bcl、loss_total，并按实际启用的损失拆到独立列。总损失为 2.0×LC + 0.6×BCL。B、C 的 LC 列均为前向校正后的 LC。空单元格表示该项未启用或原始日志没有记录。q 的逐 epoch 置信度、熵和分布没有写入原始训练历史。

## 官方 CIFAR-100 TEST 分类结果

每一行类别有 100 个 TEST 样本。每类 CSV 的差值单位为百分点，Many、Medium、Few 使用原始 TRAIN 类频定义。[逐类准确率和差值](classification/per_class_accuracy.csv) · [频次分组统计](classification/many_medium_few_summary.csv)

混淆矩阵是整数计数，行是真实类别 ID，pred_000 到 pred_099 对应预测类别 ID。[A 矩阵](classification/confusion_matrix_A.csv) · [B 矩阵](classification/confusion_matrix_B.csv) · [C 矩阵](classification/confusion_matrix_C.csv)

A 矩阵使用归档预测。原始 B、C 正式评估只写了每类正确数，因此补充发布阶段用同一 TEST、已校验的 Epoch 200 权重和原评估代码各做一次只读前向。一次 TEST 文件读取由两组共享，三组矩阵的对角线各 100 项均与原始记录逐项相等。原运行的正式评估记录和结果保留原值。[发布校验记录](publication_receipt.json)

## 验证与运行异常

初次 CPU 测试识别出转移概率先在 FP32 取对数再转 FP64 的精度差异。修复后 106 项测试全部通过，保留 1 条既有审计脚本的无效转义警告。B、C 真实 GPU Smoke 各更新一次，原正式训练退出码为 0，两组各完成 8,600 次 SGD 更新。
