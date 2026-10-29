# 最终验证报告

结论：PASS。科学决策：NO-GO（门槛未通过，非证据缺失）。

- `stage1_same_initialization_images_and_mixup_all200_epochs`：True
- `stage2_all3400_pairs_exact_controls_BN_and_frozen_BB`：True
- `no_training_validation_clean_oracle_or_official_TEST_access`：True
- `all_final_logits_recompute_exact_metrics_interactions_and_paired_counts`：True
- `historical20543_metadata_unchanged_and_locked_source_SHA256_unchanged`：True

CPU单元测试9/9通过，正式训练前原有5项通过，补充4项CPU检查后共9项通过；测试原始输出见 `CPU_TEST_RESULTS.txt`。末次评价前9个Python文件AST语法检查通过，最终含report.py共10个文件语法检查通过，逐文件SHA见 `STATIC_CHECKS.json`。

真实dryrun6次（两个条件的Stage I各1次、Stage II各2次）均通过。正式两次Stage I各200轮、八次Stage II各10轮，合计40,800次SGD更新，所有梯度有限性检查通过；训练内无验证/TEST读取，未选best。

末轮固定模型eval各1次（八个Stage II主结果、两个Stage I仅描述）；逐项重算32行主指标、8行交互、1000行逐类准确率及400行逐类改判；class0无验证样本，准确率留空。

冻结的原始作者源文件与已用历史原始TRAIN/split通过SHA256复核。历史20543项路径/大小/mtime_ns元数据无差异；完整内容SHA校验范围见 `VALIDATION.json`，不扩大声明。所有历史资产保留。

`RUN_MANIFEST.json`提供代码、数据和全部10个正式模型的SHA，以及58个仅本地文件索引。完整源代码/配置及逐run环境记录可核查，本轮未训练PI-H2T、未更改原H2T方法。
