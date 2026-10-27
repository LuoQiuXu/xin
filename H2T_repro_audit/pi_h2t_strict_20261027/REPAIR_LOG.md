# 训练前修复记录

本轮开始HEAD：5bdbe5c8749cafb613215e157e7b3290f70626d6；所有修复限定于`pi_h2t_strict_20261027/`，正式训练尚未开始，未读取新模型TEST准确率后再修改协议。

## 首次CPU测试的实际失败

命令：指定xiaoju解释器`-B -X utf8 -m unittest discover -s H2T_repro_audit/pi_h2t_strict_20261027 -p test_*.py -v`。实际9项，exit1，1 failure + 1 error。

1. `test_pif_attaches_after_extractor`夹具构造[2,1,1]，与Conv1d真实权重[1,2,1]不符。仅改夹具为[1,2,1]，PIF定义/方程未改；原测试随后仍验证输出确为extractor之后的PIF。
2. `group_stats`先在float32求5/6，再转Python float，得到83.3333313465而非计数定义83.3333333333。改为Python双精度的100×整数正确数/整数总数；未放宽容差。新增1/3精度回归测试，原分组/边界测试保留。

## 关键路径核查后的训练前补强

- 新增CPU调度回归：抽取冻结H2T的数值调度函数，与新runner在warmup、epoch160/161及180/181边界逐项比较。没有将StageI换成cosine。
- 原checkpoint loader虽strict-load并检查source hash，未把文件SHA与完成后的StageI summary绑定。现先核对SHA，再反序列化；另检查run-id、protocol、subset hash。新增正确SHA、篡改SHA、未完成summary的CPU测试。禁止历史25参数替代的门禁保留。
- Preflight扩充为官方torchvision MD5、全部100类计数、每类100 TEST标签、独立seed0类内索引重建/数据逐值核对及原文件SHA。只复用现有缓存，不下载、不改变样本。
- 真实smoke使用同alpha1输入MixUp检查PIF非零/有限梯度，检查StageII有限ratio、head梯度及骨干无梯度；记录真实GPU/环境/显存。
- 正式训练增加逐epoch持久化loss/LR/PIF系数，以及StageII每批图像/标签流的SHA、全训练r/实际donor mass数组、输入FC与冻结骨干参数哈希；两组必须通过相同checkpoint、初始FC、BN buffers及每epoch输入SHA门禁。记录candidate与actual donor mass，no_fusion实际为0。
- 保存最终logits/labels/pred及PIF系数、二进制SHA和完整环境。原数据/模型/公式、BN update、W=frozen_stage1、detach=True、两组200/10/10预算不变。

这些是数据真实性、来源门禁、日志/汇报修复，不是按准确率调机制。正式StageI启动后不再编辑执行源码。最终测试、preflight、smoke和训练退出码以run内VALIDATION_REPORT为准；若任何门禁失败则停止并公开STOP报告。
