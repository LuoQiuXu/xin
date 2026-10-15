# 真实 CIFAR100-LT 1+1 epoch dry-run：PASS

仅执行本轮批准的两项任务；未启动正式 200+10 epochs，未比较 detach 两种模式的性能。

## 本轮修改

|文件|修改|
|---|---|
|utils/logger.py|增加 PI_H2T.ratio_detach，默认 True|
|Stage-II 新 YAML|显式写入 ratio_detach: true|
|pi_h2t.py|fusion_ratio 增加开关；True 时 detach，False 时保留通向 classifier weight 的计算图，公式不变|
|train_pi_h2t.py|仅将 H2TF 计算移出 extractor 的 no_grad；仍对 backbone/PIF no_grad；增加日志与 --dry-run-one-epoch，保留200/10学习率horizon；ratio统计时 detach 防止持有计算图|
|test_pi_h2t_smoke.py|检查 True/False 前反向、False 对 weight 的 ratio 梯度非零、两种模式前向完全一致|
|pi_dry_run_audit.py|仅观测 loss/logit/gradient/state/checkpoint 的验证钩子，不修改梯度与训练目标|
|build_review_bundle.py|把新文件和配置开关加入审核diff|

没有修改 PIF 初始化、数学公式、legacy methods.py、官方训练脚本、sampler、optimizer、scheduler、BN policy、checkpoint 字段结构。
开关用于本次默认 paper_theoretical_bounds 路径；legacy H2Tv2 消融仍保持原行为，本轮没有运行该消融实验。

## 真实运行设置

数据 D:/xiangmuxunlian/新/H2T/data/cifar100；IF100，ResNet32，CE，batch128，seed123。
通过 --dry-run-one-epoch 截止运行，不把配置 num_epochs 改成1，因此 warmup/cosine horizon 未改变。
Stage-I 85个训练batch；Stage-II 85个训练batch。每阶段clean validation 79个batch（10000样本）。
Stage-II 使用 Stage-I 本轮真实生成的 current.pth.tar，ratio_detach=True。

## 结果

|阶段|Clean Overall|Many|Medium|Few|学习率|
|---|---:|---:|---:|---:|---:|
|Stage-I 1 epoch|3.95|10.972|0.000|0.000|0.02|
|Stage-II 1 epoch|4.66|8.139|2.743|2.655|0.02|

以上仅验证管线运行，不代表正式复现精度或方法性能提升。
所有观测到的loss、logits、梯度、最终参数/BN和准确率均有限，无NaN/Inf。
Stage-I criterion 调用输出范围 [3.516612,5.864060]；Stage-II [4.229689,5.357587]。
这些范围包含训练与验证，Stage-I MixUp含分开的两次CE调用，不应当作epoch平均loss。

## PIF参数和冻结检查

按 [a,b] 顺序：

- Stage-I 前：[0.2613041401, 0.6161798835]
- Stage-I 后：[0.2152823061, 0.3344149590]
- Stage-II 前后均为：[0.2152823061, 0.3344149590]

Stage-I 两个PIF参数均观察到非零有限梯度，均发生变化。
Stage-II 全部extractor state_dict（含PIF与BN buffers）变化项数=0，所有参数grad=None；classifier有非零梯度且参数变化。
两阶段保存后重新读出的checkpoint与内存模型/FC逐项完全相同；Stage-II实际strict加载成功。
验证调用原clean validate：仅 model（含PIF）→classifier，不调用H2TF。

## Stage-II ratio分布

count=10847；mean=0.4739791751，std=0.0438669100，min=0.3074677587，max=0.6541008353。
全部在[0,1]内，20-bin histogram的边界与计数保存在 stage2/ratio_epoch_001.json。
本轮不要求mean接近0.65，未为此调整任何实现。

## 验收

- [x] 原 synthetic smoke PASS
- [x] ratio_detach=True forward/backward PASS
- [x] ratio_detach=False forward/backward PASS，ratio可回传至classifier权重
- [x] 真实Stage-I 1 epoch完成，PIF更新
- [x] Stage-I checkpoint保存、读取、Stage-II严格加载成功
- [x] 真实Stage-II 1 epoch完成，backbone/PIF/BN冻结、classifier更新
- [x] ratio有限且在[0,1]，统计与直方图已保存
- [x] clean validation不使用H2TF
- [x] 无NaN/Inf
- [x] legacy H2T相关文件与官方快照逐字节一致（见review provenance）

产物：saved/dryrun_20260930_stage1/ 与 saved/dryrun_20260930_stage2/，各含 train.log、current.pth.tar、model_best.pth.tar、dry_run_audit.json、protocol.json、config.yaml。
正式训练停止在等待后续明确指令状态。
