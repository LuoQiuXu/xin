# 论文一致性与实际运行报告

run-id：pih2t_if100_seed123_42_v1；目标分支：repro/pi-h2t-cifar100-paper-parity-20261027。

## 已核实/继承/独立重建

训练前PARITY_LOCK.md在新模型TEST准确率产生前固化，根锁文件SHA=a44e7a04cf13e0b12cffdf66e3c603292381c1d5e35a313997aa9cf0aaeef31c，其SHA纳入protocol.json和新checkpoint的source_hashes。TRAINING_GATE.json记录全部门禁通过、当时未取得新模型TEST准确率；正式训练执行源码与锁定哈希保持一致。

VERIFIED_FROM_PI_PAPER：PIF沿通道取均值、[F-mean,F]单个bias-free Conv1d(2,1,1)两参数、保留PIF后GAP；H2TF为r*f_B+(1-r)*f_I，r=(1-cos)/2；BS接收/IS供体；接收者hard CE；推理不做H2TF。输入输出图为[B,64,8,8]，GAP后[B,64]，分类logits[B,100]，由冻结ResNet32结构、CPU测试与真实CIFAR smoke共同核查。

INHERITED_FROM_H2T：原始冻结模型/dataset逐字未改；Option-A ResNet32+原带bias FC，官方TRAIN内exp IF100/seed0选样；crop/flip/Normalize；StageI CE+MixUp(alpha1)、200轮/batch128/SGD/wd.0002、五轮warmup和160/180衰减；StageII classifier-only10轮/.02初始cosine/wd.0005，默认BN update。具体源SHA见protocol.json。

RECONSTRUCTION_ASSUMPTION：PIF用Torch默认Conv1d初始化；r对W不反传；W为冻结StageI head；cuDNN deterministic=True、benchmark=False；fixed epoch200而非TEST-selected best；no_fusion也前向donor，匹配BN流。作者这些精确细节保持UNVERIFIED。StageI MixUp公式与分布沿用，但本地randperm在CUDA产生、模型初始化含新PIF；不宣称逐位重放原H2T随机流或作者seed轨迹。上述执行方式在源哈希中训练前已锁定，未因结果改变。

原作者库本轮只读扫描main=`e0c849235d9655fca31cd9fa8a5f38f92a77c7e8`；URL https://github.com/Keke921/H2T 。分支身份不证明当前代码是完整官方PI发布。本轮始终标注独立重建，不以准确率接近恢复未定作者语义。

## 实际运行与checkpoint

StageI从零训练200轮，未载入旧模型。PIF新权重shape[1,2,1]、恰两参数；保存epoch200后才评估TEST。两组StageII均先核对checkpoint与StageI完成summary的SHA，再strict load，并检查run-id、protocol、subset/source hashes，拒绝历史25参数结构和漂移来源。

两组初始FC与源checkpoint一致；每轮图像/标签SHA一致，最终BN buffers SHA一致，训练前后骨干/PIF参数SHA一致。配对对照的有效性已通过，详见FINAL_INTEGRITY.json。StageII预测使用同一有序官方TEST集，100图/类；逐类变化从同序预测直接计算。

## 最终判断

可确认：本轮严格执行了锁定的两参数PIF独立重建，200/10/10预算、数据真实性、配对输入/BN流、冻结边界与模型来源均通过核验。不可确认：作者初始化、reference/detach、BN及checkpoint选择全部相同，或结果具有多seed显著性。结论是可信的**paper-aligned independent reconstruction**，不是“完全官方PI-H2T已复现”。
