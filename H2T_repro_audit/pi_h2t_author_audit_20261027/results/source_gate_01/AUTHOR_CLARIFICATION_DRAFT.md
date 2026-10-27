# 给作者的必要澄清问题草稿（未发送）

希望严格复现PI-H2T在CIFAR100-LT IF100/ResNet32/CE上的结果。当前所核对公开仓库main包含原H2T入口和PIF/H2Tv2片段，但尚未找到可运行的完整论文对应训练发布。请协助确认：

1. 能否提供与论文结果对应的官方release/commit、完整PIF/H2TF两阶段入口与CIFAR配置？论文报告+2参数PIF的卷积bias开关、a/b精确初始化、插入位置和有效返回路径分别是什么？
2. Eq8计算r时，w_y取StageI冻结分类器还是每步更新的StageII分类器？r是否detach，cosine distance及理论边界线性min-max的实际代码是什么？公开H2Tv2的dis由何处产生？
3. StageII冻结骨干后，BN使用train还是eval，running_mean/var及affine参数是否更新？接收者与供体前向顺序、分批/拼批方式及无融合对照的BN输入流是否相同？
4. 对应CNN结果的完整数据选择和训练seed、基础增强、MixUp alpha、ResNet shortcut、head bias、StageI warmup/weight_decay及StageII实际学习率/调度/weight_decay是什么？
5. 论文StageI42.19%取best还是last、具体epoch和checkpoint SHA是什么；StageII49.22% best/49.04% last使用哪份StageI模型？正式期刊与arXiv v2及补充材料是否存在公式/配置更正（含Algorithm1 line5的F/F_fuse）？

以上只请求来源和既有协议，不以本地准确率作为选择参数的依据。此文档仅本地审计产物，没有发送邮件、留言或issue，也没有推定作者联系方式。
