# 场景选择依据：先找完整方法的失败，不再拆解融合模块

## 选择的科学问题

**场景**：真实人工注释错误叠加CIFAR-100-LT IF100。
**主要检验**：在同样的受噪声训练数据、相同StageI以及相同训练预算下，H2T相对NoFusion的Few/Overall收益是否消失或反转？
**指标**：在clean和noisy两套Regime下分别计算H2T−NoFusion准确率增益，再看Noise×Method交互。

这是**原方法的泛化评价问题**，不是 donor 风险或旧分类器margin代理。

## 为什么不是继续既有路线？

- IF200固定BN两种StageII种子上，Original-BatchShared−NoFusion Few分别正向；不能拿更高IF作为已确定的失效场景。
- Office-Home供体域匹配/跨域几轮Few变化方向不稳，不能声称跨域供体导致稳定损害。
- 语义同superclass / 异superclass冻结几何筛查未通过；类覆盖不足及距离/频次混杂更不能证明最终方法失败。
- 过去的Low-VS、margin、donor-affinity、pair-conditioned ratio曾揭示局部几何现象，却未提供与最终性能一致的可靠失效原因。

## 为什么人工标注噪声有场景价值？

H2T 在StageII使用接收者hard label监督融合特征。具有错误人工标签时，监督信号可能不可靠；与此同时，class-balanced receiver sampling也可能反复采样噪声样本。这给出了可以证伪的**脆弱性假设**，但不保证H2T比NoFusion更脆弱。

CIFAR-100N是公开ICLR2022的真实人工标签集合，不是事后虚构的synthetic噪声比例；鲁棒长尾学习（RoLT等）已有相关研究，说明此类数据质量条件具有现实研究价值，但并不自动证明H2T缺陷或新颖性。

最重要的控制：noisy regime下类频也随观测标签而变化；这属于真实的场景效果，**不能把交互下降全部归因于H2T的固定监督或供体语义噪声**。

## 三种退出结果

- 对齐/数据空类/原始训练协议无法保持 -> BLOCKED 或 INCONCLUSIVE；保证数据与代码有效性优先。
- H2T在clean/noisy两条件都不低于DR、差分不明显 -> NO-GO；不添加噪声率寻找负结果。
- H2T在noisy中低于DR且两个StageII种子的Few交互都<=−2pp -> 候选失效；**仅提议新StageI seed独立确认**，不能直接设计方法，也不能推断PI-H2T失败。

## 资源与证据层级

先跑无GPU标签对齐；若通过，两个StageI 200epochs同种子不同标签，再8个StageII 10epochs。只输出一个最终固定评估口径。开发TRAIN留出验证集非官方独立TEST，正式论文必须后续独立确认。

重要区分：StageII两种子共享一份StageI，不是两次独立表示训练复现。PI-H2T作者一致性仍未完成；本次不拿本地25参数PIF来充当官方PI-H2T。
