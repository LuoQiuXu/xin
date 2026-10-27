# PI-H2T 严格复现：作者来源核查与训练暂停说明

日期：2026-10-27

## 结论

**当前还没有足够证据将本地 CIFAR100-LT PIF+H2TF 训练称为“严格复现作者 PI-H2T”。**

- PI-H2T 文章的代码说明/相关页面指向 `https://github.com/Keke921/H2T`。
- 现场通过 GitHub 仓库 API 核对：`Keke921/H2T` 的 README 自述为 AAAI 2024 H2T example code（核心 `methods.py:H2T`），公开分支列表只有 `main`，仓库文件中没有明确命名的 PI-H2T/PIF/H2TF 完整训练实现。最后代码 push 记录为 2025-03-22。**无法据此确认 PI-H2T 正式代码已释放。**
- 不得把作者 H2T 的 Stage-I/II 默认配置与我们自行推测的 PI-H2T Stage-I/II 配置合并后，宣称完整作者官方协议。
- 本地 PIF 两参数 / 200+10+10 重建程序仅支持在锁定的**本地独立方案**下，H2TF(36.50%)与 matched no_fusion(45.53%)存在差异。与作者 StageII Last49.04%的差距证明 **论文复现一致性尚未实现**，不是已发表 PI-H2T 原理失效的证明。

## 必须由作者材料确认的字段（不采用准确率倒推）

1. PI-H2T 代码的具体仓库、tag/commit、许可及发布日期；
2. PIF 网络、插入位置、初始化与损失、MixUp位置，Stage I 精确 optimizer/LR/epochs/checkpoint 筛选与 seed；
3. H2TF 融合公式与变量标记，比例中的分类器 `W` 具体来源和更新时机、是否有 detach，接收者/供体的定义；
4. BN running buffer 更新、Stage II 冻结范围、分类器 head、sampling+augmentation、总 optimizer.step 数、scheduler 更新频率；
5. 数据集 construction/index、Batch Size、处理方法、论文 Table6 Last/Best 的评估模式。

## 科学上的停止条件

- 如果 PI-H2T 完整作者实现/训练配置不可用，任何仍有歧义且会实质改变结果的字段标记为 `UNVERIFIED`，不得靠逐个试错/拟合49.04%确定“正确”实现。
- 若确实需要训练，请明确标为 `PAPER_ALIGNED_INDEPENDENT_RECONSTRUCTION`，并经用户认可该研究目的；不得称为官方严格复现。
- 若找到可信 PI 作者实现（完整版本/commit），优先在单一锁定的 CIFAR100-LT IF100/ResNet32 配置上原样运行，不添加 current-W / frozen-W 对照、不额外改增强、BN、损失或 PIF 结构。
- 若找不到，优先联系作者索取 PI-H2T 代码及配置、checkpoint 与实验协议；可以在此期间严格复现**作者已公开的原始 H2T**，作为独立可复现基线。

以上核查应优先于进一步 GPU 训练；原 `CODEX_PI_H2T_REFERENCE_W_PARITY.txt` 现在停用。
