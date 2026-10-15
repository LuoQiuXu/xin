# SMART → H2T 方法核对

日期：2026-10-13。本报告核对 WACV 2024 SMART 正文第 3 节、讨论及附录 A.4；本轮是受其启发的 H2T 机制实验，**不是 SMART 原方法复现**。

## 1. 可核验来源

- [CVF 正文](https://openaccess.thecvf.com/content/WACV2024/papers/Zhang_Semantic_Transfer_From_Head_to_Tail_Enlarging_Tail_Margin_for_WACV_2024_paper.pdf)：11 页；[本地 PDF](D:/xiangmuxunlian/新/H2T_repro_audit/smart_channel_runs/smart_cov_20261013_seed123/papers/smart_main.pdf)。SHA256：`6d4868d7c1831ae3e0ef63b9ba8511eb96236c128b8385cc8495d83a3761c8d0`。
- [CVF 附录](https://openaccess.thecvf.com/content/WACV2024/supplemental/Zhang_Semantic_Transfer_From_WACV_2024_supplemental.pdf)：4 页；[本地 PDF](D:/xiangmuxunlian/新/H2T_repro_audit/smart_channel_runs/smart_cov_20261013_seed123/papers/smart_supplement.pdf)。SHA256：`c1d5dc89b70637bf86bfc417dab644b789eb5f34254c9cd5067c41fade65c7d7`。
- 已下载官方 PDF，抽取文本并视觉核对正文 PDF 第 4、5 页及附录第 3、4 页。下载响应、大小、时间与哈希保存在 [下载证据](D:/xiangmuxunlian/新/H2T_repro_audit/smart_channel_runs/smart_cov_20261013_seed123/papers/fetch_evidence_unsandboxed.json)。正文第 8 页核对温度讨论。

## 2. 原论文明确运算

正文 §3.3（印刷页 1353，PDF 第 4 页）：特征为 `F ∈ R^(HW×C)`。式 (2)、(3) 对类别均值和协方差使用数量相关的在线更新：

```
μ_k^(t) = (1−a_k^(t)) μ_k^(t−1) + a_k^(t) μ_k^B
Σ_k^(t) = (1−a_k^(t)) Σ_k^(t−1) + a_k^(t) Σ_k^B
a_k^(t) = M_k^B / N_k^(t),   N_k^(t) = N_k^(t−1) + M_k^B
```

`M_k^B` 是当前 batch 中该类数量。论文未在这两式展开 batch 协方差分母或完整的跨均值合并修正，不能据此声称本次离线无偏 GAP 协方差与其实现完全一致。

跨通道关系及式 (4)、(5)：

```
A_th = Σ_t^T Σ_h
Γ = Softmax(A_th; τ)        # 最后一个维度，C×C 矩阵每行归一化
Σ_th = Γ ⊙ A_th
```

Softmax 不是对类别轴归一化，也不是将整个矩阵压成一组通道替换概率。正文使用带 `τ` 的 Softmax 记号，未写出指数中是乘 `τ` 还是除 `τ`。正文 PDF 第 8 页明确描述较大 `τ` 使在线分布更尖锐；**不能直接认定它等于本次常规温度 `T`，或臆造其精确实现约定**。本实验明确用 `exp(s/T)`，`T=1`，不依赖这个未展开的实现细节。

式 (6)：

```
F̃_t = ((F − μ_t) / sqrt(diag(Σ_t) + ε)) Σ_th + μ_t
ε = 10^−8
```

这是标准化后乘稠密跨通道矩阵的迁移。式 (10)、(11) 还使用依赖迁移矩阵的样本损失权重。本轮不引入该损失。

附录 A.4（PDF 第 3、4 页），式 (21)–(24) 讨论协方差对齐的 Frobenius 目标和 SVD/白化着色变换，含正交矩阵 `U`。附录左乘 `T F_t` 的记号与正文矩阵布局不同；它没有推导下面的列余弦评分或 16 通道子集最优性。高斯与 Margin 假设也不能直接证明本轮冻结 Backbone、CE 分类器实验的收益。

**SMART 原方法没有执行 H2T 式的随机选 16 个完整通道并复制供体特征值。** 它的关联矩阵、稠密迁移、语义采样及损失不能直接等同于本次通道选择。

## 3. 本次新设计的实验代理

本轮以每张训练图像 GAP 后的 64 维特征作为一个观测，固定离线类别协方差；不将 8×8 空间位置当成独立图像。

```
s_k(t,h) = cosine(Σ_t[:,k], Σ_h[:,k])
q_high = 0.5/64 + 0.5 softmax(+s/1)
q_low  = 0.5/64 + 0.5 softmax(−s/1)
```

对关系向量列的行维度计算余弦，第 `k` 列只决定实际被替换的第 `k` 通道。这不是将 `Σ_t^T Σ_h` 的行/列和冒充为替换概率。该代理关于 `t/h` 对称，丢失原 SMART 稠密变换的一部分方向信息。高分不是“语义安全”的证明，低分也不是有害融合的定义。`q_k` 是无放回采样权重，不是最终包含概率。

使用同一 Gumbel 噪声的 Top16；独立隔离 RNG；仍是原始单个 IS 供体、接收标签、`ρ=0.25`、SGD 与 CE。本轮只检验这一固定代理是否值得进一步复核。

## 4. 未被证明的假设及实验边界

- 协方差列关系是否对应可迁移判别信息，待实验验证。
- 5 张图像的 64 维样本协方差秩最多 4；正则化改善数值条件，不增加有效样本或证明总体估计准确。
- A→B 同时改变掩码共享粒度；C/B、D/B 的差异才是在同样逐样本机制下改变权重的比较。
- 本轮仅 seed 123、固定 99 类未见训练来源验证集；不使用官方测试集选策略、不网格搜索、不追加种子。

实现与最终数值见 `H2T_SMART_GUIDED_CHANNEL_RESULTS.md`。论文公式证据已取得；温度指数约定的证据限制如上保留，未冒称原 SMART 实现。
