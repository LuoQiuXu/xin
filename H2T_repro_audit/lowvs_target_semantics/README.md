# Low-VS Target-Semantics Pilot

本目录是下一轮小型机制实验：**保持 Low-VS 融合输入完全不变，只替换尾类融合特征的监督目标**。

## 研究问题

当前尾类融合特征仍统一使用接收者硬标签。对融合后沿决策路径明显朝另一个类别移动的特征，这种监督是否造成标签语义不匹配？

## 三个实验臂

- HardRecipient：原始接收者硬标签。
- DonorMixTarget：仅对尾类接收者使用 0.75 * one_hot(y_receiver) + 0.25 * one_hot(y_donor)。
- FirstCrossedCompetitor：仅对尾类接收者，如果冻结 balanced-cRT 参考头在 clean-to-fused 特征路径上有首次决策边界交叉，则将固定 0.25 标签质量分给首次跨越的竞争类；否则仍使用接收者硬标签。

软目标交叉熵按目标分布总质量归一（每行和为 1），不额外叠加 clean loss，不使用风险降权。三臂的融合输入、donor、receiver、mask、增强、优化器和训练预算必须相同。

## 文件

- PROTOCOL.md：预先固定的科学问题、实验协议和停止规则。
- policy.py：独立、可测试的软目标与首次跨界计算。
- test_policy.py：CPU 单元测试。
- CODEX_LOWVS_TARGET_SEMANTICS.txt：本地集成、预检、运行和报告任务说明。

完整训练依赖本地已有的 CIFAR-100-LT 数据、Clean Stage-I checkpoint、封存的 Low-VS 采样/随机流，以及有明确来源的 balanced cRT 参考头；这些私有/大文件不提交到 GitHub。若本地来源无法核实，Codex 必须停止，不得伪造或重新生成历史缓存。

## 运行

从仓库根目录使用原实验 Python 环境运行：

    python H2T_repro_audit/lowvs_target_semantics/test_policy.py

随后严格按 CODEX_LOWVS_TARGET_SEMANTICS.txt 中的步骤运行。主结果固定 epoch 10；禁止访问官方 CIFAR-100 TEST。
