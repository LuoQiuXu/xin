# PI-H2T 实现审核包：尚未启动正式实验

## 来源与范围

独立目录 PI_H2T_review/H2T-main，来源为 2026-09-30 下载的官方当前 main 快照：
https://github.com/Keke921/H2T

Git 连接失败，改用官方 codeload ZIP；归档的 commit comment 与 SHA256 在 review/upstream_provenance.json。
论文核对版本：https://arxiv.org/html/2506.00625v2 ，重点为 PIF、H2TF 和理论上下界归一化。
这是按论文公式和用户任务说明实现的复现候选，不是已验证达到论文精度的结果，也不把 r=d/2 称作官方代码。

## 修改清单

- models/resnet_cifar.py：新 PIF 和 resnet32_fe_pif；旧 factory 仍为无 PIF；Classifier 新增二维输入接口，四维旧路径不变。
- pi_h2t.py：GAP 后的 H2TF、两种 ratio 模式、全冻结函数、按 epoch 保存分布统计。
- train_pi_h2t.py：独立 CE 两阶段入口；复用官方 Stage-I 训练函数、两阶段原 LR 函数及 clean validate。
- 两份新 YAML：从相应原配置复制，保留数据、采样、优化器与 epoch 配方。
- test_pi_h2t_smoke.py：合成数据单步测试，无 CIFAR 数据集训练。
- utils/util.py：Python3 标准库 pathlib 替代缺失的 pathlib2，语义不变。
- build_review_bundle.py：生成 diff、上游身份记录及 smoke 输出；验证关键 legacy 文件逐字节不变。

原 methods.py（含 H2T 和 H2Tv2）、train_stage1.py、train_stage2.py、dataset 与原 YAML 均未改动。
但当前 main 的 ResNet_fe 本身含不可运行草稿：非法 Conv2d(feature_tuning=...)、未定义 feature_tuning，且计算 out_hat 却返回 out。
为保留可用 H2T baseline，移除此无效草稿，恢复原无 PIF feature map 路径，明确纳入 diff；没有直接启用该草稿。

## 数学与实现约定（请审核）

1. PIF = a(F-mean_channel(F))+bF；两个全局共享参数，Conv1d(2,1,1,bias=False)，堆叠顺序 [残差,F]。Stage-I 和 clean inference 均保留 PIF。
2. 初始化使用 PyTorch Conv1d 默认初始化。这是显式实现选择，尚未从论文确认作者的具体初值；不是额外网络或调参结果。
3. H2TF 在 GAP 后使用 r*ft+(1-r)*fh，r 是 recipient 自身保留量，1-r 才是 donor mass；不再通道替换。
4. paper_theoretical_bounds 默认：r=(1-cos(ft,w_y))/2，不依赖 batch min/max。使用当前正在训练的 classifier 权重，ratio detach，CE 只经 fused feature 对 classifier 更新，不经 ratio 额外反传。此梯度约定须保留在复现记录中。
5. legacy_h2tv2_ablation 显式调用原 H2Tv2，只作可选消融，不混入默认 PI-H2T。
6. 按用户“冻结整个 extractor”要求冻结参数和 BN buffers，Stage-II 使用 eval()。相对于官方 H2T 的 shift_bn=True，这是明确差异；原 H2T 配置保留不变。
7. 每 epoch ratio JSON 含全训练样本的 mean/std/min/max、20-bin histogram 和计数，std 为总体标准差。论文提到约 0.65 仅作后续 sanity check；不强行校准到 0.65，也不交换 r/1-r 来贴合该数值。
8. 保留原 clean validation 的分组和指标及 current/best 保存方式，另外保存 optimizer、seed、完整配置。原仓库的 test/eval 被用于 best 选择，这不是独立 validation；不能把 best 当独立无偏测试。应同时汇报 final。
9. 当前新入口仅支持单进程 DataParallel；不宣称 DDP 已验证。Stage-I 从零训练，不提供恢复中断训练。Stage-II 必须严格加载含 PIF 的 Stage-I 权重，拒绝普通 CE checkpoint，拒绝覆盖非空输出目录。

## 配方

Stage-I：CE+MixUp(alpha=1)，200 epochs，batch128，SGD lr0.1/momentum0.9/wd2e-4；原 5 epoch warmup 和 160/180 multistep。
Stage-II：10 epochs，原 balanced recipient + instance-wise donor，batch128，SGD lr0.1×0.2 cosine、momentum0.9、wd5e-4；只优化 FC。没有 donor selector 或新 loss。
两阶段明确记录 seed，默认123。固定随机种子便于复现，与官方 Stage-II deterministic=False 不同，未改变采样分布或数据划分。

## 审核后才运行的命令（尚未执行）

工作目录：D:/xiangmuxunlian/新/PI_H2T_review/H2T-main

```powershell
D:/app/anaconda/envs/pytorch/python.exe train_pi_h2t.py --cfg config/cifar100/cifar100_imb001_stage1_ce_pif.yaml --output saved/pi_stage1 --seed 123 data_path D:/xiangmuxunlian/新/H2T/data/cifar100
D:/app/anaconda/envs/pytorch/python.exe train_pi_h2t.py --cfg config/cifar100/cifar100_imb001_stage2_ce_pi_h2t.yaml --output saved/pi_stage2 --seed 123 data_path D:/xiangmuxunlian/新/H2T/data/cifar100 resume saved/pi_stage1/current.pth.tar
```

## 已执行的验证

只执行 test_pi_h2t_smoke.py：PIF shape与公式、2个参数、Stage-I梯度、Stage-II参数和BN冻结、FC梯度、二维分类接口、ratio边界和batch独立性、legacy消融一致性、strict checkpoint拒绝非PIF、两个YAML解析，以及CUDA可用时真实Stage-II函数的单组合成batch。
完整输出见 review/smoke_test.txt。未训练200/10 epoch，未生成正式实验精度或正式训练 checkpoint。
