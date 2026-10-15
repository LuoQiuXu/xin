"""Render the dataset audit from verified evidence, without importing any ML runner."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parent
WORK = Path(r'D:\xiangmuxunlian\新')
BASE = WORK/'H2T_repro_audit'
VS = BASE/'variation_shift_stage2_runs/vs_stage2_seed123'
FORMAL = BASE/'formal_runs/formal_20261011_1429_r2'
SELECTOR = BASE/'donor_stage2_runs/selector_fixed_bn_20261012'
MECHANISM = BASE/'variation_shift_runs/variation_shift_20261013_seed123'
e = json.loads((ROOT/'audit_evidence.json').read_text(encoding='utf-8'))


def link(path, name=None, line=None):
    p = Path(path)
    assert p.exists(), f'Missing evidence link: {p}'
    if line is not None:
        # Original Windows training logs contain legacy encoded text; line
        # anchors only require counting line endings, not decoding the log.
        assert p.is_file() and 1 <= line <= len(p.read_bytes().splitlines())
    return f'[{name or p.name}]({p.as_posix()}' + (f':{line}' if line else '') + ')'


counts = e['split']['counts']
count_block = '\n'.join(f'{start:02d}–{start+9:02d}: '+', '.join(map(str, counts[start:start+10]))
                        for start in range(0, 100, 10))
selection_rows = '\n'.join(f'| {arm} | {s["best_epoch"]} | {s["best_99class_validation_Overall"]:.4f} | '
                           f'{s["epoch10_99class_validation_Overall"]:.4f} |'
                           for arm, s in e['stage2_actual_data']['selections'].items())
g = e['split']['groups']
group_rows = '\n'.join(f'| {name} | {description} | {g[name]["nominal_classes"]} | '
                      f'{g[name]["effective_class_count"]} | {g[name]["samples"]} |'
                      for name, description in [('Head', 'ID 0–35'), ('OfficialMedium', 'ID 36–70'),
                          ('Tail', 'ID 71–99'), ('Many', 'LT 数量 >100：ID 0–34'),
                          ('Medium', '20≤LT 数量≤100：ID 35–69'), ('Few', 'LT 数量 <20：ID 70–99')])
raw_rows = '\n'.join(f'| {name} | {v["bytes"]:,} | `{v["md5"]}` | '
                     f'{"匹配" if v["verified"] else "不匹配"} |'
                     for name, v in e['data_integrity'].items())
snapshot = Path(e['stage1']['actual_training_source_snapshot'])
s1 = Path(e['stage1']['checkpoint']['path'])

report = f'''# VS Stage II 数据协议审计

核查时间：{e['audited_at']}。对象：{link(VS, 'VS seed123 五组 Stage II 实验')}（Original-H2T、Uniform-PerSample、High-VS、Low-VS、Shuffled-VS，各 10 epochs）。

本次依据原始文件校验、作者生成函数、完整 raw indices、实际保存数组、训练 batch 日志、预测文件、Checkpoint 元数据及 Stage I 历史日志核查。只读取既有模型和预测；未创建模型、执行模型前向、重新训练或运行官方测试集评估。官方 test 文件仅作为不透明字节流计算校验和，未反序列化测试图像或标签。历史文件保持原状。

## 1. 六个问题的结论

| 问题 | 核查结论 |
|---|---|
| 是否使用真实 CIFAR-100？ | **是。** 实际已解压 train/test/meta 全部匹配 torchvision 中的官方 MD5；LT 和验证数组逐张匹配原始 train 文件。旧下载压缩包损坏，详见第 2 节。 |
| 10,847 张 LT 是否符合作者 IF=100 实现？ | **是。** 100 类，指数衰减，最大 500、最小 5；原始生成函数重放得到的全部 10,847 个有序索引、标签均与当前文件一致，且像素哈希与 Stage I 训练前记录一致。 |
| 1,980 张验证图像来自哪里？ | **官方 TRAIN 的未进入 LT 子集的图像**，class 1–99 各 20 张；不是官方 TEST。与 LT 的索引和精确图像内容重叠均为 0。 |
| 官方 10,000 张 TEST 是否完全未参与本轮？ | **本轮 VS 的训练样本、验证样本、Stage II epoch 选择和五组比较没有直接使用 TEST。** 但所用 Stage I 已由 TEST 选择，且历史研究曾查看 TEST，因此不能称整个研究或模型来源完全没有测试信息。 |
| 缺少 class 0 的影响？ | 衡量的是 99 个有验证样本类别上的表现；无法衡量 class 0 的召回或策略对它的影响。100 维 logits 保留，预测 class 0 仍计错。可作同协议内部比较，不能与论文的 100 类官方测试准确率直接比较。 |
| Stage I epoch 198 是否用 TEST 选择？ | **是，已被代码、200 行 epoch 记录、原始日志及 Checkpoint 一致证实。** TEST 每 epoch 评估，按最高 Overall 保存最佳，形成测试集参与模型选择的信息泄漏风险。 |

机器可读总证据：{link(ROOT/'audit_evidence.json')}；全部核查索引：{link(ROOT/'raw_indices_verified.csv')}。

## 2. 实际原始数据文件与来源

实际加载根目录：`D:\\xiangmuxunlian\\新\\H2T\\data\\cifar100\\cifar-100-python`。

对照 {link(Path(r'D:\app\anaconda\envs\pytorch\Lib\site-packages\torchvision\datasets\cifar.py'), '已安装 torchvision CIFAR100 源码', 146)} 中的 `train_list`、`test_list`、`meta`、`tgz_md5`。该源码记录原始发布 URL 为 `https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz`。本审计能确认本地文件字节与官方数据匹配；现有材料不能完整恢复这些文件最初的网络下载过程。

| 文件 | 实际字节数 | 实际 MD5 | 与官方值比较 |
|---|---:|---|---|
{raw_rows}

**压缩包确实不完整：** 现存 `cifar-100-python.tar.gz` 为 80,052,224 bytes，官方预期 MD5 为 `eb9058c3a382ffc7106e4002c42a8d85`，实际为 `703db12c359bd9481a3c7b9e2e604569`。只读扫描在 train 成员处遇到 `EOFError: Compressed file ended before the end-of-stream marker was reached`。它不能作为完整可用的原始下载归档；具体中断原因无法从现存日志确定。

**这不等于实际训练数据损坏。** 训练代码读取已解压的 train，而不是每次从该压缩包提取。已解压 train、test、meta 的官方校验均通过；本轮实际训练与验证像素也逐张对上 train。核查记录见 {link(ROOT/'raw_file_integrity.json')}，包括 SHA256 和压缩包扫描异常。本次保留现状，未修复或重新下载。

原始 train 经读取确认为 `[50000,32,32,3]` uint8，100 个 fine classes，每类 500 张。官方 TEST 的 10,000 张、每类 100 张由已存的 {link(FORMAL/'best_evaluation.json', 'Stage I 全量评估记录')}与其匹配官方字节的原文件支持；本次未重新加载该数据集做评估。

## 3. CIFAR-100-LT 的生成与全部索引核对

作者实现：{link(BASE/'official/datasets/cifar100.py', '类别数量公式', 27)}、{link(BASE/'official/datasets/cifar100.py', '实例选择函数', 142)}、{link(BASE/'official/datasets/cifar100.py', '实际训练 Dataset 构造', 190)}。

类别 c=0…99 的数量为：

```text
n_c = int(500 × 0.01 ** (c / 99.0))
Imbalance Factor = 1 / 0.01 = 100
```

`int` 截断；类别编号保持原 fine label 顺序，不打乱类别。`rand_number=0` 令 NumPy 的原始全局 RandomState 序列从 seed 0 开始；逐类取得原始 train 的索引，执行 `np.random.shuffle(idx)`，取前 n_c 个，再按类别 0…99 拼接。**这不是 `default_rng(0)`，也不是 Stage II 的训练 seed 123。**

本次通过 AST 取出作者原始两个 NumPy 方法，以 raw ID 作为数据重放；未调用会创建官方 TEST loader 的 Dataset 构造器。结果：

- 100 类，合计 **10,847** 张，唯一 raw indices 数 **10,847**，最大/最小 **500/5=100**。
- 与当前 `split_indices.npz:train_raw_indices` **全量、有序、逐项一致**。
- `train_images.npy`、`train_labels.npy` 与原始 `train[raw_indices]` 全量一致。
- LT 图像字节 SHA256 与真实 Stage I 训练前 {link(FORMAL/'preflight.json', 'preflight')}记录一致，确认本轮沿用同一个训练子集。
- 作者副本、兼容副本和实际 Stage I 保存的 `codes/datasets/cifar100.py` 均为 SHA256 `7e97d7e565cc7744642553fe5d828c4a1f91ae7f02777a3e195c8609404c615d`。

全部类别数量（左侧为 class ID 区间）：

```text
{count_block}
```

含原始 class 名称、训练/剩余/验证数量的完整表：{link(ROOT/'dataset_class_counts.csv')}。完整 raw ID、所在分区、fine label、原始 filename：{link(ROOT/'raw_indices_verified.csv')}。

| 固定对象 | SHA256 |
|---|---|
| 原始 train 文件 | `735e79b04f092ca3d2e6d07f368c0a7d70d48c48d28865950cc24454cf45129b` |
| LT 图像字节（不含 .npy 文件头） | `{e['split']['LT_pixel_bytes_sha256']}` |
| LT 标签字节（int64） | `{e['split']['LT_label_bytes_sha256']}` |
| 有序 LT raw ID 数组 | `{e['split']['train_ID_array_sha256']}` |
| 有序 validation raw ID 数组 | `{e['split']['validation_ID_array_sha256']}` |
| 固定 split_indices.npz 文件 | `c8579feef60d7705472ec00aeac85fa00a66c243761e0a8d76c462c663286ca1` |

raw ID 数组哈希包含 shape、dtype、字节；文件哈希与裸数组哈希有不同含义，不能混用。

## 4. 1,980 张验证图像的真实来源与不重叠检查

首次构造源：{link(SELECTOR/'selector_stage2.py', 'unused-TRAIN 划分代码', 260)}；固定时间 **{e['split']['split_creation_time']}**，固定 split seed **20261012**，早于该轮训练。先重建上述 LT 原始索引，再取 `0…49999` 中未进入 LT 的 **39,153** 个 TRAIN 索引。对 class 1…99 的剩余图像用独立 `default_rng(20261012)` 打乱，逐类取 20 张，同时排除与 LT 或类内已选验证图像的精确像素重复。

本次从官方 train 再次重建该划分，与保存的 **全部 1,980 个有序验证索引一致**；重复排除列表为空。验证 raw ID 无重复，验证精确像素哈希也有 1,980 个不同值。取验证后仍有 **37,173** 张原始 TRAIN 图像未用于 LT 或当前验证。

实际数组逐张核查：`validation_images.npy == raw_TRAIN[validation_raw_indices]`，`validation_labels.npy == raw_TRAIN_fine_labels[validation_raw_indices]`。例如验证位置 0 对应 **TRAIN raw index 3724、class 1、`carassius_auratus_s_000524.png`**；位置 1 对应 TRAIN index 12218。**TRAIN 与 TEST 的整数索引属于不同分区；某个 raw index 小于 10000 不表示它来自 TEST。**

| 检查对象 | 结果 |
|---|---:|
| LT 与 validation 原始 TRAIN 索引交集 | 0 |
| validation 图像与 LT 精确像素哈希重叠 | 0 |
| 每类 validation 数量 | class 0：0；class 1–99：各 20 |
| 验证图像总数 / 有标签类别数 | 1,980 / 99 |
| 原供体实验 → SMART → 当前 VS 的 split/训练图像/标签/验证图像/标签/验证特征文件 | 6 个文件分别跨三目录 SHA256 完全相同 |

原始 {link(SELECTOR/'split_indices.npz', '冻结 split')}及当前 {link(VS/'split_indices.npz', 'VS split')}相同；验证特征缓存 SHA256 为 `7f07c03c35392bb1903755832d692da323843444194952f1bf9683e809855164`。旧 {link(SELECTOR/'split_audit.json', 'split_audit')}与本次独立检查结论一致。

这里的“未见”严格指这些原始图像未进入 Stage I、固定 BN DR 和本轮梯度训练子集。它不是从未用于过研究决策的新数据：同一验证集已用于供体选择、SMART 和 VS 多轮比较，存在验证集反复使用带来的策略选择过拟合风险。

## 5. 本轮实际训练、评分、验证与选模数据链

训练入口实际是本轮 {link(VS/'training.py', 'training.py', 70)}与 {link(VS/'vs_stage2.py', 'vs_stage2.py', 66)}；不是目录里备份的官方 `train_stage2.py`。{link(VS/'common.py', 'raw_train()', 109)}只读取原始 train；`imports()` 仅从作者构造函数 AST 提取变换表达式，没有实例化会加载 TEST 的 `CIFAR100_LT`。

{link(VS/'reused_helpers.py', 'augment()', 58)}按保存的 LT 索引读取 `train_images.npy`，IS 先增强、BS 后增强。实际 target 来自 `train_labels[receivers]`。本次核对了 `sampling_and_masks.npz` 的全部 10×10,847 个 BS 和 IS 索引及 **5×850=4,250 条实际 batch 日志**：每条索引及标签的 SHA256 与保存计划一致，末 batch 为 95。两分支均只引用 LT 范围；与验证 ID 的交集均为 0。IS 每个 epoch 是完整 LT 实例排列，精确保留各类频次；10 epochs 的 BS 共覆盖 10,517 个不同 LT 实例，IS 覆盖全部 10,847 个。

VS 评分统计文件来自 {link(MECHANISM/'class_feature_statistics.npz', '既有 LT 统计')}，与当前复制文件 SHA256 同为 `695aa8fb861a06f615d4ec24782545e92d83aa1f6533b8218c853314cfa287bf`。本次从已保存的 **10,847×64 LT GAP 特征**重算每类均值、ddof=0 方差，与冻结统计逐位一致；类别数量也等于 LT 数量。未做新特征提取。评分函数只涉及当前训练接收特征、供体类及这些训练统计，不读取验证或 TEST 标签。

验证通过 {link(VS/'training.py', '验证与最佳保存', 193)}和 {link(VS/'common.py', '100 类指标计算', 121)}读取冻结 `validation_feature_maps.npy`、`validation_labels.npy`。全部 **50 份已保存的验证预测**均为 `[1980,100]` logits，标签逐项等于冻结 TRAIN 验证标签，argmax 与保存 predictions 一致。

从这些**既有预测**重新计算准确率并重建最早最大值 epoch，与 status 和最佳 Checkpoint 全部一致；这是核查旧输出，不是重新运行模型评估。最佳 epoch 按验证 Overall 选择、平局取最早；主要方法比较仍是固定第 10 epoch。

| 方法 | 最佳验证 epoch | 最佳验证 Overall (%) | 固定 epoch 10 验证 Overall (%) |
|---|---:|---:|---:|
{selection_rows}

以上全部是 **99 类官方 TRAIN 留出验证结果**。数据源没有被报告标题或配置名替代。{link(ROOT/'vs_saved_validation_verified.csv', '50 份预测核查')}与 {link(ROOT/'checkpoint_metadata_verified.csv', '10 个 Checkpoint 元数据')}记录全部证据；每个保存模型引用同一 Stage I 哈希和同一 split 哈希，分类器 weight 为 `[100,64]`，验证样本数 1,980，缺失类 `[0]`，全部保存的 Backbone 状态与 Stage I 逐位一致。

**对 TEST 的判定范围：** 本轮执行路径和保存输出支持“没有直接把官方 TEST 作为训练、验证或 Stage II 选模数据”。协议字段 `official_test_opened=False` 和日志末尾 `no test` 只是辅助证据；主要证据是上述像素溯源、实际索引日志与全量预测核查。现有记录不是操作系统级文件访问审计，不能据此保证任何未记录的外部活动均未发生，也不能消除第 7 节的历史测试信息。

## 6. 缺少 class 0 的具体影响

class 0 为 **apple**，LT 使用原始 TRAIN 中该类全部 500 张，所以不存在可在保持 Stage I 和训练子集不变时取出的未见 class 0 TRAIN 样本。当前保留全部 100 个分类器输出；没有删除、屏蔽 class 0 logits，也没有在预测时只对 99 类取 argmax。固定 epoch 10 的预测 class 0 次数分别为 Original 9、Uniform 9、High 11、Low 9、Shuffled 9，全部计入错误。

| 组指标 | 完整类别定义 | 名义类别数 | 当前有效类别数 | 当前验证图像数 |
|---|---|---:|---:|---:|
{group_rows}

编号分组的 `OfficialMedium` 与按训练频次分组的 `Medium` 不相同。各组只平均有真实验证样本的类别；class 0 记为缺失，不作为 0% 准确率加入均值。由于 99 类各有 20 张，当前 Overall 与 Macro-99 数学上相同，浮点舍入可有极小差异。

影响包括：

1. 五组共享相同验证样本和 100 类决策竞争，可以比较这些 99 类上的条件性结果；无法观察一个策略是否以 class 0 的收益或损失换取其他类别的变化。
2. Head/Many 分组少了 class 0；其结果不是完整 Head/Many 的准确率。其他类别组的类别集合保留，但每类只有 20 个 TRAIN 留出样本，其数值仍不等于官方 TEST 上相应组的结果。
3. 官方 100 类 TEST 包含 class 0、每类 100 张；当前验证来自 TRAIN 且每类 20 张。缺失类别、样本来源和样本数同时不同。不能把当前 48.xx% 与 Stage I 的 40.49%、旧 DR 的 47.10%、旧 H2T 的 48.06% 或论文 Table/README 的 100 类官方测试数值相减来表示提升。
4. 缺失 class 0 的偏差方向和大小对当前模型无法由这 1,980 张图像确定；不能用一个固定加减值“修正”为论文协议。当前结果应标为“99 类未参与梯度训练的 TRAIN 留出验证，100 维输出，class 0 缺失”。

## 7. Stage I 最佳模型确实用官方 TEST 选择

锁定模型：{link(s1, 'Stage I epoch-198 最佳 Checkpoint')}。

```text
SHA256 = {e['stage1']['checkpoint']['sha256']}
Checkpoint epoch = 198（从 1 开始）
原始日志 Best epoch = 197（从 0 开始）
官方 TEST Overall = 40.49%
```

这不是从文件名推测，完整证据链为：

1. 作者和实际训练快照的 {link(snapshot/'datasets/cifar100.py', 'eval_dataset', 194)}使用 `torchvision.datasets.CIFAR100(train=False)`，随后作为 `self.eval`。
2. 真正运行并保存的 {link(snapshot/'train_stage1.py', 'Stage I 快照', 215)}将 `val_loader = dataset.eval`；第 253 行每 epoch 调用 validate；第 256 行 `is_best = acc1 > best_acc1`；随后保存最佳模型。因此变量名 `val_loader` 实际指官方 TEST。
3. 正式外部 runner {link(FORMAL/'formal_stage1.py', '观察与保存包装', 244)}调用原 `real_validate`、`real_save`，记录指标与 `is_best`；没有替换数据集划分或选模规则。实际保存快照的 SHA256 与正式训练前 preflight 对上。
4. {link(FORMAL/'epochs.csv', '正式 200 epoch 表', 199)}包含全部 200 行，每行有 test Overall；全表最大值在 epoch 198，`new_best=True`。epoch 199、200 未超过它。
5. {link(Path(e['stage1']['actual_training_source_snapshot']).parent/'logs/cifar100_imb001_stage1_ce_mixup_formal_20261011_1429_r2.txt', '实际原始训练日志', 1853)}记录 `Acc@1 40.490%`，接着 `Best epoch: 197`；CPU 读取锁定 Checkpoint 的 epoch 和 best_acc1 与之吻合。旧 {link(FORMAL/'best_evaluation.json', '最佳模型全量重评记录')}也明确有 100 类×100 张。

重建的 200 行记录保存为 {link(ROOT/'stage1_epoch_selection_verified.csv')}。本次只读取这些历史数据，未重新跑官方 TEST。

### 测试信息泄漏的具体含义

**没有证据显示 Stage I 把 TEST 图像或标签用于反向传播；已经证实 TEST 标签和准确率用于 epoch 选择。** 在 200 个候选训练状态中依据 TEST 最大值选择最佳，相当于把 TEST 用作验证集。所报最大 TEST 准确率可能含选择乐观偏差，偏差大小不能从现存单次运行可靠估计。

这个被 TEST 选择过的 Backbone 和 Stage I classifier 初始化被当前五组共享，VS 类均值/方差也依赖该 Backbone，所以测试信息通过模型来源**间接传入**当前实验。单独停止本轮 TEST 评估、改用 TRAIN 留出验证，并不能追溯地消除这一历史依赖。未来用同一官方 TEST 报出的结果不能再称“整个研究从未使用的独立保留测试”或严格无偏的最终泛化验证。

历史原始 Stage II 和 BN 审计已有官方 TEST 指标；供体选择阶段在方案锁定后也进行了历史终评，见 {link(WORK/'H2T_DONOR_STAGE2_RESULTS.md', '历史 TEST 使用披露', 230)}。当前材料不能量化这些历史结果对后续研究假设的人工选择影响，应该披露暴露历史，而不是声称整个研究没有触碰 TEST。

共享基座仍允许回答“同一固定基座、同一 99 类验证协议下五组有何差异”，但当前结果不构成在全新未暴露测试集上的独立方法有效性证明。该限制与验证样本未用于梯度训练是两个不同层面的事实。

## 8. 完整性、证据边界与复查命令

作者源码记录的 archive commit 为 `{e['official_archive_provenance']['commit']}`，来自既有官方 ZIP provenance；本次未重新联网核验远程 Git 对象。当前 VS 训练分支 `{e['VS_git_provenance']['branch']}`，训练 commit `{e['VS_git_provenance']['training_commit']}`。关键运行源码、实际 Stage I 快照和日志的 SHA256 见 {link(ROOT/'source_records.json')}。

核查前后 **{e['history_preservation']['protected_files']:,} 个历史/数据/来源文件 SHA256 全部不变**；此前 1,943 个历史文件和本轮 VS 227 个交付文件的既有清单也全部匹配，未发现审计前已有差异。结果见 {link(ROOT/'historical_verification.json')}、{link(ROOT/'historical_before.json')}。仅新增本审计目录和本报告。

复查命令（PowerShell；新的输出目录，每次写入独立审计证据）：

```powershell
$env:PYTHONIOENCODING = 'utf-8'
$auditOut = Join-Path 'D:\\xiangmuxunlian\\新\\H2T_repro_audit\\dataset_protocol_audits' ('recheck_vs_' + (Get-Date -Format 'yyyyMMdd_HHmmss'))
& 'D:\\app\\anaconda\\envs\\pytorch\\python.exe' -B -u 'D:\\xiangmuxunlian\\新\\H2T_repro_audit\\dataset_protocol_audits\\vs_stage2_20261013\\audit_dataset_protocol.py' --output-dir $auditOut
```

审计脚本仅读取原始 TRAIN/meta、对 TEST 和旧压缩包作字节校验、重放原始 NumPy 索引方法、核对既有数组和预测、读取 CPU Checkpoint 元数据。不会调用任何训练入口或模型 forward。旧损坏压缩包的失败校验被记录为问题，实际解压数据必须通过官方校验；原索引、标签、像素、batch、Checkpoint 或历史完整性不符时脚本会失败。

新增证据文件：

- {link(ROOT/'audit_dataset_protocol.py', '独立核查脚本')}、{link(ROOT/'write_dataset_report.py', '报告生成脚本')}。
- {link(ROOT/'audit_evidence.json')}、{link(ROOT/'raw_file_integrity.json')}、{link(ROOT/'source_records.json')}。
- {link(ROOT/'raw_indices_verified.csv')}、{link(ROOT/'dataset_class_counts.csv')}。
- {link(ROOT/'vs_saved_validation_verified.csv')}、{link(ROOT/'checkpoint_metadata_verified.csv')}、{link(ROOT/'stage1_epoch_selection_verified.csv')}。
- {link(ROOT/'historical_verification.json')}、{link(ROOT/'historical_before.json')}。

**审计结论：实际 CIFAR 数据、IF=100 LT 子集和 99 类 TRAIN 留出验证来源均可追溯并通过核查；本轮没有直接使用官方 TEST 训练、验证或选 Stage II 模型。主要协议限制是 class 0 缺失、同一验证集反复使用、Stage I 的历史 TEST 选模依赖，以及旧下载压缩包不完整。当前 VS 数字只能按该阶段性验证协议解释。**
'''

destination = WORK/'DATASET_PROTOCOL_AUDIT.md'
if destination.exists():
    # Only the new report created and sealed by this audit may be refreshed.
    seal = json.loads((ROOT/'audit_delivery_manifest.json').read_text(encoding='utf-8'))
    assert hashlib.sha256(destination.read_bytes()).hexdigest() == seal['root_report_sha256']
    assert destination.read_bytes() == (ROOT/'DATASET_PROTOCOL_AUDIT.md').read_bytes()
destination.write_text(report, encoding='utf-8')
(ROOT/'DATASET_PROTOCOL_AUDIT.md').write_text(report, encoding='utf-8')
manifest = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(ROOT.iterdir())
            if p.is_file() and p.name != 'audit_delivery_manifest.json'}
(ROOT/'audit_delivery_manifest.json').write_text(json.dumps({'files': manifest,
    'root_report': str(destination), 'root_report_sha256': hashlib.sha256(destination.read_bytes()).hexdigest(),
    'historical_files_unchanged': True, 'training_or_test_evaluation_started': False}, indent=2, ensure_ascii=False), encoding='utf-8')
print(json.dumps({'report': str(destination), 'evidence_directory': str(ROOT), 'files': len(manifest),
                  'report_bytes': destination.stat().st_size}, ensure_ascii=False))
