# IF200 固定 BN 四臂压力测试：科学报告

## 1. 准确命题与执行范围

本轮只检验：在 CIFAR-100-LT IF200、同一套新训 Stage I、固定 Backbone/BN、固定第10 epoch 的条件下，整 batch 共享随机通道替换相比 DR 是否出现尾类负迁移；以及 Low-VS 相比逐样本 Uniform 是否存在额外收益。
Original-BatchShared 是作者式掩码粒度与 BS/IS 语义的对照，使用独立 Gumbel 流，不能称逐随机数完全复现。作者 IF200 YAML 为 shift_bn=True；本轮明确固定全部 BN，因此不能把本面板的结果写成作者完整方法成功或失败。

## 2. 数据、模型与固定预算

- 真实 CIFAR-100 官方 TRAIN 文件 SHA256：`735e79b04f092ca3d2e6d07f368c0a7d70d48c48d28865950cc24454cf45129b`。原文件仅解码 TRAIN，未读取 TEST，亦未对 TEST 做不透明哈希。
- 官方 seed0 按类顺序 shuffle；IF200 每类取 int(500×0.005^(c/99)) 个，与该顺序的 IF100 类内前缀严格相等。原始 raw IDs、像素及标签均逐项核对；9,502 张、100 类、最少2/最多500张。
- 沿用 1,980 张原始 TRAIN 留出验证，class1..99 各20张；class0 缺失，但分类器始终保留100维 logits。IF200 与验证的 raw ID 和完整像素重叠均为0。
- Stage I：全新随机初始化 ResNet32+FC，seed42，CE+MixUp α=1；SGD lr0.1/momentum0.9/wd0.0002，batch128；200 epochs、15,000次更新，5epoch warmup，epoch161/181起下降。只保存固定 epoch200，不按验证选 best。
- 新 Stage I checkpoint：`stage1/stage1_fixed_epoch200.pth`，SHA256 `5abef38180bf85cd5c60c38bae08efe23173bdf4775fe44bc38299458448548f`。未载入旧 IF100 Stage I 权重。
- Stage II：seed124/125各四臂，从上述同一 Stage I 的同一 FC 状态独立初始化，SGD状态为空；仅训练6,500个FC参数，冻结463,504个Backbone参数及31个BN。每组10epochs×75更新=750次，末batch30张，总共8组、6,000次更新。
- 作者 IF200 YAML lr0.2，lr_factor0.1；单FC组初始有效lr0.02，10epoch cosine，momentum0.9、wd0.0002、CE，无新增损失或超参数搜索。
- 使用新模型的 IF200 TRAIN-only eval GAP 统计：ddof0类方差，pooled=Σn×var0/(9502−100)，λ=5/(n+5)，v_reg=(1−λ)var0+λpooled；epsilon=1e−6×全LT通道方差(ddof1)均值=4.04904647995e-07。供体n<20的类别39个，按既定规则回退到同一个逐样本Uniform掩码；不删除或改变IS供体类别分布。
- 环境：`NVIDIA GeForce RTX 5070 Laptop GPU`；Python `D:\app\minicondaaa\envs\xiaoju\python.exe`；PyTorch 2.14.1+cu132 / CUDA 13.2。各步骤峰值显存见 `guards/*.json`；最大 allocated 379.49 MiB。

### 验证分组定义（不能混用）

|分组|有效类别数|验证图像数|定义|
|---|---|---|---|
|Head|35|700|类别ID 0..35，但0无样本|
|OfficialMedium|35|700|类别ID 36..70|
|Tail|29|580|类别ID 71..99|
|Many|29|580|训练数量>100|
|Medium|31|620|训练数量20..100|
|Few|39|780|训练数量<20|

Overall 是1980张图像的 micro accuracy；每类恰好20张，所以等于99类 Macro。WorstClass 是这99类的最小准确率，不能表述为包含缺失class0的100类最差结果。

## 3. 四臂与逐 batch 配对

- DR-NoFusion：只将原接收特征送入FC，不执行替换；为配对审计仍计算相同供体特征、评分和全部掩码。
- Original-BatchShared：每个真实minibatch共享一个随机16/64完整8×8通道掩码。
- Uniform-PerSample：每样本独立随机16/64完整通道。
- LowVS-PerSample：供体类统计的 Bottom48 池中随机取16个完整通道；采用同一逐样本Gumbel、相同fallback。接收者标签始终不变。
- 四臂共享每seed的BS类别均匀接收序列、IS每epoch无放回全实例排列、隔离的crop/flip增强流、逐样本与batch共享Gumbel。每seed先DR建立750行基准；另三臂在更新前比较联合哈希，报告时再次逐行核对，总计4,500行非DR配对全部一致。
- 联合哈希实际包含：接收/供体索引、两套增强像素、两套冻结特征、逐样本/共享Gumbel及三个掩码。评分本身未另列独立哈希；其全部输入（GAP、供体标签、同一锁定统计和代码）已固定，不可声称CSV单独存有评分哈希字段。
- 每epoch比较完整Backbone state hash，包含Conv、BN affine、running_mean/var和num_batches_tracked；所有参数被冻结且特征提取在no_grad中。31个BN计数均保持Stage I结束时15,000。每组FC的weight与bias均有变化。
- 每臂保存final_fc.pth、10行epochs.csv、750行paired_batches.csv、1980×100 logits；训练脚本重新加载final FC后验证全部预测和指标一致。额外报告脚本只核对已有数组和权重，不执行推理或训练。

## 4. 固定第10 epoch主结果（%）

|seed|策略|Overall|Head|OfficialMedium|Tail|Many|Medium|Few|Macro|WorstClass|
|---|---|---|---|---|---|---|---|---|---|---|
|124|DR-NoFusion|41.6162|63.2857|40.5714|16.7241|67.0690|43.0645|21.5385|41.6162|0.0000|
|124|Original-BatchShared|42.8283|62.7143|43.5714|17.9310|66.3793|45.3226|23.3333|42.8283|0.0000|
|124|Uniform-PerSample|42.7273|62.8571|43.2857|17.7586|66.2069|45.3226|23.2051|42.7273|0.0000|
|124|LowVS-PerSample|42.4242|61.2857|43.7143|18.1034|64.3103|45.4839|23.7179|42.4242|0.0000|
|125|DR-NoFusion|41.8687|63.7143|40.7143|16.8966|67.4138|43.0645|21.9231|41.8687|0.0000|
|125|Original-BatchShared|42.6263|62.4286|43.2857|17.9310|66.5517|44.5161|23.3333|42.6263|0.0000|
|125|Uniform-PerSample|42.7273|62.7143|43.0000|18.2759|66.5517|44.5161|23.5897|42.7273|0.0000|
|125|LowVS-PerSample|42.2222|61.0000|43.4286|18.1034|64.3103|44.8387|23.7179|42.2222|0.0000|

新 Stage I 固定epoch200的描述性验证：Overall=35.3535%, Head=65.5714%, OfficialMedium=32.7143%, Tail=2.0690%, Many=71.2069%, Medium=37.4194%, Few=7.0513%, Macro=35.3535%, WorstClass=0.0000%。该结果不参与checkpoint选择。

## 5. 配对准确率差与净得失

以下均为 A−B，单位pp。两seed取算术平均仅供描述；同一1980张验证图像重复出现，不能将3960次预测当作独立观测，也不作显著性检验。

|seed|A − B|Overall pp|Head pp|Tail pp|wrong→right|right→wrong|净正确数|Tail净正确数|Tail→Head Δ|Head→Tail Δ|
|---|---|---|---|---|---|---|---|---|---|---|
|124|Original-BatchShared − DR-NoFusion|+1.2121|-0.5714|+1.2069|56|32|24|7|-46|15|
|124|Uniform-PerSample − Original-BatchShared|-0.1010|+0.1429|-0.1724|7|9|-2|-1|2|3|
|124|LowVS-PerSample − Uniform-PerSample|-0.3030|-1.5714|+0.3448|13|19|-6|2|-9|6|
|124|LowVS-PerSample − Original-BatchShared|-0.4040|-1.4286|+0.1724|12|20|-8|1|-7|9|
|125|Original-BatchShared − DR-NoFusion|+0.7576|-1.2857|+1.0345|47|32|15|6|-42|20|
|125|Uniform-PerSample − Original-BatchShared|+0.1010|+0.2857|+0.3448|10|8|2|2|-1|-1|
|125|LowVS-PerSample − Uniform-PerSample|-0.5051|-1.7143|-0.1724|10|20|-10|-1|-13|8|
|125|LowVS-PerSample − Original-BatchShared|-0.4040|-1.4286|+0.1724|17|25|-8|1|-14|7|

### 两seed平均差（pp，n=2不代表显著）

|比较|Overall|Head|OfficialMedium|Tail|Many|Medium|Few|Macro|WorstClass|
|---|---|---|---|---|---|---|---|---|---|
|Original-BatchShared - DR-NoFusion|+0.9848|-0.9286|+2.7857|+1.1207|-0.7759|+1.8548|+1.6026|+0.9848|+0.0000|
|Uniform-PerSample - Original-BatchShared|+0.0000|+0.2143|-0.2857|+0.0862|-0.0862|+0.0000|+0.0641|+0.0000|+0.0000|
|LowVS-PerSample - Uniform-PerSample|-0.4040|-1.6429|+0.4286|+0.0862|-2.0690|+0.2419|+0.3205|-0.4040|+0.0000|
|LowVS-PerSample - Original-BatchShared|-0.4040|-1.4286|+0.1429|+0.1724|-2.1552|+0.2419|+0.3846|-0.4040|+0.0000|

### 核心判读

Original-BatchShared 相对 DR 未在两个seed均出现Tail下降，本轮未发现稳定的IF200固定BN负迁移。停止此单点失败假说，不追加IF50、外部数据或其他seed寻找负结果。
- 掩码粒度（Uniform-PerSample−Original-BatchShared）：Tail -0.1724/+0.3448pp；Overall -0.1010/+0.1010pp；Tail跨seed变号，Overall跨seed变号。只能解释该固定BN、固定预算下的受控比较。
- VS池身份（LowVS-PerSample−Uniform-PerSample）：Tail +0.3448/-0.1724pp；Overall -0.3030/-0.5051pp；Tail跨seed变号，Overall同向。只能解释该固定BN、固定预算下的受控比较。
- 混合对照（LowVS-PerSample−Original-BatchShared）：Tail +0.1724/+0.1724pp；Overall -0.4040/-0.4040pp；Tail同向，Overall同向。此比较同时改变粒度与评分，不能独立归因于VS。
LowVS未在两seed的Tail均超过逐样本Uniform；本轮不支持稳定的Tail VS额外收益，不据此设计新门控。

## 6. 类别得失与Head/Tail混淆

每类别只有20张，单张改判即该类5pp。以下只按事先定义的负净正确数列出两seed持续受损类别，不筛选有利类别证明结论。所有99类的绝对准确率见 `per_class_accuracy.csv`，全部比较见 `per_class.csv`。

### Original-BatchShared − DR-NoFusion

两seed都受损的Tail类别：4个；全部99类中14个。

|类别ID|seed124净正确数|seed125净正确数|
|---|---|---|
|72|-1|-1|
|77|-1|-2|
|79|-1|-2|
|87|-2|-1|

### Uniform-PerSample − Original-BatchShared

两seed都受损的Tail类别：0个；全部99类中1个。

无两seed均负净正确数的Tail类别。

### LowVS-PerSample − Uniform-PerSample

两seed都受损的Tail类别：2个；全部99类中9个。

|类别ID|seed124净正确数|seed125净正确数|
|---|---|---|
|73|-1|-1|
|76|-1|-1|

### LowVS-PerSample − Original-BatchShared

两seed都受损的Tail类别：2个；全部99类中9个。

|类别ID|seed124净正确数|seed125净正确数|
|---|---|---|
|71|-1|-1|
|76|-1|-1|

绝对Head/Tail互相混淆计数如下；预测class0归入Head，尽管验证真值无class0。Tail→Head是错误去向计数，不等于Tail的全部错误；三组完整混淆表见 `head_tail_confusion.csv`。

|seed|策略|Tail真→Head预测|Head真→Tail预测|
|---|---|---|---|
|124|DR-NoFusion|272|63|
|124|Original-BatchShared|226|78|
|124|Uniform-PerSample|228|81|
|124|LowVS-PerSample|219|87|
|125|DR-NoFusion|274|61|
|125|Original-BatchShared|232|81|
|125|Uniform-PerSample|231|80|
|125|LowVS-PerSample|218|88|

精度分辨率：Overall一张=0.050505pp；Head一张=0.142857pp；Tail一张=0.172414pp；类别内一张=5pp。小净变化不能夸大，最差类别为零也不能仅靠这一指标归因于融合。

## 7. 失败、修复与可复查证据

- 静态审查发现 cfg() 用 Path + str 拼接YAML路径；只对文件名加括号，不改任何配置值。
- 首次preflight在导入前SyntaxError退出：报告两处Head/Tail混淆计数表达式缺失右括号。修复括号后全部新Python源通过AST解析；第二次preflight通过。失败日志 `console/preflight.log`、首次 `FAILED_STOP.json` 和 `preflight_repair.json` 原样保留；未删除/覆盖失败输出，也没有失败训练或恢复训练。
- 原guard只匹配一条TEST绝对路径且无历史写保护；新增执行期审计防护，拦截任何cifar-100-python/test打开及工作区旧文件写操作，并保存各步骤状态、时间、显存和完整stdout。未放松数据、哈希、冻结或配对条件。
- 一次只读rg搜索使用Windows不支持的字面通配目录而报错，后续改为真实路径。该异常未运行任何实验；已记入静态审查。
- 原作者IF200 YAML的name字符串仍含imb001，但实际imb_factor为0.005，且9502张raw indices与官方seed0 IF200逐项相符。本轮按真实配置值、数据数量和索引判断IF；未修改历史YAML显示名称。
- `blocking_fixes.patch` 与 `provided_source/` 保存新脚本的修复差异和原提供版本。`policy.py`与原提供文件逐字节一致；没有改变rho、池规模、供体分布、损失、学习率、训练预算或Backbone结构。
- 全部正式训练成功，未OOM、未出现配对不一致；12个成功主脚本进程的TEST/历史写拦截事件均为0。824个相关历史文件前后SHA256相同。源文件锁见 `execution_source_lock.json`；最终全文件校验见 `delivery_manifest.json`。

## 8. 因果解释和停止条件

- B−A识别本IF200固定模型/BN/预算下加入共享掩码特征替换的整体影响；不是跨IF因果效应，也不是原作者shift_bn=True策略的检验。
- C−B识别本面板中掩码粒度改变；D−C识别既定VS池选择相对逐样本随机的整体影响。D−C仍包含通道身份引起的替换幅度与方向共同变化，不能称已匹配范数后的纯方向因果效应。
- 历史IF100使用不同训练子集、不同Stage I训练执行及验证选best epoch184；IF200本轮为新模型固定epoch200。不能直接比较绝对准确率，也不能把跨实验差异全部归因于IF。
- 只训了一套Stage I seed42；两个Stage II seed在同一模型、同一验证图像上，不能称独立Stage I复现或统计显著。该99类开发验证集已多次复用，class0缺失，不能称官方100类TEST精度或最终独立泛化证据。
- Backbone与BN固定，任何本轮提升均是冻结特征上的分类器训练结果，不能称Backbone表征提升。没有使用验证图像估计类别均值/方差/epsilon，也没有按验证结果调参或选Stage I epoch。
- 本轮停止：没有稳定Tail负迁移时，不追加IF、seed或外部数据寻找失败条件，不启动新门控或新模型。
- 已完成指定一套200epoch Stage I和8组10epoch Stage II，报告后停止。未执行任何版本控制操作。

## 9. 原始结果与可复现入口

本报告相对目录为 `H2T_repro_audit/if200_stress/results/`。正式模型、缓存和统计均保留在本地该目录，不覆盖历史文件。

- `stage1/stage1_fixed_epoch200.pth`、`stage1/epochs.csv`、`stage1/summary.json`。
- `stats.npz`、`stats_provenance.json`、`validation_maps.npy`、`if200_split_indices.npz`、`dataset_attestation.json`。
- `seed{124,125}/{四臂}/epochs.csv`、`paired_batches.csv`、`final_fc.pth`、`validation_epoch10.npz`、`summary.json`。
- `epoch10.csv`、`paired.csv`、`per_class.csv`、`mean_paired_differences.csv`、`per_class_accuracy.csv`、`head_tail_confusion.csv`。
- `scientific_summary.json`、`scientific_verification.json`、`classifier_state_verification.csv`、`execution_registry.json`、`guards/*.json`、`console/*.log`。
- 从项目根目录使用指定Python，严格按 `user_task_instructions.txt` 命令顺序执行：policy.py → preflight → stage1 → prepare → seed124四臂（DR先）→ seed125四臂（DR先）→ report。既有results拒绝覆盖；此处只是记录本轮命令，不表示可重复覆盖封存输出。
- 本轮完整命令、起止时间、退出码及所有日志路径见 `execution_registry.json`；日志封装 `execute_step.py` 只传递这些原始命令，不实现训练方法。
