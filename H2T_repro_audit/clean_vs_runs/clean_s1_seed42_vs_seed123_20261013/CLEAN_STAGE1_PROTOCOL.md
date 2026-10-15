# Clean Stage I 数据与模型选择协议

完成时间：2026-10-13T22:01:01+08:00。独立目录：`D:\xiangmuxunlian\新\H2T_repro_audit\clean_vs_runs\clean_s1_seed42_vs_seed123_20261013`。

## 核心审计结论

1. **训练期间没有加载官方 TEST 图像/标签，没有调用 CIFAR100(train=False)，没有计算 TEST accuracy。** Python open 审计钩子及 Dataset 构造器拦截全程生效；训练阶段允许的 TEST 校验读取次数也为0。prepare阶段只进行了1次不透明字节 MD5 核查。
2. **最佳依据：** 同一1980张官方 TRAIN 留出验证的 Overall，200 epochs 内严格变大才替换最佳；平局取最早。最佳epoch=184，验证Overall=40.050505%。固定200epoch预算，没有early stopping；逐epoch记录距最近最佳的epoch数。
3. **无重叠：** LT与验证raw ID交集=0，精确像素重叠=0；同一已冻结 split，SHA256 `c8579feef60d7705472ec00aeac85fa00a66c243761e0a8d76c462c663286ca1`。100类输出保留，class0无验证样本；99类各20张。
4. **LT正确：** 10847张、100类、max500/min5、IF100；官方seed0逐类指数采样索引、分布、像素与过去Stage I一致。训练自身Python/NumPy/Torch RNG均seed42；LT seed0在prepare中隔离验证，冻结数组加载不重置训练RNG。
5. **最佳Checkpoint：** [clean_stage1_best.pth](D:/xiangmuxunlian/新/H2T_repro_audit/clean_vs_runs/clean_s1_seed42_vs_seed123_20261013/stage1/checkpoints/clean_stage1_best.pth)；SHA256 `7ced9d07f71335357c08623573f038acd0dfdcdeadb55d95d57b332a14f83a88`。从随机初始化训练，没有加载旧epoch198或旧StageII权重。最终Checkpoint另存 `D:\xiangmuxunlian\新\H2T_repro_audit\clean_vs_runs\clean_s1_seed42_vs_seed123_20261013\stage1\checkpoints\clean_stage1_final.pth`。

## 保持的训练配置

ResNet32、100维classifier、CE+MixUp alpha1、SGD baseLR0.1/momentum0.9/weight_decay0.0002、batch128、workers0、200epochs；原5epoch warmup和160/180 multistep逐字复用。每epoch85次更新，最后batch95，共17000次有限loss与全部梯度检查。原train函数与scheduler以AST直接编译运行；没有调用会创建官方TEST loader的原main。DataParallel、FP32与StageI cudnn.deterministic=True保留。

## 可复核文件

- [epoch_validation_metrics.csv](D:/xiangmuxunlian/新/H2T_repro_audit/clean_vs_runs/clean_s1_seed42_vs_seed123_20261013/stage1/epoch_validation_metrics.csv)（train loss、Overall、Head、OfficialMedium、Tail、Many/Medium/Few、Macro与早停记录）。
- [protocol JSON](D:/xiangmuxunlian/新/H2T_repro_audit/clean_vs_runs/clean_s1_seed42_vs_seed123_20261013/clean_stage1_protocol.json)、[原始数据preflight](D:/xiangmuxunlian/新/H2T_repro_audit/clean_vs_runs/clean_s1_seed42_vs_seed123_20261013/preflight.json)、[TEST运行拦截记录](D:/xiangmuxunlian/新/H2T_repro_audit/clean_vs_runs/clean_s1_seed42_vs_seed123_20261013/test_guard_stage1_train.json)。
- [源码锁](D:/xiangmuxunlian/新/H2T_repro_audit/clean_vs_runs/clean_s1_seed42_vs_seed123_20261013/source_lock.json)、stage1/source_snapshot、完整training.log和full_console.log、200份validation_epoch预测。

## 解释范围

Clean指新基座的训练和checkpoint选择不依赖官方TEST。该TRAIN验证集仍是历史多轮复用的99类集合，并非全新未暴露数据；缺少class0且不能与100类官方测试准确率直接比较。新seed42与旧seed123的初始化也不同，所以后续变化不能仅归因于删除TEST选模。

协议核查通过后才允许进入本任务已授权的Uniform/High-VS/Low-VS三组StageII。发现任意TEST访问拦截或完整性失败时必须停止。
