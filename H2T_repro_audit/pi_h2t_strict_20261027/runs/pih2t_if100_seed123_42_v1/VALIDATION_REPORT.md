# 运行验证报告

日期2026-10-27；run-id pih2t_if100_seed123_42_v1。开始HEAD：5bdbe5c8749cafb613215e157e7b3290f70626d6；分支repro/pi-h2t-cifar100-paper-parity-20261027。允许保留的三个历史未跟踪目录未纳入本轮提交。

## 测试和退出码

初次CPU测试9项，exit1（1 failure/1 error）：夹具权重shape错误及float32准确率中间舍入；只在本study修复，保留失败记录，新增回归。最终14项CPU测试exit0；4文件py_compile exit0；preflight exit0；真实CIFAR GPU smoke exit0；StageI、两组StageII及report四条正式命令退出码均0。完整输出与检查事实在相邻pre_run目录TEST_AND_PREFLIGHT_LOG.json、各exit.json及本地.log中。修复理由见根REPAIR_LOG.md，执行源码在正式StageI后未改。

解释器：D:\app\minicondaaa\envs\xiaoju\python.exe；Python 3.12.12 | packaged by Anaconda, Inc. | (main, Oct 21 2025, 20:05:38) [MSC v.1929 64 bit (AMD64)]；torch 2.14.1+cu132，torchvision 0.29.1+cu132，CUDA build 13.2，cuDNN 92400。GPU：NVIDIA GeForce RTX 5070 Laptop GPU，显存8546484224字节，compute capability [12, 0]。实际cuDNN deterministic=True、benchmark=False、allow_tf32=True；matmul allow_tf32=False，precision=highest；AMP=False。preflight进程在seeding前的deterministic可能为False，正式环境与smoke seeding后为True，分别保留记录。

真实smoke为官方TRAIN中的batch128，StageI MixUp+PIF梯度与StageII H2TF/冻结骨干/FC梯度均通过，PIF grad abs sum=4.7254481，记录峰值CUDA分配375.03MiB。一次float(loss)日志打印有requires_grad Tensor转scalar提示，未改变梯度/公式或触发OOM。未降batch、未开启AMP、未调模型/损失/比例。

## 数据真实性

现有官方缓存：D:\xiangmuxunlian\xin\H2T\data\cifar100，未下载。torchvision官方MD5检查、所有100类IF100计数、独立RandomState(0)类内索引重建与原dataset数据/标签逐值相同。TRAIN LT=10847，TEST=10000、100图/类。LT像素+标签SHA256 `e81d10803e261af2ad81ed93676c3e8a615f35ef968220f7213a169147ada37e`；选定TRAIN索引SHA256 `e0d58a4d2700c46062da852554e5d185bbee74d3c612b052ea2a839bdadf939e`。train/test/meta原文件SHA与大小见pre_run/data_preflight.json，训练前、StageII前及评估前复核。官方TEST已按授权用于数据预检和三个固定模型终点的准确率汇报，未进入训练/选样、未据其调参或选checkpoint；它也不是一个未曾被本项目使用的独立最终测试集。

## 来源、二进制与完整性

StageI新checkpoint SHA256 `64f53e45a1b7f73f494f9c88374607e714b58dc6eb55118fa31154c4389f93ef`，具体字节大小、所有二进制相对路径/大小/SHA列于RESULTS_INDEX.json。StageII初始FC/最终BN/冻结骨干参数及流SHA在各summary.json与train_epochs.csv。20份完整训练比例NPZ和三个终点logits/labels/pred NPZ保留本地，不推送；报告、CSV/JSON与脚本显式暂存，不使用git add -A，不PR/merge或force push。

所有文件SHA均针对运行时本地原始字节；本仓库core.autocrlf=True，Git可能对文本做LF/CRLF转换，文本SHA不是Git blob标识。二进制SHA不受行尾转换影响。跨系统复查源哈希时须同时核对运行时行尾和编码，不得把文本规范化差异当作模型数值重放通过。

最终核验重算了所有epoch/LR/有限性、checkpoint/PIF二参数、10轮真实配对输入SHA、预测logits argmax、标签顺序、每类100图、全部分组准确率和逐类胜负；通过。1519份历史已跟踪文本SHA不变，三个未跟踪历史目录264个文件大小/mtime不变；没有把后者说成全内容哈希。原始H2T/历史H2T/PI review与旧结果Git diff为空。最终git diff --check及暂存范围另核对后提交。

## 精确命令

所有Python命令均以`D:/app/minicondaaa/envs/xiaoju/python.exe`执行；训练用-B -u -X utf8，以下entry均为H2T_repro_audit/pi_h2t_strict_20261027/run_reproduction.py。

```text
-m unittest discover -s H2T_repro_audit/pi_h2t_strict_20261027 -p test_*.py -v
-m py_compile paper_components.py run_reproduction.py test_paper_components.py test_run_reproduction.py
preflight --run-id pih2t_if100_seed123_42_v1 --data-root D:/xiangmuxunlian/xin/H2T/data/cifar100
smoke --run-id pih2t_if100_seed123_42_v1 --data-root D:/xiangmuxunlian/xin/H2T/data/cifar100
stage1 --run-id pih2t_if100_seed123_42_v1 --data-root D:/xiangmuxunlian/xin/H2T/data/cifar100
stage2 --run-id pih2t_if100_seed123_42_v1 --data-root D:/xiangmuxunlian/xin/H2T/data/cifar100 --arm no_fusion
stage2 --run-id pih2t_if100_seed123_42_v1 --data-root D:/xiangmuxunlian/xin/H2T/data/cifar100 --arm h2tf
report --run-id pih2t_if100_seed123_42_v1
```

上面py_compile的4个文件实际均带study前缀，从仓库根运行。科学/作者parity限制见PARITY_RUN_REPORT与SCIENTIFIC_RESULTS，不将测试通过等同于全部作者细节已核实。
