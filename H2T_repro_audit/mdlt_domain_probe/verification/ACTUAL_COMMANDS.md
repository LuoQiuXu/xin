# 本轮实际运行命令

解释器固定为 D:\app\minicondaaa\envs\xiaoju\python.exe。
以下目录已存在，复跑时必须使用新的唯一输出目录，不能覆盖本轮结果。

    $py = 'D:\app\minicondaaa\envs\xiaoju\python.exe'
    $module = 'D:\xiangmuxunlian\xin\H2T_repro_audit\mdlt_domain_probe'
    $images = 'D:\xiangmuxunlian\xin\data\OfficeHomeDataset_10072016'
    $env:PYTHONDONTWRITEBYTECODE = '1'
    $env:PYTHONIOENCODING = 'utf-8'

    & $py -m compileall -q $module
    & $py -m unittest discover -s $module -p 'test_officehome_domain_probe.py' -v
    & $py "$module\officehome_domain_probe.py" audit --data-root $images --out "$module\results\metadata_20261018_01"
    & $py "$module\officehome_domain_probe.py" smoke --data-root $images --out "$module\results\smoke_20261018_01"
    & $py "$module\officehome_domain_probe.py" run --data-root $images --smoke-dir "$module\results\smoke_20261018_01" --out "$module\results\pilot_20261018_01"

所有命令均返回0。主流程的逐字 argv、完整配置、源文件 SHA256、环境和基础提交在各 run 的 config.json / environment.json。
CPU 检查日志在本目录；离线核验读取保存的 checkpoint、sampling plan 和最终预测，不新增训练、模型推理或测试选模。

## 改动清单

只新增 H2T_repro_audit/mdlt_domain_probe/ 下的文件：

- officehome_data.py：锁定CSV、路径审计、训练/验证变换、域标签、测试访问锁、指标与配对计数。
- officehome_domain_probe.py：固定预算 Stage I、冻结特征缓存、四组独立 FC 训练、统一最终测试与报告。
- test_officehome_domain_probe.py：10项关键CPU正确性测试。
- README.md：预注册协议、运行命令及限制。
- .gitignore：排除本轮 output/ 下的权重、缓存和计划。
- .gitattributes：固定三份Python脚本和README为LF；官方CSV按二进制保留字节，避免换行转换破坏SHA256。
- protocol/：固定官方 CSV 以及来源/源码URL与哈希。
- results/metadata_20261018_01/：数据元信息与路径审计。
- results/smoke_20261018_01/：20更新/每组2批的smoke证据。
- results/pilot_20261018_01/：三seed、12组主结果、完整stdout文本、指标和checkpoint锁定清单。
- verification/：编译/CPU检查、离线模型与配对复核、本文。

本地 output/ 继续保留三个 seed 各自的 Stage I 最佳与最终模型（共6个）、12个正式epoch10 FC模型、特征缓存、采样计划和预测。
这些文件、原始图像和本轮smoke模型不进入代码提交。历史H2T源码及既有实验未改动。
