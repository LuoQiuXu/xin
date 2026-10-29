# 实际执行命令与顺序

工作目录：`D:\xiangmuxunlian\xin`。Python始终为 `D:\app\minicondaaa\envs\xiaoju\python.exe`，以下以 `$python` 表示该绝对路径；未安装新依赖、未改变环境。

```powershell
$python = 'D:\app\minicondaaa\envs\xiaoju\python.exe'
$task = 'H2T_repro_audit/noisy_lt_failure_screen'
$out = "$task/results/cifar100n_lt_seed42_20261029_01"
& $python -B -X utf8 -m unittest discover -s $task -p 'test_*.py' -v
& $python -B -X utf8 "$task/prepare_inputs.py"
& $python -B -X utf8 "$task/prepare_protocol.py" --original-labels "$out/primary_cache/inputs/raw_clean_labels.npy" --human-labels "$out/primary_cache/inputs/human_labels.npz" --split "$out/primary_cache/inputs/split.npz" --output "$out/protocol"
& $python -B -u -X utf8 "$task/train.py" initialize
& $python -B -u -X utf8 "$task/train.py" dryrun
& $python -B -u -X utf8 "$task/train.py" campaign
# 只有 campaign 完成所有10个正式训练后，才执行：
& $python -B -u -X utf8 "$task/evaluate.py"
& $python -B -X utf8 "$task/validate.py"
& $python -B -X utf8 "$task/report.py"
```

`campaign` 为每个训练启动全新子进程，顺序是 Clean Stage I、Noisy Stage I，然后对clean/noisy、seed124/125依次做 No-Fusion 和 H2T。所有原生训练进度、epoch CSV、START/RUN JSON分别落盘。`evaluate.py` 在锁定全部最终checkpoint后评价，并调用提供的 `compare_outcomes.py` 产生独立 `analysis/`。`validate.py` 从持久化权重、控制日志和logits重新核验，不追加训练。

这是本轮命令记录，不应对已有结果目录重新执行。脚本通过文件独占创建/目录不存在检查拒绝覆盖；若未来需要独立复核，应预先把新增适配器 `prepare_inputs.py` 的输出常量指向一个全新目录，并在训练前重新锁定该次代码SHA，保留本轮目录和所有历史资产。大数组与checkpoint不在Git中，路径、字节数、SHA256见 `RUN_MANIFEST.json` 的 `local_only_files`。
