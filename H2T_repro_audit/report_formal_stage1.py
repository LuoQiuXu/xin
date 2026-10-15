"""Write the Stage I completion report from verified machine-readable evidence."""
from pathlib import Path
import csv
import hashlib
import json
import sys

BASE = Path(__file__).resolve().parent
RUN = BASE / "formal_runs" / sys.argv[1]


def read(name):
    return json.loads((RUN / name).read_text(encoding="utf-8"))


def link(label, path):
    return f"[{label}]({Path(path).resolve().as_posix()})"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


s, p, v = read("status.json"), read("preflight.json"), read("artifact_verification.json")
b, f = read("best_evaluation.json"), read("final_evaluation.json")
assert s["state"] == "completed" and s["completed_epochs"] == 200 and v["passed"]
assert b["matches_training_epoch_evaluation"] and f["matches_training_epoch_evaluation"]
checkpoints = v["checkpoints"]
with (RUN / "epochs.csv").open("w", encoding="utf-8-sig", newline="") as stream:
    writer = csv.writer(stream)
    writer.writerow(["epoch", "lr", "train_loss", "loss_min", "loss_max", "overall", "head", "medium", "tail", "ece", "new_best", "train_seconds", "eval_seconds", "steps"])
    for e in s["epochs"]:
        m = e["eval_metrics_percent_rounded_3dp"]
        writer.writerow([e["epoch"], e["lr"], e["train_loss"], e["train_loss_min"], e["train_loss_max"],
            e["eval_overall_percent"], m["head"], m["medium"], m["tail"], e["ece_percent"],
            e["new_best"], e["train_seconds"], e["eval_seconds"], e["steps"]])

for filename in ["verify_formal_stage1.py", "report_formal_stage1.py"]:
    (RUN / filename).write_bytes((BASE / filename).read_bytes())
(RUN / "H2T_DR_COMPARISON_PLAN.md").write_bytes((BASE.parent / "H2T_DR_COMPARISON_PLAN.md").read_bytes())
# Preserve the exact executed launcher already copied during preflight;
# archive the later bookkeeping-only repair under a different filename.
(RUN / "formal_stage1_after_run_fix.py").write_bytes((BASE / "formal_stage1.py").read_bytes())
for kind in ["best", "final"]:
    matches = list((BASE / "runtime/saved").glob(s["run_name"] + "_" + kind + "_eval_*/logs/*.txt"))
    assert len(matches) == 1
    (RUN / (kind + "_evaluation.official.log")).write_bytes(matches[0].read_bytes())

bm, fm = b["metrics_percent"], f["metrics_percent"]
checkpoint_table = "\n".join(f"| {k} | {x['epoch']} | `{x['sha256']}` |" for k, x in checkpoints.items())
min_loss = min(e["train_loss_min"] for e in s["epochs"])
max_loss = max(e["train_loss_max"] for e in s["epochs"])
report = f"""# H2T Stage I 正式训练结果

## 完成状态

- 完成 **200/200 epochs**，执行 17,000 个 SGD step，累计训练样本访问 2,169,400 次；未加载旧 Checkpoint。
- 用户授权范围为 Stage I。本轮没有启动任何 Stage II 正式训练，也没有改变 H2T 融合策略、供体/通道规则或 rho。
- 运行 ID：`{sys.argv[1]}`；开始：`{s['started_at']}`；结束：`{s['finished_at']}`；训练及每 epoch 评估耗时 {s['elapsed_seconds']/60:.2f} 分钟（不含之后的独立模型复核评估）。

## 最佳与最终模型

| Checkpoint | Epoch（从 1 开始） | Overall | Head | Medium | Tail |
|---|---:|---:|---:|---:|---:|
| 最佳 model_best.pth.tar | {b['epoch']} | {bm['overall']:.3f}% | {bm['head']:.3f}% | {bm['medium']:.3f}% | {bm['tail']:.3f}% |
| 最终 model_final.pth.tar | 200 | {fm['overall']:.3f}% | {fm['head']:.3f}% | {fm['medium']:.3f}% | {fm['tail']:.3f}% |

最佳 Epoch 为 **{s['best_epoch']}**；原日志 `Best epoch` 从 0 开始，对应值为 **{s['official_best_epoch_zero_based']}**。每 epoch 以官方 CIFAR-100 test split 的 Overall 监控最佳模型；它不是独立 validation split 上的选择。正式保存的最佳和最终模型均经过原 `eval-modifed.py` 独立重新加载、全量评估，结果与对应训练 epoch 一致。

评估共 10,000 张图像，每类 100 张。Head [0,36)、Medium [36,71)、Tail [71,100) 为官方 YAML 的零基半开区间；各组按逐类准确率平均，Overall 按全部样本计算。没有改用按样本数阈值划分的其他 Head/Medium/Tail 定义。评估不使用 H2T 或 MixUp，且 BN buffers 不变。

- {link('最佳 Checkpoint', checkpoints['model_best.pth.tar']['path'])}
- {link('最终 Checkpoint', checkpoints['model_final.pth.tar']['path'])}
- {link('最佳模型逐类结果', RUN / 'best_evaluation.json')}
- {link('最终模型逐类结果', RUN / 'final_evaluation.json')}

## 配置与算法一致性

使用已审计独立兼容副本 `H2T_repro_audit/runtime`。官方归档 Commit 为 `e0c849235d9655fca31cd9fa8a5f38f92a77c7e8`；该值来自本地官方 ZIP comment，网络限制使此前未能以 Git 对象/远程 HEAD 独立核验。48 个官方归档文件及兼容副本的逐文件 SHA256 已留存，本轮运行前后核对兼容副本和运行源码快照，均通过。

| 项目 | 本轮实际设置 |
|---|---|
| 原 YAML | cifar100_imb001_stage1_ce_mixup.yaml，字节未修改 |
| 数据 / Backbone | CIFAR-100-LT IF=100 / resnet32_fe（ResNet-32） |
| epochs / batch / workers | 200 / 128 / 0 |
| Loss | 原 CrossEntropyLoss，无类别重权重 |
| MixUp | True，alpha=1.0，原 Beta 采样和双标签加权 CE |
| SGD | lr=0.1，momentum=0.9，weight_decay=2e-4，原默认 nesterov=False |
| LR | 5 epoch warmup；第 1–5 epoch 为 0.02/0.04/0.06/0.08/0.1；6–160 为 0.1；161–180 为 0.01；181–200 为 0.001 |
| 随机设置 | 官方 deterministic=True，SetRandomSeed(123)；长尾子集构造 NumPy seed=0 |
| GPU 包装 | YAML gpu=null，原 DataParallel CUDA:0 路径 |
| Backbone / classifier | 两者都进入 Stage I SGD，PIF/feature_tuning=False |
| resume | 显式空字符串；训练期间 checkpoint load 调用为 0 |
| CLI 覆盖 | 仅绝对 data_path、空 resume、独立运行 name |

没有新增或修改核心训练代码。本轮新增外部运行记录脚本，调用原 Stage I 的 train / validate / scheduler / save_checkpoint，记录 Loss、指标、显存和输出位置；未改动采样、梯度、优化器、随机数流、前向结果或 epoch 数。额外保存的 model_final.pth.tar 是官方 current.pth.tar 的字节复制。

独立副本沿用原审计的五个兼容文件修改：ResNet_fe 构造参数错误修补、pathlib2→标准库 pathlib，以及训练/评估 torch.load 显式 weights_only=False。兼容 Diff 见 {link('compatibility.diff', RUN / 'compatibility.diff')}。

首次外部启动器在参数解析阶段因 `sys.argv` 多包含 Python 的 `-u` 参数而退出；当时尚未创建模型/训练目录，也未执行训练。已仅修复外部启动器，并使用新的运行 ID 从头启动；失败记录保存在 `formal_runs/formal_20261011_1429`，未与本轮结果合并。

本轮原 Stage I 主函数完整完成 200 epochs 后，外部归档核验因官方 save_code 默认忽略 PDF/PPTX，而外部脚本要求核验全部归档文件，触发 FileNotFoundError，启动器因此退出码为 1。随后仅补齐 3 个非代码演示文档、复制 epoch-200 current 为 final，重新核验全部源码、Checkpoint 和指标；没有重跑训练、没有修改模型 tensor。原异常状态保存在 status_before_finalize_recovery.json，恢复记录保存在 status.json 的 postprocessing_recovery。完成的训练结果有效，外部记录错误已修复。实际执行的启动器快照保持原字节；后续修正版本另存为 formal_stage1_after_run_fix.py，未冒充本次执行源码。

## 环境、数据与显存

- Python：`{p['python']}`，版本 {p['python_version'].split()[0]}。
- 系统：{p['platform']}；torch {p['torch']}、torchvision {p['torchvision']}、NumPy {p['numpy']}。
- GPU：{p['gpu']}，显存 {p['gpu_memory_bytes']/1024**2:.0f} MiB；CUDA runtime {p['cuda_runtime']}，compute capability={p['compute_capability']}；驱动见保存的 nvidia-smi。
- 数据路径：`{p['data_path']}`。train/test/meta 完整性验证通过；训练 10,847 张、100 类、最大 500 / 最小 5、实际 IF=100；每 epoch 85 batch（末 batch=95）。测试 10,000 张，每类 100。
- PyTorch 峰值 allocated：**{s['cuda_peak_allocated_mib']:.2f} MiB**；reserved：**{s['cuda_peak_reserved_mib']:.2f} MiB**。
- nvidia-smi 每 5 秒采样的 GPU 全局占用最高：**{s['gpu_global_peak_sampled_mib']:.0f} MiB**，包含桌面及其他 GPU 程序；不是本训练进程的独占显存。WDDM 下 nvidia-smi 不提供可用的逐进程显存数字。
- 所有 17,000 个 MixUp 训练 Loss 均为有限值，batch Loss 范围 {min_loss:.6f}–{max_loss:.6f}；最终 epoch 平均 Loss={s['epochs'][-1]['train_loss']:.6f}。全部最佳/最终 Checkpoint 状态 tensor 有限。
- 最终每个 BN num_batches_tracked=17,000，符合 Stage I 每个训练 batch 一次前向；31 个 BN 均符合此计数。评估不增加计数。

## 运行命令

实际执行的记录运行器（内部调用下列官方入口，配置一致）：

```powershell
Set-Location -LiteralPath 'D:\\xiangmuxunlian\\新'
& 'D:\\app\\anaconda\\envs\\pytorch\\python.exe' -u .\\H2T_repro_audit\\formal_stage1.py run --run-id {sys.argv[1]}
```

等价的官方入口配置命令（展示用；**本轮已经运行完成，不要重复执行**）：

```powershell
Set-Location -LiteralPath 'D:\\xiangmuxunlian\\新\\H2T_repro_audit\\runtime'
& 'D:\\app\\anaconda\\envs\\pytorch\\python.exe' -u .\\train_stage1.py --cfg .\\config\\cifar100\\cifar100_imb001_stage1_ce_mixup.yaml data_path 'D:\\xiangmuxunlian\\新\\H2T\\data\\cifar100' resume "''" name {s['run_name']}
```

独立评估通过外部只读观察运行器调用原 `eval-modifed.py`；best/final 两次分别保存官方评估日志和机器可读结果，未执行任何训练。

## 保存清单与追溯

- {link('完整官方训练日志', s['official_log'])}
- {link('完整 stdout/stderr', RUN / 'stdout_stderr.log')}
- {link('原 YAML', RUN / 'original.yaml')}、{link('实际完整配置', RUN / 'effective.yaml')}
- {link('运行源码快照目录', s['code_dir'])}、{link('逐文件源码 hash', RUN / 'runtime_manifest.json')}
- {link('环境与数据预检', RUN / 'preflight.json')}、{link('显存采样', RUN / 'gpu_samples.jsonl')}
- {link('完整运行状态与逐 epoch 数据', RUN / 'status.json')}、{link('epoch CSV', RUN / 'epochs.csv')}
- {link('Checkpoint / LR / 源码验证', RUN / 'artifact_verification.json')}
- {link('最佳模型独立评估日志', RUN / 'best_evaluation.official.log')}、{link('最终模型独立评估日志', RUN / 'final_evaluation.official.log')}
- 本轮外部运行、验证及报告脚本各自在证据目录保存了一份；源文件与输出文件 hash 见 artifact_manifest.json。

| 文件 | Checkpoint epoch | SHA256 |
|---|---:|---|
{checkpoint_table}

官方 Checkpoint 不保存 optimizer/RNG 状态，其 resume 分支也不能精确续训。本轮从头连续完成 200 epochs，不存在将不完整 resume 误报为完整训练的问题。

## 下一阶段方案与 BatchNorm 记录

{link('DR 与 DR+H2T 公平对照方案', BASE.parent / 'H2T_DR_COMPARISON_PLAN.md')} 已给出：同一 Stage I Checkpoint / 相同初始分类器 / 相同 BS 标签 / 原 SGD 与 10 epoch 调度，DR 使用已有 `H2T 0.0` 开关，DR+H2T 保持官方 `H2T=0.25`。

官方 Stage II 的 shift_bn=True 会更新 Backbone BN 统计量。DR 每 batch 做 BS 一次前向，每个 BN 每 epoch +85；DR+H2T 做 BS→IS 两次前向，每个 BN每 epoch +170。Conv 和 BN affine 权重不更新，但 running_mean / running_var / num_batches_tracked 会变动。因此仓库默认对照的性能变化包含 BN 更新流程的影响。

方案另外列出了冻结 BN 统计量的配置对照，以及通过独立运行器匹配 BS→IS BN 更新的控制对照；均明确区分于官方默认流程，且本轮未实施。相同 seed 也不自动保证采样/增强相同，因为 H2T 的 CPU randperm 会额外消耗 RNG；严格配对方案对此作了记录。

**Stage I 已完成，Stage II 未启动，等待下一步确认。**
"""
report_path = BASE.parent / "H2T_STAGE1_RESULTS.md"
report_path.write_text(report, encoding="utf-8")
(RUN / "H2T_STAGE1_RESULTS.md").write_text(report, encoding="utf-8")
artifacts = {p.name: {"bytes": p.stat().st_size, "sha256": sha(p)} for p in RUN.iterdir() if p.is_file() and p.name != "artifact_manifest.json"}
write_manifest = {"run_id": sys.argv[1], "files": artifacts, "checkpoints": checkpoints}
(RUN / "artifact_manifest.json").write_text(json.dumps(write_manifest, indent=2, ensure_ascii=False), encoding="utf-8")
print(report_path)
