"""Verify Stage II artifacts, compare actual runs, and write the result report."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import re
import shutil

import numpy as np
import torch

from formal_stage1 import check_sources, dump, sha
from formal_stage2 import delta

BASE = Path(__file__).resolve().parent
RUNTIME = BASE/"runtime"
KEYS = ["overall","head","medium","tail"]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def link(label,path):
    return f"[{label}]({Path(path).resolve().as_posix()})"


def buffer_stats(before,after):
    output = {}
    for suffix in ["running_mean","running_var","num_batches_tracked"]:
        names = [k for k in before if k.endswith(suffix)]
        diffs = [(after[k].double()-before[k].double()).abs() for k in names]
        output[suffix] = {"changed_tensors":sum(not torch.equal(before[k],after[k]) for k in names),
            "total_tensors":len(names),"max_abs_delta":max(float(v.max()) for v in diffs),
            "mean_abs_delta":float(torch.cat([v.reshape(-1) for v in diffs]).mean())}
    return output


def verify(folder):
    preflight = read(folder/"preflight.json")
    stage1 = torch.load(preflight["stage1_checkpoint"]["path"],map_location="cpu",weights_only=False)
    assert sha(preflight["stage1_checkpoint"]["path"]) == preflight["stage1_checkpoint"]["sha256"]
    manifest = check_sources()
    assert manifest == preflight["runtime_files"]
    report = {"passed":True,"same_stage1_checkpoint":preflight["stage1_checkpoint"],"arms":{}}
    loaded_final = {}
    for arm in ["dr","h2t"]:
        status = read(folder/arm/"status.json")
        assert status["state"] == "completed" and status["completed_epochs"] == 10
        assert len(status["epochs"]) == 10 and len(status["checkpoint_loads"]) == 1
        assert status["checkpoint_loads"][0]["sha256"] == preflight["stage1_checkpoint"]["sha256"]
        assert status["h2t_calls"] == (0 if arm=="dr" else 850)
        assert status["initial_backbone_equals_stage1"] and status["initial_classifier_equals_stage1"]
        assert status["seed"] == 123 and not status["deterministic_yaml"] and status["shift_bn"]
        assert [e["epoch"] for e in status["epochs"]] == list(range(1,11))
        rates = [.2*.5*.1*(1+math.cos(e/10*3.1415926535)) for e in range(10)]
        assert np.allclose([e["lr"] for e in status["epochs"]],rates,atol=1e-15,rtol=0)
        assert all(e["steps"]==85 and e["samples"]==10847 and e["finite_train_loss"] for e in status["epochs"])
        assert all(e["backbone_parameters_delta"]["changed_count"]==0 and e["classifier_delta_from_stage1"]["changed_count"]==2 for e in status["epochs"])
        assert all(e["ce_labels_equal_bs"] and e["backbone_gradients_absent"] and e["classifier_gradients_finite"] and e["bn_unchanged_during_eval"] for e in status["epochs"])
        assert status["stage2_best_epoch"] == int(np.argmax([e["eval_overall_percent"] for e in status["epochs"]]))+1
        assert all(sha(Path(status["code_dir"])/p)==v for p,v in manifest.items())
        checkpoints = {}
        for filename,value in status["checkpoints_sha256"].items():
            path = Path(status["model_dir"])/filename
            assert sha(path)==value
            state = torch.load(path,map_location="cpu",weights_only=False)
            assert all(torch.isfinite(v).all() for key in ["state_dict_model","state_dict_classifier"] for v in state[key].values())
            if filename == "model_final.pth.tar":
                assert state["epoch"] == 10
                loaded_final[arm] = state
            if filename == "model_best_stage2.pth.tar":
                assert state["epoch"] == status["stage2_best_epoch"]
            buffer_names = {k for k in stage1["state_dict_model"] if k.endswith(("running_mean","running_var","num_batches_tracked"))}
            params0 = {k:v for k,v in stage1["state_dict_model"].items() if k not in buffer_names}
            params1 = {k:v for k,v in state["state_dict_model"].items() if k not in buffer_names}
            pdelta = delta(params0,params1)
            assert pdelta["changed_count"] == 0 and pdelta["total_count"] == 93
            cdelta = delta(stage1["state_dict_classifier"],state["state_dict_classifier"])
            assert cdelta["changed_count"] == 2
            buffers0 = {k:stage1["state_dict_model"][k] for k in buffer_names}
            buffers1 = {k:state["state_dict_model"][k] for k in buffer_names}
            counts = {k:int(buffers1[k]-buffers0[k]) for k in buffer_names if k.endswith("num_batches_tracked")}
            expected = state["epoch"]*(85 if arm=="dr" else 170)
            assert all(v==expected for v in counts.values())
            checkpoints[filename] = {"path":str(path),"sha256":value,"epoch":int(state["epoch"]),
                "all_state_tensors_finite":True,"backbone_parameter_delta":pdelta,
                "classifier_parameter_delta":cdelta,"bn_delta_from_stage1":buffer_stats(buffers0,buffers1),
                "bn_per_tensor_delta_from_stage1":delta(buffers0,buffers1),
                "bn_count_delta_from_stage1":counts,"bn_counter_absolute":int(next(v for k,v in buffers1.items() if k.endswith("num_batches_tracked")))}
        assert status["checkpoints_sha256"]["model_final.pth.tar"] == status["checkpoints_sha256"]["current.pth.tar"]
        if "model_best.pth.tar" in status["checkpoints_sha256"]:
            assert status["checkpoints_sha256"]["model_best.pth.tar"] == status["checkpoints_sha256"]["model_best_stage2.pth.tar"]
        report["arms"][arm] = {"completed_epochs":10,"optimizer_steps":850,"checkpoints":checkpoints,
            "h2t_calls":status["h2t_calls"],"source_snapshot_verified":True}
    a,b = read(folder/"dr/status.json"),read(folder/"h2t/status.json")
    assert a["initial_rng_hashes"] == b["initial_rng_hashes"]
    assert a["backend_settings"] == b["backend_settings"]
    for key in ["initial_backbone_parameters_sha256","initial_backbone_buffers_sha256","initial_classifier_sha256"]:
        assert a[key] == b[key]
    pairs = [(x,y) for ea,eb in zip(a["epochs"],b["epochs"]) for x,y in zip(ea["batch_fingerprints"],eb["batch_fingerprints"])]
    report["randomness_comparison"] = {"same_initial_rng_states":True,"same_backend_settings":True,"seed":123,
        "batches_compared":len(pairs),"same_bs_labels_batches":sum(x["bs_labels_sha256"]==y["bs_labels_sha256"] for x,y in pairs),
        "same_is_labels_batches":sum(x["is_labels_sha256"]==y["is_labels_sha256"] for x,y in pairs),
        "same_cpu_rng_at_batch_ready_batches":sum(x["cpu_rng_at_batch_ready_sha256"]==y["cpu_rng_at_batch_ready_sha256"] for x,y in pairs),
        "image_batches_checked":sum("bs_images_sha256" in x for x,y in pairs),
        "same_bs_images_in_checked_batches":sum(x.get("bs_images_sha256")==y.get("bs_images_sha256") for x,y in pairs if "bs_images_sha256" in x),
        "same_is_images_in_checked_batches":sum(x.get("is_images_sha256")==y.get("is_images_sha256") for x,y in pairs if "is_images_sha256" in x),
        "first_batch_bs_images_equal":pairs[0][0]["bs_images_sha256"]==pairs[0][1]["bs_images_sha256"],
        "second_batch_bs_images_equal":pairs[1][0]["bs_images_sha256"]==pairs[1][1]["bs_images_sha256"],
        "fully_paired_batches_claimed":False}
    buffers_a = {k:v for k,v in loaded_final["dr"]["state_dict_model"].items() if k.endswith(("running_mean","running_var","num_batches_tracked"))}
    buffers_b = {k:v for k,v in loaded_final["h2t"]["state_dict_model"].items() if k in buffers_a}
    report["bn_final_h2t_vs_dr"] = buffer_stats(buffers_a,buffers_b)
    report["bn_final_h2t_vs_dr_per_tensor"] = delta(buffers_a,buffers_b)
    dump(folder/"artifact_verification.json",report)
    print("VERIFY PASS: both 10 epochs, same Stage I, frozen weights / changed classifier / BN verified",flush=True)


def render(folder):
    verified = read(folder/"artifact_verification.json")
    assert verified["passed"]
    status = {arm:read(folder/arm/"status.json") for arm in ["dr","h2t"]}
    evaluated = {arm:{kind:read(folder/arm/(kind+"_evaluation.json")) for kind in ["best","final"]} for arm in status}
    stage1 = read(folder/"stage1_reference.json")
    preflight = read(folder/"preflight.json")
    reference = read(folder/"literature_reference.json")
    baseline = stage1["metrics_percent"]
    for arm in status:
        for kind in evaluated[arm]:
            assert evaluated[arm][kind]["test_samples"]==10000 and evaluated[arm][kind]["matches_training_epoch_evaluation"]
        with (folder/arm/"epochs.csv").open("w",encoding="utf-8-sig",newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["epoch","lr","loss","overall","head","medium","tail","bn_updates","h2t_calls_total"])
            for e in status[arm]["epochs"]:
                m = e["eval_metrics_percent_rounded_3dp"]
                writer.writerow([e["epoch"],e["lr"],e["train_loss"],*[m[k] for k in KEYS],e["backbone_forward_calls"],e["epoch"]*(85 if arm=="h2t" else 0)])
    tables = []
    delta_rows = []
    comparison_rows = []
    for kind,label in [("best","最佳"),("final","最终")]:
        for arm,title in [("dr","DR"),("h2t","DR+H2T")]:
            item = evaluated[arm][kind]
            values = item["metrics_percent"]
            tables.append(f"| {title} {label} | {item['epoch']} | "+" | ".join(f"{values[k]:.3f}%" for k in KEYS)+" |")
            delta_rows.append(f"| {title} {label} − Stage I | "+" | ".join(f"{values[k]-baseline[k]:+.3f}" for k in KEYS)+" |")
        comparison_rows.append(f"| {label}：H2T − DR | "+" | ".join(f"{evaluated['h2t'][kind]['metrics_percent'][k]-evaluated['dr'][kind]['metrics_percent'][k]:+.3f}" for k in KEYS)+" |")
    final_gain = evaluated["h2t"]["final"]["metrics_percent"]["overall"]-evaluated["dr"]["final"]["metrics_percent"]["overall"]
    best_gain = evaluated["h2t"]["best"]["metrics_percent"]["overall"]-evaluated["dr"]["best"]["metrics_percent"]["overall"]
    conclusion = "本次单 seed 重跑中，原始 DR+H2T 流程在固定第 10 epoch 的 Overall 上带来额外收益" if final_gain>0 else "本次单 seed 重跑中，原始 DR+H2T 流程在固定第 10 epoch 的 Overall 上没有带来额外收益"
    checkpoint_links = []
    for arm,title in [("dr","DR"),("h2t","DR+H2T")]:
        checkpoints = verified["arms"][arm]["checkpoints"]
        checkpoint_links.append(f"- {title}："+link("最佳",checkpoints["model_best_stage2.pth.tar"]["path"])+"、"+link("最终",checkpoints["model_final.pth.tar"]["path"])+"；最佳 Epoch="+str(status[arm]["stage2_best_epoch"])+"。")
    stage1_ref = reference["table5"]["stage1"]
    paper_rows = ["| 论文 Table 5 Stage I CE | "+" | ".join(f"{stage1_ref[k]:.2f}%" for k in KEYS)+" |",
                  "| 本次 Stage I − 论文 | "+" | ".join(f"{baseline[k]-stage1_ref[k]:+.3f} pp" for k in KEYS)+" |"]
    for arm,name,reference_key in [("dr","DR","dr"),("h2t","DR+H2T","h2t")]:
        values = reference["table5"][reference_key]
        paper_rows.append(f"| 论文 Table 5 {name} | "+" | ".join(f"{values[k]:.2f}%" for k in KEYS)+" |")
        for kind,label in [("best","最佳"),("final","最终")]:
            data = evaluated[arm][kind]["metrics_percent"]
            paper_rows.append(f"| 本次 {name} {label} − 论文 | "+" | ".join(f"{data[k]-values[k]:+.3f} pp" for k in KEYS)+" |")
    rn = verified["randomness_comparison"]
    bn_rows = []
    for arm,title in [("dr","DR"),("h2t","DR+H2T")]:
        info = verified["arms"][arm]["checkpoints"]["model_final.pth.tar"]
        bn_rows.append(f"| {title} | 0/93 | {info['classifier_parameter_delta']['changed_count']}/2 | "+
            str(info["bn_delta_from_stage1"]["running_mean"]["changed_tensors"])+"/31 | "+
            str(info["bn_delta_from_stage1"]["running_var"]["changed_tensors"])+f"/31 | {'850' if arm=='dr' else '1700'} | {info['bn_counter_absolute']} |")
    details = []
    for arm,title in [("dr","DR"),("h2t","DR+H2T")]:
        info = verified["arms"][arm]["checkpoints"]["model_final.pth.tar"]
        bn = info["bn_delta_from_stage1"]
        details.append(f"- {title}：classifier 最大绝对参数变化 {info['classifier_parameter_delta']['max_abs_delta']:.6f}；BN running_mean 最大绝对变化 {bn['running_mean']['max_abs_delta']:.6f}，running_var 最大绝对变化 {bn['running_var']['max_abs_delta']:.6f}。")
    hbest = evaluated["h2t"]["best"]["metrics_percent"]["overall"]
    hfinal = evaluated["h2t"]["final"]["metrics_percent"]["overall"]
    report = f"""# H2T Stage II 正式复现结果

## 1. 完成状态与结论

完成实验 A（DR）和实验 B（DR+H2T），各 **10 epochs / 850 个 SGD step**。两组分别在新进程中从同一 Stage I 最佳 Checkpoint 独立初始化，未加载对方的 Stage II 权重。所有训练 Loss 有限，源码 hash、参数冻结、分类器更新及 BN 更新核验通过。所有最佳和最终模型均已通过原 `eval-modifed.py` 在 **全部 10,000 张测试图像** 上独立重新加载评估。

{conclusion}：**{final_gain:+.3f} 个百分点**；最佳模型对比为 **{best_gain:+.3f} 个百分点**。这是单次固定 seed、仓库默认训练流程的观测结果。两臂 BN 更新次数和数据增强 RNG 消耗不同，结果不能单独归因于 H2T 通道融合或分类器权重调整。

本轮没有供体/通道筛选、自适应 rho、Backbone 结构更改或其他创新性修改。训练完成后停止；**未执行额外 BN 控制实验或机制审计**。

## 2. 输入 Checkpoint、配置与种子

Stage I epoch=198，Overall={baseline['overall']:.3f}%。输入文件为 {link('同一 Stage I 最佳模型',preflight['stage1_checkpoint']['path'])}，SHA256=`{preflight['stage1_checkpoint']['sha256']}`。两臂分别核对 Backbone、classifier 和全部 BN buffers 与此 Checkpoint 逐 tensor 完全相等；SGD 从新的空状态开始。原源码加载 epoch=198 后，Stage II 仍执行自己的 0–9 epoch 循环，Checkpoint epoch 字段为 Stage II 的 1–10。

| 项目 | A：DR | B：DR+H2T |
|---|---|---|
| 数据 / 模型 | CIFAR-100-LT IF=100 / ResNet-32 | 同左 |
| Stage II / batch / workers | 10 epochs / 128 / 0 | 同左 |
| Loss / MixUp | CE，无类别重权重 / False | 同左 |
| SGD | lr=0.1，momentum=0.9，wd=5e-4，nesterov=False | 同左 |
| 分类器实际 LR | 原 cosine × lr_factor=0.2，首 epoch=0.02 | 同左 |
| 训练对象 | 仅 classifier 进入 SGD，Backbone no_grad | 同左 |
| 采样 | BS 接收；保留原 IS 元数据生成 | BS 接收 + IS 供体 |
| H2T | 仅 CLI 设 0.0；融合调用=0 | 原 YAML 0.25；融合调用=850 |
| Backbone 前向 / batch | BS 一次 | BS→IS 两次 |
| shift_bn | True | True |
| Seed | Python / NumPy / torch CPU / CUDA 初始 seed=123，PYTHONHASHSEED=123 | 同左 |
| deterministic | YAML False；原 CuDNN / deterministic algorithms 默认值保留 | 同左 |
| gpu | YAML null，单 GPU DataParallel | 同左 |

两臂的最终完整配置只在 `H2T` 和输出 `name` 上不同。数据路径和 resume 均一致。A 保留原 `sampler.dual_sample.enable=True`，数据集仍生成 IS 图像和标签，但原 `H2T>0` 条件为 False，跳过供体 GPU 前向和 H2T；CE 仅使用 BS 特征及 BS 标签。没有再关闭双采样配置而引入额外差异。

完整配置如下，并分别保存为独立 effective.yaml。

### 实验 A 完整配置

```yaml
{(folder/'dr/effective.yaml').read_text(encoding='utf-8').rstrip()}
```

### 实验 B 完整配置

```yaml
{(folder/'h2t/effective.yaml').read_text(encoding='utf-8').rstrip()}
```

## 3. 准确率、最佳 Epoch 与 Checkpoint

| 模型 | Stage II Epoch（从 1 开始） | Overall | Head | Medium | Tail |
|---|---:|---:|---:|---:|---:|
| Stage I 最佳 | 输入 epoch 198 | {baseline['overall']:.3f}% | {baseline['head']:.3f}% | {baseline['medium']:.3f}% | {baseline['tail']:.3f}% |
{chr(10).join(tables)}

独立评估的每类样本数均为 100；Overall 为全部样本准确率，Head [0,36)、Medium [36,71)、Tail [71,100) 为官方 YAML 的逐类准确率均值。评估使用干净特征，不执行 MixUp 或 H2T；BN buffers 在评估期间完全不变。独立评估四组指标与对应训练 epoch 一致。

{chr(10).join(checkpoint_links)}

“最佳”按 Stage II 10 个 epoch 内最高 Overall 选择，首次达到最高值时保留。外部记录器只复制原 current 文件为 model_best_stage2，未改变原 best_acc1、训练或保存逻辑。原 Stage II best_acc1 继承 Stage I 的 40.49% 门槛；本轮两个实验均超过该门槛，原 model_best.pth.tar 与独立 Stage II 最佳副本 SHA256 完全一致。最终 model_final.pth.tar 与原 current.pth.tar 字节相同。最佳和最终均保存并重新评估。

原日志 Best epoch 为零基编号；本报告最佳 Epoch 已加 1。主要 DR/H2T 对比采用固定第 10 epoch 的最终模型，最佳模型作为同时报告的结果。每 epoch 监控使用的是官方 CIFAR-100 test split，最佳选择并非独立 validation split。

## 4. 相对 Stage I 的变化与 H2T 的额外收益

以下均为**准确率百分点变化（pp）**，不是相对百分比增长。

| 对比 | Overall | Head | Medium | Tail |
|---|---:|---:|---:|---:|
{chr(10).join(delta_rows)}
{chr(10).join(comparison_rows)}

必须同时观察 Overall 和各类别组变化；Head 的下降与 Medium/Tail 的提升反映原训练流程对分类决策的重新平衡。本报告只陈述本轮测得的差值，不把单 seed 结果解释为多次实验的统计显著性。

## 5. 学习参数与 BN 统计量

| 实验（最终模型） | Backbone 学习参数变化 | classifier 参数变化 | running_mean 变化 | running_var 变化 | 每个 BN 计数增量 | 最终计数 |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(bn_rows)}

输入 Stage I 每个 BN num_batches_tracked=16,830。两组均包含 31 个 BatchNorm2d。每 epoch A 增加 85、B 增加 170；10 epochs 分别增加 850/1700。Conv 权重和 BN affine 参数逐 tensor 保持与 Stage I 完全一致，Backbone 梯度始终 None；classifier 的权重和 bias 都变化，梯度有限。源码的 Backbone requires_grad 标志仍为 True，实际冻结由 no_grad 和 classifier-only SGD 实现，未另行改写该机制。

{chr(10).join(details)}

`shift_bn=True` 保持 Backbone.train()；torch.no_grad 并不冻结 running_mean / running_var / num_batches_tracked。DR 只执行 BS 前向，H2T 先 BS 再 IS。默认 BN momentum=0.1 的均值累计分别为：

```text
DR:       μ_new = 0.9 μ_old + 0.1 μ_BS
DR+H2T:   μ_new = 0.81 μ_old + 0.09 μ_BS + 0.1 μ_IS
```

因此差异涉及统计量更新次数、输入分布和先后顺序。最终模型间 running_mean / running_var 的差异及逐层最大变化保存在 artifact_verification.json。这些 buffers 会参与 eval 模式推理，不能将 Stage II 的全部收益或 DR/H2T 差值都归因于 classifier 权重。本轮按要求保留此官方行为，没有统一或冻结 BN 统计量。

## 6. RNG 消耗与采样记录

两组初始 Python、NumPy、torch CPU/CUDA RNG 状态 hash 完全一致，后端设置一致。原 CIFAR 子集构造会将 NumPy seed 重设为 0，两组均经历此过程。种子控制通过独立运行器实现；未将 YAML deterministic 改为 True，也没有强制启用额外确定性算法。

H2T 的原 CPU torch.randperm(64) 每个训练 batch 消耗一次 CPU RNG；DR 没有该调用。数据增强和 DataLoader 也使用 CPU RNG，所以不能宣称全程逐 batch 图像完全相同。本轮没有增加虚拟 randperm、重放数据或隔离随机流。

- 实际比较了 {rn['batches_compared']} 对 batch 的指纹：BS 标签相同 {rn['same_bs_labels_batches']} 对，IS 标签顺序相同 {rn['same_is_labels_batches']} 对，batch 就绪时 CPU RNG 状态相同 {rn['same_cpu_rng_at_batch_ready_batches']} 对。
- 每个 epoch 的前 2 个 batch 共 {rn['image_batches_checked']} 对图像做 hash 检查：BS 图像相同 {rn['same_bs_images_in_checked_batches']} 对，IS 图像相同 {rn['same_is_images_in_checked_batches']} 对。这只是被检查的图像 batch 范围，没有对其余图像作“完全配对”声明。
- 首个 BS 图像 batch 相同：{rn['first_batch_bs_images_equal']}；第二个 BS 图像 batch 相同：{rn['second_batch_bs_images_equal']}。
- IS 每 epoch 的类别直方图均严格等于完整 10,847 张长尾子集计数；BS 每 epoch 的类别直方图和全部标签指纹分别记录。相同标签序列不等于相同的增强图像。

## 7. 与论文 Table 5 / README 的对比

论文 Table 5 给出的 CIFAR-100-LT IF=100 结果如下。[论文 Table 5](https://arxiv.org/html/2306.06963v3)

| 对比 | Overall | Head | Medium | Tail |
|---|---:|---:|---:|---:|
{chr(10).join(paper_rows)}

论文 Table 5 的 DR→BS+IS Overall 增益为 +2.05 pp。本轮最佳对比 {best_gain:+.3f} pp，固定最终对比 {final_gain:+.3f} pp；用户指定的 rho=0.25 为仓库 YAML 值，论文 §4.4 最佳比例为 0.3，两个设置不能视为完全相同的超参数复现。[论文 §4.4](https://arxiv.org/html/2306.06963v3)

官方 README 的 IF=100 Stage I=39.55%、Stage II CE+H2T=47.80%，没有提供对应 DR-only 结果。本轮 Stage I=40.49%（+0.94 pp），H2T 最佳 {hbest:.3f}%（相对 README {hbest-47.80:+.3f} pp）、最终 {hfinal:.3f}%（{hfinal-47.80:+.3f} pp）。作者说明仓库配置重跑略优于论文。[官方 README](https://github.com/Keke921/H2T#results-and-models)

## 8. 复现偏差与解释边界

1. 使用本机 Stage I epoch-198 最佳模型，不是作者发布的 39.55% 模型；上游表征不同，后续绝对准确率可能不同。两实验内部始终使用同一文件。
2. 按用户和原 YAML 使用 rho=0.25；论文最佳 0.3 对 64 通道替换 19 个，而当前替换 16 个。本轮没有据结果调参。
3. 运行环境为 Windows、{preflight['gpu']}、torch {preflight['torch']}、torchvision {preflight['torchvision']}、CUDA runtime {preflight['cuda_runtime']}、Python {preflight['python_version'].split()[0]}。原论文训练环境未由仓库完整锁定；不同库/硬件的数值路径可能导致偏差。
4. Stage II 原 YAML 没固定 seed，本轮按用户要求固定为 123；依然保留 deterministic=False。报告为一次原始流程重跑，不提供重复 seed 的均值/方差。
5. 相同种子不消除 H2T 对 CPU RNG 的额外消耗。BN 统计量也存在官方规定的单/双前向差异。这两点限制“纯融合机制”的因果归因。
6. 仍使用审计锁定的官方归档 Commit `e0c849235d9655fca31cd9fa8a5f38f92a77c7e8`，兼容 Diff 仅修复构造器、pathlib 导入、现代 torch.load 的运行问题。Commit 来自 ZIP comment，未以远程 Git 对象独立核验；逐文件 SHA256、运行快照和方法/采样/模型源文件均核对未变。

## 9. 保存位置、日志和命令

证据目录：`{folder}`。

"""
    for arm,title in [("dr","DR"),("h2t","DR+H2T")]:
        s = status[arm]
        report += f"""### {title}

- 开始 {s['started_at']}，结束 {s['finished_at']}；耗时 {s['elapsed_seconds']:.2f} 秒。
- PyTorch 峰值 allocated={s['cuda_peak_allocated_mib']:.2f} MiB、reserved={s['cuda_peak_reserved_mib']:.2f} MiB；GPU 全局采样占用峰值={s['gpu_global_peak_sampled_mib']:.0f} MiB（包含桌面程序，非本进程独占显存）。
- {link('原训练日志',s['official_log'])}、{link('完整 stdout/stderr',folder/arm/'stdout_stderr.log')}、{link('逐 epoch / BN / RNG 记录',folder/arm/'status.json')}、{link('epoch CSV',folder/arm/'epochs.csv')}。
- {link('完整配置',folder/arm/'effective.yaml')}、{link('原 YAML',folder/arm/'original.yaml')}、{link('运行源码快照',s['code_dir'])}。
- {link('最佳模型评估',folder/arm/'best_evaluation.json')}、{link('最终模型评估',folder/arm/'final_evaluation.json')}，含全部逐类正确计数；对应 stdout 日志位于相同证据目录。

实际运行命令（已执行完毕，展示追溯用途）：

```powershell
Set-Location -LiteralPath 'D:\\xiangmuxunlian\\新'
$env:PYTHONHASHSEED = '123'
& 'D:\\app\\anaconda\\envs\\pytorch\\python.exe' -u .\\H2T_repro_audit\\formal_stage2.py run --suite-id {folder.name} --arm {arm}
```

运行器调用原 train_stage2.py；完整原入口 argv 保存于 status.json。最佳模型保存副本及证据 hook 不更改训练计算。直接单独运行原入口而不设置所有 RNG 的种子，会失去本轮的相同随机种子设置。

"""
    report += f"""{link('环境 / 数据 / 输入预检',folder/'preflight.json')}、{link('参数 / BN / Checkpoint 核验',folder/'artifact_verification.json')}、{link('官方源码兼容 Diff',folder/'compatibility.diff')}。

全部 Checkpoint 的 SHA256、epoch 和参数/BN 比较记录在核验 JSON，输出证据逐文件 SHA256 记录在 delivery_manifest.json。Stage I 输入 hash 在训练前后均保持不变。外部运行、评估、报告脚本及引用数据各保存一份。

**任务完成后停止；等待下一步机制审计任务。**
"""
    destination = BASE.parent/"H2T_STAGE2_RESULTS.md"
    destination.write_text(report,encoding="utf-8")
    (folder/"H2T_STAGE2_RESULTS.md").write_text(report,encoding="utf-8")
    for filename in ["evaluate_stage2.py","summarize_stage2.py"]:
        shutil.copy2(BASE/filename,folder/filename)
    files = {p.relative_to(folder).as_posix():{"bytes":p.stat().st_size,"sha256":sha(p)}
             for p in folder.rglob("*") if p.is_file() and p.name!="delivery_manifest.json"}
    dump(folder/"delivery_manifest.json",{"suite_id":folder.name,"files":files,
        "checkpoint_hashes":{arm:status[arm]["checkpoints_sha256"] for arm in status}})
    targets = re.findall(r"\]\((D:/[^)]+)\)",report)
    assert all(Path(re.sub(r":\d+$","",target)).exists() for target in targets)
    print(f"REPORT COMPLETE: {destination}; final H2T gain {final_gain:+.3f} pp",flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action",choices=["verify","report"])
    parser.add_argument("--suite-id",required=True)
    args = parser.parse_args()
    folder = BASE/"stage2_runs"/args.suite_id
    if args.action=="verify":
        verify(folder)
    else:
        render(folder)
