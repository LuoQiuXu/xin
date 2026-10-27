"""Read completed outputs, validate paired invariants, write scientific reports.

No training, model forward or method selection. Only this run's completed,
digest-verified checkpoint is deserialized on CPU for provenance checks.
"""
from __future__ import annotations
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
RUN=HERE/"runs/pih2t_if100_seed123_42_v1"
CHECKS=HERE/"runs/pre_run_pih2t_if100_seed123_42_v1"


def read(path):return json.loads(path.read_text(encoding="utf-8-sig"))
def sha(path):
    with path.open("rb") as handle:return hashlib.file_digest(handle,"sha256").hexdigest()
def write_new(name,text):
    with (RUN/name).open("x",encoding="utf-8",newline="\n") as handle:handle.write(text.rstrip()+"\n")
def json_new(name,value):write_new(name,json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False))
def csv_rows(path):
    with path.open(encoding="utf-8-sig",newline="") as handle:return list(csv.DictReader(handle))
def git(*args):
    result=subprocess.run(["git",*args],cwd=REPO,capture_output=True,text=True,encoding="utf-8")
    if result.returncode:raise RuntimeError(result.stderr)
    return result.stdout
def metric(pred,labels,low,high):
    mask=(labels>=low)&(labels<high)
    return 100.*int((pred[mask]==labels[mask]).sum())/int(mask.sum())


def verify():
    from run_reproduction import stage1_lr,cosine_lr,source_hashes,verify_checkpoint_digest
    if git("branch","--show-current").strip()!="repro/pi-h2t-cifar100-paper-parity-20261027":
        raise RuntimeError("Wrong branch")
    plan=read(RUN/"protocol.json")
    if plan["source_hashes"]!=source_hashes():raise RuntimeError("Execution source drift")
    gate=read(CHECKS/"TRAINING_GATE.json")
    for name,value in gate["source_hashes"].items():
        if sha(HERE/name)!=value:raise RuntimeError("Pre-TEST protocol/source changed")
    for command in ["stage1","stage2_no_fusion","stage2_h2tf","report"]:
        if read(CHECKS/f"{command}_exit.json")["exit_code"]!=0:raise RuntimeError(f"Incomplete command: {command}")
    for command in ["stage1","stage2_no_fusion","stage2_h2tf"]:
        if not read(CHECKS/f"{command}_completion_gate.json")["passed"]:
            raise RuntimeError(f"Completion integrity gate failed: {command}")
    if not read(CHECKS/"POST_RUN_STATIC_CHECKS.json")["passed"]:
        raise RuntimeError("Post-run static checks failed")
    history=csv_rows(RUN/"stage1_epochs.csv")
    if len(history)!=200:raise RuntimeError("StageI is not 200 epochs")
    for i,row in enumerate(history):
        if int(row["epoch"])!=i+1 or abs(float(row["lr"])-stage1_lr(i))>1e-15:
            raise RuntimeError("StageI scheduler/count mismatch")
        if not all(math.isfinite(float(row[k])) for k in ["train_loss","pif_a_residual","pif_b_feature"]):
            raise RuntimeError("Nonfinite StageI history")
    source_digest=verify_checkpoint_digest(RUN)
    checkpoint=torch.load(RUN/"stage1_final_epoch200.pth",map_location="cpu",weights_only=False)
    if checkpoint["run_id"]!=RUN.name or checkpoint["epoch"]!=200:raise RuntimeError("Wrong checkpoint lineage")
    coefficients=checkpoint["network"]["pif.fuse.weight"]
    if tuple(coefficients.shape)!=(1,2,1) or coefficients.numel()!=2:raise RuntimeError("Not 2-parameter PIF")
    stage1=read(RUN/"stage1_summary.json")
    if coefficients.flatten().tolist()!=stage1["pif_coefficients_final_residual_feature"]:
        raise RuntimeError("Checkpoint/summary PIF coefficients differ")
    with np.load(RUN/"stage1_epoch200_predictions.npz",allow_pickle=False) as z:
        stage1_labels=z["labels"].copy();stage1_pred=z["pred"].copy();stage1_logits=z["logits"].copy()
    if stage1_labels.shape!=(10000,) or stage1_logits.shape!=(10000,100) or not np.isfinite(stage1_logits).all():
        raise RuntimeError("Invalid StageI TEST output shape/values")
    if not np.array_equal(np.bincount(stage1_labels,minlength=100),np.full(100,100)) or not np.array_equal(stage1_pred,stage1_logits.argmax(1)):
        raise RuntimeError("Invalid StageI prediction/class counts")
    for name,a,b in [("Overall",0,100),("Many",0,36),("Medium",36,71),("Few",71,100)]:
        if abs(metric(stage1_pred,stage1_labels,a,b)-stage1["official_test_diagnostic"][name])>1e-10:
            raise RuntimeError("StageI TEST metric recomputation mismatch")
    actual={}; predictions={}; ratio_facts={}; histories={}
    for arm in ["no_fusion","h2tf"]:
        base=RUN/"stage2"/arm;info=read(base/"summary.json");rows=csv_rows(base/"train_epochs.csv")
        if len(rows)!=10 or info["source_stage1_sha256"]!=source_digest:raise RuntimeError("StageII budget/source mismatch")
        if info["initial_backbone_parameters_hash"]!=info["final_backbone_parameters_hash"]:
            raise RuntimeError("Backbone parameters changed")
        if info["pif_coefficients"]!=stage1["pif_coefficients_final_residual_feature"]:
            raise RuntimeError("StageII PIF coefficients differ from StageI")
        if sha(base/"classifier_final_epoch10.pth")!=info["stage2_final_classifier_sha256"]:
            raise RuntimeError("StageII checkpoint digest mismatch")
        all_r=[];all_used=[]
        for i,row in enumerate(rows):
            if int(row["epoch"])!=i+1 or abs(float(row["lr"])-.2*cosine_lr(.1,i,10))>1e-15:
                raise RuntimeError("StageII schedule mismatch")
            if not all(math.isfinite(float(row[k])) for k in ["train_loss","candidate_donor_mass_mean","used_donor_mass_mean"]):
                raise RuntimeError("Nonfinite StageII history")
            with np.load(base/f"train_ratios_epoch{i+1:02d}.npz",allow_pickle=False) as z:
                r=z["r"].copy();used=z["donor_mass_used"].copy();y=z["receiver_labels"].copy()
                if len(r)!=10847 or len(y)!=len(r) or not np.isfinite(r).all() or not ((r>=0)&(r<=1)).all():
                    raise RuntimeError("Invalid full ratio records")
                wanted=1-r if arm=="h2tf" else np.zeros_like(r)
                if not np.array_equal(used,wanted):raise RuntimeError("Ratio polarity/arm mismatch")
                all_r.append(r);all_used.append(used)
        rr=np.concatenate(all_r).astype(np.float64);uu=np.concatenate(all_used).astype(np.float64)
        ratio_facts[arm]={"count":len(rr),"r_mean":float(rr.mean()),"candidate_donor_mean":float((1-rr).mean()),
                          "used_donor_mean":float(uu.mean()),"used_donor_std":float(uu.std()),
                          "used_donor_min":float(uu.min()),"used_donor_max":float(uu.max()),
                          "used_donor_quantiles_0_10_25_50_75_90_100":np.quantile(uu,[0,.1,.25,.5,.75,.9,1]).tolist()}
        actual[arm]=info;histories[arm]=rows
        with np.load(base/"test_epoch10_predictions.npz",allow_pickle=False) as z:
            y=z["labels"].copy();pred=z["pred"].copy();logits=z["logits"].copy()
        if y.shape!=(10000,) or logits.shape!=(10000,100) or not np.isfinite(logits).all():
            raise RuntimeError("Invalid TEST output shape/values")
        if not np.array_equal(np.bincount(y,minlength=100),np.full(100,100)) or not np.array_equal(pred,logits.argmax(1)):
            raise RuntimeError("Prediction ordering/class counts mismatch")
        if not np.array_equal(y,stage1_labels):raise RuntimeError("StageI/StageII TEST label order mismatch")
        expected={name:metric(pred,y,a,b) for name,a,b in [("Overall",0,100),("Many",0,36),("Medium",36,71),("Few",71,100)]}
        if any(abs(expected[k]-info["metrics"][k])>1e-10 for k in expected):raise RuntimeError("Metric re-computation mismatch")
        predictions[arm]=(pred,y)
    left,right=actual["no_fusion"],actual["h2tf"]
    if left["stage2_initial_classifier_hash"]!=right["stage2_initial_classifier_hash"] or left["final_bn_buffers_hash"]!=right["final_bn_buffers_hash"]:
        raise RuntimeError("Paired classifier/BN identity failed")
    if any(a["input_stream_sha256"]!=b["input_stream_sha256"] for a,b in zip(histories["no_fusion"],histories["h2tf"])):
        raise RuntimeError("Paired training streams differ")
    p0,y0=predictions["no_fusion"];p1,y1=predictions["h2tf"]
    if not np.array_equal(y0,y1):raise RuntimeError("TEST label order mismatch")
    classes=csv_rows(RUN/"per_class_paired_changes.csv")
    if len(classes)!=100:raise RuntimeError("Per-class rows missing")
    for klass,row in enumerate(classes):
        m=y0==klass;wins=int(((p1[m]==klass)&(p0[m]!=klass)).sum());losses=int(((p1[m]!=klass)&(p0[m]==klass)).sum())
        if int(row["class"])!=klass or int(row["h2tf_wins"])!=wins or int(row["h2tf_losses"])!=losses:
            raise RuntimeError("Per-class transition mismatch")
    baseline=read(CHECKS/"historical_protection_before.json")
    for item in baseline["tracked_text"]:
        if sha(REPO/item["path"])!=item["sha256"]:raise RuntimeError(f"Historical source changed: {item['path']}")
    prior={r["path"]:r for r in baseline["historical_metadata"]};current={}
    for name in ["fusion_semantic_audit","lowvs_mask_parity","mdlt_domain_probe"]:
        for p in (REPO/"H2T_repro_audit"/name).rglob("*"):
            if p.is_file():
                key=p.relative_to(REPO).as_posix();current[key]={"path":key,"bytes":p.stat().st_size,"mtime_ns":p.stat().st_mtime_ns}
    if prior!=current:raise RuntimeError("Historical untracked assets metadata changed")
    git("diff","--check")
    validation={"passed":True,"StageI_epochs":200,"StageII_epochs_per_arm":10,"PIF_parameters":2,
                "source_stage1_sha256":source_digest,"same_checkpoint":True,"same_initial_classifier":True,
                "same_final_BN_buffers":True,"same_training_stream_each_epoch":True,"backbone_parameters_frozen":True,
                "official_TEST_shape":[10000,100],"TEST_class_counts":[100]*100,"TEST_order_matched":True,
                "TEST_metrics_recomputed":True,"per_class_transitions_recomputed":True,
                "protected_text_hashes_unchanged":len(baseline["tracked_text"]),
                "protected_untracked_metadata_unchanged":len(prior),"source_hashes":source_hashes(),
                "parity":"PAPER_ALIGNED_INDEPENDENT_RECONSTRUCTION_NOT_AUTHOR_VERIFIED"}
    json_new("FINAL_INTEGRITY.json",validation)
    json_new("RATIO_DISTRIBUTIONS.json",ratio_facts)
    return stage1,actual,ratio_facts,classes,validation


def reports(stage1,arms,ratios,classes,validation):
    comparison=read(RUN/"comparison.json");plan=read(RUN/"protocol.json")
    data=read(CHECKS/"data_preflight.json");smoke=read(CHECKS/"smoke.json")
    env=read(RUN/"environment_stage1.json")
    initial=stage1["pif_coefficients_initial_residual_feature"]
    final=stage1["pif_coefficients_final_residual_feature"]
    transitions={"improved_classes":sum(int(r["delta_pp"])>0 for r in classes),
                 "worsened_classes":sum(int(r["delta_pp"])<0 for r in classes),
                 "unchanged_classes":sum(int(r["delta_pp"])==0 for r in classes),
                 "image_wins":sum(int(r["h2tf_wins"]) for r in classes),
                 "image_losses":sum(int(r["h2tf_losses"]) for r in classes)}
    json_new("CLASS_TRANSITION_SUMMARY.json",transitions)
    table=["|固定终点|Overall %|Many %|Medium %|Few %|实际平均donor mass|",
           "|---|---:|---:|---:|---:|---:|"]
    metrics=stage1["official_test_diagnostic"]
    table.append("|Stage I epoch200|"+"|".join(f"{metrics[k]:.2f}" for k in ["Overall","Many","Medium","Few"])+"|无融合|")
    for arm in ["no_fusion","h2tf"]:
        m=arms[arm]["metrics"]
        table.append(f"|Stage II {arm} epoch10|"+"|".join(f"{m[k]:.2f}" for k in ["Overall","Many","Medium","Few"])+f"|{ratios[arm]['used_donor_mean']:.8f}|")
    delta=comparison["h2tf_minus_no_fusion_pp"]
    table.append("|H2TF − matched no_fusion（百分点）|"+"|".join(f"{delta[k]:+.2f}" for k in ["Overall","Many","Medium","Few"])+"|—|")
    if delta["Overall"]>0:interpretation="本轮H2TF相对匹配no_fusion有局部Overall增益"
    elif delta["Overall"]<0:interpretation="本轮H2TF相对匹配no_fusion出现局部Overall退化"
    else:interpretation="本轮两组Overall相同"
    scientific=f"""# PI-H2T 可信复现科学结果

日期：2026-10-27；run-id：`{RUN.name}`。状态：**200 + 10 + 10 epochs已实际完成；PAPER_ALIGNED_INDEPENDENT_RECONSTRUCTION_NOT_AUTHOR_VERIFIED**。

## 固定终点指标

{chr(10).join(table)}

所有数值来自本run真实官方CIFAR100 TEST预测，不从历史报告抄入；以保存logits的argmax重新核算。每类100图，Many/Medium/Few为[0,36)/[36,71)/[71,100)。StageI只选固定epoch200，两组StageII只选固定epoch10，没有逐epoch TEST挑best或按准确率更改机制。

{interpretation}，差值为{delta['Overall']:+.2f}个百分点。这只适用于本checkpoint/这对seed及预先锁定的配置，不等于总体统计显著性、作者全流程精度匹配、独立测试集确认或创新模块收益。若存在退化，不自动归因论文错误；若有收益，也不据此填补尚未核实的作者配置。

相对StageI固定终点，no_fusion Overall变化={arms['no_fusion']['metrics']['Overall']-metrics['Overall']:+.2f}个百分点，H2TF变化={arms['h2tf']['metrics']['Overall']-metrics['Overall']:+.2f}个百分点。StageI与StageII还同时改变了采样、优化目标输入和BN训练流，不能把这一跨stage差值视作只改变一个因素的归因结果。

## PIF与融合比例

PIF仅2个无bias参数，顺序[a_residual,b_feature]：初始={initial}；epoch200={final}；有效F系数a+b={sum(final):.10g}。完整200轮轨迹在stage1_epochs.csv，两组StageII均保持这套骨干/PIF参数哈希不变。

H2TF的训练接收者权重r均值={ratios['h2tf']['r_mean']:.8f}，实际供体权重1-r均值={ratios['h2tf']['used_donor_mean']:.8f}，std={ratios['h2tf']['used_donor_std']:.8f}，min/max={ratios['h2tf']['used_donor_min']:.8f}/{ratios['h2tf']['used_donor_max']:.8f}。分位数见RATIO_DISTRIBUTIONS.json。这里汇总全部10轮共{ratios['h2tf']['count']}个训练样本出现次数；no_fusion实际donor mass为0，其candidate ratio只用于相同计算/日志，不参与混合。

每轮完整r、实际使用的donor mass、receiver/donor标签保存在stage2各组train_ratios_epochXX.npz；二进制留本地，SHA/大小见RESULTS_INDEX.json。未交换r与1-r，也未把论文约.65的接收者r作为待拟合目标。

## 逐类/逐图变化

H2TF净改善类别{transitions['improved_classes']}，净下降{transitions['worsened_classes']}，不变{transitions['unchanged_classes']}。在完全相同TEST标签顺序上，no_fusion错→H2TF对{transitions['image_wins']}张，no_fusion对→H2TF错{transitions['image_losses']}张。每类计数/组别/胜负在per_class_paired_changes.csv；每类100图，因此该类净正确数差恰为百分点差。

两组checkpoint源SHA、初始FC、全部10轮真实图像/标签流SHA和最终BN buffers严格相同；骨干/PIF参数未更新。差异是StageII采用接收者向量还是Eq7融合向量。no_fusion也前向donor是预先声明的matched BN traffic控制，不能把它称作作者未改动的原DR实现。

本配对运行可支持“启用这套已锁定H2TF算子后的条件变化”判断；供体权重分布、Many/Medium/Few差值是可直接复核的事实。它不能单独证明某个比例定义、PIF初始化、冻结reference W、BN update或作者checkpoint选择是论文精度差距的根因。作者配置未定项与复现误差来源仍未解决，不能由一次总体下降反推论文错误，也没有据此再搜索配置。

## 与作者论文分别陈列

PI-H2T arXiv2506.00625v2（PDF p10 Table6、p12 Fig9）给CIFAR100-LT IF100/ResNet32/CE：PIF StageI42.19%，StageII best49.22%、last49.04%。这些是作者报告，不是本run指标；作者PIF精确初始化、Eq8 W来源/detach、BN buffers和checkpoint选择没有完全恢复。本run固定StageI epoch200，与作者可能使用的best路径不同，因此不以数值相等作为唯一parity门禁，不将两套协议混合汇总。

历史25参数PIF实验未用于本次模型初始化、选择、对照或指标合并。新checkpoint为本run内stage1_final_epoch200.pth，SHA256 `{validation['source_stage1_sha256']}`。

## 科学边界

核心Eq3/6/7/8的字面实现已核实；完整作者程序parity仍未建立。本轮是一个StageI seed123与StageII seed42的干净初步复现，没有追加seed、第三臂、参数搜索、TailOff、MarginProtect或新注意力机制。仅就本次已锁定重建报告现象，不据此宣称发表方法缺陷或新颖研究结论。
"""
    write_new("SCIENTIFIC_RESULTS.md",scientific)
    parity=f"""# 论文一致性与实际运行报告

run-id：{RUN.name}；目标分支：repro/pi-h2t-cifar100-paper-parity-20261027。

## 已核实/继承/独立重建

训练前PARITY_LOCK.md在新模型TEST准确率产生前固化，根锁文件SHA={sha(HERE/'PARITY_LOCK.md')}，其SHA纳入protocol.json和新checkpoint的source_hashes。TRAINING_GATE.json记录全部门禁通过、当时未取得新模型TEST准确率；正式训练执行源码与锁定哈希保持一致。

VERIFIED_FROM_PI_PAPER：PIF沿通道取均值、[F-mean,F]单个bias-free Conv1d(2,1,1)两参数、保留PIF后GAP；H2TF为r*f_B+(1-r)*f_I，r=(1-cos)/2；BS接收/IS供体；接收者hard CE；推理不做H2TF。输入输出图为[B,64,8,8]，GAP后[B,64]，分类logits[B,100]，由冻结ResNet32结构、CPU测试与真实CIFAR smoke共同核查。

INHERITED_FROM_H2T：原始冻结模型/dataset逐字未改；Option-A ResNet32+原带bias FC，官方TRAIN内exp IF100/seed0选样；crop/flip/Normalize；StageI CE+MixUp(alpha1)、200轮/batch128/SGD/wd.0002、五轮warmup和160/180衰减；StageII classifier-only10轮/.02初始cosine/wd.0005，默认BN update。具体源SHA见protocol.json。

RECONSTRUCTION_ASSUMPTION：PIF用Torch默认Conv1d初始化；r对W不反传；W为冻结StageI head；cuDNN deterministic=True、benchmark=False；fixed epoch200而非TEST-selected best；no_fusion也前向donor，匹配BN流。作者这些精确细节保持UNVERIFIED。StageI MixUp公式与分布沿用，但本地randperm在CUDA产生、模型初始化含新PIF；不宣称逐位重放原H2T随机流或作者seed轨迹。上述执行方式在源哈希中训练前已锁定，未因结果改变。

原作者库本轮只读扫描main=`e0c849235d9655fca31cd9fa8a5f38f92a77c7e8`；URL https://github.com/Keke921/H2T 。分支身份不证明当前代码是完整官方PI发布。本轮始终标注独立重建，不以准确率接近恢复未定作者语义。

## 实际运行与checkpoint

StageI从零训练200轮，未载入旧模型。PIF新权重shape[1,2,1]、恰两参数；保存epoch200后才评估TEST。两组StageII均先核对checkpoint与StageI完成summary的SHA，再strict load，并检查run-id、protocol、subset/source hashes，拒绝历史25参数结构和漂移来源。

两组初始FC与源checkpoint一致；每轮图像/标签SHA一致，最终BN buffers SHA一致，训练前后骨干/PIF参数SHA一致。配对对照的有效性已通过，详见FINAL_INTEGRITY.json。StageII预测使用同一有序官方TEST集，100图/类；逐类变化从同序预测直接计算。

## 最终判断

可确认：本轮严格执行了锁定的两参数PIF独立重建，200/10/10预算、数据真实性、配对输入/BN流、冻结边界与模型来源均通过核验。不可确认：作者初始化、reference/detach、BN及checkpoint选择全部相同，或结果具有多seed显著性。结论是可信的**paper-aligned independent reconstruction**，不是“完全官方PI-H2T已复现”。
"""
    write_new("PARITY_RUN_REPORT.md",parity)
    validation_report=f"""# 运行验证报告

日期2026-10-27；run-id {RUN.name}。开始HEAD：5bdbe5c8749cafb613215e157e7b3290f70626d6；分支repro/pi-h2t-cifar100-paper-parity-20261027。允许保留的三个历史未跟踪目录未纳入本轮提交。

## 测试和退出码

初次CPU测试9项，exit1（1 failure/1 error）：夹具权重shape错误及float32准确率中间舍入；只在本study修复，保留失败记录，新增回归。最终14项CPU测试exit0；4文件py_compile exit0；preflight exit0；真实CIFAR GPU smoke exit0；StageI、两组StageII及report四条正式命令退出码均0。完整输出与检查事实在相邻pre_run目录TEST_AND_PREFLIGHT_LOG.json、各exit.json及本地.log中。修复理由见根REPAIR_LOG.md，执行源码在正式StageI后未改。

解释器：{env['executable']}；Python {env['python']}；torch {env['torch']}，torchvision {env['torchvision']}，CUDA build {env['cuda_build']}，cuDNN {env['cudnn']}。GPU：{env['gpu']}，显存{env['total_memory_bytes']}字节，compute capability {env['compute_capability']}。实际cuDNN deterministic={env['cudnn_deterministic']}、benchmark={env['cudnn_benchmark']}、allow_tf32={env['cudnn_allow_tf32']}；matmul allow_tf32={env['cuda_matmul_allow_tf32']}，precision={env['float32_matmul_precision']}；AMP={env['AMP']}。preflight进程在seeding前的deterministic可能为False，正式环境与smoke seeding后为True，分别保留记录。

真实smoke为官方TRAIN中的batch128，StageI MixUp+PIF梯度与StageII H2TF/冻结骨干/FC梯度均通过，PIF grad abs sum={smoke['stage1']['PIF_gradient_abs_sum']:.8g}，记录峰值CUDA分配{smoke['stage2']['peak_cuda_allocated_mb']:.2f}MiB。一次float(loss)日志打印有requires_grad Tensor转scalar提示，未改变梯度/公式或触发OOM。未降batch、未开启AMP、未调模型/损失/比例。

## 数据真实性

现有官方缓存：{data['data_root']}，未下载。torchvision官方MD5检查、所有100类IF100计数、独立RandomState(0)类内索引重建与原dataset数据/标签逐值相同。TRAIN LT={data['train_length']}，TEST={data['test_length']}、100图/类。LT像素+标签SHA256 `{data['lt_data_and_labels_sha256']}`；选定TRAIN索引SHA256 `{data['selected_train_indices_sha256']}`。train/test/meta原文件SHA与大小见pre_run/data_preflight.json，训练前、StageII前及评估前复核。官方TEST已按授权用于数据预检和三个固定模型终点的准确率汇报，未进入训练/选样、未据其调参或选checkpoint；它也不是一个未曾被本项目使用的独立最终测试集。

## 来源、二进制与完整性

StageI新checkpoint SHA256 `{validation['source_stage1_sha256']}`，具体字节大小、所有二进制相对路径/大小/SHA列于RESULTS_INDEX.json。StageII初始FC/最终BN/冻结骨干参数及流SHA在各summary.json与train_epochs.csv。20份完整训练比例NPZ和三个终点logits/labels/pred NPZ保留本地，不推送；报告、CSV/JSON与脚本显式暂存，不使用git add -A，不PR/merge或force push。

所有文件SHA均针对运行时本地原始字节；本仓库core.autocrlf=True，Git可能对文本做LF/CRLF转换，文本SHA不是Git blob标识。二进制SHA不受行尾转换影响。跨系统复查源哈希时须同时核对运行时行尾和编码，不得把文本规范化差异当作模型数值重放通过。

最终核验重算了所有epoch/LR/有限性、checkpoint/PIF二参数、10轮真实配对输入SHA、预测logits argmax、标签顺序、每类100图、全部分组准确率和逐类胜负；通过。{validation['protected_text_hashes_unchanged']}份历史已跟踪文本SHA不变，三个未跟踪历史目录{validation['protected_untracked_metadata_unchanged']}个文件大小/mtime不变；没有把后者说成全内容哈希。原始H2T/历史H2T/PI review与旧结果Git diff为空。最终git diff --check及暂存范围另核对后提交。

## 精确命令

所有Python命令均以`D:/app/minicondaaa/envs/xiaoju/python.exe`执行；训练用-B -u -X utf8，以下entry均为H2T_repro_audit/pi_h2t_strict_20261027/run_reproduction.py。

```text
-m unittest discover -s H2T_repro_audit/pi_h2t_strict_20261027 -p test_*.py -v
-m py_compile paper_components.py run_reproduction.py test_paper_components.py test_run_reproduction.py
preflight --run-id {RUN.name} --data-root D:/xiangmuxunlian/xin/H2T/data/cifar100
smoke --run-id {RUN.name} --data-root D:/xiangmuxunlian/xin/H2T/data/cifar100
stage1 --run-id {RUN.name} --data-root D:/xiangmuxunlian/xin/H2T/data/cifar100
stage2 --run-id {RUN.name} --data-root D:/xiangmuxunlian/xin/H2T/data/cifar100 --arm no_fusion
stage2 --run-id {RUN.name} --data-root D:/xiangmuxunlian/xin/H2T/data/cifar100 --arm h2tf
report --run-id {RUN.name}
```

上面py_compile的4个文件实际均带study前缀，从仓库根运行。科学/作者parity限制见PARITY_RUN_REPORT与SCIENTIFIC_RESULTS，不将测试通过等同于全部作者细节已核实。
"""
    write_new("VALIDATION_REPORT.md",validation_report)
    artifacts=[]
    for base in [RUN,CHECKS]:
        for path in sorted(base.rglob("*")):
            if path.is_file():
                local_only=path.suffix.lower() in {".pth",".npz",".log"} or path.name=="historical_protection_before.json"
                artifacts.append({"path":path.relative_to(HERE).as_posix(),"bytes":path.stat().st_size,
                                  "sha256":sha(path),"publish":not local_only})
    index={"run_id":RUN.name,"status":"COMPLETED_INDEPENDENT_RECONSTRUCTION","budgets":[200,10,10],
           "stage1_metrics":stage1["official_test_diagnostic"],"stage2_metrics":{k:v["metrics"] for k,v in arms.items()},
           "h2tf_minus_no_fusion_pp":delta,"PIF_initial":initial,"PIF_final":final,
           "ratios":ratios,"per_class_changes":transitions,"artifacts":artifacts,
           "self_hash_excluded":True,"large_binaries":"kept local; every file has bytes/SHA256 above",
           "hash_scope":"raw local runtime bytes; text hashes may differ from Git-normalized LF blobs (core.autocrlf=True)",
           "report_generator":{"path":"finalize_results.py","bytes":(HERE/"finalize_results.py").stat().st_size,"sha256":sha(HERE/"finalize_results.py")},
           "parity":validation["parity"]}
    json_new("RESULTS_INDEX.json",index)
    print(json.dumps({"metrics":index["stage2_metrics"],"delta_pp":delta,"PIF_final":final,
                      "ratios":ratios,"transitions":transitions,"reports_written":True},ensure_ascii=False,indent=2))


if __name__=="__main__":reports(*verify())
