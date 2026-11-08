"""CPU verification and original fixed-epoch analysis, only after all native runs end."""
import contextlib
import csv
import hashlib
import json
import runpy
import sys
from pathlib import Path

OUT=Path(__file__).resolve().parent
ROOT=OUT.parent
sys.path.insert(0,str(ROOT))
import ccsam_common as c


def read(path):return json.loads(path.read_text(encoding="utf-8"))
def write(path,payload):
    with path.open("x",encoding="utf-8",newline="\n") as stream:
        json.dump(payload,stream,ensure_ascii=False,indent=2,allow_nan=False)


def main():
    c.install_test_guard(c.DEFAULT_DATA_ROOT)
    end=read(OUT/"END.json");start=read(OUT/"START.json")
    assert end["state"]=="COMPLETED" and end["return_code"]==0
    assert end["actual_completed_runs"]==9 and end["actual_recorded_epochs"]==270
    for name,value in start["source_sha256"].items():assert c.sha256_file(ROOT/name)==value,name
    assert c.sha256_file(ROOT/"smoke_report.json")==start["smoke_report_sha256"]
    assert c.sha256_file(ROOT/"prepared/prepared_manifest.json")==start["prepared_manifest_sha256"]
    manifest=read(ROOT/"prepared/prepared_manifest.json")
    assert c.sha256_file(c.resolve_checkpoint(manifest["checkpoint_path"]))==c.EXPECTED_CLEAN_SHA256
    for name,record in manifest["prepared_array_actions"].items():
        assert c.sha256_file(ROOT/"prepared"/name)==record["sha256"],name
    for seed,record in manifest["sampling_plans"].items():
        assert c.sha256_file(ROOT/"prepared"/f"sampling_plan_seed{seed}.npy")==record["sha256"]
    outputs=["final_epoch_metrics.csv","paired_deltas.csv","summary.json","CCSAM_STAGE2_RESULTS.md"]
    assert not any((ROOT/name).exists() for name in outputs),"Refuse to overwrite analysis"
    with (OUT/"analysis.log").open("x",encoding="utf-8",newline="\n") as log:
        with contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
            runpy.run_path(str(ROOT/"analyze_ccsam.py"),run_name="__main__")
    summary=read(ROOT/"summary.json")
    guard=read(ROOT/"formal_test_guard.json")
    assert guard["official_test_file_open_attempts"]==0 and guard["CIFAR100_train_false_attempts"]==0
    observed_epochs=0;observed_steps=0;observed_samples=0;run_records=[]
    for seed in c.SEEDS:
        for arm in c.ARMS:
            folder=ROOT/"results"/f"seed_{seed}"/arm
            status=read(folder/"status.json")
            with (folder/"metrics.csv").open(encoding="utf-8",newline="") as stream:metrics=list(csv.DictReader(stream))
            with (folder/"batch_trace.csv").open(encoding="utf-8",newline="") as stream:trace=list(csv.DictReader(stream))
            assert status["state"]=="completed" and len(status["epochs"])==len(metrics)==30
            assert [int(row["epoch"]) for row in metrics]==list(range(1,31))
            assert len(trace)==5100
            assert all(epoch["frozen_backbone_and_bn_bitwise_unchanged"] for epoch in status["epochs"])
            for index,row in enumerate(metrics):
                assert int(row["steps"])==170 and int(row["samples"])==10847
                assert float(row["alpha"])==min((index+1)/30,0.3) and float(row["learning_rate"])==0.01
                if arm!="C1_CCSAM_FEATURE_GRAD":assert float(row["mean_delta_l2_per_sample"])==0
                observed_steps+=int(row["steps"]);observed_samples+=int(row["samples"])
                rows=trace[index*170:(index+1)*170]
                assert sum(int(r["batch_size"]) for r in rows)==10847 and int(rows[-1]["batch_size"])==31
                assert [int(r["batch"]) for r in rows]==list(range(170))
            observed_epochs+=len(metrics)
            assert c.sha256_file(folder/"final_classifier.pth")==status["final_classifier_sha256"]
            run_records.append(dict(seed=seed,arm=arm,epochs=len(metrics),trace_rows=len(trace),
                final_classifier_sha256=status["final_classifier_sha256"],final_validation=status["final_validation"]))
    assert (observed_epochs,observed_steps,observed_samples)==(270,45900,2928690)
    write(OUT/"VERIFICATION.json",dict(status="PASS",runs=run_records,actual_epochs=observed_epochs,
        actual_optimizer_steps=observed_steps,actual_sample_occurrences=observed_samples,
        paired_trace_checks=summary["batch_and_initialization_pairing_verification"],
        checkpoint_and_prepared_unchanged=True,source_unchanged=True,formal_test_guard=guard,
        analysis_test_guard=c.guard_pass(),smoke_rerun=False,training_rerun=False))
    lines=["# CC-SAM Stage II 正式执行完成记录","","九组正式实验全部完成，原生命令退出码 0；固定末轮 epoch=30。","",
        "| Seed | 条件 | Overall (%) | Few (%) | Tail (%) |","| --- | --- | ---: | ---: | ---: |"]
    for item in run_records:
        m=item["final_validation"]
        lines.append(f"| {item['seed']} | {item['arm']} | {m['Overall']:.6f} | {m['Few']:.6f} | {m['Tail']:.6f} |")
    lines += ["","## 配对效应（百分点，3 个 seed 的均值）","",
        "| 比较 | Overall | Few | Tail |","| --- | ---: | ---: | ---: |"]
    for name,metrics in summary["paired_delta_summary"].items():
        lines.append(f"| {name} | {metrics['Overall']['mean_delta_pp']:+.6f} | {metrics['Few']['mean_delta_pp']:+.6f} | {metrics['Tail']['mean_delta_pp']:+.6f} |")
    lines += ["","实际完成 270 个 Epoch、45,900 次 optimizer.step、2,928,690 次样本使用；每组 5,100 个批次轨迹，C1/C2 与同 seed C0 的索引、增强图像、GAP 特征和初始化均精确配对。",
        "","官方 TEST 打开/实例化尝试均为 0；原始 checkpoint、prepared、采样计划和生产源码哈希不变。原 Smoke 未重跑，未扩展任何种子、轮数或机制。",
        "","主比较为 C1−C2，隔离特征扰动的增量效应；C2−C0 为类别加权和 alpha 混合的效应。数值来自原分析程序的固定末轮评价，未选择最佳 validation epoch。",
        "","三个 seed 共享一个 Clean Stage I backbone；1980 张 validation 来自官方 TRAIN 且不含 class 0，并非独立 TEST。三种子结果不构成独立表征或跨数据集推广证据，不宣称统计显著性。",
        "","完整机器结果见 ../summary.json、../final_epoch_metrics.csv、../paired_deltas.csv；原分析报告为 ../CCSAM_STAGE2_RESULTS.md。执行证据见 START.json、END.json、formal.log、VERIFICATION.json；所有本地 checkpoint/批次轨迹 SHA256 见 ARTIFACT_SHA256.csv。",""]
    with (OUT/"FORMAL_EXECUTION_REPORT.md").open("x",encoding="utf-8",newline="\n") as stream:stream.write("\n".join(lines))
    paths=[p for p in (ROOT/"results").rglob("*") if p.is_file()]
    paths.extend(p for p in OUT.iterdir() if p.is_file() and p.name!="ARTIFACT_SHA256.csv")
    paths.extend(ROOT/name for name in outputs+["formal_test_guard.json","formal_completed.json"])
    with (OUT/"ARTIFACT_SHA256.csv").open("x",encoding="utf-8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=["path","bytes","sha256"],lineterminator="\n");writer.writeheader()
        for path in sorted(paths):writer.writerow(dict(path=path.relative_to(ROOT).as_posix(),bytes=path.stat().st_size,sha256=c.sha256_file(path)))
    print(json.dumps(dict(status="ALL_NINE_RUNS_AND_ANALYSIS_VERIFIED",actual_epochs=observed_epochs,
        actual_optimizer_steps=observed_steps,actual_sample_occurrences=observed_samples,test_guard=guard)),flush=True)


if __name__=="__main__":main()
