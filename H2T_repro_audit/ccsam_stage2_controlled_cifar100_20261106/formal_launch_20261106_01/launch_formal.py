"""Launch the unchanged native formal command once and preserve its real execution state."""
import csv
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

OUT=Path(__file__).resolve().parent
ROOT=OUT.parent
REPO=ROOT.parents[1]
PYTHON=r"D:\app\minicondaaa\envs\xiaoju\python.exe"
SEEDS=(123,124,125)
ARMS=("C0_BALANCED_CE","C1_CCSAM_FEATURE_GRAD","C2_WEIGHTED_CE_NO_PERTURB")


def now():return datetime.now().astimezone().isoformat()
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert str(Path(sys.executable).resolve()).lower()==str(Path(PYTHON).resolve()).lower()
    assert not (ROOT/"results").exists() and not (ROOT/"formal_test_guard.json").exists()
    assert not (OUT/"END.json").exists()
    argv=[PYTHON,"-B","-X","utf8","-u",str(ROOT/"train_ccsam.py"),"--formal"]
    started=now()
    source_hashes={name:sha(ROOT/name) for name in ("train_ccsam.py","ccsam_common.py","experiment_protocol.json")}
    record=dict(started=started,argv=argv,cwd=str(REPO),parent_pid=os.getpid(),
        branch=subprocess.check_output(["git","branch","--show-current"],cwd=REPO,text=True).strip(),
        head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=REPO,text=True).strip(),
        source_sha256=source_hashes,smoke_report_sha256=sha(ROOT/"smoke_report.json"),
        prepared_manifest_sha256=sha(ROOT/"prepared/prepared_manifest.json"),
        budget=dict(seeds=list(SEEDS),arms=list(ARMS),epochs_per_run=30),
        smoke_rerun=False,automatic_retry=False)
    with (OUT/"formal.log").open("x",encoding="utf-8",newline="\n") as log, \
            (OUT/"START.json").open("x",encoding="utf-8",newline="\n") as start_file:
        process=subprocess.Popen(argv,cwd=REPO,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                 text=True,encoding="utf-8",errors="backslashreplace")
        record["training_pid"]=process.pid
        json.dump(record,start_file,ensure_ascii=False,indent=2)
        start_file.flush()
        print(json.dumps(dict(state="NATIVE_FORMAL_PROCESS_STARTED",pid=process.pid,started=started,log=str(OUT/"formal.log"))),flush=True)
        for line in process.stdout:
            log.write(line);log.flush()
            print(line,end="",flush=True)
        return_code=process.wait()
    runs=[]
    for seed in SEEDS:
        for arm in ARMS:
            folder=ROOT/"results"/f"seed_{seed}"/arm
            path=folder/"status.json"
            status=json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
            metrics=folder/"metrics.csv"
            if metrics.exists():
                with metrics.open(encoding="utf-8",newline="") as f:rows=list(csv.DictReader(f))
            else:rows=[]
            runs.append(dict(seed=seed,arm=arm,state=status.get("state","NOT_STARTED"),
                epochs_recorded=len(status.get("epochs",[])),metric_rows=len(rows),
                final_epoch=status.get("final_epoch"),final_classifier_sha256=status.get("final_classifier_sha256"),
                final_validation=status.get("final_validation")))
    guard_path=ROOT/"formal_test_guard.json"
    guard=json.loads(guard_path.read_text(encoding="utf-8")) if guard_path.exists() else None
    all_completed=all(r["state"]=="completed" and r["epochs_recorded"]==30 and r["metric_rows"]==30 for r in runs)
    source_unchanged=all(sha(ROOT/name)==value for name,value in source_hashes.items())
    complete=(return_code==0 and all_completed and (ROOT/"formal_completed.json").exists()
              and guard is not None and guard["official_test_file_open_attempts"]==0
              and guard["CIFAR100_train_false_attempts"]==0 and source_unchanged)
    result=dict(state="COMPLETED" if complete else "FAILED_STOPPED",started=started,ended=now(),
        return_code=return_code,all_nine_runs_completed=all_completed,runs=runs,
        actual_completed_runs=sum(r["state"]=="completed" for r in runs),
        actual_recorded_epochs=sum(r["epochs_recorded"] for r in runs),
        formal_training_started=any(r["state"]!="NOT_STARTED" for r in runs),
        source_unchanged=source_unchanged,test_guard=guard,formal_log_sha256=sha(OUT/"formal.log"),
        automatic_retry=False)
    with (OUT/"END.json").open("x",encoding="utf-8",newline="\n") as end_file:
        json.dump(result,end_file,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps(dict(state=result["state"],return_code=return_code,completed_runs=result["actual_completed_runs"],
                         recorded_epochs=result["actual_recorded_epochs"],source_unchanged=source_unchanged)),flush=True)
    if not complete:sys.exit(return_code or 2)


if __name__=="__main__":main()
