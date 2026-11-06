"""Read-only formal-entry checks. Never call main, Smoke, training, forward or backward."""
import ast
import copy
import csv
import inspect
import json
import os
import shutil
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

OUT=Path(__file__).resolve().parent
ROOT=OUT.parent
REPO=ROOT.parents[1]
sys.path.insert(0,str(ROOT))
import ccsam_common as c
import train_ccsam as production

CHECKS=[]
DETAILS={}
BRANCH="experiment/ccsam-gradient-numerics-fix-20261106"
PARENT="cc7e60a29068a3548cf2fefd6ebd776c1eb9d8c8"


def git(*args):
    return subprocess.run(["git",*args],cwd=REPO,check=True,capture_output=True,
                          text=True,encoding="utf-8").stdout


def check(name,condition,evidence):
    CHECKS.append(dict(item=name,status="PASS" if condition else "FAIL",evidence=evidence))
    if not condition:raise RuntimeError(name+": "+str(evidence))


def exclusive_json(path,payload):
    with path.open("x",encoding="utf-8",newline="\n") as stream:
        json.dump(payload,stream,ensure_ascii=False,indent=2,allow_nan=False)


def native_readonly_prefix():
    node=ast.parse(inspect.getsource(production.main)).body[0]
    boundary=next(i for i,n in enumerate(node.body) if isinstance(n,ast.For)
                  and isinstance(n.target,ast.Name) and n.target.id=="seed"
                  and isinstance(n.iter,ast.Name) and n.iter.id=="SEEDS")
    prefix=copy.deepcopy(node)
    prefix.name="_isolated_native_preconditions"
    prefix.body=prefix.body[:boundary]
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=="train_one_arm"
                   for n in ast.walk(prefix))
    prefix.body.append(ast.parse("return dict(passed=True, data_root=str(data_root), manifest_sha256=sha256_file(manifest_path), train_shape=list(train_images.shape), labels_shape=list(train_labels.shape), validation_gap_shape=list(validation_gap.shape))").body[0])
    calls=[]
    def prohibited(*args,**kwargs):
        calls.append("prohibited_callback")
        raise RuntimeError("Smoke/training/gradient verification/output writing forbidden in prelaunch")
    namespace=dict(vars(production))
    for name in ("train_one_arm","smoke_test","verify_analytic_gradient","write_json"):
        namespace[name]=prohibited
    module=ast.fix_missing_locations(ast.Module(body=[prefix],type_ignores=[]))
    exec(compile(module,str(ROOT/"train_ccsam.py")+":read-only-prefix","exec"),namespace)
    argv=sys.argv
    try:
        # Only this isolated pre-loop function sees the argparse flag. Original main is never called.
        sys.argv=[str(ROOT/"train_ccsam.py"),"--formal"]
        result=namespace["_isolated_native_preconditions"]()
    finally:
        sys.argv=argv
    assert not calls
    result.update(original_main_called=False,training_loop_compiled=False,smoke_or_train_callbacks=0,
                  stop_boundary="before outer for seed in SEEDS; no training loop or later statements included")
    return result


def run_checks():
    before=json.loads((OUT/"BEFORE.json").read_text(encoding="utf-8"))
    check("Branch and baseline",git("branch","--show-current").strip()==BRANCH
          and git("rev-parse","HEAD").strip()==PARENT,dict(branch=BRANCH,head=PARENT))
    # The unmodified native prefix installs the original TEST guard before its actual data loads.
    native=native_readonly_prefix()
    DETAILS["native_entry_preconditions"]=native
    check("Native formal-entry preconditions",native["passed"] and native["smoke_or_train_callbacks"]==0,native)
    validation=ROOT/"validation_20261106_05"
    source=validation/"smoke_report.json";target=ROOT/"smoke_report.json"
    report=production.read_json(target)
    same=source.read_bytes()==target.read_bytes()
    check("Smoke report original bytes",same,dict(source=str(source),destination=str(target),
          sha256=c.sha256_file(target),bytes=target.stat().st_size,byte_equal=same,json_equal=report==production.read_json(source)))
    check("Smoke result flags",report["passed"] is True and report["formal_training_started"] is False
          and report["official_TEST_used"] is False,dict(passed=report["passed"],formal_training_started=report["formal_training_started"]))
    manifest_path=ROOT/"prepared/prepared_manifest.json"
    manifest=c.read_json(manifest_path);manifest_sha=c.sha256_file(manifest_path)
    check("Smoke/prepared manifest binding",report["prepared_manifest_sha256"]==manifest_sha,
          dict(actual=manifest_sha,expected=report["prepared_manifest_sha256"]))
    for name in ("train_ccsam.py","ccsam_common.py","experiment_protocol.json"):
        check("Unchanged "+name,c.sha256_file(ROOT/name)==before["source_hashes"][name],
              dict(sha256=c.sha256_file(ROOT/name)))
    checkpoint=c.resolve_checkpoint(manifest["checkpoint_path"])
    checkpoint_sha=c.sha256_file(checkpoint)
    check("Clean Stage I checkpoint",checkpoint_sha==c.EXPECTED_CLEAN_SHA256,
          dict(path=str(checkpoint),sha256=checkpoint_sha,expected=c.EXPECTED_CLEAN_SHA256))
    arrays={}
    for name,shape in (("train_images.npy",(10847,32,32,3)),("train_labels.npy",(10847,)),
                       ("validation_images.npy",(1980,32,32,3)),("validation_labels.npy",(1980,)),
                       ("validation_gap_features.npy",(1980,64))):
        path=ROOT/"prepared"/name;value=np.load(path,mmap_mode="r",allow_pickle=False)
        actual=c.sha256_file(path);expected=manifest["prepared_array_actions"][name]["sha256"]
        check("Prepared "+name,tuple(value.shape)==shape and actual==expected,
              dict(shape=list(value.shape),dtype=str(value.dtype),sha256=actual,expected=expected))
        arrays[name]=dict(shape=list(value.shape),dtype=str(value.dtype),sha256=actual)
    labels=np.load(ROOT/"prepared/train_labels.npy",allow_pickle=False)
    plans={}
    for seed in c.SEEDS:
        path=ROOT/"prepared"/f"sampling_plan_seed{seed}.npy"
        plan=np.load(path,mmap_mode="r",allow_pickle=False)
        actual=c.sha256_file(path);expected=manifest["sampling_plans"][str(seed)]["sha256"]
        stats=c.validate_class_aware_plan(labels,seed,plan)
        check("Sampling plan seed "+str(seed),actual==expected and stats["mismatched_indices"]==0
              and stats["indices_compared"]==325410,
              dict(sha256=actual,expected=expected,shape=list(plan.shape),dtype=str(plan.dtype),
                   matches_official=stats["matches_official_sequence"],indices_compared=stats["indices_compared"]))
        plans[str(seed)]=dict(sha256=actual,indices_compared=stats["indices_compared"],mismatched_indices=0)
    protocol=c.read_json(ROOT/"experiment_protocol.json")
    stage=protocol["stage2"]
    expected=dict(epochs=30,batch_size=64,learning_rate=0.01,momentum=0.9,weight_decay=0.0002,
                  nesterov=False,amp=False,mixup=False,gradient_accumulation=False,seeds=[123,124,125])
    config_ok=all(stage[k]==v for k,v in expected.items()) and tuple(stage["seeds"])==c.SEEDS
    config_ok &= (c.BATCH_SIZE,c.STAGE2_EPOCHS,c.STAGE2_LR,c.MOMENTUM,c.WEIGHT_DECAY,c.ADVER_REG,c.CLASS_WEIGHT_DIV)==(64,30,0.01,0.9,0.0002,1.0,2.0)
    check("C0/C1/C2 and frozen hyperparameters",config_ok and tuple(a["id"] for a in protocol["arms"])==c.ARMS,
          dict(stage2=expected,arms=list(c.ARMS),precision=stage["precision"],adver_reg=c.ADVER_REG,class_weight_divisor=c.CLASS_WEIGHT_DIV))
    check("Alpha schedule",all(c.alpha_for_epoch(e)==min((e+1)/30,0.3) for e in range(30)),"Original 30-epoch ramp, cap 0.3; no changes")
    conflicts=[str(ROOT/name) for name in ("results","formal_test_guard.json","formal_completed.json",
        "formal_test_guard.json.writing","formal_completed.json.writing","smoke_report.json.writing",
        "final_epoch_metrics.csv","paired_deltas.csv","summary.json","CCSAM_STAGE2_RESULTS.md") if (ROOT/name).exists()]
    runs=[ROOT/"results"/f"seed_{seed}"/arm for seed in c.SEEDS for arm in c.ARMS]
    check("Output paths have no old runs or conflicts",not conflicts and not any(path.exists() for path in runs),
          dict(conflicts=conflicts,nine_run_paths=[str(path) for path in runs]))
    check("Output parent writable",os.access(ROOT,os.W_OK),dict(parent=str(ROOT),free_bytes=shutil.disk_usage(ROOT).free))
    # Model construction/load only; no forward/backward/update. Returned parameters reside on CPU.
    backbone,classifier,metadata=c.build_models(checkpoint,torch.device("cpu"))
    state={k:v.detach().cpu().clone() for k,v in backbone.state_dict().items()}
    bh=c.state_hash(backbone);ch=c.state_hash(classifier)
    frozen=all(not p.requires_grad and p.grad is None and p.device.type=="cpu" for p in backbone.parameters())
    bn=[(name,module) for name,module in backbone.named_modules() if isinstance(module,torch.nn.modules.batchnorm._BatchNorm)]
    c.check_backbone_unchanged(backbone,state)
    check("Backbone/BN freeze and checkpoint state",frozen and all(not m.training for _,m in bn)
          and bh==report["seed_smoke"]["controls"]["initial_backbone_sha256"]
          and ch==report["seed_smoke"]["controls"]["initial_classifier_sha256"]
          and all(v["backbone_and_bn_bitwise_unchanged"] for v in report["seed_smoke"]["arms"].values()),
          dict(backbone_sha256=bh,classifier_sha256=ch,bn_modules=len(bn),checkpoint_seed=metadata["seed"],
               checkpoint_epoch=metadata["epoch"],current_check="CPU construction/load and eval/frozen state; no model forward",historical_real_smoke_bn_unchanged=True))
    c.seed_everything(123,deterministic=True)
    previous=c.read_json(validation/"PROTECTION_BEFORE.json")["effective_seeded_environment"]
    env=dict(torch=torch.__version__,cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),
        gpu=torch.cuda.get_device_name(0),cudnn_deterministic=torch.backends.cudnn.deterministic,
        cudnn_benchmark=torch.backends.cudnn.benchmark,cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
        matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32)
    check("Validated numerical environment",all(env[k]==previous[k] for k in env),env)
    old=c.read_json(validation/"PROTECTION_REPORT.json")
    allowed={(ROOT/name).relative_to(REPO).as_posix() for name in ("README.md",".gitattributes")}
    changed=[]
    for name,item in old["protected_files"].items():
        if name not in allowed and c.sha256_file(REPO/name)!=item["before"]["sha256"]:changed.append(name)
    with (validation/"PUBLICATION_SHA256.csv").open(encoding="utf-8") as stream:
        publication=list(csv.DictReader(stream))
    for row in publication:
        path=validation/row["path"]
        if c.sha256_file(path)!=row["sha256"]:changed.append(str(path))
    check("Historical results, inputs and sources protected",not changed,
          dict(unexpected_changes=changed,previous_protected_files=len(old["protected_files"]),authorized_doc_attribute_changes=sorted(allowed),publication_entries_verified=len(publication)))
    runtime_guard=c.guard_pass()
    check("Official TEST isolation",runtime_guard["official_test_file_open_attempts"]==0
          and runtime_guard["CIFAR100_train_false_attempts"]==0 and report["test_guard"]["official_test_file_open_attempts"]==0,
          runtime_guard)
    check("No Smoke rerun, forward or formal training",not (ROOT/"results").exists()
          and source.read_bytes()==target.read_bytes(),dict(new_smoke_calls=0,model_forward_calls=0,
          backward_calls=0,optimizer_instances=0,optimizer_steps=0,formal_training_started=False))
    DETAILS.update(checkpoint_sha256=checkpoint_sha,prepared_manifest_sha256=manifest_sha,prepared_arrays=arrays,
        sampling_plans=plans,environment=env,test_guard=runtime_guard,unexpected_protected_changes=changed,
        train_source_sha256=c.sha256_file(ROOT/"train_ccsam.py"),source_smoke_sha256=c.sha256_file(source),
        copied_smoke_sha256=c.sha256_file(target),formal_training_started=False)


def main():
    started=datetime.now().astimezone().isoformat()
    failure=None
    try:
        run_checks()
    except BaseException as exc:
        failure=exc
        DETAILS["exception"]=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
        if not CHECKS or CHECKS[-1]["status"]!="FAIL":CHECKS.append(dict(item="Unexpected check failure",status="FAIL",evidence=str(exc)))
    ready=failure is None and all(row["status"]=="PASS" for row in CHECKS)
    payload=dict(status="READY_FOR_FORMAL_TRAINING" if ready else "BLOCKED",started=started,
        completed=datetime.now().astimezone().isoformat(),branch=BRANCH,checked_head=PARENT,
        checks=CHECKS,details=DETAILS,smoke_rerun=False,formal_training_started=False)
    exclusive_json(OUT/"CHECK_RESULT.json",payload)
    lines=["# Final CC-SAM Stage II prelaunch check","",payload["status"],"",
        "Branch: `"+BRANCH+"`; checked HEAD: `"+PARENT+"`.","",
        "| Check | Result |","| --- | --- |"]
    lines += ["| "+row["item"]+" | "+row["status"]+" |" for row in CHECKS]
    lines += ["","The existing 2026-11-06 22:44:34 passing Smoke report was copied byte-for-byte; it was not regenerated or rerun.",
        "Both copies have SHA256 `"+DETAILS.get("copied_smoke_sha256","UNVERIFIED")+"`.",
        "Prepared manifest SHA256: `"+DETAILS.get("prepared_manifest_sha256","UNVERIFIED")+"`.",
        "Clean checkpoint SHA256: `"+DETAILS.get("checkpoint_sha256","UNVERIFIED")+"`.","",
        "The native main function was never called. Its original pre-loop statements were executed in an isolated function, stopping before `for seed in SEEDS`; the training loop and subsequent statements were not compiled into that function. Smoke/training callbacks were blocked and never invoked. This proves the existing formal gates can read the installed report without starting a run.","",
        "The current model check only constructed/loaded the sealed model and inspected frozen CPU state. No forward, backward or optimizer was executed. Real forward/BN invariance and gradient parity remain supported by the immutable validation_20261106_05 evidence: 510 real batches, 10 stress cases, and C0/C1/C2 Smoke PASS.","",
        "Authorized preparation edits: README status/usage wording, byte-identical root smoke_report.json, and a narrow .gitattributes rule to preserve the Smoke JSON bytes across Git checkouts. Gradient code, common implementation, protocol, data, sampler, checkpoint and old reports remain unchanged.","",
        "Detailed PASS/FAIL evidence, input hashes, frozen-model state, native gate execution and TEST counters: `prelaunch_20261106_01/CHECK_RESULT.json`.","",
        "Formal training has NOT started. No results directory was created. Starting the nine runs requires a new user confirmation.",""]
    if failure:lines += ["Failure: `"+str(failure).replace("\n"," ")+"`",""]
    with (ROOT/"FINAL_PRELAUNCH_CHECK.md").open("x",encoding="utf-8",newline="\n") as stream:stream.write("\n".join(lines))
    if failure:
        print(DETAILS["exception"]["traceback"])
        sys.exit(1)
    print(json.dumps(dict(status=payload["status"],checks=len(CHECKS),failed_checks=0,
        smoke_rerun=False,formal_training_started=False,test_guard=DETAILS["test_guard"])),flush=True)


if __name__=="__main__":main()
