"""Fixed-epoch campaign; unchanged author train/scheduler/model/loss/H2T cores.

No validation loader is constructed in training processes. No best-model path.
Each command runs in a fresh process. Existing run directories are never reused.
"""
import argparse
import csv
import datetime
import platform
import subprocess
import sys
import traceback

import numpy as np
import torch
import torchvision

from runtime import *

BRANCH = "experiment/h2t-cifar100n-lt-failure-screen-20261029"
AUTHOR_FILES = ["train_stage1.py", "train_stage2.py", "datasets/cifar100.py", "methods.py",
                "models/resnet_cifar.py", "loss.py", "utils/logger.py", "utils/metric.py", "utils/meter.py",
                "config/cifar100/cifar100_imb001_stage1_ce_mixup.yaml", "config/cifar100/cifar100_imb001_stage2_ce_H2T.yaml"]
OWN_FILES = ["runtime.py", "train.py", "prepare_inputs.py", "prepare_protocol.py", "compare_outcomes.py",
             "CODEX_CIFAR100N_LT_FAILURE_SCREEN.txt", "SCIENTIFIC_RATIONALE.md"]


def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()


def check_lock():
    assert subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip() == BRANCH
    lock = read(OUT / "TRAINING_LOCK.json")
    for name, expected in lock["code_sha256"].items():
        assert sha(ROOT / name) == expected, "Locked source changed: " + name
    assert lock["stage1_seed"] == 42 and lock["stage2_seeds"] == [124, 125]
    return lock


def environment(cfg):
    return dict(python=sys.version, interpreter=sys.executable, platform=platform.platform(),
        torch=torch.__version__, torchvision=torchvision.__version__, CUDA=torch.version.cuda,
        cuDNN=torch.backends.cudnn.version(), GPU=torch.cuda.get_device_name(0),
        GPU_bytes=torch.cuda.get_device_properties(0).total_memory,
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        cudnn_benchmark=torch.backends.cudnn.benchmark, cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
        matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        float32_matmul_precision=torch.get_float32_matmul_precision(),
        AMP=False, batch_size=128, workers=0, DataParallel=True, config=cfg.dump())


def save_weights(path, bb, fc, epoch):
    b, c = cpu_state(bb), cpu_state(fc)
    assert all(torch.isfinite(t).all() for t in list(b.values()) + list(c.values()))
    with path.open("xb") as f: torch.save(dict(bb=b, fc=c, epoch=epoch), f)
    return dict(file=str(path.relative_to(ROOT)), sha256=sha(path), epoch=epoch,
                bb_state_sha=state_hash(b), fc_state_sha=state_hash(c))


def load_weights(path, bb, fc):
    data = torch.load(path, map_location="cpu", weights_only=True)
    bb.module.load_state_dict(data["bb"], strict=True)
    fc.module.load_state_dict(data["fc"], strict=True)
    return data


def counters(bb):
    return {k:int(v) for k,v in bb.module.state_dict().items() if k.endswith("num_batches_tracked")}


def bn_state(bb):
    return {k:v for k,v in cpu_state(bb).items() if any(x in k for x in ("running_mean", "running_var", "num_batches_tracked"))}


def parameter_hash(bb):
    return state_hash({k:p for k,p in bb.module.named_parameters()})


def initialize():
    assert not (OUT / "TRAINING_LOCK.json").exists()
    g = install_guard("initialize")
    files = [OFFICIAL / p for p in AUTHOR_FILES] + [TASK / p for p in OWN_FILES]
    write_json(OUT / "TRAINING_LOCK.json", dict(created_UTC=now(), branch=BRANCH,
        code_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in files},
        regime_label_sha256={r:sha(OUT / "protocol" / (r+"_train.npz")) for r in ("clean", "noisy")},
        lt_image_sha256=sha(CACHE / "inputs/lt_images.npy"),
        stage1_seed=42, stage2_seeds=[124,125], stage1_epochs=200, stage2_epochs=10,
        selection="fixed final epoch only; no validation during either training stage",
        paired_control="Both arms take author H2T>0 two-forward BN-shift branch; no_fusion substitutes identity for replacement and consumes same unused CPU randperm(64) draw.",
        H2T="Unchanged author methods.H2T; batch-shared replacement of16/64 complete channels, rho=.25",
        labels="Noisy training and receiver class pools use only observed noisy labels; no clean training oracle or validation reads"))
    check_lock()
    c=config(1); seed(42,c); bb,fc=models()
    meta=save_weights(CACHE / "initial_seed42.pth",bb,fc,0)
    write_json(OUT / "INITIALIZATION.json",dict(checkpoint=meta, environment=environment(c), IO_guard=g))
    assert_guard(g)
    print("INITIALIZATION LOCKED " + meta["sha256"], flush=True)


def execute(stage, regime, s2seed=124, method="no_fusion", dryrun=False):
    lock=check_lock()
    if not dryrun: assert read(OUT / "DRYRUN.json")["passed"] is True
    name=("stage1_"+regime if stage==1 else f"stage2_{regime}_seed{s2seed}_{method}")
    if dryrun: name="dryrun_"+name
    run=OUT / name;run.mkdir(exist_ok=False)
    local=CACHE / name;local.mkdir(exist_ok=False)
    g=install_guard("dryrun" if dryrun else "stage"+str(stage),regime)
    try:
        c=config(stage);seed(42 if stage==1 else s2seed,c)
        bb,fc=models()
        parent=CACHE / ("initial_seed42.pth" if stage==1 or dryrun else "stage1_"+regime+"/epoch200.pth")
        parent_sha=sha(parent); loaded=load_weights(parent,bb,fc)
        assert loaded["epoch"]==(0 if stage==1 or dryrun else 200)
        if stage==1: assert parent_sha==read(OUT / "INITIALIZATION.json")["checkpoint"]["sha256"]
        else:
            if not dryrun: assert parent_sha==read(OUT / ("stage1_"+regime) / "RUN.json")["checkpoint"]["sha256"]
            for p in bb.parameters():p.requires_grad_(False)
        initial=dict(bb_state_sha=state_hash(cpu_state(bb)),fc_state_sha=state_hash(cpu_state(fc)),
                     bb_parameters_sha=parameter_hash(bb),BN_counters=counters(bb))
        ds=ObservedDataset(regime,c);loader=AuditedLoader(ds,stage,1 if dryrun else None)
        groups=[{"params":bb.parameters()},{"params":fc.parameters()}] if stage==1 else [{"params":fc.parameters()}]
        optimizer=AuditedSGD(groups,c.lr,momentum=c.momentum,weight_decay=c.weight_decay)
        assert not optimizer.state
        write_json(run / "START.json",dict(start_UTC=now(),parent_sha256=parent_sha,parent_file=str(parent.relative_to(ROOT)),
            source_epoch=loaded["epoch"],initial=initial,environment=environment(c),
            optimizer_initial_state_entries=0,trainable_parameters=sum(p.numel() for group in optimizer.param_groups for p in group["params"]),
            label_counts_observed=ds.cls_num_list))
        baseline=None;nf_run=OUT / name.replace("_h2t","_no_fusion")
        if stage==2 and method=="h2t":
            nf=read(nf_run / "RUN.json")
            assert nf["initial"]==initial and nf["parent_sha256"]==parent_sha
            with (CACHE / name.replace("_h2t","_no_fusion") / "batch_controls.csv").open(newline="",encoding="utf-8") as f: baseline=list(csv.DictReader(f))
            assert len(baseline)==(1 if dryrun else 850)
        nforward=[0]
        bb.register_forward_hook(lambda *unused:nforward.__setitem__(0,nforward[0]+1))
        torch.cuda.reset_peak_memory_stats()
        epoch_rows=[];all_controls=[]
        with (run / "native_train.log").open("x",encoding="utf-8") as log, (run / "epochs.csv").open("x",newline="",encoding="utf-8") as ef:
            core=Core(stage,c,loader,method,log); writer=None
            for epoch in range(1 if dryrun else c.num_epochs):
                row=core.epoch_run(optimizer,epoch,bb,fc,c,baseline)
                row={"epoch":epoch+1,**row}
                assert len(loader.rows)==(1 if dryrun else 85)
                assert optimizer.updates==(epoch+1)*(1 if dryrun else 85)
                assert all(v==(epoch+1)*(1 if dryrun else 85)*(1 if stage==1 else 2) for k,v in {k:v-initial["BN_counters"][k] for k,v in counters(bb).items()}.items())
                if stage==2:assert parameter_hash(bb)==initial["bb_parameters_sha"] and all(p.grad is None for p in bb.parameters())
                all_controls.extend(loader.rows);epoch_rows.append(row)
                if writer is None:writer=csv.DictWriter(ef,fieldnames=list(row));writer.writeheader()
                writer.writerow(row);ef.flush()
                print(f"{name} epoch={epoch+1}/{1 if dryrun else c.num_epochs} loss={row['Loss']:.5f} train_top1={row['Acc@1']:.3f} lr={row['lr']:.8f} seconds={row['seconds']:.2f}",flush=True)
        with (local / "batch_controls.csv").open("x",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f,fieldnames=list(all_controls[0]));w.writeheader();w.writerows(all_controls)
        assert nforward[0]==optimizer.updates*(1 if stage==1 else 2)
        if stage==2:
            assert core.replacements==(optimizer.updates if method=="h2t" else 0)
            if method=="h2t":assert state_hash(bn_state(bb))==nf["BN_final_sha"]
        assert_guard(g)
        final_epoch=1 if dryrun else c.num_epochs
        checkpoint=save_weights(local / f"epoch{final_epoch}.pth",bb,fc,final_epoch)
        proof=dict(completed_UTC=now(),stage=stage,regime=regime,stage1_seed=42,stage2_seed=s2seed if stage==2 else None,
            method=method if stage==2 else "CE_MixUp",dryrun=dryrun,initial=initial,parent_sha256=parent_sha,checkpoint=checkpoint,
            optimizer_updates=optimizer.updates,finite_gradient_checks=optimizer.finite_gradient_checks,BB_forwards=nforward[0],
            BN_final_counters=counters(bb),BN_final_sha=state_hash(bn_state(bb)),
            final_BB_parameters_sha=parameter_hash(bb),H2T_replacements=core.replacements,
            controls_file=str((local / "batch_controls.csv").relative_to(ROOT)),controls_file_sha256=sha(local / "batch_controls.csv"),
            all_batch_control_epoch_hashes=[r["control_epoch_sha"] for r in epoch_rows],
            input_image_epoch_hashes=[r["input_images_epoch_sha"] for r in epoch_rows],
            mixup_lambda_epoch_hashes=[r["mixup_lambdas_sha"] for r in epoch_rows],
            pair_checked_before_all_updates=bool(baseline is not None),
            peak_GPU_bytes=torch.cuda.max_memory_allocated(),IO_guard=g)
        write_json(run / "RUN.json",proof)
        return proof
    except BaseException:
        write_json(run / "FAILURE.json",dict(UTC=now(),traceback=traceback.format_exc(),IO_guard=g))
        raise


def dryrun():
    proofs={}
    for regime in ("clean","noisy"):
        for stage,method in [(1,"no_fusion"),(2,"no_fusion"),(2,"h2t")]:
            cmd=[sys.executable,"-B","-u","-X","utf8",str(Path(__file__)),"stage"+str(stage),"--regime",regime,"--method",method,"--dryrun"]
            subprocess.run(cmd,check=True,cwd=ROOT)
            name="dryrun_stage1_"+regime if stage==1 else f"dryrun_stage2_{regime}_seed124_{method}"
            proofs[name]=read(OUT / name / "RUN.json")
    a,b=[proofs["dryrun_stage1_"+r] for r in ("clean","noisy")]
    assert a["initial"]==b["initial"] and a["input_image_epoch_hashes"]==b["input_image_epoch_hashes"] and a["mixup_lambda_epoch_hashes"]==b["mixup_lambda_epoch_hashes"]
    assert all(p["finite_gradient_checks"]==1 and p["optimizer_updates"]==1 for p in proofs.values())
    write_json(OUT / "DRYRUN.json",dict(passed=True,UTC=now(),checks="Both regimes, one real S1 batch and paired S2 NF/H2T real batches, exact pre-update control hashes and final BN equality; smoke states not reused",runs={k:v["checkpoint"]["sha256"] for k,v in proofs.items()}))
    print("REAL BATCH DRYRUN PASSED",flush=True)


def campaign():
    check_lock();assert read(OUT / "DRYRUN.json")["passed"]
    commands=[("stage1",r,124,"no_fusion") for r in ("clean","noisy")]
    commands += [("stage2",r,s,m) for r in ("clean","noisy") for s in (124,125) for m in ("no_fusion","h2t")]
    for phase,r,s,m in commands:
        subprocess.run([sys.executable,"-B","-u","-X","utf8",str(Path(__file__)),phase,"--regime",r,"--seed",str(s),"--method",m],check=True,cwd=ROOT)
    write_json(OUT / "TRAINING_COMPLETED.json",dict(UTC=now(),completed_runs=10,stage1_epochs=400,stage2_epochs=80,evaluation_started=False))


def main():
    p=argparse.ArgumentParser();p.add_argument("phase",choices=("initialize","dryrun","stage1","stage2","campaign"))
    p.add_argument("--regime",choices=("clean","noisy"));p.add_argument("--method",choices=("no_fusion","h2t"),default="no_fusion")
    p.add_argument("--seed",type=int,choices=(124,125),default=124);p.add_argument("--dryrun",action="store_true")
    a=p.parse_args()
    if a.phase=="initialize":initialize()
    elif a.phase=="dryrun":dryrun()
    elif a.phase=="campaign":campaign()
    else:
        assert a.regime is not None
        execute(int(a.phase[-1]),a.regime,a.seed,a.method,a.dryrun)


if __name__=="__main__":main()
