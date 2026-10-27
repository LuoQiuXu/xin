#!/usr/bin/env python3
"""CIFAR100-LT IF100 PI-H2T v2 paper-aligned independent reconstruction.

This is NOT an official PI-H2T reproduction until open protocol questions
(W-reference, detach, PIF initialization and BN running buffers) are resolved.
Pre-register those choices BEFORE evaluating accuracy. No extra model modules.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR100

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
ORIGINAL=REPO/"H2T_official_clean/H2T-main"
if not (ORIGINAL/"models/resnet_cifar.py").is_file():
    raise FileNotFoundError(f"Frozen original H2T source missing: {ORIGINAL}")
sys.path.insert(0,str(ORIGINAL))
from models.resnet_cifar import resnet32_fe, Classifier  # noqa: E402
from datasets.cifar100 import IMBALANCECIFAR100  # noqa: E402
from paper_components import PIExtractor, pool_feature, receiver_ratio, h2tf, classify_vector, group_stats  # noqa: E402

GROUP={"Many":(0,36),"Medium":(36,71),"Few":(71,100)}
STAGE1_EPOCHS=200
STAGE2_EPOCHS=10
BATCH=128
ALLOWED_W=("frozen_stage1","current_stage2")
ALLOWED_BN=("update","frozen")
MEAN=(.4914,.4822,.4465)
STD=(.2023,.1994,.2010)


def hashfile(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as inp:
        for piece in iter(lambda:inp.read(1024*1024),b""):
            h.update(piece)
    return h.hexdigest()


def dump(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf-8")


def seeding(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cosine_lr(lr: float, epoch: int, total: int) -> float:
    return lr*.5*(1+math.cos(math.pi*epoch/total))


def stage1_lr(epoch: int) -> float:
    # Original H2T logger defaults (YAML omits lr_scheduler):
    # 5-epoch linear warm-up, milestones 160/180.
    ep=epoch+1
    if ep<=5:
        return 0.1*ep/5.
    if ep>180:
        return .001
    if ep>160:
        return .01
    return .1


def source_hashes() -> dict:
    paths=[
        HERE/"paper_components.py", HERE/"run_reproduction.py",
        ORIGINAL/"models/resnet_cifar.py",ORIGINAL/"datasets/cifar100.py",
        ORIGINAL/"train_stage1.py",ORIGINAL/"train_stage2.py",
        ORIGINAL/"utils/logger.py"
    ]
    return {str(p.relative_to(REPO)).replace("\\","/"):hashfile(p) for p in paths}


def paths_for(args):
    path=HERE/"runs"/args.run_id
    if not args.run_id or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for ch in args.run_id):
        raise ValueError("run-id may only contain letters, digits, underscore and hyphen")
    return path


def check_plan(args) -> dict:
    if args.w_reference not in ALLOWED_W or args.bn_mode not in ALLOWED_BN:
        raise ValueError("invalid mechanism setting")
    if args.stage1_seed!=123 or args.stage2_seed!=42:
        raise ValueError("This initial replication has locked seeds 123 (stage1), 42 (stage2)")
    return {
        "status":"PAPER_ALIGNED_INDEPENDENT_RECONSTRUCTION_NOT_AUTHOR_VERIFIED",
        "paper":"PI-H2T v2, Eq(3),(6),(7),(8); independent reconstruction",
        "official_author_repo":"Keke921/H2T publicly shows original H2T; no complete PI-H2T verified",
        "dataset":"CIFAR-100-LT IF=100, exp, original H2T selection seed=0",
        "model":"original ResNet-32 CIFAR feature extractor + 2-weight unbiased Conv1d PIF",
        "loss":"CE both stages; StageI alpha=1 MixUp; StageII receiver hard labels",
        "stage1_epochs":STAGE1_EPOCHS,
        "stage2_epochs":STAGE2_EPOCHS,
        "batch_size":BATCH,
        "stage1_seed":args.stage1_seed,
        "stage2_seed":args.stage2_seed,
        "stage1_lr":"SGD .1, momentum .9, weight decay .0002; warmup5, >160 x.1, >180 x.01",
        "stage2_lr":"SGD .1 x .2 initial=.02, momentum .9, weight decay .0005; cosine",
        "w_reference":args.w_reference,
        "bn_running_buffers":args.bn_mode,
        "ratio_detach":True,
        "pif_initialization":"torch.nn.Conv1d default (author init unknown)",
        "checkpoint_selection":"fixed final epoch200 (not TEST-selected best)",
        "evaluation":"official CIFAR100 TEST for paper-comparison ONLY; no tuning or model selection",
        "unverified_author_details":["PIF initializer","ratio W-reference/current-vs-frozen",
                                     "ratio gradient detach","StageII BN behavior","paper best-checkpoint selection"],
        "source_hashes":source_hashes(),
    }


def transforms_pair():
    train=transforms.Compose([
        transforms.RandomCrop(32,padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN,STD),
    ])
    test=transforms.Compose([transforms.ToTensor(),transforms.Normalize(MEAN,STD)])
    return train,test


def build_loader(args,stage: int):
    train_tf,test_tf=transforms_pair()
    dsconf=SimpleNamespace(sampler=SimpleNamespace(
        type="default" if stage==1 else "weighted sampler",
        weighted_sampler=SimpleNamespace(type="balance"),
        dual_sample=SimpleNamespace(enable=stage==2,type="long-tailed"),
    ))
    ds=IMBALANCECIFAR100(dsconf,root=args.data_root,imb_factor=.01,rand_number=0,
                          train=True,download=args.download,transform=train_tf)
    nums=ds.get_cls_num_list()
    assert len(nums)==100 and nums[0]==500 and nums[-1]==5, "CIFAR-LT sample counts not IF100"
    train=DataLoader(ds,batch_size=BATCH,shuffle=True,num_workers=0,pin_memory=True,drop_last=False)
    evalset=CIFAR100(root=args.data_root,train=False,download=args.download,transform=test_tf)
    test=DataLoader(evalset,batch_size=256,shuffle=False,num_workers=0,pin_memory=True)
    lt_hash=hashlib.sha256(ds.data.tobytes()+np.asarray(ds.targets,dtype=np.int64).tobytes()).hexdigest()
    return train,test,nums,lt_hash


def make_models(device):
    # Original backbone factory and classifier, not historical 25-param PIF.
    network=PIExtractor(resnet32_fe()).to(device)
    classifier=Classifier(feat_in=64,num_classes=100).to(device)
    n=sum(p.numel() for p in network.pif.parameters())
    assert n==2, f"unexpected PIF parameter count {n}"
    return network,classifier


@torch.no_grad()
def evaluate(network,head,loader,device):
    old_train1,old_train2=network.training,head.training
    network.eval();head.eval()
    pp=[];yy=[]
    for im,y in loader:
        im=im.to(device,non_blocking=True)
        logits=head(network(im))
        pp.append(logits.argmax(1).cpu())
        yy.append(y.cpu())
    pred=torch.cat(pp); labels=torch.cat(yy)
    assert len(pred)==10000 and len(labels)==10000, "Official CIFAR100 TEST expected"
    out=group_stats(pred,labels)
    if old_train1:network.train()
    if old_train2:head.train()
    return out,pred,labels


def preflight(args):
    print(json.dumps(check_plan(args),indent=2,ensure_ascii=False))
    # Do not auto-download or train during preflight.
    data=Path(args.data_root)
    if not (data/"cifar-100-python"/"train").is_file() or not (data/"cifar-100-python"/"test").is_file():
        if not args.download:
            raise FileNotFoundError(f"CIFAR100 cache missing under {data}; explicit --download is needed")
        CIFAR100(root=args.data_root,train=True,download=True)
        CIFAR100(root=args.data_root,train=False,download=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not detected; do not silently run a 200-epoch CPU training")
    print("PREFLIGHT PASS: original CIFAR files present, CUDA detected; unresolved author details remain explicitly labelled")


def stage1(args):
    run=paths_for(args)
    if run.exists(): raise FileExistsError(f"Refuse overwrite: {run}")
    preflight(args)
    plan=check_plan(args)
    seeding(args.stage1_seed)
    device=torch.device("cuda:0")
    net,head=make_models(device)
    # Dataset ctor resets NumPy seed to zero as in frozen H2T.
    loader,test,counts,lt_hash=build_loader(args,1)
    run.mkdir(parents=True,exist_ok=False)
    dump(run/"protocol.json",plan)
    dump(run/"dataset_counts.json",{"class_counts":counts,"train_length":len(loader.dataset),
        "test_length":len(test.dataset),"lt_data_and_labels_sha256":lt_hash})
    opt=torch.optim.SGD(list(net.parameters())+list(head.parameters()),
                        lr=.1,momentum=.9,weight_decay=2e-4)
    record=[]
    for ep in range(STAGE1_EPOCHS):
        net.train();head.train()
        lr=stage1_lr(ep)
        for pg in opt.param_groups:pg["lr"]=lr
        weighted_loss=0.
        for im,lab,_ in loader:
            im=im.to(device,non_blocking=True);lab=lab.to(device,non_blocking=True)
            lam=float(np.random.beta(1.,1.))
            idx=torch.randperm(len(lab),device=device)
            mixed=lam*im+(1-lam)*im[idx]
            logits=head(net(mixed))
            loss=lam*F.cross_entropy(logits,lab)+(1-lam)*F.cross_entropy(logits,lab[idx])
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"nonfinite StageI loss epoch={ep+1}")
            opt.zero_grad(set_to_none=True);loss.backward();opt.step()
            weighted_loss+=float(loss.detach())*len(lab)
        line={"epoch":ep+1,"lr":lr,"train_loss":weighted_loss/len(loader.dataset)}
        record.append(line)
        print("StageI",line,flush=True)
    checkpoint=run/"stage1_final_epoch200.pth"
    torch.save({
        "stage":1,"epoch":200,
        "stage1_seed":args.stage1_seed,
        "network":{k:v.detach().cpu() for k,v in net.state_dict().items()},
        "classifier":{k:v.detach().cpu() for k,v in head.state_dict().items()},
        "source_hashes":plan["source_hashes"],
        "protocol":plan,
    },checkpoint)
    save_rows(run/"stage1_epochs.csv",record)
    metrics,_,_=evaluate(net,head,test,device)
    dump(run/"stage1_summary.json",{"epoch":200,"official_test_diagnostic":metrics,
                                    "checkpoint_file_sha256":hashfile(checkpoint)})
    print("StageI FINAL epoch200",metrics,flush=True)


def save_rows(path,rows):
    with path.open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def read_stage1(run,device):
    ckpath=run/"stage1_final_epoch200.pth"
    if not ckpath.is_file():raise FileNotFoundError(ckpath)
    # Only read locally generated trusted checkpoint files.
    ck=torch.load(ckpath,map_location="cpu",weights_only=False)
    if ck.get("stage")!=1 or ck.get("epoch")!=200 or ck.get("stage1_seed")!=123:
        raise ValueError("Wrong StageI checkpoint lineage")
    if ck["source_hashes"]!=source_hashes():
        raise RuntimeError("Source code changed since StageI training; stop and investigate")
    net,head=make_models(device)
    net.load_state_dict(ck["network"],strict=True)
    head.load_state_dict(ck["classifier"],strict=True)
    return net,head,hashfile(ckpath)


def stage2(args):
    run=paths_for(args)
    planpath=run/"protocol.json"
    if not planpath.is_file():raise FileNotFoundError(planpath)
    plan=json.loads(planpath.read_text(encoding="utf-8"))
    if plan!=check_plan(args):
        raise RuntimeError("Protocol drift since StageI: no changes to BN/W-reference/seed/plan")
    target=run/"stage2"/args.arm
    if target.exists():raise FileExistsError(f"Refuse overwrite: {target}")
    if args.arm not in ("no_fusion","h2tf"):raise ValueError("arm must be no_fusion or h2tf")
    if not torch.cuda.is_available():raise RuntimeError("CUDA required")
    seeding(args.stage2_seed)
    device=torch.device("cuda:0")
    net,head,ckhash=read_stage1(run,device)
    for p in net.parameters():p.requires_grad_(False)
    fixed_stage1_w=head.fc.weight.detach().clone()
    input_head_hash=hashlib.sha256(b"".join(x.detach().cpu().numpy().tobytes() for x in head.state_dict().values())).hexdigest()
    loader,test,_,lt_hash=build_loader(args,2)
    reference=json.loads((run/"dataset_counts.json").read_text(encoding="utf-8"))
    if lt_hash!=reference["lt_data_and_labels_sha256"]:
        raise RuntimeError("StageII CIFAR-LT subset differs from StageI; stop")
    # run's stage2/arm is created only AFTER checkpoint, protocol and loader preflight.
    target.mkdir(parents=True,exist_ok=False)
    opt=torch.optim.SGD(head.parameters(),lr=.02,momentum=.9,weight_decay=5e-4)
    history=[]
    for ep in range(STAGE2_EPOCHS):
        if args.bn_mode=="update":net.train()
        else:net.eval()
        head.train()
        lr=.2*cosine_lr(.1,ep,STAGE2_EPOCHS)
        for pg in opt.param_groups:pg["lr"]=lr
        weighted_loss=0.; ratio_sum=0.; ratio_n=0
        for im,y,meta in loader:
            im=im.to(device,non_blocking=True);y=y.to(device,non_blocking=True)
            donor=meta["sample_image"].to(device,non_blocking=True)
            # Always forward donor too in no-fusion: keep BN input traffic
            # matched across H2TF/no-fusion arms (labelled paired-BN control).
            with torch.no_grad():
                a=pool_feature(net(im))
                b=pool_feature(net(donor))
                w=(fixed_stage1_w if args.w_reference=="frozen_stage1"
                   else head.fc.weight.detach())
                r=receiver_ratio(a,y,w)
                mixed=(h2tf(a,b,r) if args.arm=="h2tf" else a)
                ratio_sum+=float((1-r).sum().cpu())
                ratio_n+=len(y)
            logits=classify_vector(head,mixed)
            loss=F.cross_entropy(logits,y)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"nonfinite StageII loss epoch={ep+1}")
            opt.zero_grad(set_to_none=True);loss.backward();opt.step()
            weighted_loss+=float(loss.detach())*len(y)
        row={"epoch":ep+1,"lr":lr,"train_loss":weighted_loss/len(loader.dataset),
             "candidate_donor_mass_mean":ratio_sum/max(ratio_n,1)}
        history.append(row)
        print("StageII",args.arm,row,flush=True)
    network_bn_hash=hashlib.sha256(b"".join(v.detach().cpu().numpy().tobytes()
                        for k,v in net.state_dict().items() if "running_" in k or "num_batches_tracked" in k)).hexdigest()
    ckpath=target/"classifier_final_epoch10.pth"
    torch.save({"stage":2,"epoch":10,"arm":args.arm,"stage1_sha256":ckhash,
                "w_reference":args.w_reference,"bn_mode":args.bn_mode,
                "classifier":{k:v.detach().cpu() for k,v in head.state_dict().items()},
                "network_bn":{k:v.detach().cpu() for k,v in net.state_dict().items()
                              if "running_" in k or "num_batches_tracked" in k}},ckpath)
    save_rows(target/"train_epochs.csv",history)
    metrics,pred,labels=evaluate(net,head,test,device)
    np.savez_compressed(target/"test_epoch10_predictions.npz",
                         labels=labels.numpy(),pred=pred.numpy())
    dump(target/"summary.json",{
        "arm":args.arm,"epoch":10,"metrics":metrics,
        "source_stage1_sha256":ckhash,
        "stage2_initial_classifier_hash":input_head_hash,
        "stage2_final_classifier_sha256":hashfile(ckpath),
        "final_bn_buffers_hash":network_bn_hash,
        "paired_bn_traffic":True,
        "official_test_evaluated":True,
        "label":"paper-aligned independent reconstruction, unresolved author protocol",
    })
    print("StageII FINAL",args.arm,metrics,flush=True)


def report(args):
    run=paths_for(args)
    records={}
    for arm in ("no_fusion","h2tf"):
        p=run/"stage2"/arm/"summary.json"
        if not p.is_file():raise FileNotFoundError(p)
        records[arm]=json.loads(p.read_text(encoding="utf-8"))
    a=records["no_fusion"]["metrics"];b=records["h2tf"]["metrics"]
    comparison={k:round(b[k]-a[k],5) for k in GROUP.keys()|{"Overall"}}
    # Paired per-image changes, not just headline accuracy. Never select a new
    # tuning target from these official-TEST observations.
    with np.load(run/"stage2/no_fusion/test_epoch10_predictions.npz",allow_pickle=False) as z:
        labels=z["labels"].copy();baseline=z["pred"].copy()
    with np.load(run/"stage2/h2tf/test_epoch10_predictions.npz",allow_pickle=False) as z:
        if not np.array_equal(z["labels"],labels):
            raise RuntimeError("Evaluation label/order mismatch across arms")
        fused=z["pred"].copy()
    rows=[]
    for klass in range(100):
        mask=labels==klass
        if int(mask.sum())!=100:raise ValueError(f"class {klass} TEST count differs from 100")
        correct0=baseline[mask]==klass;correct1=fused[mask]==klass
        wins=int(np.count_nonzero(correct1 & ~correct0))
        losses=int(np.count_nonzero(~correct1 & correct0))
        rows.append({"class":klass,"group":"Many" if klass<36 else ("Medium" if klass<71 else "Few"),
                     "no_fusion_correct":int(correct0.sum()),"h2tf_correct":int(correct1.sum()),
                     "h2tf_wins":wins,"h2tf_losses":losses,
                     "delta_pp":wins-losses})
    save_rows(run/"per_class_paired_changes.csv",rows)
    dump(run/"comparison.json",{"arms":records,"h2tf_minus_no_fusion_pp":comparison,
        "per_class_file":"per_class_paired_changes.csv",
        "paired_total_wins":sum(x["h2tf_wins"] for x in rows),
        "paired_total_losses":sum(x["h2tf_losses"] for x in rows),
        "caution":"One StageI seed + one StageII seed. Reused official CIFAR TEST is NOT independent science confirmation; no tuning by this table."})
    print("H2TF - NO FUSION [pp]",comparison)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command",choices=("preflight","stage1","stage2","report"))
    ap.add_argument("--run-id",required=True)
    ap.add_argument("--data-root",default="H2T/data/cifar100")
    ap.add_argument("--download",action="store_true",
                    help="explicit opt-in to download CIFAR100 using torchvision")
    ap.add_argument("--stage1-seed",type=int,default=123)
    ap.add_argument("--stage2-seed",type=int,default=42)
    ap.add_argument("--arm",choices=("no_fusion","h2tf"),default="no_fusion")
    ap.add_argument("--bn-mode",choices=ALLOWED_BN,default="update")
    ap.add_argument("--w-reference",choices=ALLOWED_W,default="frozen_stage1")
    args=ap.parse_args()
    if args.command=="preflight":preflight(args)
    elif args.command=="stage1":stage1(args)
    elif args.command=="stage2":stage2(args)
    elif args.command=="report":report(args)


if __name__=="__main__":
    main()
