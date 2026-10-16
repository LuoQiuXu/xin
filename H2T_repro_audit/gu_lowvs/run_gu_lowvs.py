"""GU-LowVS: one four-arm classifier-augmentation iteration on frozen Clean Stage I.

Actions:
    python H2T_repro_audit/gu_lowvs/run_gu_lowvs.py preflight
    python H2T_repro_audit/gu_lowvs/run_gu_lowvs.py prepare
    python H2T_repro_audit/gu_lowvs/run_gu_lowvs.py train --seed 124 --arm LowVS-Full
    python H2T_repro_audit/gu_lowvs/run_gu_lowvs.py report

All input/feature paths from prior experiments are read-only. Results remain in
this new directory. No Stage I re-training, no official CIFAR100 TEST evaluation.
"""
from __future__ import annotations
import argparse
import csv
import json
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
RAVS = PARENT / "ravs_v1"
sys.path.insert(0, str(RAVS))
import run_ravs_v1 as base
from run_safeguards import protect, history_manifest
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from common import N, BATCH, sha, state_hash, array_hash, evaluate, check_frozen
from stability_models import check_sources, plan_dir
from reused_train_helpers import augment, fuse, scheduler, optimizer_for
from online_policy import online_masks
from utility_policy import ARMS, reference_gradient, utility_actions, apply_actions, test_policy

SEEDS = (124,125)
RESULTS = HERE / "results"
BANK = RESULTS / "balanced_clean_train_bank.npz"
KEYS = ("indices_sha256","images_sha256","features_sha256",
        "scores_sha256","gumbel_sha256")
METRICS = ("Overall","Head","OfficialMedium","Tail","Many","Medium","Few","Macro","validation_CE")
HISTORY_EVENTS = []
PROTOCOL = {
    "question": "Does train-only first-order balanced gradient utility improve Low-VS augmentation?",
    "methods": list(ARMS), "seeds": list(SEEDS), "epochs": 10,
    "primary_epoch":10,"batch_size":128,"stage1":"same clean seed42 epoch184",
    "backbone":"frozen ResNet-32, BN in eval throughout",
    "sampling":"BS receiver, IS donor, paired Gumbel and augmentations",
    "vs":"exact original Bottom48 -> uniformly 16 channels, donor fallback unchanged",
    "alphas":[0.0,0.5,1.0],
    "utility":"argmax of analytic per-example CE gradient dot balanced clean-TRAIN reference gradient, no validation",
    "reference":"once per epoch exact class-balanced CE gradient on all original TRAIN eval-transform GAP vectors",
    "permitted_outputs":"only this results directory",
    "stage1_train":False,"test_access":False,
    "validation":"TRAIN-only reused development holdout, 1980 images of 99 classes, class0 absent",
    "limitations":"utility is local surrogate for clean TRAIN risk, not future generalization; momentum and weight decay not in score",
}


def save_json(path, x):
    p=Path(path);assert p.resolve().is_relative_to(HERE.resolve()),p
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")


def save_csv(path, rows):
    p=Path(path);assert rows and p.resolve().is_relative_to(HERE.resolve()),p
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with Path(path).open("r",newline="",encoding="utf-8") as f:
        return list(csv.DictReader(f))


def inputs(seed):
    y,vy,bank,plan,noise=base.load_data(seed)
    check_sources()
    assert len(y)==N and vy.shape==(1980,)
    return y,vy,bank,plan,noise


def preflight():
    test_policy()
    assert torch.cuda.is_available()
    assert not (RESULTS/"preflight.json").exists(), "Never overwrite existing experiment"
    data={}
    for seed in SEEDS:
        y,vy,vbank,plan,noise=inputs(seed)
        cfg,tr,bb,cls,CE,initial,s1=base.fresh(seed)
        base.warmup(tr,bb)
        rec,don=plan["receivers"][0,:BATCH],plan["donors"][0,:BATCH]
        aug=torch.Generator().manual_seed(seed+300009)
        a,b=augment(rec,don,aug,tr["train_transform"])
        with torch.no_grad():receiver,donor=bb(a.cuda()),bb(b.cuda())
        trace=online_masks(receiver,y[don],noise[0,:BATCH])
        selection=trace["masks"]["Low-VS"]
        full,mask=fuse(receiver,donor,selection)
        assert (mask.sum(1)==16).all()
        assert torch.equal(full[mask[:,:,None,None].expand_as(full)],
                           donor[mask[:,:,None,None].expand_as(full)])
        assert all(not p.requires_grad for p in bb.parameters())
        assert all(p.grad is None for p in cls.parameters())
        data[str(seed)]={
            "backbone_hash":initial["backbone"],
            "initial_classifier_hash":initial["classifier"],
            "stage1_sha":sha(s1),
            "first_batch_signature":base.signatures(rec,don,y,a,b,receiver,donor,noise[0,:BATCH],trace),
        }
    save_json(RESULTS/"preflight.json",{
        "status":"PASS", "protocol":PROTOCOL, "data":data,
        "train_source_sha256":sha(base.OLD/"train_images.npy"),
        "test_used":False,"validation_used_for_reference":False,
    })
    print("PREFLIGHT PASS: policy + paired source + train-only protocol",flush=True)


def prepare():
    """Freeze a clean TRAIN-only reference bank; never use validation features."""
    if not (RESULTS/"preflight.json").is_file():raise RuntimeError("preflight first")
    if BANK.exists():raise FileExistsError(BANK)
    y,vy,vbank,plan,noise=inputs(124)
    cfg,tr,bb,cls,CE,initial,s1=base.fresh(124)
    images=np.load(base.OLD/"train_images.npy",mmap_mode="r",allow_pickle=False)
    assert len(images)==N
    before=state_hash(bb)
    chunks=[]
    with torch.no_grad():
        for start in range(0,N,BATCH):
            x=torch.stack([tr["eval_transform"](Image.fromarray(im))
                           for im in images[start:start+BATCH]]).cuda()
            f=bb(x)
            gap=F.avg_pool2d(f,f.shape[-1]).flatten(1)
            chunks.append(gap.detach().cpu().numpy().astype(np.float32))
    x=np.concatenate(chunks)
    assert x.shape==(N,64) and np.isfinite(x).all()
    assert state_hash(bb)==before
    BANK.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(BANK,features=x,labels=y,
                        stage1_sha256=np.array(sha(s1)),split=np.array("TRAIN only"))
    print("PREPARE PASS:",BANK,"shape",x.shape,"sha256",sha(BANK),flush=True)


def load_reference(y,s1):
    if not BANK.is_file():raise FileNotFoundError("Prepare TRAIN reference bank first")
    data=np.load(BANK,allow_pickle=False)
    x=data["features"];label=data["labels"]
    assert x.shape==(N,64) and np.array_equal(label,y)
    assert str(data["stage1_sha256"])==sha(s1)
    assert str(data["split"])=="TRAIN only"
    assert np.isfinite(x).all()
    return torch.from_numpy(x.copy()).cuda(),torch.from_numpy(y.copy()).long().cuda()


def train(seed,arm):
    assert seed in SEEDS and arm in ARMS
    pre=RESULTS/"preflight.json"
    if not pre.is_file():raise RuntimeError("preflight first")
    assert json.loads(pre.read_text(encoding="utf-8"))["status"]=="PASS"
    y,vy,vbank,plan,noise=inputs(seed)
    output=RESULTS/f"seed{seed}"/arm
    if output.exists():raise FileExistsError(f"No overwrite/resume: {output}")
    cfg,tr,bb,cls,CE,initial,s1=base.fresh(seed)
    assert initial["backbone"]==json.loads(pre.read_text(encoding="utf-8"))["data"][str(seed)]["backbone_hash"]
    assert initial["classifier"]==json.loads(pre.read_text(encoding="utf-8"))["data"][str(seed)]["initial_classifier_hash"]
    reference_x,reference_y=load_reference(y,s1)
    weights_initial={k:v.detach().cpu().clone() for k,v in bb.state_dict().items()}
    base.warmup(tr,bb)
    generator=torch.Generator().manual_seed(seed+300009)
    optimizer=optimizer_for(cfg,cls)
    criterion=CE(cfg,np.bincount(y,minlength=100)).cuda()
    adjust=scheduler()
    previous=None
    if arm!="LowVS-Full":
        p=RESULTS/f"seed{seed}"/"LowVS-Full"/"batch_signatures.csv"
        if not p.is_file():raise FileNotFoundError("Run paired LowVS-Full first: "+str(p))
        previous=read_csv(p);assert len(previous)==850
    output.mkdir(parents=True)
    (output/"checkpoints").mkdir()
    save_json(output/"config.json",{**PROTOCOL,"seed":seed,"arm":arm,
              "bank_sha256":sha(BANK),"stage1_sha256":sha(s1),
              "initial_hashes":initial})
    rows=[];signatures=[];best=-1.0;best_epoch=0;total_steps=0;pair_count=0
    with (output/"train.log").open("w",encoding="utf-8",buffering=1) as log:
        for epoch in range(10):
            cls.train()
            adjust(optimizer,epoch,cfg)
            gW,gb=reference_gradient(reference_x,reference_y,cls)
            assert torch.isfinite(gW).all() and torch.isfinite(gb).all()
            nr=np.zeros(3,dtype=np.int64)
            advantage_sum=0.;advantage_n=0;train_loss=0.
            for batch,start in enumerate(range(0,N,BATCH)):
                end=min(start+BATCH,N)
                rec=plan["receivers"][epoch,start:end]
                don=plan["donors"][epoch,start:end]
                target=torch.from_numpy(y[rec].copy()).long().cuda()
                ri,di=augment(rec,don,generator,tr["train_transform"])
                with torch.no_grad():
                    receiver,donor=bb(ri.cuda()),bb(di.cuda())
                g=noise[epoch,start:end]
                trace=online_masks(receiver,y[don],g)
                full,mask=fuse(receiver,donor,trace["masks"]["Low-VS"])
                sig=base.signatures(rec,don,y,ri,di,receiver,donor,g,trace)
                if previous is not None:
                    for key in KEYS:
                        if sig[key]!=previous[total_steps][key]:
                            raise RuntimeError(f"paired batch mismatch seed={seed} "
                                               f"arm={arm} epoch={epoch+1} batch={batch} key={key}")
                        pair_count+=1
                with torch.no_grad():
                    if arm=="LowVS-Full":
                        action=torch.full((len(rec),),2,device="cuda",dtype=torch.uint8)
                    elif arm=="LowVS-Half":
                        action=torch.full((len(rec),),1,device="cuda",dtype=torch.uint8)
                    else:
                        action,scores=utility_actions(receiver,full,target,cls,gW,gb)
                        gain=scores.max(1).values-scores[:,0]
                        advantage_sum+=float(gain.sum());advantage_n+=len(rec)
                        if arm=="GU-Shuffled":
                            # Same action histogram for this batch, but recipients permuted
                            # deterministically without touching shared torch/augmentation RNG.
                            perm=np.random.default_rng(
                                seed*100000+epoch*1000+batch).permutation(len(rec))
                            action=action[torch.as_tensor(perm,device=action.device)]
                    nr+=np.bincount(action.cpu().numpy(),minlength=3)
                    mixed=apply_actions(receiver,full,action)
                    if arm=="LowVS-Full":assert torch.equal(mixed,full)
                    if arm=="LowVS-Half":assert torch.equal(mixed,0.5*(receiver+full))
                loss=criterion(cls(mixed),target,extra_info=None)
                assert torch.isfinite(loss)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in cls.parameters())
                optimizer.step()
                train_loss+=float(loss.detach())*len(rec)
                signatures.append({"epoch":epoch+1,"batch":batch,
                                   "size":len(rec),**sig,
                                   "alpha0":int((action==0).sum()),
                                   "alpha_half":int((action==1).sum()),
                                   "alpha_full":int((action==2).sum())})
                total_steps+=1
            assert nr.sum()==N
            check_frozen(bb,weights_initial)
            m,logits,pred=evaluate(cls,vy,feature_maps=vbank)
            if m["Overall"]>best:
                best=m["Overall"];best_epoch=epoch+1
                torch.save({"seed":seed,"arm":arm,"epoch":epoch+1,
                    "classifier":{k:v.detach().cpu().clone() for k,v in cls.state_dict().items()},
                    "test_used":False},output/"checkpoints/best.pth")
            if epoch==9:
                torch.save({"seed":seed,"arm":arm,"epoch":10,
                    "classifier":{k:v.detach().cpu().clone() for k,v in cls.state_dict().items()},
                    "test_used":False},output/"checkpoints/final.pth")
                np.savez_compressed(output/"validation_epoch10.npz",
                                    labels=vy,logits=logits,predictions=pred)
            record={"seed":seed,"arm":arm,"epoch":epoch+1,"train_loss":train_loss/N,
                    "alpha0_pct":100*nr[0]/N,"alphaHalf_pct":100*nr[1]/N,
                    "alphaFull_pct":100*nr[2]/N,
                    "mean_predicted_gain":advantage_sum/advantage_n if advantage_n else None,
                    **{key:m[key] for key in METRICS}}
            rows.append(record)
            save_csv(output/"epochs.csv",rows)
            save_csv(output/"batch_signatures.csv",signatures)
            line=(f"seed{seed} {arm} ep{epoch+1}/10 "
                  f"overall={m['Overall']:.4f} Tail={m['Tail']:.4f} "
                  f"Head={m['Head']:.4f} actions={nr.tolist()}")
            print(line,flush=True);log.write(line+"\n")
    assert total_steps==850 and len(signatures)==850
    assert pair_count==(0 if arm=="LowVS-Full" else 850*len(KEYS))
    assert state_hash(bb)==initial["backbone"]
    for name in ("best","final"):
        ck=torch.load(output/f"checkpoints/{name}.pth",map_location="cpu",weights_only=False)
        assert ck["arm"]==arm and ck["test_used"] is False
        cls.load_state_dict(ck["classifier"],strict=True)
        metrics,_,_=evaluate(cls,vy,feature_maps=vbank)
        expected=rows[ck["epoch"]-1]
        for key in METRICS:
            assert abs(metrics[key]-expected[key])<1e-9,(arm,name,key)
    save_json(output/"summary.json",{
        "seed":seed,"arm":arm,"status":"complete","updates":850,
        "last":rows[-1],"best_epoch":best_epoch,"best_Overall":best,
        "stage1_sha256":sha(s1),"train_bank_sha256":sha(BANK),
        "paired_fields_verified":pair_count,"frozen_backbone":state_hash(bb)==initial["backbone"],
        "official_TEST_used":False})
    print("TRAIN PASS",seed,arm,flush=True)


def report():
    rows=[];paired=[];per_class=[]
    for seed in SEEDS:
        reference=None;by_arm={};labels=None
        for arm in ARMS:
            folder=RESULTS/f"seed{seed}"/arm
            summ=json.loads((folder/"summary.json").read_text(encoding="utf-8"))
            assert summ["status"]=="complete" and summ["updates"]==850
            sig=read_csv(folder/"batch_signatures.csv")
            if reference is None:reference=sig
            else:
                assert len(reference)==len(sig)==850
                for old,new in zip(reference,sig):
                    assert all(old[k]==new[k] for k in KEYS),(seed,arm,"pairing")
            v=np.load(folder/"validation_epoch10.npz",allow_pickle=False)
            p=v["predictions"];yy=v["labels"]
            if labels is None:labels=yy.copy()
            else:assert np.array_equal(labels,yy)
            assert np.array_equal(p,v["logits"].argmax(1))
            by_arm[arm]=(summ,p)
            rows.append({"seed":seed,"arm":arm,**{k:summ["last"][k] for k in METRICS},
                        "best_epoch":summ["best_epoch"],"best_Overall":summ["best_Overall"],
                        "alpha0_pct":summ["last"]["alpha0_pct"],
                        "alphaHalf_pct":summ["last"]["alphaHalf_pct"],
                        "alphaFull_pct":summ["last"]["alphaFull_pct"]})
        for a in ARMS[1:]:
            for b in ("LowVS-Full","LowVS-Half","GU-Shuffled"):
                if a==b:continue
                sa,pa=by_arm[a];sb,pb=by_arm[b]
                win=(pb!=labels)&(pa==labels)
                lose=(pb==labels)&(pa!=labels)
                tail=(labels>=71)&(labels<=99)
                head=(labels>=1)&(labels<=35)
                delta={"seed":seed,"comparison":f"{a} - {b}",
                       "wins":int(win.sum()),"losses":int(lose.sum()),
                       "net":int(win.sum()-lose.sum()),
                       "tail_net":int(win[tail].sum()-lose[tail].sum()),
                       "head_net":int(win[head].sum()-lose[head].sum()),
                       **{f"d_{k}_pp":sa["last"][k]-sb["last"][k] for k in
                          ("Overall","Tail","Few","Head")}}
                paired.append(delta)
                for c in range(1,100):
                    yy=labels==c
                    per_class.append({"seed":seed,"comparison":f"{a} - {b}",
                                      "class_id":c,"net_correct":int((pa[yy]==labels[yy]).sum()-
                                                               (pb[yy]==labels[yy]).sum())})
    save_csv(RESULTS/"epoch10.csv",rows)
    save_csv(RESULTS/"paired.csv",paired)
    save_csv(RESULTS/"per_class.csv",per_class)
    def avg(comp,metric):
        vals=[p[metric] for p in paired if p["comparison"]==comp]
        assert len(vals)==2
        return sum(vals)/2
    core="GU-LowVS - LowVS-Full"
    control="GU-LowVS - GU-Shuffled"
    mean_full=avg(core,"d_Overall_pp");mean_perm=avg(control,"d_Overall_pp")
    gu_rows=[r for r in rows if r["arm"]=="GU-LowVS"]
    not_collapsed=all(max(r["alpha0_pct"],r["alphaHalf_pct"],r["alphaFull_pct"])<95.0
                      for r in gu_rows)
    gate=(mean_full>0 and mean_perm>0 and
          avg(core,"d_Tail_pp")>=0 and avg(control,"d_Head_pp")>=-0.15 and
          all(r["d_Overall_pp"]>=0 for r in paired if r["comparison"]==core)
          and not_collapsed)
    summary={"status":"complete","protocol":PROTOCOL,"epoch10":rows,
             "pairwise":paired,"go_exploratory":gate,
             "mean_gain_vs_full":mean_full,"mean_gain_vs_shuffled":mean_perm,
             "head_delta_vs_shuffled":avg(control,"d_Head_pp"),
             "utility_noncollapsed":not_collapsed,
             "assessment":"development-only: two StageII seeds, same StageI, same reused TRAIN validation"}
    save_json(RESULTS/"scientific_summary.json",summary)
    lines=["# GU-LowVS V1 | primary epoch10, Clean TRAIN validation",
           "StageI frozen, stageII seeds124/125; 99 classes, 1980 TRAIN holdout.",
           "Official TEST not used. Clean reference gradient uses TRAIN only.",
           "All paired per-batch indices/images/features/VS scores/Gumbel verified.",
           "", "|Seed|Arm|Overall|Head|Tail|Few|Alpha0%|Alpha0.5%|Alpha1%|Best ep|",
           "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"|{r['seed']}|{r['arm']}|{r['Overall']:.3f}|{r['Head']:.3f}|"
                     f"{r['Tail']:.3f}|{r['Few']:.3f}|{r['alpha0_pct']:.1f}|"
                     f"{r['alphaHalf_pct']:.1f}|{r['alphaFull_pct']:.1f}|{r['best_epoch']}|")
    lines+=["",f"Exploratory method gate: {'PASS' if gate else 'NO-GO'}",
            f"Mean Overall vs full: {mean_full:+.4f}pp; vs shuffled: {mean_perm:+.4f}pp",
            "Read paired.csv + per_class.csv for actual wins/losses, not only means.",
            "High clean TRAIN gradient alignment is NOT proof of unseen utility.",
            "If GU loses to shuffled or almost always chooses alpha=0, stop this V1.",
            "Do not tune candidate alphas, class weights, or reference bank on this validation set.",
            "No additional StageI retraining and no claims about learned backbone features."]
    (RESULTS/"GU_LOWVS_V1_RESULTS.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print("REPORT PASS",RESULTS/"GU_LOWVS_V1_RESULTS.md",flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("action",choices=("preflight","prepare","train","report"))
    parser.add_argument("--seed",type=int,choices=SEEDS)
    parser.add_argument("--arm",choices=ARMS)
    args=parser.parse_args()
    base.forbid_official_test()
    protect(PARENT,HERE,HISTORY_EVENTS)
    if args.action=="preflight":preflight()
    elif args.action=="prepare":prepare()
    elif args.action=="train":
        if args.seed is None or args.arm is None:parser.error("train requires --seed and --arm")
        train(args.seed,args.arm)
    else:report()
    base.check_test_guard()
    assert not HISTORY_EVENTS

if __name__=="__main__":
    main()
