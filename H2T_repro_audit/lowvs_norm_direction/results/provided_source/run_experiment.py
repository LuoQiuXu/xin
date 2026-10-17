"""Four-arm matched GAP-L2 study: original Uniform/LowVS vs paired norm controls.

From repository root: run_experiment.py preflight; train --seed ... --arm ...;
run_experiment.py report. See CODEX_LOWVS_NORM_DIRECTION.txt for exact order.
Source artifacts are READ ONLY. Writes to this directory's results/ only.
"""
from __future__ import annotations
import argparse,csv,json,sys
from pathlib import Path
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"ravs_v1"))
import run_ravs_v1 as old
from common import N,BATCH,sha,state_hash,evaluate,check_frozen,array_hash,metrics
from stability_models import check_sources
from reused_train_helpers import augment,scheduler,optimizer_for
from online_policy import online_masks
from norm_policy import ARMS,paired_norm_control,test_policy
import numpy as np
import torch

SEEDS=(124,125)
RESULTS=HERE/"results"
PAIR_KEYS=("indices_sha256","images_sha256","features_sha256",
           "scores_sha256","gumbel_sha256")
METRICS=("Overall","Head","OfficialMedium","Tail","Many","Medium",
         "Few","Macro","validation_CE")
PROTOCOL={
    "question":"After equal GAP perturbation magnitude, does LowVS beat Uniform?",
    "seeds":list(SEEDS),"arms":list(ARMS),"stage1":"Clean seed42 epoch184",
    "epochs":10,"primary_epoch":10,"batch_size":128,
    "stage1_backbone_bn":"frozen eval","only_train":"100-way linear FC",
    "score_and_masks":"reviewed online VS: Uniform64/Low48 -> random16 whole channels",
    "target_norm":"r=min(norm(GAP(deltaU_full)),norm(GAP(deltaL_full)))",
    "scaling":"alphaU=r/nU, alphaL=r/nL; alpha in [0,1], zero-norm safe",
    "controls":"paired BS/IS, augmentations, VS scores, both masks and Gumbel",
    "validation":"1980 TRAIN holdout, class1..99; class0 absent, 100 logits",
    "official_TEST_used":False,
    "interpretation":"GAP L2 matched, NOT guaranteed feature-map L2 or semantic orientation match",
}

def write_json(p,obj):
    p=Path(p);assert p.resolve().is_relative_to(RESULTS.resolve())
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")

def write_csv(p,rows):
    p=Path(p);assert rows and p.resolve().is_relative_to(RESULTS.resolve())
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def read_csv(p):
    with Path(p).open("r",encoding="utf-8",newline="") as f:
        return list(csv.DictReader(f))

def source(seed):
    old.inputs(seed);check_sources()
    return old.load_data(seed)

def fused(receiver,donor,trace):
    assert receiver.shape==donor.shape and receiver.shape[1:]==(64,8,8)
    masks={}
    for name,key in (("uniform","Uniform-PerSample"),("lowvs","Low-VS")):
        indices=trace["masks"][key].astype(np.int64,copy=True)
        assert indices.shape==(len(receiver),16)
        m=torch.zeros((len(receiver),64),device=receiver.device,dtype=torch.bool)
        m.scatter_(1,torch.from_numpy(indices).to(receiver.device),True)
        masks[name]=m
    return paired_norm_control(receiver,donor,masks["uniform"],masks["lowvs"])

def preflight():
    test_policy()
    if not torch.cuda.is_available():raise RuntimeError("CUDA required")
    if (RESULTS/"preflight.json").exists():raise FileExistsError("preflight already exists")
    checks={}
    for seed in SEEDS:
        y,vy,vbank,plan,noise=source(seed)
        cfg,tr,bb,cls,CE,initial,s1=old.fresh(seed)
        frozen={k:v.detach().cpu().clone() for k,v in bb.state_dict().items()}
        old.warmup(tr,bb)
        aug=torch.Generator().manual_seed(seed+300009)
        rec,don=plan["receivers"][0,:BATCH],plan["donors"][0,:BATCH]
        ri,di=augment(rec,don,aug,tr["train_transform"])
        with torch.no_grad():
            receiver,donor=bb(ri.cuda()),bb(di.cuda())
            trace=online_masks(receiver,y[don],noise[0,:BATCH])
            mixtures,d=fused(receiver,donor,trace)
            tgt=torch.from_numpy(y[rec].copy()).long().cuda()
            ce=CE(cfg,np.bincount(y,minlength=100)).cuda()
            losses={a:float(ce(cls(v),tgt,extra_info=None)) for a,v in mixtures.items()}
        assert all(np.isfinite(v) for v in losses.values())
        check_frozen(bb,frozen)
        assert state_hash(cls)==initial["classifier"]
        checks[str(seed)]={"initial":initial,"stage1_sha256":sha(s1),
            "first_batch_signature":old.signatures(rec,don,y,ri,di,receiver,donor,noise[0,:BATCH],trace),
            "no_grad_losses":losses,
            "max_gap_norm_discrepancy":float((d["matched_norm_uniform"]-d["matched_norm_lowvs"]).abs().max())}
    write_json(RESULTS/"preflight.json",{"status":"PASS","protocol":PROTOCOL,"checks":checks})
    old.check_test_guard();print("PREFLIGHT PASS",flush=True)

def train(seed,arm):
    assert seed in SEEDS and arm in ARMS
    pre=RESULTS/"preflight.json"
    if not pre.is_file():raise FileNotFoundError("Run preflight first")
    expected=json.loads(pre.read_text(encoding="utf-8"))
    y,vy,vbank,plan,noise=source(seed)
    folder=RESULTS/f"seed{seed}"/arm
    if folder.exists():raise FileExistsError(f"Refuse overwrite {folder}")
    cfg,tr,bb,cls,CE,initial,s1=old.fresh(seed)
    assert initial==expected["checks"][str(seed)]["initial"]
    opt=optimizer_for(cfg,cls);assert not opt.state
    frozen={k:v.detach().cpu().clone() for k,v in bb.state_dict().items()}
    old.warmup(tr,bb)
    aug=torch.Generator().manual_seed(seed+300009)
    criterion=CE(cfg,np.bincount(y,minlength=100)).cuda()
    schedule=scheduler()
    control=None
    if arm!="Uniform-Full":
        p=RESULTS/f"seed{seed}"/"Uniform-Full"/"batch_signatures.csv"
        if not p.is_file():raise FileNotFoundError("Run Uniform-Full first: "+str(p))
        control=read_csv(p);assert len(control)==850
    folder.mkdir(parents=True)
    (folder/"checkpoints").mkdir()
    write_json(folder/"config.json",{**PROTOCOL,"seed":seed,"arm":arm,
        "stage1_sha256":sha(s1),"statistics_sha256":sha(old.OLD/"class_feature_statistics.npz"),
        "initial_hashes":initial})
    (folder/"effective.yaml").write_text(cfg.dump(),encoding="utf-8")
    rows=[];batches=[];best=-1.0;best_epoch=0
    for epoch in range(10):
        cls.train();schedule(opt,epoch,cfg)
        loss_sum=0.0
        keys=("original_norm_uniform","original_norm_lowvs","matched_norm_uniform",
              "matched_norm_lowvs","alpha_uniform","alpha_lowvs","target_norm","zero_norm_pairs")
        agg={k:0.0 for k in keys}
        head_alpha=[0.0,0.0,0];tail_alpha=[0.0,0.0,0]
        for batch,start in enumerate(range(0,N,BATCH)):
            end=min(N,start+BATCH)
            rec,don=plan["receivers"][epoch,start:end],plan["donors"][epoch,start:end]
            tgt=torch.from_numpy(y[rec].copy()).long().cuda()
            ri,di=augment(rec,don,aug,tr["train_transform"])
            with torch.no_grad():
                receiver,donor=bb(ri.cuda()),bb(di.cuda())
                g=noise[epoch,start:end]
                trace=online_masks(receiver,y[don],g)
                mixes,diag=fused(receiver,donor,trace)
            sig=old.signatures(rec,don,y,ri,di,receiver,donor,g,trace)
            sig["uniform_mask_sha256"]=array_hash(trace["masks"]["Uniform-PerSample"])
            sig["lowvs_mask_sha256"]=array_hash(trace["masks"]["Low-VS"])
            if control is not None:
                baseline=control[len(batches)]
                for k in (*PAIR_KEYS,"uniform_mask_sha256","lowvs_mask_sha256"):
                    if sig[k]!=baseline[k]:
                        raise RuntimeError(f"Pairing mismatch seed{seed} {arm} e{epoch+1} b{batch}: {k}")
            logits=cls(mixes[arm]);loss=criterion(logits,tgt,extra_info=None)
            assert torch.isfinite(loss).item() and torch.isfinite(logits).all()
            opt.zero_grad(set_to_none=True);loss.backward();opt.step()
            loss_sum+=float(loss.detach())*len(rec)
            for k in keys:agg[k]+=float(diag[k].sum())
            for cond,group in ((y[rec]<=35,head_alpha),(y[rec]>=71,tail_alpha)):
                idx=torch.as_tensor(cond,device=receiver.device)
                group[0]+=float(diag["alpha_uniform"][idx].sum())
                group[1]+=float(diag["alpha_lowvs"][idx].sum())
                group[2]+=int(idx.sum())
            batches.append({"epoch":epoch+1,"batch":batch,"size":end-start,**sig})
        check_frozen(bb,frozen)
        val,vl,vp=evaluate(cls,vy,feature_maps=vbank)
        if val["Overall"]>best:
            best=val["Overall"];best_epoch=epoch+1
            torch.save({"epoch":epoch+1,"arm":arm,"seed":seed,
                "classifier":{k:v.detach().cpu().clone() for k,v in cls.state_dict().items()},
                "TEST_used":False},folder/"checkpoints/best.pth")
        if epoch==9:
            torch.save({"epoch":10,"arm":arm,"seed":seed,
                "classifier":{k:v.detach().cpu().clone() for k,v in cls.state_dict().items()},
                "TEST_used":False},folder/"checkpoints/final.pth")
            np.savez_compressed(folder/"validation_epoch10.npz",labels=vy,logits=vl,predictions=vp)
        row={"seed":seed,"arm":arm,"epoch":epoch+1,"train_loss":loss_sum/N,
             **{k:val[k] for k in METRICS},
             **{"mean_"+k:agg[k]/N for k in keys},
             "head_alpha_uniform":head_alpha[0]/max(1,head_alpha[2]),
             "head_alpha_lowvs":head_alpha[1]/max(1,head_alpha[2]),
             "tail_alpha_uniform":tail_alpha[0]/max(1,tail_alpha[2]),
             "tail_alpha_lowvs":tail_alpha[1]/max(1,tail_alpha[2])}
        rows.append(row);write_csv(folder/"epochs.csv",rows)
        print(f"seed{seed} {arm} epoch {epoch+1}/10 Overall={val['Overall']:.4f} Tail={val['Tail']:.4f}",flush=True)
    assert len(batches)==850
    assert state_hash(bb)==initial["backbone"]
    write_csv(folder/"batch_signatures.csv",batches)
    for which in ("best","final"):
        ck=torch.load(folder/f"checkpoints/{which}.pth",map_location="cpu",weights_only=False)
        assert ck["arm"]==arm and not ck["TEST_used"]
        cls.load_state_dict(ck["classifier"],strict=True)
        check,_,_=evaluate(cls,vy,feature_maps=vbank)
        assert all(abs(check[k]-rows[ck["epoch"]-1][k])<1e-9 for k in METRICS)
    write_json(folder/"summary.json",{"status":"completed","seed":seed,"arm":arm,
        "steps":850,"best_epoch":best_epoch,"best_Overall":best,
        "last":rows[-1],"TEST_used":False,"backbone_frozen":True,"all_batches_paired":control is not None})
    old.check_test_guard();print("TRAIN PASS",seed,arm,flush=True)

def weights(path):
    ck=torch.load(path,map_location="cpu",weights_only=False)["classifier"]
    w=ck["module.fc.weight"].numpy().astype(np.float64)
    b=ck["module.fc.bias"].numpy().astype(np.float64)
    assert w.shape==(100,64) and b.shape==(100,)
    return w,b

def report():
    all_rows=[];pairs=[];hybrids=[]
    valbank=np.load(old.OLD/"validation_feature_maps.npy",mmap_mode="r",allow_pickle=False)
    gap=np.asarray(valbank.mean(axis=(2,3)),dtype=np.float64)
    for seed in SEEDS:
        by={};reference=None;y=None
        for arm in ARMS:
            folder=RESULTS/f"seed{seed}"/arm
            s=json.loads((folder/"summary.json").read_text(encoding="utf-8"))
            assert s["status"]=="completed" and s["steps"]==850
            v=np.load(folder/"validation_epoch10.npz",allow_pickle=False)
            yy=v["labels"]
            if y is None:y=yy.copy()
            else:assert np.array_equal(y,yy)
            p=v["predictions"];assert np.array_equal(p,v["logits"].argmax(1))
            trace=read_csv(folder/"batch_signatures.csv")
            assert len(trace)==850
            if reference is None:reference=trace
            else:assert all(all(a[k]==b[k] for k in (*PAIR_KEYS,"uniform_mask_sha256","lowvs_mask_sha256"))
                            for a,b in zip(reference,trace))
            w,b=weights(folder/"checkpoints/final.pth")
            assert np.allclose(gap@w.T+b,v["logits"],atol=1e-4,rtol=1e-4)
            by[arm]=(s,p,w,b)
            all_rows.append({"seed":seed,"arm":arm,**{k:s["last"][k] for k in METRICS},
                "best_epoch":s["best_epoch"],"best_Overall":s["best_Overall"],
                **{k:s["last"][k] for k in ("mean_original_norm_uniform","mean_original_norm_lowvs",
                    "mean_matched_norm_uniform","mean_matched_norm_lowvs",
                    "mean_alpha_uniform","mean_alpha_lowvs")}})
        for a,b in (("LowVS-Full","Uniform-Full"),("LowVS-Matched","Uniform-Matched"),
                    ("LowVS-Matched","LowVS-Full"),("Uniform-Matched","Uniform-Full")):
            sa,pa,_,_=by[a];sb,pb,_,_=by[b]
            wins=(pa==y)&(pb!=y);losses=(pa!=y)&(pb==y)
            ta=(y>=71);he=(y<=35)
            pairs.append({"seed":seed,"comparison":a+" - "+b,
                "wins":int(wins.sum()),"losses":int(losses.sum()),
                "net_correct":int(wins.sum()-losses.sum()),
                "tail_net":int(wins[ta].sum()-losses[ta].sum()),
                "head_net":int(wins[he].sum()-losses[he].sum()),
                **{"delta_"+k+"_pp":sa["last"][k]-sb["last"][k] for k in ("Overall","Tail","Few","Head")}})
        for a,b in (("LowVS-Full","Uniform-Full"),("LowVS-Matched","Uniform-Matched")):
            _,_,wu,bu=by[b];_,_,wl,bl=by[a]
            for name,w,bias in (("U_W+U_b",wu,bu),("L_W+L_b",wl,bl),
                                ("U_W+L_b",wu,bl),("L_W+U_b",wl,bu)):
                pred=(gap@w.T+bias).argmax(1)
                m=metrics(pred,y)
                hybrids.append({"seed":seed,"comparison":a+" vs "+b,"hybrid":name,
                    **{k:m[k] for k in ("Overall","Head","Tail","Few")},
                    "weight_frobenius":float(np.linalg.norm(w)),
                    "bias_l2":float(np.linalg.norm(bias))})
    write_csv(RESULTS/"epoch10.csv",all_rows)
    write_csv(RESULTS/"paired.csv",pairs)
    write_csv(RESULTS/"classifier_hybrids.csv",hybrids)
    write_json(RESULTS/"scientific_summary.json",{
        "protocol":PROTOCOL,"results":all_rows,"paired":pairs,"hybrids":hybrids,"TEST_used":False,
        "hybrid_warning":"W-b swaps are diagnostic hybrids, not identified causal contributions"})
    lines=["# Low-VS vs Uniform: norm matching vs direction",
        "", "Locked Clean StageI, fixed epoch10, 1980 reused TRAIN validation images (class0 absent).",
        "Same input/VS/masks checked across four arms, 850 batches per seed. No official TEST.",
        "", "|Seed|Arm|Overall|Head|Tail|Few|Avg alphaU|Avg alphaL|",
        "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in all_rows:lines.append(
        f"|{r['seed']}|{r['arm']}|{r['Overall']:.4f}|{r['Head']:.4f}|{r['Tail']:.4f}|{r['Few']:.4f}|{r['mean_alpha_uniform']:.3f}|{r['mean_alpha_lowvs']:.3f}|")
    lines+=["","## Paired changes, fixed epoch10",
        "|Seed|Contrast|Overall pp|Tail pp|Head pp|Net images|",
        "|---|---|---:|---:|---:|---:|"]
    for r in pairs:lines.append(
        f"|{r['seed']}|{r['comparison']}|{r['delta_Overall_pp']:+.4f}|{r['delta_Tail_pp']:+.4f}|{r['delta_Head_pp']:+.4f}|{r['net_correct']:+d}|")
    lines+=["","Read paired.csv and classifier_hybrids.csv before attributing mechanisms.",
        "Key comparison: LowVS-Matched vs Uniform-Matched, next to the two Full arms.",
        "Matched GAP norm does not match full-map L2, spatial structure, or gradient distributions.",
        "If matched advantage persists, GAP norm alone is insufficient, NOT proof of useful semantics.",
        "If matched advantage vanishes, norm is plausible, NOT uniquely established.",
        "Do not sweep strengths or masks on this repeatedly-used dev validation split.",""]
    (RESULTS/"NORM_DIRECTION_RESULTS.md").write_text("\n".join(lines),encoding="utf-8")
    old.check_test_guard();print("REPORT PASS",flush=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("action",choices=("preflight","train","report"))
    p.add_argument("--seed",type=int,choices=SEEDS)
    p.add_argument("--arm",choices=ARMS)
    q=p.parse_args()
    old.forbid_official_test()
    if q.action=="preflight":preflight()
    elif q.action=="train":
        if q.seed is None or q.arm is None:p.error("train requires --seed and --arm")
        train(q.seed,q.arm)
    else:report()
    old.check_test_guard()

if __name__=="__main__":main()
