"""Low-VS fixed augmentation + tail clean discriminative preservation.

NO re-training StageI.  No mask/VS/alpha changes. Three arms are:
 LowVS | LowVS-CleanCE | LowVS-TopCompetitor.
First train LowVS for each seed; the other arms are fully paired 850-batch
replays against the fresh in-round baseline and the prior LowVS-Full replay.
The repeat of historical LowVS is a strong positive-control reproducibility test.
Outputs only to lowvs_disc_preserve/results/. Official CIFAR TEST prohibited.
"""
from __future__ import annotations
import argparse,csv,json,sys,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
BASE=HERE.parent
sys.path.insert(0,str(BASE/"ravs_v1"))
import run_ravs_v1 as old
from common import N,BATCH,sha,array_hash,state_hash,evaluate,metrics,check_frozen
from stability_models import check_sources
from reused_train_helpers import augment,fuse,scheduler,optimizer_for
from online_policy import online_masks
from losses import ARMS,LAMBDA,loss_terms,self_test

SEEDS=(124,125)
RESULTS=HERE/"results"
HISTORY=BASE/"lowvs_norm_direction"/"results"
PAIR_KEYS=("indices_sha256","images_sha256","features_sha256",
           "scores_sha256","gumbel_sha256","uniform_mask_sha256",
           "lowvs_mask_sha256")
METRICS=("Overall","Head","OfficialMedium","Tail","Many","Medium","Few",
         "Macro","validation_CE")
PROTOCOL={
 "question":"Does clean tail nearest-competitor discrimination add value beyond extra clean CE on a frozen Low-VS sampler?",
 "arms":list(ARMS),"seeds":list(SEEDS),"stage1":"CLEAN seed42 epoch184 frozen all backbone/BN",
 "loss":"fused 100-class CE always; auxiliary only tail IDs>=71 on clean receiver; batch-normalized sum",
 "lambda":LAMBDA,"margin":0.0,"competitor":"current clean logits, detach argmax excluding target class",
 "control":"additional CleanCE uses same tail subset, unperturbed receiver and lambda as competitor arm",
 "immutable_lowvs":"original online donor-stat Bottom48 -> random16 hard full-channel swaps, fallback n<20",
 "train":"same BS receiver/IS donor, crops/flips, Gumbel, seed, 850 steps, SGD/Cosine as historical Clean",
 "primary":"TopCompetitor - CleanCE on epoch10 Tail, Overall, Head; paired validation predictions",
 "secondary":"CleanCE - LowVS; TopCompetitor - LowVS",
 "holdout":"1980 original TRAIN val images (20 each classes1..99), no class0; full 100 logits",
 "scope":"pilot only; one StageI, two correlated stageII runs; not independent test",
 "official_TEST_used":False,
 "limitations":"Aux CE and pairwise competitor softplus have different gradient geometry and scale; fixed lambda is not gradient norm matching. Pairwise competitor loss is not by itself novel."
}

def jwrite(path,obj):
    path=Path(path)
    assert path.resolve().is_relative_to(RESULTS.resolve())
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")

def cwrite(path,rows):
    path=Path(path)
    assert path.resolve().is_relative_to(RESULTS.resolve()) and rows
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)

def cread(path):
    with Path(path).open(encoding="utf-8",newline="") as f:
        return list(csv.DictReader(f))

def file_hashes():
    return {p:sha(HERE/p) for p in ("run_pilot.py","losses.py")}

def oldsignatures(seed):
    path=HISTORY/f"seed{seed}/LowVS-Full/batch_signatures.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing SEALED history {path}; restore old CSV, don't manufacture history")
    rows=cread(path)
    assert len(rows)==850 and all(k in rows[0] for k in PAIR_KEYS)
    return path,rows

def input_data(seed):
    old.inputs(seed);check_sources()
    return old.load_data(seed)

def signatures(rec,don,y,ri,di,receiver,donor,g,trace):
    sig=old.signatures(rec,don,y,ri,di,receiver,donor,g,trace)
    sig["uniform_mask_sha256"]=array_hash(trace["masks"]["Uniform-PerSample"])
    sig["lowvs_mask_sha256"]=array_hash(trace["masks"]["Low-VS"])
    return sig

def equal_signatures(current,control,seed,arm,epoch,batch):
    for k in PAIR_KEYS:
        if current[k]!=control[k]:
            raise RuntimeError(f"PAIRED INPUT/MASK mismatch seed{seed} {arm} epoch{epoch} batch{batch} field{k}; STOP.")

def forward_pair(bb,ri,di):
    with torch.no_grad():
        receiver=bb(ri.cuda());donor=bb(di.cuda())
    return receiver,donor

def preflight():
    old.forbid_official_test()
    self_test()
    if not torch.cuda.is_available():raise RuntimeError("CUDA is required")
    if (RESULTS/"preflight.json").exists():raise FileExistsError("Cannot replace preflight")
    splits=np.load(old.OLD/"split_indices.npz",allow_pickle=False)
    assert not np.intersect1d(splits["train_raw_indices"],splits["validation_raw_indices"]).size
    checks={}
    for seed in SEEDS:
        _,historic=oldsignatures(seed)
        y,vy,vbank,plan,noise=input_data(seed)
        cfg,tr,bb,cls,CE,initial,s1=old.fresh(seed)
        saved={k:v.detach().cpu().clone() for k,v in bb.state_dict().items()}
        old.warmup(tr,bb)
        aug=torch.Generator().manual_seed(seed+300009)
        rec,don=plan["receivers"][0,:BATCH],plan["donors"][0,:BATCH]
        ri,di=augment(rec,don,aug,tr["train_transform"])
        receiver,donor=forward_pair(bb,ri,di)
        trace=online_masks(receiver,y[don],noise[0,:BATCH])
        sig=signatures(rec,don,y,ri,di,receiver,donor,noise[0,:BATCH],trace)
        equal_signatures(sig,historic[0],seed,"preflight",1,0)
        fused,mask=fuse(receiver,donor,trace["masks"]["Low-VS"])
        chosen=mask[...,None,None].expand_as(receiver)
        assert torch.equal(fused[chosen],donor[chosen])
        assert torch.equal(fused[~chosen],receiver[~chosen])
        with torch.no_grad():
            y_t=torch.from_numpy(y[rec].copy()).long().cuda()
            losses={}
            for arm in ARMS:
                total,diag=loss_terms(arm,cls(fused),cls(receiver),y_t)
                assert torch.isfinite(total)
                losses[arm]=float(total)
        check_frozen(bb,saved)
        assert state_hash(cls)==initial["classifier"]
        checks[str(seed)]={"initial":initial,"stage1_sha256":sha(s1),
             "historical_signature_sha256":sha(HISTORY/f"seed{seed}/LowVS-Full/batch_signatures.csv"),
             "first_batch_signature":sig,"losses":losses,"tail_count":int((y_t>=71).sum())}
    jwrite(RESULTS/"preflight.json",{"passed":True,"checks":checks,"source_files":file_hashes(),
        "protocol":PROTOCOL,"test_guard_events":old.AUDIT_EVENTS,"official_TEST_used":False})
    old.check_test_guard()
    print("PREFLIGHT PASS, historical first batch paired, source and TEST guard OK",flush=True)

def train(seed,arm):
    old.forbid_official_test()
    if seed not in SEEDS or arm not in ARMS:raise ValueError("Fixed seed/arm only")
    pre_path=RESULTS/"preflight.json"
    if not pre_path.is_file():raise FileNotFoundError("Must run preflight")
    pre=json.loads(pre_path.read_text(encoding="utf-8"))
    assert pre["passed"] and pre["source_files"]==file_hashes()
    y,vy,vbank,plan,noise=input_data(seed)
    hpath,historic=oldsignatures(seed)
    assert sha(hpath)==pre["checks"][str(seed)]["historical_signature_sha256"]
    folder=RESULTS/f"seed{seed}"/arm
    if folder.exists():raise FileExistsError("No overwrite/restart: "+str(folder))
    baseline=None
    if arm!="LowVS":
        ctrl=RESULTS/f"seed{seed}/LowVS/batch_signatures.csv"
        if not ctrl.is_file():raise FileNotFoundError("Run LowVS first to establish paired control")
        baseline=cread(ctrl);assert len(baseline)==850
    cfg,tr,bb,cls,CE,initial,s1=old.fresh(seed)
    assert initial==pre["checks"][str(seed)]["initial"]
    optim=optimizer_for(cfg,cls)
    assert not optim.state
    frozen={k:v.detach().cpu().clone() for k,v in bb.state_dict().items()}
    initial_classifier=state_hash(cls)
    old.warmup(tr,bb)
    aug=torch.Generator().manual_seed(seed+300009)
    lr_schedule=scheduler()
    rows=[];signatures_all=[]
    folder.mkdir(parents=True)
    jwrite(folder/"config.json",{**PROTOCOL,"seed":seed,"arm":arm,"stage1_sha":sha(s1),
           "source_files":file_hashes(),"initial_state":initial,"historic_signature_sha":sha(hpath)})
    (folder/"effective.yaml").write_text(cfg.dump(),encoding="utf-8")
    best=-1.;best_epoch=0;steps=0
    for epoch in range(10):
        cls.train();lr_schedule(optim,epoch,cfg)
        sums={"total":0.,"fused_ce":0.,"clean_tail_aux":0., "n_tail":0,
              "competitor_head":0,"competitor_medium":0,"competitor_tail":0}
        for batch,start in enumerate(range(0,N,BATCH)):
            end=min(start+BATCH,N)
            rec,don=plan["receivers"][epoch,start:end],plan["donors"][epoch,start:end]
            target=torch.from_numpy(y[rec].copy()).long().cuda()
            ri,di=augment(rec,don,aug,tr["train_transform"])
            receiver,donor=forward_pair(bb,ri,di)
            g=noise[epoch,start:end]
            trace=online_masks(receiver,y[don],g)
            chosen=trace["masks"]["Low-VS"]
            fused,mask=fuse(receiver,donor,chosen)
            sig=signatures(rec,don,y,ri,di,receiver,donor,g,trace)
            ref=historic[steps]
            if int(ref["epoch"])!=epoch+1 or int(ref["batch"])!=batch:
                raise RuntimeError("Historical batch ordering incorrect")
            equal_signatures(sig,ref,seed,arm,epoch+1,batch)
            if baseline is not None:equal_signatures(sig,baseline[steps],seed,arm,epoch+1,batch)
            # Only FC is trainable; logits on clean receiver are differentiated
            # for the two auxiliary arms, NEVER used to choose VS masks.
            fused_logits=cls(fused)
            clean_logits=cls(receiver) if arm!="LowVS" else torch.zeros_like(fused_logits)
            loss,diag=loss_terms(arm,fused_logits,clean_logits,target)
            assert torch.isfinite(loss).item()
            optim.zero_grad(set_to_none=True)
            loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in cls.parameters())
            optim.step()
            steps+=1
            sums["total"]+=float(loss.detach())*len(rec)
            sums["fused_ce"]+=diag["fused_ce"]*len(rec)
            sums["clean_tail_aux"]+=diag["clean_tail_aux"]*len(rec)
            sums["n_tail"]+=diag["n_tail"]
            if diag["n_competitor_head"] is not None:
                sums["competitor_head"]+=diag["n_competitor_head"]
                sums["competitor_medium"]+=diag["n_competitor_medium"]
                sums["competitor_tail"]+=diag["n_competitor_tail"]
            signatures_all.append({"epoch":epoch+1,"batch":batch,
                                    "size":end-start,**sig})
        assert steps==(epoch+1)*85
        check_frozen(bb,frozen)
        val,vl,vp=evaluate(cls,vy,feature_maps=vbank)
        if val["Overall"]>best:
            best=val["Overall"];best_epoch=epoch+1
        if epoch==9:
            torch.save({"seed":seed,"arm":arm,"epoch":10,
                        "classifier":{k:v.detach().cpu().clone() for k,v in cls.state_dict().items()},
                        "stage1_sha":sha(s1),"official_TEST_used":False},
                       folder/"final_fc.pth")
            np.savez_compressed(folder/"validation_epoch10.npz",labels=vy,logits=vl,predictions=vp)
        row={"seed":seed,"arm":arm,"epoch":epoch+1,
             "lr":optim.param_groups[0]["lr"],
             "train_total_loss":sums["total"]/N,
             "train_fused_ce":sums["fused_ce"]/N,
             "train_tail_clean_aux_weighted_by_lambda":LAMBDA*sums["clean_tail_aux"]/N,
             "train_tail_receiver_draws":sums["n_tail"],
             "top_competitor_head":sums["competitor_head"],
             "top_competitor_medium":sums["competitor_medium"],
             "top_competitor_tail":sums["competitor_tail"],
             **{k:val[k] for k in METRICS}}
        rows.append(row);cwrite(folder/"epochs.csv",rows)
        print(f"seed={seed} {arm} ep={epoch+1} Overall={val['Overall']:.4f} Tail={val['Tail']:.4f}",flush=True)
    assert steps==850
    cwrite(folder/"batch_signatures.csv",signatures_all)
    assert state_hash(bb)==initial["backbone"] and state_hash(cls)!=initial_classifier
    final=torch.load(folder/"final_fc.pth",map_location="cpu",weights_only=False)
    assert final["epoch"]==10 and not final["official_TEST_used"]
    cls.load_state_dict(final["classifier"],strict=True)
    replay,rl,rp=evaluate(cls,vy,feature_maps=vbank)
    assert np.array_equal(rp,vp)
    assert np.allclose(rl,vl,rtol=1e-6,atol=1e-6)
    assert all(abs(replay[k]-rows[-1][k])<1e-8 for k in METRICS)
    jwrite(folder/"summary.json",{"completed":True,"seed":seed,"arm":arm,"steps":steps,
        "best_Overall_diagnostic":best,"best_epoch_diagnostic":best_epoch,
        "epoch10":rows[-1],"initial_classifier_sha":initial_classifier,
        "final_classifier_sha":state_hash(cls),"final_checkpoint_sha":sha(folder/"final_fc.pth"),
        "pairing_historical_850":True,"pairing_internal_850":baseline is not None,
        "frozen_backbone_BN":True,"official_TEST_used":False})
    old.check_test_guard()
    print("PASS",seed,arm,"historical signatures 850/850 matched",flush=True)

def report():
    old.forbid_official_test()
    all_rows=[];changes=[];perclass=[];curves=[]
    for seed in SEEDS:
        by={};val_y=None;control=None
        for arm in ARMS:
            folder=RESULTS/f"seed{seed}"/arm
            summary=json.loads((folder/"summary.json").read_text(encoding="utf-8"))
            assert summary["completed"] and summary["steps"]==850
            arr=np.load(folder/"validation_epoch10.npz",allow_pickle=False)
            y,p,lg=arr["labels"],arr["predictions"],arr["logits"]
            assert np.array_equal(p,lg.argmax(1)) and lg.shape==(1980,100)
            if val_y is None:val_y=y.copy()
            else:assert np.array_equal(val_y,y)
            sig=cread(folder/"batch_signatures.csv")
            assert len(sig)==850
            if control is None:control=sig
            else:
                for a,b in zip(sig,control):
                    equal_signatures(a,b,seed,arm,int(a["epoch"]),int(a["batch"]))
            by[arm]=(p,summary["epoch10"],lg)
            all_rows.append({"seed":seed,"arm":arm,**{k:summary["epoch10"][k] for k in METRICS}})
            curves+=cread(folder/"epochs.csv")
        comparisons=(("LowVS-CleanCE","LowVS"),
                     ("LowVS-TopCompetitor","LowVS"),
                     ("LowVS-TopCompetitor","LowVS-CleanCE"))
        for a,b in comparisons:
            pa,ma,la=by[a];pb,mb,lb=by[b]
            wins=(pa==val_y)&(pb!=val_y)
            losses=(pa!=val_y)&(pb==val_y)
            rec={"seed":seed,"comparison":a+" - "+b,
                 "wins":int(wins.sum()),"losses":int(losses.sum()),
                 "net_correct":int(wins.sum()-losses.sum()),
                 "head_net":int((wins[val_y<=35]).sum()-(losses[val_y<=35]).sum()),
                 "tail_net":int((wins[val_y>=71]).sum()-(losses[val_y>=71]).sum()),
                 "tail_true_to_head_pred_delta":int(((val_y>=71)&(pa<=35)).sum())-int(((val_y>=71)&(pb<=35)).sum()),
                 "head_true_to_tail_pred_delta":int(((val_y<=35)&(pa>=71)).sum())-int(((val_y<=35)&(pb>=71)).sum())}
            for name in ("Overall","Head","OfficialMedium","Tail","Few","validation_CE"):
                rec["delta_"+name]=ma[name]-mb[name]
            changes.append(rec)
            for c in range(1,100):
                sel=val_y==c
                perclass.append({"seed":seed,"comparison":rec["comparison"],"class_id":c,"n":int(sel.sum()),
                      "wins":int(wins[sel].sum()),"losses":int(losses[sel].sum()),
                      "net":int(wins[sel].sum()-losses[sel].sum())})
    cwrite(RESULTS/"epoch10.csv",all_rows)
    cwrite(RESULTS/"paired.csv",changes)
    cwrite(RESULTS/"per_class.csv",perclass)
    cwrite(RESULTS/"curves.csv",curves)
    jwrite(RESULTS/"scientific_summary.json",{"protocol":PROTOCOL,"results":all_rows,"pairs":changes,
        "limits":"Same backbone/same repeated dev images, 2 StageII seeds, fixed margin=0 lambda=0.5; cannot prove representation gains."})
    md=["# Low-VS + clean discriminative preservation: three-arm pilot",
        "","Primary contrast: TopCompetitor vs CleanCE, NOT merely vs LowVS.",
        "Historical LowVS Full signature 850/850 checked in each arm and each seed.",
        "Frozen StageI seed42 epoch184; StageII fixed epoch10; 1980 TRAIN holdout class1..99 (20/class); no official TEST.",
        "","| Seed | Arm | Overall | Head | OfficialMedium | Tail | Few | validation CE |",
        "|---|---|---:|---:|---:|---:|---:|---:|"]
    for x in all_rows:
        md.append(f"|{x['seed']}|{x['arm']}|{x['Overall']:.4f}|{x['Head']:.4f}|{x['OfficialMedium']:.4f}|{x['Tail']:.4f}|{x['Few']:.4f}|{x['validation_CE']:.4f}|")
    md+=["","|Seed|Contrast|Overall pp|Head pp|Tail pp|Net images|Tail net|Tail→Head change|",
        "|---|---|---:|---:|---:|---:|---:|---:|"]
    for x in changes:
        md.append(f"|{x['seed']}|{x['comparison']}|{x['delta_Overall']:+.4f}|{x['delta_Head']:+.4f}|{x['delta_Tail']:+.4f}|{x['net_correct']:+d}|{x['tail_net']:+d}|{x['tail_true_to_head_pred_delta']:+d}|")
    md+=["","**Interpretation precommitted**:",
        "- If TopCompetitor beats CleanCE on Tail in both seeds without consequential Overall/Head loss, targeted clean discrimination earns follow-up consideration, not immediate novelty.",
        "- If both auxiliary arms behave alike, added clean-label supervision is a sufficient explanation: do not claim a special competitor mechanism.",
        "- If both harm LowVS, adding clean constraints is not justified on this frozen FC; stop this module pilot.",
        "- If results change sign by seed or net few images, do not tune lambda or margins on this reused dev split.",
        "- Auxiliary CE and pairwise softplus have different gradient magnitudes; same lambda alone does not equate gradient budget.",
        "- The classifier alone is trained: cannot claim Backbone feature learning or publication-grade generalization."]
    (RESULTS/"LOWVS_DISC_RESULTS.md").write_text("\n".join(md),encoding="utf-8")
    old.check_test_guard()
    print("REPORT PASS",flush=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("action",choices=("unit","preflight","train","report"))
    p.add_argument("--seed",type=int,choices=SEEDS)
    p.add_argument("--arm",choices=ARMS)
    x=p.parse_args()
    if x.action=="unit":self_test()
    elif x.action=="preflight":preflight()
    elif x.action=="train":
        if x.seed is None or x.arm is None:p.error("Specify --seed and --arm")
        train(x.seed,x.arm)
    else:report()
if __name__=="__main__":main()
