"""Four-arm matched GAP-L2 study: original Uniform/LowVS vs paired norm controls.

From repository root: run_experiment.py preflight; train --seed ... --arm ...;
run_experiment.py report. See CODEX_LOWVS_NORM_DIRECTION.txt for exact order.
Source artifacts are READ ONLY. Writes to this directory's results/ only.
"""
from __future__ import annotations
import argparse,csv,json,sys,time,traceback
from pathlib import Path
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"ravs_v1"))
import run_ravs_v1 as old
from run_safeguards import protect,history_manifest,environment,stamp
from common import N,BATCH,sha,state_hash,evaluate,check_frozen,array_hash,metrics
from stability_models import check_sources
from reused_train_helpers import augment,scheduler,optimizer_for
from online_policy import online_masks
from norm_policy import ARMS,paired_norm_control,test_policy
import numpy as np
import torch

SEEDS=(124,125)
RESULTS=HERE/"results"
HISTORY_EVENTS=[]
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

def source_hashes():
    return {name:sha(HERE/name) for name in ("run_experiment.py","norm_policy.py")}


def verify_history():
    manifest=json.loads((RESULTS/"historical_manifest.json").read_text(encoding="utf-8"))
    for path,expected in manifest.items():
        assert history_manifest(Path(path),sha)==expected,("Historical files changed",path)
    return sum(len(v) for v in manifest.values())


def record_norms(diag,receiver,donor,trace,mixtures):
    # Diagnostics only; the supplied paired_norm_control remains untouched.
    target=diag["target_norm"]
    err=(diag["matched_norm_uniform"]-diag["matched_norm_lowvs"]).abs()
    eu=(diag["matched_norm_uniform"]-target).abs()
    el=(diag["matched_norm_lowvs"]-target).abs()
    tol=1e-5+1e-4*target
    assert (err<=tol).all() and (eu<=tol).all() and (el<=tol).all()
    diag["gap_norm_abs_error"]=err
    diag["target_error_uniform"]=eu
    diag["target_error_lowvs"]=el
    for label,key in (("uniform","Uniform-PerSample"),("lowvs","Low-VS")):
        mask=torch.zeros((len(receiver),64),device=receiver.device,dtype=torch.bool)
        mask.scatter_(1,torch.from_numpy(trace["masks"][key].astype(np.int64).copy()).to(receiver.device),True)
        delta=torch.where(mask[:,:,None,None],donor-receiver,torch.zeros_like(receiver))
        full_norm=torch.linalg.vector_norm(delta.flatten(1),dim=1)
        diag["full_map_norm_"+label]=full_norm
        matched_key="Uniform-Matched" if label=="uniform" else "LowVS-Matched"
        diag["full_map_norm_"+label+"_matched"]=torch.linalg.vector_norm((mixtures[matched_key]-receiver).flatten(1),dim=1)
    return diag


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
    # Verify original TRAIN provenance without opening the official TEST file.
    import pickle,common
    split=np.load(old.OLD/"split_indices.npz",allow_pickle=False)
    train_ids=split["train_raw_indices"];val_ids=split["validation_raw_indices"]
    assert train_ids.shape==(N,) and val_ids.shape==(1980,)
    assert len(np.unique(train_ids))==N and len(np.unique(val_ids))==1980
    assert len(np.intersect1d(train_ids,val_ids))==0
    train_path=common.DATA/"cifar-100-python/train"
    with train_path.open("rb") as f:raw=pickle.load(f,encoding="latin1")
    raw_images=raw["data"].reshape(-1,3,32,32).transpose(0,2,3,1)
    raw_labels=np.asarray(raw["fine_labels"],dtype=np.int64)
    assert np.array_equal(np.load(old.OLD/"train_images.npy",mmap_mode="r",allow_pickle=False),raw_images[train_ids])
    assert np.array_equal(np.load(old.OLD/"train_labels.npy",allow_pickle=False),raw_labels[train_ids])
    assert np.array_equal(np.load(old.OLD/"validation_labels.npy",allow_pickle=False),raw_labels[val_ids])
    write_json(RESULTS/"dataset_protocol.json",{
        "official_TRAIN":str(train_path),"official_TRAIN_sha256":sha(train_path),
        "LT_samples":N,"validation_samples":1980,"train_validation_index_overlap":0,
        "raw_TRAIN_pixel_and_label_match":True,"missing_validation_classes":[0],
        "LT_class_counts":np.bincount(raw_labels[train_ids],minlength=100).tolist(),
        "TEST_used":False})
    paths=(old.OLD,HERE.parent/"ravs_v1",HERE.parent/"gu_lowvs")
    write_json(RESULTS/"historical_manifest.json",{str(path):history_manifest(path,sha) for path in paths})
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
            d=record_norms(d,receiver,donor,trace,mixtures)
            for name,key in (("Uniform-Full","Uniform-PerSample"),("LowVS-Full","Low-VS")):
                mask=torch.zeros((len(receiver),64),dtype=torch.bool,device=receiver.device)
                mask.scatter_(1,torch.from_numpy(trace["masks"][key].astype(np.int64).copy()).to(receiver.device),True)
                selected=mask[:,:,None,None].expand_as(receiver)
                assert torch.equal(mixtures[name][selected],donor[selected])
                assert torch.equal(mixtures[name][~selected],receiver[~selected])
            tgt=torch.from_numpy(y[rec].copy()).long().cuda()
            ce=CE(cfg,np.bincount(y,minlength=100)).cuda()
            losses={a:float(ce(cls(v),tgt,extra_info=None)) for a,v in mixtures.items()}
        assert all(np.isfinite(v) for v in losses.values())
        check_frozen(bb,frozen)
        assert state_hash(cls)==initial["classifier"]
        assert all(p.grad is None for p in cls.parameters())
        assert np.array_equal(tgt.cpu().numpy(),y[rec])
        checks[str(seed)]={"initial":initial,"stage1_sha256":sha(s1),
            "first_batch_signature":old.signatures(rec,don,y,ri,di,receiver,donor,noise[0,:BATCH],trace),
            "no_grad_losses":losses,
            "max_gap_norm_discrepancy":float(d["gap_norm_abs_error"].max()),
            "max_target_error_uniform":float(d["target_error_uniform"].max()),
            "max_target_error_lowvs":float(d["target_error_lowvs"].max()),
            "uniform_mask_sha256":array_hash(trace["masks"]["Uniform-PerSample"]),
            "lowvs_mask_sha256":array_hash(trace["masks"]["Low-VS"])}
    write_json(RESULTS/"preflight.json",{"status":"PASS","protocol":PROTOCOL,"checks":checks,
        "environment":environment(),"new_source_sha256":source_hashes(),
        "protected_historical_files":verify_history(),
        "matching_tolerance":"1e-5 + 1e-4 * per-example target", "norm_epsilon":1e-12})
    old.check_test_guard();print("PREFLIGHT PASS",flush=True)

def train(seed,arm):
    assert seed in SEEDS and arm in ARMS
    started=time.perf_counter()
    pre=RESULTS/"preflight.json"
    if not pre.is_file():raise FileNotFoundError("Run preflight first")
    expected=json.loads(pre.read_text(encoding="utf-8"))
    assert expected["status"]=="PASS" and source_hashes()==expected["new_source_sha256"]
    verify_history()
    y,vy,vbank,plan,noise=source(seed)
    folder=RESULTS/f"seed{seed}"/arm
    if folder.exists():raise FileExistsError(f"Refuse overwrite {folder}")
    cfg,tr,bb,cls,CE,initial,s1=old.fresh(seed)
    assert initial==expected["checks"][str(seed)]["initial"]
    opt=optimizer_for(cfg,cls);assert not opt.state
    frozen={k:v.detach().cpu().clone() for k,v in bb.state_dict().items()}
    initial_classifier={k:v.detach().cpu().clone() for k,v in cls.state_dict().items()}
    all_norms={};norm_rows=[];pair_counts={key:0 for key in (*PAIR_KEYS,"uniform_mask_sha256","lowvs_mask_sha256")}
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
        "initial_hashes":initial,"environment":environment(),"source_sha256":source_hashes()})
    torch.cuda.reset_peak_memory_stats()
    (folder/"effective.yaml").write_text(cfg.dump(),encoding="utf-8")
    rows=[];batches=[];best=-1.0;best_epoch=0
    for epoch in range(10):
        epoch_started=time.perf_counter()
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
                diag=record_norms(diag,receiver,donor,trace,mixes)
                for key,value in diag.items():
                    value_cpu=value.cpu().numpy()
                    if key not in all_norms:all_norms[key]=np.empty((10,N),dtype=value_cpu.dtype)
                    all_norms[key][epoch,start:end]=value_cpu
                diag_sha=array_hash(*[diag[k].cpu().numpy() for k in sorted(diag)])
            sig=old.signatures(rec,don,y,ri,di,receiver,donor,g,trace)
            sig["uniform_mask_sha256"]=array_hash(trace["masks"]["Uniform-PerSample"])
            sig["lowvs_mask_sha256"]=array_hash(trace["masks"]["Low-VS"])
            if control is not None:
                baseline=control[len(batches)]
                for k in (*PAIR_KEYS,"uniform_mask_sha256","lowvs_mask_sha256"):
                    if sig[k]!=baseline[k]:
                        raise RuntimeError(f"Pairing mismatch seed{seed} {arm} e{epoch+1} b{batch}: {k}")
                    pair_counts[k]+=1
                assert diag_sha==baseline["norm_diagnostics_sha256"],"Norm diagnostic parity failed"
            logits=cls(mixes[arm]);loss=criterion(logits,tgt,extra_info=None)
            assert torch.isfinite(loss).item() and torch.isfinite(logits).all()
            opt.zero_grad(set_to_none=True);loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in cls.parameters())
            opt.step()
            loss_sum+=float(loss.detach())*len(rec)
            for k in keys:agg[k]+=float(diag[k].sum())
            for cond,group in ((y[rec]<=35,head_alpha),(y[rec]>=71,tail_alpha)):
                idx=torch.as_tensor(cond,device=receiver.device)
                group[0]+=float(diag["alpha_uniform"][idx].sum())
                group[1]+=float(diag["alpha_lowvs"][idx].sum())
                group[2]+=int(idx.sum())
            batches.append({"epoch":epoch+1,"batch":batch,"size":end-start,**sig,"norm_diagnostics_sha256":diag_sha})
            norm_rows.append({"epoch":epoch+1,"batch":batch,"size":end-start,
                "max_pair_gap_error":float(diag["gap_norm_abs_error"].max()),
                "max_target_error_uniform":float(diag["target_error_uniform"].max()),
                "max_target_error_lowvs":float(diag["target_error_lowvs"].max()),
                "norm_tolerance_failures":0,"zero_norm_pairs":int(diag["zero_norm_pairs"].sum())})
            if batch%40==0:
                print(f"seed{seed} {arm} epoch{epoch+1}/10 batch{batch}/85 loss={float(loss.detach()):.6f}",flush=True)
        torch.cuda.synchronize()
        epoch_seconds=time.perf_counter()-epoch_started
        check_frozen(bb,frozen)
        val,vl,vp=evaluate(cls,vy,feature_maps=vbank)
        if val["Overall"]>best:
            best=val["Overall"];best_epoch=epoch+1
            torch.save({"epoch":epoch+1,"arm":arm,"seed":seed,
                "classifier":{k:v.detach().cpu().clone() for k,v in cls.state_dict().items()},
                "TEST_used":False,"stage1_sha256":sha(s1)},folder/"checkpoints/best.pth")
        if epoch==9:
            torch.save({"epoch":10,"arm":arm,"seed":seed,
                "classifier":{k:v.detach().cpu().clone() for k,v in cls.state_dict().items()},
                "TEST_used":False,"stage1_sha256":sha(s1)},folder/"checkpoints/final.pth")
            np.savez_compressed(folder/"validation_epoch10.npz",labels=vy,logits=vl,predictions=vp)
        row={"seed":seed,"arm":arm,"epoch":epoch+1,"train_loss":loss_sum/N,
             "lr":opt.param_groups[0]["lr"],"training_seconds":epoch_seconds,
             "peak_allocated_mib":torch.cuda.max_memory_allocated()/1024**2,
             "peak_reserved_mib":torch.cuda.max_memory_reserved()/1024**2,
             **{k:val[k] for k in METRICS},
             **{"mean_"+k:agg[k]/N for k in keys},
             "head_alpha_uniform":head_alpha[0]/max(1,head_alpha[2]),
             "head_alpha_lowvs":head_alpha[1]/max(1,head_alpha[2]),
             "tail_alpha_uniform":tail_alpha[0]/max(1,tail_alpha[2]),
             "tail_alpha_lowvs":tail_alpha[1]/max(1,tail_alpha[2])}
        rows.append(row);write_csv(folder/"epochs.csv",rows)
        write_csv(folder/"batch_signatures.csv",batches)
        write_csv(folder/"batch_norm_diagnostics.csv",norm_rows)
        print(f"seed{seed} {arm} epoch {epoch+1}/10 Overall={val['Overall']:.4f} Tail={val['Tail']:.4f}",flush=True)
    assert len(batches)==850
    assert state_hash(bb)==initial["backbone"]
    changes={k:float((v.detach().cpu()-initial_classifier[k]).abs().max()) for k,v in cls.state_dict().items()}
    assert all(v>0 for v in changes.values())
    if control is not None:assert all(n==850 for n in pair_counts.values())
    if arm=="Uniform-Full":
        np.savez_compressed(folder/"norm_diagnostics.npz",**all_norms,
            receiver_labels=y[plan["receivers"]].astype(np.int16),
            receiver_indices=plan["receivers"],donor_indices=plan["donors"])
    protected=verify_history()
    write_csv(folder/"batch_signatures.csv",batches)
    for which in ("best","final"):
        ck=torch.load(folder/f"checkpoints/{which}.pth",map_location="cpu",weights_only=False)
        assert ck["arm"]==arm and not ck["TEST_used"]
        cls.load_state_dict(ck["classifier"],strict=True)
        check,_,_=evaluate(cls,vy,feature_maps=vbank)
        assert all(abs(check[k]-rows[ck["epoch"]-1][k])<1e-9 for k in METRICS)
    write_json(folder/"summary.json",{"status":"completed","seed":seed,"arm":arm,
        "steps":850,"best_epoch":best_epoch,"best_Overall":best,
        "last":rows[-1],"TEST_used":False,"backbone_frozen":True,"all_batches_paired":control is not None,
        "paired_field_counts":pair_counts if control is not None else {k:850 for k in pair_counts},
        "backbone_state_tensors_bitwise_unchanged":len(frozen),
        "BN_counters":sorted({int(v) for k,v in frozen.items() if k.endswith("num_batches_tracked")}),
        "classifier_parameter_max_abs_changes":changes,
        "checkpoint_sha256":{k:sha(folder/f"checkpoints/{k}.pth") for k in ("best","final")},
        "checkpoint_reload_exact":True,"elapsed_seconds":time.perf_counter()-started,
        "max_sample_gap_norm_error":max(r["max_pair_gap_error"] for r in norm_rows),
        "norm_tolerance_failures":0,"norm_diagnostics_sha256":sha(RESULTS/f"seed{seed}/Uniform-Full/norm_diagnostics.npz"),
        "protected_historical_files":protected,"official_TEST_events":old.AUDIT_EVENTS,
        "historical_write_events":HISTORY_EVENTS})
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
    protect(HERE.parent,RESULTS,HISTORY_EVENTS)
    if q.action=="train" and (q.seed is None or q.arm is None):p.error("train requires --seed and --arm")
    registry_path=RESULTS/"run_registry.json"
    registry=json.loads(registry_path.read_text(encoding="utf-8")) if registry_path.is_file() else []
    entry={"action":q.action,"seed":q.seed,"arm":q.arm,"started_at":stamp(),"status":"running"}
    registry.append(entry);write_json(registry_path,registry)
    try:
        if q.action=="preflight":preflight()
        elif q.action=="train":train(q.seed,q.arm)
        else:report()
        old.check_test_guard();assert not HISTORY_EVENTS
        entry.update(status="completed",finished_at=stamp(),TEST_events=old.AUDIT_EVENTS,historical_write_events=HISTORY_EVENTS)
    except BaseException:
        entry.update(status="failed",finished_at=stamp(),traceback=traceback.format_exc(),TEST_events=old.AUDIT_EVENTS,historical_write_events=HISTORY_EVENTS)
        raise
    finally:write_json(registry_path,registry)

if __name__=="__main__":main()
