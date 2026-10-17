"""One-epoch-per-seed, no-training boundary-direction probe.

Run from repository root:
 python H2T_repro_audit/lowvs_boundary_direction/geometry.py
 python H2T_repro_audit/lowvs_boundary_direction/run_probe.py run --seed 124
 python H2T_repro_audit/lowvs_boundary_direction/run_probe.py run --seed 125
 python H2T_repro_audit/lowvs_boundary_direction/run_probe.py report

Never trains; Stage-I weights W,b are the fixed reference classifier. Replays
epoch ONE only, with all original paired augmentations and Gumbel masks.
"""
from __future__ import annotations
import argparse,csv,json,sys
from pathlib import Path
import numpy as np
from geometry import probe,self_test

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
RESULTS=HERE/"results"
OLD_NORM=ROOT/"lowvs_norm_direction"/"results"
SEEDS=(124,125)
PAIR_KEYS=("indices_sha256","images_sha256","features_sha256","scores_sha256",
           "gumbel_sha256","uniform_mask_sha256","lowvs_mask_sha256")
METRIC_KEYS=("nu","nl","r","alpha_u","alpha_l","q_u","q_l","cos_u","cos_l",
             "q_uh","q_lh","full_margin_delta_u","full_margin_delta_l",
             "matched_margin_delta_u","matched_margin_delta_l",
             "margin_diff_direction_at_r","margin_diff_extra_magnitude",
             "loss_delta_u","loss_delta_l","loss_delta_um","loss_delta_lm")
BOOL_KEYS=("clean_correct","correct_full_u","correct_full_l",
           "correct_matched_u","correct_matched_l",
           "prediction_change_full_u","prediction_change_full_l",
           "prediction_change_matched_u","prediction_change_matched_l",
           "original_margin_cross_u","original_margin_cross_l",
           "matched_margin_cross_u","matched_margin_cross_l")
PROTOCOL={
    "question":"Does Low-VS select GAP perturbation directions systematically different in fixed Stage-I true-vs-competitor classifier geometry?",
    "primary_reference":"Clean Stage-I seed42 epoch184 (W,b), before any StageII update",
    "sample_plan":"StageII plan seed124/125 epoch1 only, all 10847 receiver/donor draws per seed; same augmented images, VS statistics and masks",
    "primary_population":"Tail receiver (71-99), h != y, and no VS fallback",
    "primary_quantities":["mean(qL-qU)","fraction(qL<qU)","mean cosL-cosU",
                          "mean matched_margin(L-U)"],
    "q":"(W_y-W_j*) dot delta / ||delta||, j* fixed from clean Stage-I logits on receiver excluding true label",
    "secondary":"y vs donor h margin when h != y; Head/Medium; clean correctly/incorrectly classified; instantaneous CE; fixed algebraic decomposition",
    "matched":"r=min(||deltaU||2,||deltaL||2); same per-draw GAP norm; no StageII training",
    "scope":"original LT TRAIN only; existing dev accuracy not used to choose samples or policies",
    "limits":"Same underlying backbone and repeated source images; descriptive, not independent sample-level significance",
    "no_training":True,"official_TEST_used":False,"new_module":False,
}

def save_json(path,obj):
    assert path.resolve().is_relative_to(RESULTS.resolve())
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False),encoding="utf-8")

def save_csv(path,rows):
    assert rows and path.resolve().is_relative_to(RESULTS.resolve())
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def get_old():
    sys.dont_write_bytecode=True
    sys.path.insert(0,str(ROOT/"ravs_v1"))
    import run_ravs_v1 as base
    from common import N,BATCH,sha,array_hash,state_hash,check_frozen
    from stability_models import check_sources
    from reused_train_helpers import augment
    from online_policy import online_masks
    import torch
    return base,N,BATCH,sha,array_hash,state_hash,check_frozen,check_sources,augment,online_masks,torch

def run(seed):
    assert seed in SEEDS
    self_test()
    folder=RESULTS/f"seed{seed}"
    if folder.exists():raise FileExistsError("Do not overwrite or resume: "+str(folder))
    (folder).mkdir(parents=True)
    (base,N,BATCH,sha,array_hash,state_hash,check_frozen,check_sources,
     augment,online_masks,torch)=get_old()
    base.forbid_official_test()
    assert torch.cuda.is_available(),"CUDA unavailable"
    check_sources()
    base.inputs(seed)
    y,vy,vbank,plan,noise=base.load_data(seed)
    prev=OLD_NORM/f"seed{seed}"/"Uniform-Full"/"batch_signatures.csv"
    if not prev.is_file():raise FileNotFoundError("Need genuine historical paired epoch1 signatures: "+str(prev))
    with prev.open(newline="",encoding="utf-8") as f:
        sigs=list(csv.DictReader(f))
    assert len(sigs)==850
    cfg,tr,bb,cls,CE,initial,stage1=base.fresh(seed)
    assert cfg.num_epochs==10 and len(y)==N
    assert all(not p.requires_grad for p in bb.parameters())
    frozen={k:v.detach().cpu().clone() for k,v in bb.state_dict().items()}
    W=cls.module.fc.weight.detach().cpu().numpy().astype(np.float64)
    b=cls.module.fc.bias.detach().cpu().numpy().astype(np.float64)
    assert W.shape==(100,64) and b.shape==(100,)
    base.warmup(tr,bb)
    generator=torch.Generator().manual_seed(seed+300009)
    out=[];identity=0
    # The plan is indexed by epoch, and generator initially reproduces epoch1.
    for batch,start in enumerate(range(0,N,BATCH)):
        end=min(start+BATCH,N)
        rec=plan["receivers"][0,start:end]
        don=plan["donors"][0,start:end]
        ri,di=augment(rec,don,generator,tr["train_transform"])
        with torch.no_grad():
            receiver,donor=bb(ri.cuda()),bb(di.cuda())
            g=noise[0,start:end]
            trace=online_masks(receiver,y[don],g)
        sig=base.signatures(rec,don,y,ri,di,receiver,donor,g,trace)
        sig["uniform_mask_sha256"]=array_hash(trace["masks"]["Uniform-PerSample"])
        sig["lowvs_mask_sha256"]=array_hash(trace["masks"]["Low-VS"])
        historical=sigs[batch]
        assert int(historical["epoch"])==1 and int(historical["batch"])==batch
        for key in PAIR_KEYS:
            if sig[key]!=historical[key]:
                raise RuntimeError(f"Exact historical draw mismatch seed={seed} batch={batch} field={key}: STOP")
        identity+=1
        rz=receiver.mean((-1,-2)).detach().cpu().numpy()
        hz=donor.mean((-1,-2)).detach().cpu().numpy()
        d=probe(rz,hz,y[rec],y[don],trace["masks"]["Uniform-PerSample"],
                trace["masks"]["Low-VS"],W,b,trace["fallback"])
        d["receiver_index"]=rec.astype(np.int32)
        d["donor_index"]=don.astype(np.int32)
        d["batch_index"]=np.full(len(rec),batch,dtype=np.int16)
        out.append(d)
        if batch%25==0:print("seed",seed,"epoch1",batch,"/85",flush=True)
    assert identity==85
    data={k:np.concatenate([q[k] for q in out]) for k in out[0]}
    assert all(len(v)==N for v in data.values())
    assert np.all(data["same_mask"][data["fallback"]>0])
    assert np.isfinite(data["q_u"]).all() and np.isfinite(data["q_l"]).all()
    assert np.allclose(data["matched_margin_delta_u"]-data["matched_margin_delta_l"],
                       -data["margin_diff_direction_at_r"],atol=1e-8)
    check_frozen(bb,frozen)
    assert state_hash(bb)==initial["backbone"] and state_hash(cls)==initial["classifier"]
    file=folder/"train_epoch1_boundary_probe.npz"
    np.savez_compressed(file,**data)
    save_json(folder/"provenance.json",{
        "seed":seed,"status":"complete","draws":N,"paired_historical_batches":identity,
        "train_only":True,"official_TEST_used":False,"stage1_sha256":sha(stage1),
        "stage1_classifier_hash":initial["classifier"],"stage1_backbone_hash":initial["backbone"],
        "norm_dir_signatures_csv_sha256":sha(prev),
        "probe_file_sha256":sha(file),
        "frozen":True,"optimizer_steps":0})
    base.check_test_guard()
    print("PROBE PASS",seed,"draws",N,"optimizer_steps=0",flush=True)

def mean(a):
    a=np.asarray(a,dtype=np.float64);x=a[np.isfinite(a)]
    return float(x.mean()) if len(x) else None

def summaries(seed,data,select,name,class_label=None):
    n=int(select.sum())
    if n==0:return None
    q_u=data["q_u"][select];q_l=data["q_l"][select]
    r=data["r"][select]
    d=data["matched_margin_delta_l"][select]-data["matched_margin_delta_u"][select]
    extra=data["margin_diff_extra_magnitude"][select]
    fields={
        "seed":seed,"group":name,"class_id":class_label,"n":n,
        "fraction_fallback":mean(data["fallback"][select]>0),
        "fraction_masks_identical":mean(data["same_mask"][select]),
        "fraction_clean_correct":mean(data["clean_correct"][select]),
        "mean_nU":mean(data["nu"][select]),"mean_nL":mean(data["nl"][select]),
        "mean_r":mean(r),"mean_alpha_U":mean(data["alpha_u"][select]),
        "mean_alpha_L":mean(data["alpha_l"][select]),
        "mean_qU":mean(q_u),"mean_qL":mean(q_l),
        "mean_qL_minus_qU":mean(q_l-q_u),
        "fraction_qL_less_qU":mean(q_l<q_u),
        "mean_cosL_minus_cosU":mean(data["cos_l"][select]-data["cos_u"][select]),
        "mean_donor_qL_minus_qU":mean(data["q_lh"][select]-data["q_uh"][select]),
        "mean_matched_margin_L_minus_U":mean(d),
        "mean_full_margin_L_minus_U":mean(data["full_margin_delta_l"][select]-data["full_margin_delta_u"][select]),
        "mean_extra_magnitude_term":mean(extra),
        "mean_loss_full_L_minus_U":mean(data["loss_delta_l"][select]-data["loss_delta_u"][select]),
        "mean_loss_matched_L_minus_U":mean(data["loss_delta_lm"][select]-data["loss_delta_um"][select]),
        "flip_full_L_minus_U":mean(data["prediction_change_full_l"][select])-mean(data["prediction_change_full_u"][select]),
        "flip_matched_L_minus_U":mean(data["prediction_change_matched_l"][select])-mean(data["prediction_change_matched_u"][select]),
        "correct_full_L_minus_U":mean(data["correct_full_l"][select])-mean(data["correct_full_u"][select]),
        "correct_matched_L_minus_U":mean(data["correct_matched_l"][select])-mean(data["correct_matched_u"][select]),
        "fraction_donor_is_competitor":mean(data["donor_is_competitor"][select]),
    }
    if not np.allclose(fields["mean_full_margin_L_minus_U"],
        fields["mean_matched_margin_L_minus_U"]+fields["mean_extra_magnitude_term"],atol=1e-8):
        raise AssertionError("Decomposition averaging mismatch")
    return fields

def report():
    rows=[];classes=[];sha={}
    for seed in SEEDS:
        folder=RESULTS/f"seed{seed}"
        prop=json.loads((folder/"provenance.json").read_text(encoding="utf-8"))
        assert prop["status"]=="complete" and prop["optimizer_steps"]==0
        import hashlib
        file=folder/"train_epoch1_boundary_probe.npz"
        h=hashlib.sha256()
        with file.open("rb") as f:
            for chunk in iter(lambda:f.read(1024*1024),b""):h.update(chunk)
        assert h.hexdigest()==prop["probe_file_sha256"]
        sha[str(seed)]=h.hexdigest()
        z=np.load(file,allow_pickle=False)
        data={k:z[k] for k in z.files};y=data["receiver_class"];h=data["donor_class"]
        tail=y>=71;head=y<=35;medium=(y>=36)&(y<=70)
        eligible=(h!=y)&(data["fallback"]==0)&(~data["same_mask"])
        groups={
            "All":np.ones(len(y),dtype=bool),
            "AllEligible":eligible,
            "Head":head,
            "Medium":medium,
            "Tail":tail,
            "TailEligible":tail&eligible,
            "TailEligibleDonorHead":tail&eligible&(h<=35),
            "TailEligibleCleanCorrect":tail&eligible&data["clean_correct"],
            "TailEligibleCleanWrong":tail&eligible&(~data["clean_correct"]),
            "HeadEligible":head&eligible,
        }
        for name,flag in groups.items():
            record=summaries(seed,data,flag,name)
            if record is not None:rows.append(record)
        for cls in range(100):
            flag=(y==cls)&eligible
            record=summaries(seed,data,flag,"PerClassEligible",cls)
            if record is not None:classes.append(record)
    RESULTS.mkdir(exist_ok=True)
    save_csv(RESULTS/"group_summary.csv",rows)
    save_csv(RESULTS/"per_class_summary.csv",classes)
    save_json(RESULTS/"scientific_summary.json",{
        "protocol":PROTOCOL,"stage1_probe_sha256":sha,
        "primary_TailEligible":[r for r in rows if r["group"]=="TailEligible"],
        "explanation":"Matched signed-margin decomposition is algebraically exact at the FROZEN Stage-I W,b; not an attribution of Stage-II generalization.",
        "conclusion_warning":"No metric here independently proves positive long-tail generalization or semantic feature importance."})
    md=["# Low-VS 边界方向机制审计（TRAIN-only；无需训练）",
        "","固定Clean Stage-I分类器的竞争类别 j*，不使用验证/TEST选择类别、掩码、参数或阈值。",
        "主指标是尾类有效跨类融合的归一化方向投影 qL-qU；更负只代表更强的即时判别干扰，不代表泛化改善。",
        "两个StageII seed在同一个StageI骨干上产生不同增强视图，不是独立训练模型。",
        "",
        "|Seed|Group|Draws|mean(qL-qU)|P(qL<qU)|mean(cosL-cosU)|matched Δmargin L-U|full Δmargin L-U|extra magnitude term|",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for s in rows:
        if s["group"] not in ("AllEligible","HeadEligible","TailEligible","TailEligibleDonorHead"):continue
        md.append(f"|{s['seed']}|{s['group']}|{s['n']}|{s['mean_qL_minus_qU']:+.6f}|"
            f"{s['fraction_qL_less_qU']:.3f}|{s['mean_cosL_minus_cosU']:+.6f}|"
            f"{s['mean_matched_margin_L_minus_U']:+.6f}|"
            f"{s['mean_full_margin_L_minus_U']:+.6f}|"
            f"{s['mean_extra_magnitude_term']:+.6f}|")
    md.extend(["","**Mandatory scientific reading:**",
        "q_yj<0 means the chosen direction compresses the existing true-vs-fixed-competitor margin. It can provide a difficult training example OR destructive label noise; this audit cannot decide which.",
        "Identity: Full margin contrast = matched-at-r direction contribution + extra-magnitude contribution. The identity describes a frozen instant-margin, NOT final accuracy percentages.",
        "Examine both TailEligible and TailEligibleDonorHead before proposing an algorithm. If full-margin contrast mostly comes from extra magnitude and matched direction contrast is tiny/inconsistent, do not claim VS provides useful boundary direction.",
        "Original 1980-image TRAIN development set has already been reused to choose prior methods; no independent inference or significance claims.",
        "Any subsequent transfer assessment must compare DR/Original-H2T/LowVS at same dataset, imbalance, Stage-I, preprocessing and training schedule, measuring negative transfer and head-tail impact.",
        ""])
    (RESULTS/"BOUNDARY_DIRECTION_RESULTS.md").write_text("\n".join(md),encoding="utf-8")
    print("REPORT COMPLETE. Inspect group_summary.csv and BOUNDARY_DIRECTION_RESULTS.md",flush=True)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("action",choices=("unit","run","report"))
    ap.add_argument("--seed",type=int,choices=SEEDS)
    x=ap.parse_args()
    if x.action=="unit":self_test()
    elif x.action=="run":
        if x.seed is None:ap.error("--seed required")
        run(x.seed)
    else:report()

if __name__=="__main__":
    main()
