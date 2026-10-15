"""RAVS-H2T V1: matched five-arm Clean Stage-II experiment (no Stage-I retraining).

python H2T_repro_audit/ravs_v1/run_ravs_v1.py preflight
python H2T_repro_audit/ravs_v1/run_ravs_v1.py train --seed 124 --arm RAVS-V1
python H2T_repro_audit/ravs_v1/run_ravs_v1.py report
Actual full sequence: see CODEX_RAVS_V1_INSTRUCTIONS.txt.
Requires original locally complete clean_vs_stability_runs; does not modify it.
Never evaluates official TEST. Outputs only under ravs_v1/results/.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
OLD = HERE.parent / "clean_vs_stability_runs" / "clean_s1_seed42_s2_124_125_20261014"
if not (OLD / "stability_models.py").is_file():
    raise RuntimeError(f"Missing locally complete Clean stability experiment: {OLD}")
sys.path.insert(0, str(OLD))

import numpy as np
import torch
import torchvision
from PIL import Image
from common import N, BATCH, array_hash, sha, state_hash, evaluate, check_frozen
from stability_models import frozen_models, plan_dir, check_sources
from reused_train_helpers import augment, fuse, scheduler, optimizer_for
from online_policy import online_masks
from ravs_policy import ARMS, choose, self_test

SEEDS = (124, 125)
RESULTS = HERE / "results"
GROUPS = ("Overall", "Head", "OfficialMedium", "Tail",
          "Many", "Medium", "Few", "Macro", "validation_CE")
AUDIT_EVENTS = []
PROTOCOL = {
    "method": "RAVS-H2T V1 receiver-adaptive VS candidate pooling",
    "stage1_seed": 42, "stage1_selection": "TRAIN-only validation, epoch184",
    "stage2_seeds": list(SEEDS), "arms": list(ARMS),
    "epochs": 10, "primary_epoch": 10, "batch_size": 128,
    "channels": 64, "replace_whole_channels": 16, "pool": [48,64],
    "hardness": "sigmoid(-clean_receiver_margin/(std(logits)+1e-8)), detach",
    "rarity": "clip(log(500/n_y)/log(100),0,1)",
    "omega_ravs": "rarity*(0.5+0.5*hardness)",
    "omega_frequency_only": "rarity*0.75",
    "pool_size": "clip(floor(64-16*omega+0.5),48,64)",
    "fallback": "donor count<20 or flat/nonfinite score: old Uniform rule; nonfinite aborts in online_policy",
    "sample_size_control": "exactly 16 full 8x8 channels except No-Fusion",
    "stage1_backbone_bn": "completely frozen eval",
    "trainable": "StageII 100-way linear classifier only",
    "loss": "original CE; recipient hard label; no auxiliary loss",
    "optimizer": "original SGD + original StageII LR scheduler",
    "control": "shared fixed receiver/donor/augmentation/Gumbel by seed, fresh same StageI FC init",
    "validation": "TRAIN holdout class1..99 20 images/class; missing class0; full 100 logits",
    "TEST_access": False,
    "not_claimed": "No official full-H2T/PIF reproduction; not end-to-end representation learning",
}


def json_write(path, obj):
    path = Path(path)
    assert path.resolve().is_relative_to(HERE.resolve()), path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2,
                               allow_nan=False), encoding="utf-8")


def csv_write(path, rows):
    path = Path(path)
    assert rows and path.resolve().is_relative_to(HERE.resolve()), path
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def csv_read(path):
    with Path(path).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def forbid_official_test():
    import common
    forbidden = os.path.normcase(os.path.abspath(common.DATA / "cifar-100-python/test"))
    def audit(event, args):
        if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
            checked = os.path.normcase(os.path.abspath(os.fsdecode(args[0])))
            if checked == forbidden:
                AUDIT_EVENTS.append(checked)
                raise RuntimeError("Official CIFAR100 TEST file access is prohibited")
    sys.addaudithook(audit)
    original = torchvision.datasets.CIFAR100.__init__
    def blocked(self, root, train=True, *args, **kwargs):
        if not train:
            AUDIT_EVENTS.append("CIFAR100(train=False)")
            raise RuntimeError("Official TEST dataset construction is prohibited")
        return original(self, root, train, *args, **kwargs)
    torchvision.datasets.CIFAR100.__init__ = blocked


def check_test_guard():
    assert not AUDIT_EVENTS, AUDIT_EVENTS


def configure_compute():
    # Common across ALL five new arms, not historically bit-identical to old runs.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False


def inputs(seed):
    assert seed in SEEDS
    needed = [
        OLD / "locked/clean_stage1_best.pth",
        OLD / "class_feature_statistics.npz",
        OLD / "stage2_original_effective.yaml",
        OLD / "train_images.npy", OLD / "train_labels.npy",
        OLD / "validation_feature_maps.npy", OLD / "validation_labels.npy",
        OLD / "split_indices.npz",
        plan_dir(seed) / "sampling_plan.npz",
        plan_dir(seed) / "gumbel_noise_float64.npy",
    ]
    missing = [str(x) for x in needed if not x.is_file()]
    if missing:
        raise FileNotFoundError("Restore these EXISTING local source files; do not regenerate old experiment:\n"+
                                "\n".join(missing))
    return needed


def load_data(seed):
    inputs(seed)
    train_labels = np.load(OLD / "train_labels.npy", allow_pickle=False)
    val_labels = np.load(OLD / "validation_labels.npy", allow_pickle=False)
    val_maps = np.load(OLD / "validation_feature_maps.npy", mmap_mode="r", allow_pickle=False)
    plan = np.load(plan_dir(seed) / "sampling_plan.npz", allow_pickle=False)
    noise = np.load(plan_dir(seed) / "gumbel_noise_float64.npy", mmap_mode="r", allow_pickle=False)
    assert train_labels.shape == (N,) and np.bincount(train_labels, minlength=100).min() >= 5
    assert val_labels.shape == (1980,)
    assert np.array_equal(np.bincount(val_labels, minlength=100),
                          np.r_[0, np.repeat(20, 99)])
    assert val_maps.shape == (1980, 64, 8, 8)
    assert plan["receivers"].shape == plan["donors"].shape == (10, N)
    assert noise.shape == (10, N, 64)
    assert np.isfinite(noise).all()
    return train_labels, val_labels, val_maps, plan, noise


def fresh(seed):
    cfg,tr,bb,cls,CE,methods,p,q = frozen_models(seed)
    configure_compute()
    initial = {"backbone":state_hash(bb), "classifier":state_hash(cls)}
    assert not any(p.requires_grad for p in bb.parameters())
    return cfg,tr,bb,cls,CE,initial,p


def warmup(tr, bb):
    # Each arm executes identical two no-grad forwards before train augmentation.
    images = np.load(OLD/"train_images.npy", mmap_mode="r",allow_pickle=False)[:BATCH]
    with torch.no_grad():
        x = torch.stack([tr["eval_transform"](Image.fromarray(im)) for im in images]).cuda()
        bb(x)
        bb(x)
    torch.cuda.synchronize()


def signatures(rec,don,labels,ri,di,receiver,donor,noise,trace):
    return {
        "indices_sha256":array_hash(rec,don,labels[rec],labels[don]),
        "images_sha256":array_hash(ri.numpy(),di.numpy()),
        "features_sha256":array_hash(receiver.detach().cpu().numpy(), donor.detach().cpu().numpy()),
        "scores_sha256":trace["score_sha256"],
        "gumbel_sha256":array_hash(noise),
    }


def preflight():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for this existing H2T code path")
    self_test()
    check_sources()
    results = {}
    for seed in SEEDS:
        y,val_y,val_bank,plan,noise = load_data(seed)
        cfg,tr,bb,cls,CE,initial,p = fresh(seed)
        aug = torch.Generator().manual_seed(seed + 300009)
        warmup(tr,bb)
        rec,don = plan["receivers"][0,:BATCH],plan["donors"][0,:BATCH]
        ri,di = augment(rec,don,aug,tr["train_transform"])
        with torch.no_grad():
            receiver,donor = bb(ri.cuda()),bb(di.cuda())
            logits = cls(receiver).detach().float().cpu().numpy()
        trace = online_masks(receiver,y[don],noise[0,:BATCH])
        counts = np.bincount(y,minlength=100)
        masks = {}
        for arm in ARMS:
            chosen,diag = choose(arm,trace,y[rec],counts,logits,noise[0,:BATCH])
            if chosen is not None:
                mixed,mask=fuse(receiver,donor,chosen)
                m = mask[:,:,None,None].expand_as(mixed)
                assert torch.equal(mixed[m],donor[m])
                assert torch.equal(mixed[~m],receiver[~m])
                masks[arm] = array_hash(chosen)
            else:
                masks[arm]="no-fusion"
            if arm in ("Frequency-Only","RAVS-V1"):
                assert (diag["candidate_pool_size"]>=48).all()
                assert (diag["candidate_pool_size"]<=64).all()
        check_test_guard()
        results[str(seed)] = {
            "inputs_sha":{p.name:sha(p) for p in inputs(seed)},
            "initial_state_hashes":initial,
            "first_batch_signature":signatures(rec,don,y,ri,di,receiver,donor,noise[0,:BATCH],trace),
            "mask_examples_sha":masks,
            "TEST_used":False
        }
        print(f"PREFLIGHT seed {seed}: 5 policies, 16 full channels, model frozen",flush=True)
    json_write(RESULTS/"preflight.json", {"status":"PASS","seeds":results,"protocol":PROTOCOL})
    print("PREFLIGHT PASS. Historical sources were not modified.",flush=True)


def run(seed,arm):
    if not (RESULTS/"preflight.json").is_file():
        raise RuntimeError("Run preflight first.")
    y,val_y,val_bank,plan,noise=load_data(seed)
    output=RESULTS/f"seed{seed}"/arm
    if output.exists():
        raise FileExistsError(f"Refuse to overwrite or implicitly resume {output}")
    cfg,tr,bb,cls,CE,initial,s1=fresh(seed)
    optimizer=optimizer_for(cfg,cls)
    assert not optimizer.state
    frozen={k:v.detach().cpu().clone() for k,v in bb.state_dict().items()}
    warmup(tr,bb)
    aug=torch.Generator().manual_seed(seed+300009)
    criterion=CE(cfg,np.bincount(y,minlength=100)).cuda()
    schedule=scheduler()
    output.mkdir(parents=True)
    (output/"checkpoints").mkdir()
    json_write(output/"config.json", {
        **PROTOCOL,"seed":seed,"arm":arm,"initial_hashes":initial,
        "StageI_sha256":sha(s1),
        "VS_statistics_sha256":sha(OLD/"class_feature_statistics.npz"),
    })
    (output/"effective.yaml").write_text(cfg.dump(),encoding="utf-8")
    epoch_rows=[]
    batch_rows=[]
    best=-1.
    best_epoch=None
    n_updates=0
    started=time.perf_counter()
    with (output/"train.log").open("w",encoding="utf-8",buffering=1) as log:
        for epoch in range(10):
            cls.train()
            schedule(optimizer,epoch,cfg)
            loss_sum=0.
            diagnostics={"hardness":[], "omega":[], "candidate_pool_size":[],
                         "fallback":[], "frequency_need":[]}
            for batch,start in enumerate(range(0,N,BATCH)):
                end=min(N,start+BATCH)
                rec,don=plan["receivers"][epoch,start:end],plan["donors"][epoch,start:end]
                target=torch.from_numpy(y[rec].copy()).cuda()
                ri,di=augment(rec,don,aug,tr["train_transform"])
                with torch.no_grad():
                    receiver,donor=bb(ri.cuda()),bb(di.cuda())
                    clean_logits=cls(receiver).detach().float().cpu().numpy()
                g=noise[epoch,start:end]
                trace=online_masks(receiver,y[don],g)
                chosen,diag=choose(arm,trace,y[rec],np.bincount(y,minlength=100),
                                   clean_logits,g)
                with torch.no_grad():
                    mixed=receiver if chosen is None else fuse(receiver,donor,chosen)[0]
                logits=cls(mixed)
                loss=criterion(logits,target,extra_info=None)
                assert torch.isfinite(loss).item()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in cls.parameters())
                optimizer.step()
                sig=signatures(rec,don,y,ri,di,receiver,donor,g,trace)
                batch_rows.append({"epoch":epoch+1,"batch":batch,"size":end-start,
                                   **sig,"mask_sha256":array_hash(chosen) if chosen is not None else "none"})
                for key,values in diag.items():
                    if arm in ("Frequency-Only","RAVS-V1") or key in ("hardness","frequency_need","fallback"):
                        arr=np.asarray(values)
                        if np.isfinite(arr).all():
                            diagnostics[key].extend(arr.tolist())
                loss_sum+=float(loss.detach())*(end-start)
                n_updates+=1
            check_frozen(bb,frozen)
            val,logits_val,pred_val=evaluate(cls,val_y,feature_maps=val_bank)
            if val["Overall"]>best:
                best=val["Overall"];best_epoch=epoch+1
                torch.save({"epoch":epoch+1,"seed":seed,"arm":arm,
                            "classifier":{k:v.detach().cpu().clone() for k,v in cls.state_dict().items()},
                            "StageI_sha256":sha(s1),"TEST_used":False},
                           output/"checkpoints/best.pth")
            if epoch==9:
                torch.save({"epoch":10,"seed":seed,"arm":arm,
                            "classifier":{k:v.detach().cpu().clone() for k,v in cls.state_dict().items()},
                            "StageI_sha256":sha(s1),"TEST_used":False},
                           output/"checkpoints/final.pth")
                np.savez_compressed(output/"validation_epoch10.npz",
                                    labels=val_y,logits=logits_val,predictions=pred_val)
            record={"seed":seed,"arm":arm,"epoch":epoch+1,"lr":optimizer.param_groups[0]["lr"],
                    "train_loss":loss_sum/N,
                    **{key:val[key] for key in GROUPS},
                    **{f"mean_{key}":float(np.mean(v)) if v else None
                        for key,v in diagnostics.items()}}
            epoch_rows.append(record)
            csv_write(output/"epoch_metrics.csv",epoch_rows)
            line=(f"{arm} seed{seed} epoch {epoch+1}/10 "
                  f"Overall={val['Overall']:.4f} Tail={val['Tail']:.4f} "
                  f"Few={val['Few']:.4f} Head={val['Head']:.4f}")
            print(line,flush=True)
            log.write(line+"\n")
    csv_write(output/"batch_signatures.csv",batch_rows)
    assert n_updates==850 and len(batch_rows)==850
    assert state_hash(bb)==initial["backbone"]
    for kind in ("best","final"):
        check=torch.load(output/f"checkpoints/{kind}.pth",map_location="cpu",weights_only=False)
        assert check["TEST_used"] is False and check["arm"]==arm
        cls.load_state_dict(check["classifier"],strict=True)
        reloaded,_,_=evaluate(cls,val_y,feature_maps=val_bank)
        at=epoch_rows[check["epoch"]-1]
        assert all(abs(reloaded[k]-at[k])<1e-9 for k in GROUPS)
    json_write(output/"summary.json",{
        "status":"completed","seed":seed,"arm":arm,"updates":850,
        "last":epoch_rows[-1],"best_epoch":best_epoch,"best_overall":best,
        "initial_hashes":initial,
        "final_backbone_hash":state_hash(bb),
        "TEST_used":False,"elapsed_seconds":time.perf_counter()-started,
    })
    check_test_guard()


def report():
    rows=[]
    pair_rows=[]
    for seed in SEEDS:
        by_arm={}
        labels=None
        reference_sig=None
        for arm in ARMS:
            folder=RESULTS/f"seed{seed}"/arm
            summary=json.loads((folder/"summary.json").read_text(encoding="utf-8"))
            assert summary["status"]=="completed" and not summary["TEST_used"]
            sig=csv_read(folder/"batch_signatures.csv")
            assert len(sig)==850
            if reference_sig is None: reference_sig=sig
            matching={}
            for key in ("indices_sha256","images_sha256","features_sha256","scores_sha256","gumbel_sha256"):
                matching[key]=sum(ref[key]==other[key] for ref,other in zip(reference_sig,sig))
            vals=np.load(folder/"validation_epoch10.npz",allow_pickle=False)
            assert np.array_equal(vals["predictions"],vals["logits"].argmax(1))
            if labels is None:labels=vals["labels"].copy()
            else:assert np.array_equal(labels,vals["labels"])
            prediction=vals["predictions"]
            by_arm[arm]=(summary["last"],prediction,matching)
            rows.append({"seed":seed,"arm":arm,"best_epoch":summary["best_epoch"],
                         "best_overall":summary["best_overall"],
                         **{k:summary["last"][k] for k in GROUPS},
                         **{f"matched_{k}":v for k,v in matching.items()}})
        for arm in ARMS[1:]:
            for base in ("No-Fusion","Uniform-PerSample","Low-VS","Frequency-Only"):
                if arm==base:continue
                m,p,_=by_arm[arm]
                ref,rp,_=by_arm[base]
                wins=(rp!=labels)&(p==labels)
                losses=(rp==labels)&(p!=labels)
                tail=(labels>=71)&(labels<=99)
                head=(labels<=35)
                pair_rows.append({
                    "seed":seed,"comparison":f"{arm} - {base}",
                    "wins":int(wins.sum()),"losses":int(losses.sum()),
                    "delta_net_correct":int(wins.sum()-losses.sum()),
                    "tail_net_correct":int(wins[tail].sum()-losses[tail].sum()),
                    "head_net_correct":int(wins[head].sum()-losses[head].sum()),
                    "delta_overall_pp":float(m["Overall"]-ref["Overall"]),
                    "delta_tail_pp":float(m["Tail"]-ref["Tail"]),
                    "delta_few_pp":float(m["Few"]-ref["Few"]),
                    "delta_head_pp":float(m["Head"]-ref["Head"]),
                })
    csv_write(RESULTS/"all_arms_epoch10.csv",rows)
    csv_write(RESULTS/"paired_epoch10.csv",pair_rows)
    mismatch=[r for r in rows if any(r[f"matched_{k}"]!=850 for k in
                ("indices_sha256","images_sha256","features_sha256","scores_sha256","gumbel_sha256"))]
    summary={"status":"complete" if not mismatch else "COMPARABILITY_WARNING",
             "fully_matched":not mismatch,
             "unpaired_arms":[{"seed":r["seed"],"arm":r["arm"]} for r in mismatch],
             "evaluated":"TRAIN holdout class1..99, not official TEST",
             "rows":rows,"paired":pair_rows}
    json_write(RESULTS/"scientific_summary.json",summary)
    lines=["# RAVS-H2T V1: fixed-epoch10 experiment",
           "", "TRAIN-only 99-class validation (1980 images), class0 missing.",
           "Clean StageI fixed, all arms retrained in the same execution environment.",
           "Official TEST never loaded; best epochs are diagnostic, not the primary endpoint.",
           "",f"All per-batch paired inputs/features exact: **{not mismatch}**.",
           ""]
    if mismatch:
        lines+=["WARNING: some input/feature hashes differ across arms.",
                 "Do not claim single-intervention causality until this is resolved.",""]
    lines+=["| Seed | Arm | Overall | Head | Tail | Few | Best epoch |",
            "|---|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"|{r['seed']}|{r['arm']}|{r['Overall']:.4f}|{r['Head']:.4f}|{r['Tail']:.4f}|{r['Few']:.4f}|{r['best_epoch']}|")
    lines+=["","## Paired gains at epoch10 (percentage points)",
            "| Seed | Pair | Overall | Tail | Few | Head | Net imgs |",
            "|---|---|---:|---:|---:|---:|---:|"]
    for r in pair_rows:
        lines.append(f"|{r['seed']}|{r['comparison']}|{r['delta_overall_pp']:+.4f}|{r['delta_tail_pp']:+.4f}|{r['delta_few_pp']:+.4f}|{r['delta_head_pp']:+.4f}|{r['delta_net_correct']:+d}|")
    lines+=["","## Interpretation requirements",
            "Use V1 against BOTH Low-VS and Frequency-Only. If its advantage exists only against",
            "Uniform, there is no evidence the receiver-hardness gate adds value.",
            "Check No-Fusion and Head/Few tradeoffs. A few flipped images or two StageII",
            "seeds under one StageI do not establish stable generalization.",
            "If V1 fails, stop this particular gated-pool rule; do not tune on the same holdout.",
            "No further experiment was launched by this report."]
    (RESULTS/"RAVS_V1_RESULTS.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print("REPORT",RESULTS/"RAVS_V1_RESULTS.md",flush=True)
    print("MATCHED",not mismatch,flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("action", choices=("preflight","train","report"))
    parser.add_argument("--seed",type=int,choices=SEEDS)
    parser.add_argument("--arm",choices=ARMS)
    args=parser.parse_args()
    forbid_official_test()
    if args.action=="preflight":
        preflight()
    elif args.action=="train":
        if args.seed is None or args.arm is None:
            parser.error("train requires --seed and --arm")
        run(args.seed,args.arm)
    else:
        report()
    check_test_guard()


if __name__=="__main__":
    main()
