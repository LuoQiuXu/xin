"""One-question iteration: Original-like BatchShared vs paired PerSample vs LowVS.

Run from anywhere: python run_mask_granularity.py preflight
                   python run_mask_granularity.py train --seed 124
                   python run_mask_granularity.py train --seed 125
                   python run_mask_granularity.py report
Requires the *locally complete* October 14 run, including omitted large files.
All historical paths are read-only. No Git operations. No official TEST access.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
import os
import torchvision

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / "clean_vs_stability_runs" / "clean_s1_seed42_s2_124_125_20261014"
if not (OLD / "common.py").is_file():
    raise RuntimeError("Expected previous clean stability experiment adjacent to this folder")
sys.path.insert(0, str(OLD))
import numpy as np
import torch
from common import (N, BATCH, seed_all, array_hash, state_hash, sha, metrics, evaluate,
                    check_frozen)
from stability_models import (frozen_models, outdir, plan_dir, PAIR_KEYS, check_sources)
from reused_train_helpers import augment, fuse, tensor_hash, scheduler, optimizer_for
from online_policy import online_masks, selected_statistics
from mask_policy import choose, self_test

_TEST_EVENTS = []
def install_local_test_guard():
    forbidden = os.path.normcase(os.path.abspath(OLD.parent.parent.parent / "H2T/data/cifar100/cifar-100-python/test"))
    def audit(event, args):
        if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
            checked = os.path.normcase(os.path.abspath(os.fsdecode(args[0])))
            if checked == forbidden:
                _TEST_EVENTS.append(checked)
                raise RuntimeError("Official CIFAR100 TEST file access prohibited")
    sys.addaudithook(audit)
    previous = torchvision.datasets.CIFAR100.__init__
    def protected(self, root, train=True, *args, **kwargs):
        if not train:
            _TEST_EVENTS.append("CIFAR100(train=False)")
            raise RuntimeError("Official CIFAR100 TEST loading prohibited")
        return previous(self, root, train, *args, **kwargs)
    torchvision.datasets.CIFAR100.__init__ = protected

def guard_check():
    assert not _TEST_EVENTS, _TEST_EVENTS

def csv_write(path, rows):
    assert rows and path.resolve().is_relative_to(HERE), "Outputs must remain under new experiment directory"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

ARM = "Uniform-BatchShared"
SEEDS = (124, 125)
RESULTS = HERE / "results"
# Deterministically use first receiver's Uniform Gumbel mask for all samples in batch.
# This is batch-sharing semantics only, NOT torch.randperm RNG parity with official H2T.
PROTOCOL = {
    "purpose": "Separate mask granularity from VS filtering",
    "existing_controls": ["Uniform-PerSample", "Low-VS"],
    "new_arm": ARM,
    "batch_shared_rule": "tile Uniform-PerSample mask of first item of each actual minibatch",
    "nominal_original_H2T_4D_semantics": "one mask per minibatch",
    "not_equivalent_to_official_sampling_rng": True,
    "stage1_seed": 42, "stage2_seeds": list(SEEDS),
    "epoch_primary": 10, "epochs": 10, "rho": 0.25, "batch_size": 128,
    "backbone_bn": "frozen eval", "classifier": "SGD only",
    "dataset": "clean CIFAR100-LT train-only 99-class holdout, class0 missing; 100 logits",
    "test_access": False, "no_stage1_retraining": True,
    "run_only_new_arm": True,
}


def ensure_inputs(seed):
    assert seed in SEEDS
    needed = [
        OLD/"source_lock.json", OLD/"locked/clean_stage1_best.pth",
        OLD/"class_feature_statistics.npz", OLD/"train_images.npy",
        OLD/"train_labels.npy", OLD/"validation_labels.npy",
        OLD/"validation_feature_maps.npy", OLD/"split_indices.npz",
        OLD/"stage2_original_effective.yaml",
        OLD/"plans"/str(seed)/"sampling_plan.npz",
        OLD/"plans"/str(seed)/"gumbel_noise_float64.npy",
        outdir(seed, "Uniform-PerSample")/"batches.csv",
        outdir(seed, "Uniform-PerSample")/"epoch_validation_metrics.csv",
        outdir(seed, "Low-VS")/"epoch_validation_metrics.csv",
        outdir(seed, "Uniform-PerSample")/"validation_epoch10.npz",
        outdir(seed, "Low-VS")/"validation_epoch10.npz",
    ]
    missing = [str(p) for p in needed if not p.is_file()]
    if missing:
        raise FileNotFoundError("Local complete previous run is required; missing:\n" + "\n".join(missing))
    check_sources()
    return needed


def control_rows(seed):
    p = outdir(seed, "Uniform-PerSample")/"batches.csv"
    with p.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 850, len(rows)
    return rows


def preflight():
    self_test()
    for seed in SEEDS:
        ensure_inputs(seed)
        plan = np.load(plan_dir(seed)/"sampling_plan.npz", allow_pickle=False)
        gumbel = np.load(plan_dir(seed)/"gumbel_noise_float64.npy", mmap_mode="r")
        assert plan["receivers"].shape == (10, N)
        assert plan["donors"].shape == (10, N)
        assert gumbel.shape == (10, N, 64)
        assert len(control_rows(seed)) == 850
        for epoch in (0, 9):
            for start in (0, 128, N-95):
                end = min(N, start+128)
                g = gumbel[epoch, start:end]
                uniform = np.argsort(-g, axis=1, kind="stable")[:, :16].astype(np.uint8)
                shared = choose(ARM, {"masks":{"Uniform-PerSample":uniform, "Low-VS":uniform}})
                assert shared.shape[0] == end-start
                assert np.array_equal(shared[0], uniform[0])
        print(f"PREFLIGHT PASS seed={seed}, inputs locked and 850 reference batches", flush=True)
    print("PREFLIGHT PASS: policy, local checkpoints, reference controls", flush=True)


def train(seed):
    ensure_inputs(seed)
    output = RESULTS / f"shared_seed{seed}"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite previous run: {output}")
    output.mkdir(parents=True)
    (output/"checkpoints").mkdir()
    (output/"protocol.json").write_text(json.dumps({**PROTOCOL, "seed": seed}, indent=2), encoding="utf-8")
    reference = control_rows(seed)
    cfg, tr, bb, cls, CE, methods, s1, q = frozen_models(seed)
    cfg.name = f"mask_granularity_shared_seed{seed}"
    cfg.model_dir = str(output/"checkpoints")
    cfg.shift_bn = False
    (output/"effective.yaml").write_text(cfg.dump(), encoding="utf-8")
    init_bb = state_hash(bb)
    init_cls = state_hash(cls)
    frozen = {k:v.detach().cpu().clone() for k,v in bb.state_dict().items()}
    optimizer = optimizer_for(cfg,cls)
    assert not optimizer.state
    labels = np.load(OLD/"train_labels.npy",allow_pickle=False)
    val_labels = np.load(OLD/"validation_labels.npy",allow_pickle=False)
    val_bank = np.load(OLD/"validation_feature_maps.npy", mmap_mode="r",allow_pickle=False)
    plan = np.load(plan_dir(seed)/"sampling_plan.npz",allow_pickle=False)
    noise = np.load(plan_dir(seed)/"gumbel_noise_float64.npy",mmap_mode="r",allow_pickle=False)
    aug = torch.Generator().manual_seed(seed+300009)
    criterion = CE(cfg,np.bincount(labels,minlength=100)).cuda()
    lr_schedule = scheduler()
    records, masks = [], np.empty((10,N,16),dtype=np.uint8)
    equality = {key:0 for key in PAIR_KEYS}
    best = -1.0
    batch_counter = 0
    for epoch in range(10):
        cls.train()
        lr_schedule(optimizer,epoch,cfg)
        weighted_loss = 0.
        for batch,start in enumerate(range(0,N,BATCH)):
            end = min(N,start+BATCH)
            rec,don = plan["receivers"][epoch,start:end],plan["donors"][epoch,start:end]
            target = torch.from_numpy(labels[rec].copy()).cuda()
            ri,di = augment(rec,don,aug,tr["train_transform"])
            with torch.no_grad():
                receiver,donor = bb(ri.cuda()),bb(di.cuda())
            g = noise[epoch,start:end]
            trace = online_masks(receiver,labels[don],g)
            row = {
                "epoch": epoch+1, "batch":batch, "size":end-start,
                "indices_sha256":array_hash(rec,don,labels[rec],labels[don]),
                "images_sha256":array_hash(ri.numpy(),di.numpy()),
                "features_sha256":array_hash(receiver.cpu().numpy(),donor.cpu().numpy()),
                "augmentation_state_after_sha256":tensor_hash(aug.get_state()),
                "ambient_cpu_rng_sha256":tensor_hash(torch.get_rng_state()),
                "ambient_cuda_rng_sha256":tensor_hash(torch.cuda.get_rng_state()),
                "gumbel_sha256":array_hash(g),
                "online_gap_sha256":trace["GAP_sha256"],
                "online_scores_sha256":trace["score_sha256"],
                "all_two_masks_sha256":trace["all_two_masks_sha256"],
            }
            for key in PAIR_KEYS:
                if str(row[key]) != reference[batch_counter][key]:
                    raise AssertionError(f"Control mismatch seed={seed} batch={batch_counter} key={key}")
                equality[key] += 1
            chosen = choose(ARM,trace)
            with torch.no_grad():
                mixed,_ = fuse(receiver,donor,chosen)
            assert np.array_equal(target.cpu().numpy(),labels[rec])
            masks[epoch,start:end] = chosen
            logits = cls(mixed)
            loss = criterion(logits,target,extra_info=None)
            assert torch.isfinite(loss)
            optimizer.zero_grad()
            loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in cls.parameters())
            optimizer.step()
            weighted_loss += float(loss.detach())*(end-start)
            batch_counter += 1
        check_frozen(bb,frozen)
        val,logits,preds = evaluate(cls,val_labels,feature_maps=val_bank)
        improved = val["Overall"] > best
        if improved: best=val["Overall"]
        checkpoint = {
            "epoch":epoch+1,
            "state_dict_model":frozen,
            "state_dict_classifier":{k:v.detach().cpu().clone() for k,v in cls.state_dict().items()},
            "validation":val, "arm":ARM, "seed":seed, "TEST_used":False,
            "Clean_StageI_sha256":sha(s1),
            "VS_statistics_sha256":sha(OLD/"class_feature_statistics.npz"),
        }
        if improved: torch.save(checkpoint,output/"checkpoints/model_best.pth.tar")
        if epoch==9: torch.save(checkpoint,output/"checkpoints/model_final.pth.tar")
        np.savez_compressed(output/f"validation_epoch{epoch+1:02d}.npz",
                            logits=logits,predictions=preds,labels=val_labels)
        records.append({"arm":ARM,"seed":seed,"epoch":epoch+1,
                        "train_loss":weighted_loss/N, "lr":optimizer.param_groups[0]["lr"],
                        **{k:val[k] for k in ("Overall","Head","OfficialMedium","Tail",
                                                "Many","Medium","Few","Macro","validation_CE")}})
        csv_write(output/"epoch_validation_metrics.csv",records)
        guard_check()
        print(f"shared seed={seed} epoch={epoch+1}/10 "
              f"overall={val['Overall']:.4f} Tail={val['Tail']:.4f} Few={val['Few']:.4f}", flush=True)
    assert batch_counter==850 and all(v==850 for v in equality.values())
    np.savez_compressed(output/"actual_masks.npz",masks=masks)
    for kind in ("best","final"):
        ck = torch.load(output/f"checkpoints/model_{kind}.pth.tar",
                        map_location="cpu",weights_only=False)
        assert ck["TEST_used"] is False and ck["arm"]==ARM
        assert all(torch.equal(v,frozen[k]) for k,v in ck["state_dict_model"].items())
        cls.load_state_dict(ck["state_dict_classifier"],strict=True)
        loaded,_,_=evaluate(cls,val_labels,feature_maps=val_bank)
        assert loaded==ck["validation"], f"Checkpoint reload mismatch {kind}"
    summary = {
        "status":"complete", "seed":seed, "epoch10":records[-1],
        "best":max(records,key=lambda r:r["Overall"]),
        "initial_backbone_hash":init_bb,"initial_classifier_hash":init_cls,
        "paired_control_fields_verified":equality, "paired_batches":850,
        "stage1_frozen":state_hash(bb)==init_bb, "TEST_used":False,
        "final_checkpoint_sha256":sha(output/"checkpoints/model_final.pth.tar"),
    }
    (output/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(f"TRAIN PASS seed={seed}: 850 fully paired batches, frozen BB/BN, 10 epochs", flush=True)


def csv_records(path):
    with path.open(encoding="utf-8",newline="") as f:
        return list(csv.DictReader(f))


def report():
    for seed in SEEDS:
        ensure_inputs(seed)
        assert (RESULTS/f"shared_seed{seed}/summary.json").is_file(), f"train seed{seed} first"
    rows=[];flip_rows=[]
    for seed in SEEDS:
        paths={
            "Uniform-BatchShared":RESULTS/f"shared_seed{seed}",
            "Uniform-PerSample":outdir(seed,"Uniform-PerSample"),
            "Low-VS":outdir(seed,"Low-VS"),
        }
        metrics_by_arm={}
        pred_by_arm={}
        labels=None
        for arm,folder in paths.items():
            epochs=csv_records(folder/"epoch_validation_metrics.csv")
            at10=next(r for r in epochs if int(r["epoch"])==10)
            best=max(epochs,key=lambda r:float(r["Overall"]))
            metrics_by_arm[arm]={k:float(at10[k]) for k in
                                 ("Overall","Head","OfficialMedium","Tail","Many","Medium","Few","Macro")}
            bundle=np.load(folder/"validation_epoch10.npz",allow_pickle=False)
            y=bundle["labels"];p=bundle["predictions"]
            if labels is None: labels=y
            else: assert np.array_equal(labels,y)
            pred_by_arm[arm]=p
            rows.append({"seed":seed,"arm":arm,**metrics_by_arm[arm],
                         "best_epoch":int(best["epoch"]),"best_Overall":float(best["Overall"])})
        for arm in ("Uniform-PerSample","Low-VS"):
            a=pred_by_arm["Uniform-BatchShared"]
            b=pred_by_arm[arm]
            flip_rows.append({
                "seed":seed,"comparison":f"{arm} - Uniform-BatchShared",
                "gain_wrong_to_right":int(((a!=labels)&(b==labels)).sum()),
                "loss_right_to_wrong":int(((a==labels)&(b!=labels)).sum()),
                "changed_prediction":int((a!=b).sum()),
                "delta_Overall_pp":metrics_by_arm[arm]["Overall"]-metrics_by_arm["Uniform-BatchShared"]["Overall"],
                "delta_Tail_pp":metrics_by_arm[arm]["Tail"]-metrics_by_arm["Uniform-BatchShared"]["Tail"],
                "delta_Few_pp":metrics_by_arm[arm]["Few"]-metrics_by_arm["Uniform-BatchShared"]["Few"],
            })
    csv_write(RESULTS/"all_three_epoch10.csv",rows)
    csv_write(RESULTS/"paired_differences.csv",flip_rows)
    print("Report written:",RESULTS/"all_three_epoch10.csv")
    print("Paired comparisons:",RESULTS/"paired_differences.csv")
    for row in flip_rows:print(json.dumps(row,ensure_ascii=False))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("action",choices=("preflight","train","report"))
    parser.add_argument("--seed",type=int,choices=SEEDS)
    args=parser.parse_args()
    if args.action=="train" and args.seed is None:
        parser.error("--seed 124 or --seed 125 required")
    # Preserve clean train-only rule for every action, including reporting.
    install_local_test_guard()
    if args.action=="preflight":preflight()
    elif args.action=="train":train(args.seed)
    else:report()
    guard_check()


if __name__=="__main__":
    main()
