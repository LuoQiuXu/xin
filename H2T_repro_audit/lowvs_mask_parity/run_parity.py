"""Minimal SAME-ENV 2x2 H2T mask-choice experiment (Clean Stage I; no TEST).

Commands:
    python run_parity.py unit
    python run_parity.py preflight
    python run_parity.py train --seed 124 --arm U-Shared
    python run_parity.py train --seed 124 --arm VS-Shared
    python run_parity.py train --seed 124 --arm U-Individual
    python run_parity.py train --seed 124 --arm VS-Individual
    ... repeat for seed 125
    python run_parity.py report

Historical Clean Stage I and sample/augmentation/Gumbel plans are READ-ONLY.
Unlike the older 3-arm audit, NONE of the four reported arms is reused from a
historically different numerical environment: each is freshly retrained here.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
OLD = HERE.parent / "clean_vs_stability_runs" / "clean_s1_seed42_s2_124_125_20261014"
RESULTS = HERE / "results"
REPO = HERE.parent.parent
SEEDS = (124, 125)
EPOCHS, N, BATCH = 10, 10847, 128
METRICS = ("Overall", "Head", "OfficialMedium", "Tail", "Many", "Medium", "Few", "Macro", "validation_CE")

sys.path.insert(0, str(OLD))
import numpy as np
import torch
import torchvision
from PIL import Image
from common import array_hash, state_hash, sha, evaluate, check_frozen
from stability_models import frozen_models, check_sources, plan_dir, PAIR_KEYS
from reused_train_helpers import augment, fuse, tensor_hash, scheduler, optimizer_for
from online_policy import online_masks, selected_statistics
from mask_policy import ARMS, choose_masks

_test_events = []


def install_test_guard():
    forbidden = str((REPO / "H2T" / "data" / "cifar100" / "cifar-100-python" / "test").resolve()).casefold()
    def audit(event, args):
        if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
            p = str(Path(os.fsdecode(args[0])).resolve()).casefold()
            if p == forbidden:
                _test_events.append("Official CIFAR-100 test file access")
                raise RuntimeError("Official TEST data access is prohibited")
    sys.addaudithook(audit)
    real_init = torchvision.datasets.CIFAR100.__init__
    def guarded(self, root, train=True, *args, **kwargs):
        if not train:
            _test_events.append("CIFAR100(train=False)")
            raise RuntimeError("Official TEST dataset loading is prohibited")
        return real_init(self, root, train, *args, **kwargs)
    torchvision.datasets.CIFAR100.__init__ = guarded


def dump(path, data, *, replace=False):
    path = Path(path)
    if not path.resolve().is_relative_to(RESULTS.resolve()):
        raise ValueError(f"Attempted write outside results/: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if replace else "x"
    with path.open(mode, encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def write_csv(path, rows):
    if not rows:
        raise ValueError("Cannot write empty CSV")
    path = Path(path)
    if not path.resolve().is_relative_to(RESULTS.resolve()):
        raise ValueError("CSV write escaped output folder")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def run_dir(seed, arm):
    return RESULTS / f"seed{seed}" / arm


def input_manifest(seed):
    inputs = {
        "stage1": OLD / "locked" / "clean_stage1_best.pth",
        "statistics": OLD / "class_feature_statistics.npz",
        "source_lock": OLD / "source_lock.json",
        "train_labels": OLD / "train_labels.npy",
        "train_images": OLD / "train_images.npy",
        "validation_labels": OLD / "validation_labels.npy",
        "validation_maps": OLD / "validation_feature_maps.npy",
        "split": OLD / "split_indices.npz",
        "config": OLD / "stage2_original_effective.yaml",
        "plan": plan_dir(seed) / "sampling_plan.npz",
        "noise": plan_dir(seed) / "gumbel_noise_float64.npy",
    }
    missing = [str(p) for p in inputs.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError("The complete local Clean experiment is required:\n" + "\n".join(missing))
    check_sources()
    digest = {key: sha(path) for key, path in inputs.items()}
    return inputs, digest


def preflight():
    import unittest
    suite = unittest.defaultTestLoader.discover(str(HERE), pattern="test_mask_policy.py")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise RuntimeError("CPU tests did not pass")
    for seed in SEEDS:
        inputs, digests = input_manifest(seed)
        labels = np.load(inputs["train_labels"], allow_pickle=False)
        val = np.load(inputs["validation_labels"], allow_pickle=False)
        plan = np.load(inputs["plan"], allow_pickle=False)
        noise = np.load(inputs["noise"], mmap_mode="r", allow_pickle=False)
        maps = np.load(inputs["validation_maps"], mmap_mode="r", allow_pickle=False)
        if labels.shape != (N,) or np.unique(labels).size != 100:
            raise ValueError("Unexpected LT train labels")
        if val.shape != (1980,) or sorted(set(val.tolist())) != list(range(1, 100)):
            raise ValueError("Expected 99-class TRAIN holdout, missing class 0")
        if maps.shape[0] != 1980 or maps.shape[1:] != (64, 8, 8):
            raise ValueError(f"Invalid validation feature bank: {maps.shape}")
        if plan["receivers"].shape != (EPOCHS, N) or plan["donors"].shape != (EPOCHS, N):
            raise ValueError("Invalid paired sampling plan")
        if noise.shape != (EPOCHS, N, 64):
            raise ValueError("Invalid shared Gumbel stream")
        if np.min(plan["receivers"]) < 0 or np.max(plan["receivers"]) >= N:
            raise ValueError("Invalid recipient raw indices")
        if np.min(plan["donors"]) < 0 or np.max(plan["donors"]) >= N:
            raise ValueError("Invalid donor raw indices")
        print(f"seed={seed}: preflight PASS; 10 x {N} paired draws; 99-class TRAIN holdout; hashes={digests}")
    print("Preflight complete: no new training and no official TEST access.")


def signature(epoch, batch, rec, don, labels, ri, di, receiver, donor, g, trace, aug):
    return {
        "epoch": epoch+1, "batch": batch, "size": len(rec),
        "indices_sha256": array_hash(rec, don, labels[rec], labels[don]),
        "images_sha256": array_hash(ri.numpy(), di.numpy()),
        "features_sha256": array_hash(receiver.detach().cpu().numpy(), donor.detach().cpu().numpy()),
        "augmentation_state_after_sha256": tensor_hash(aug.get_state()),
        "ambient_cpu_rng_sha256": tensor_hash(torch.get_rng_state()),
        "ambient_cuda_rng_sha256": tensor_hash(torch.cuda.get_rng_state()),
        "gumbel_sha256": array_hash(g),
        "online_gap_sha256": trace["GAP_sha256"],
        "online_scores_sha256": trace["score_sha256"],
        "all_two_masks_sha256": trace["all_two_masks_sha256"],
    }


def train(seed, arm):
    if seed not in SEEDS or arm not in ARMS:
        raise ValueError("Unexpected seed or arm")
    inputs, input_sha = input_manifest(seed)
    folder = run_dir(seed, arm)
    baseline = run_dir(seed, "U-Shared") / "batch_signatures.csv"
    if arm != "U-Shared" and not baseline.is_file():
        raise FileNotFoundError(f"Run U-Shared seed{seed} first: {baseline}")
    reference = read_csv(baseline) if arm != "U-Shared" else None
    if reference is not None and len(reference) != 850:
        raise ValueError("Incomplete U-Shared reference; abort")
    folder.mkdir(parents=True, exist_ok=False)
    dump(folder / "status.json", {"state":"started","seed":seed,"arm":arm}, replace=True)

    cfg, transform, bb, cls, CE, methods, stage1, q = frozen_models(seed)
    bb_init, fc_init = state_hash(bb), state_hash(cls)
    if reference is not None:
        saved = json.loads((run_dir(seed, "U-Shared") / "summary.json").read_text(encoding="utf-8"))
        if bb_init != saved["initial_backbone_hash"] or fc_init != saved["initial_classifier_hash"]:
            raise RuntimeError("Different initialization from U-Shared; stop")
        if input_sha != saved["input_sha256"]:
            raise RuntimeError("Input checksums changed since U-Shared; stop")
    snapshot = {key:value.detach().cpu().clone() for key,value in bb.state_dict().items()}
    opt = optimizer_for(cfg, cls)
    schedule = scheduler()
    labels = np.load(inputs["train_labels"], allow_pickle=False)
    val_labels = np.load(inputs["validation_labels"], allow_pickle=False)
    val_maps = np.load(inputs["validation_maps"], mmap_mode="r", allow_pickle=False)
    plan = np.load(inputs["plan"], allow_pickle=False)
    gumbel = np.load(inputs["noise"], mmap_mode="r", allow_pickle=False)
    augment_rng = torch.Generator().manual_seed(seed+300009)
    criterion = CE(cfg, np.bincount(labels, minlength=100)).cuda()
    # Reproduce the former Clean experiment's deterministic backbone warm-up.
    images = np.load(inputs["train_images"], mmap_mode="r", allow_pickle=False)[:BATCH]
    with torch.no_grad():
        warm = torch.stack([transform["eval_transform"](Image.fromarray(v)) for v in images]).cuda()
        bb(warm); bb(warm)
    torch.cuda.synchronize()

    log_rows, batch_rows = [], []
    mask_records = np.empty((EPOCHS,N,16), dtype=np.uint8)
    composition_sum = {"V":0.,"B":0.,"ratio":0.,"score":0.,"fallback":0.,\n                       "common_with_uniform":0.}
    best=-1.0
    best_epoch=None
    step=0
    start=time.perf_counter()
    for epoch in range(EPOCHS):
        cls.train()
        schedule(opt, epoch, cfg)
        for batch, lower in enumerate(range(0,N,BATCH)):
            upper=min(lower+BATCH,N)
            rec=plan["receivers"][epoch,lower:upper]
            don=plan["donors"][epoch,lower:upper]
            target=torch.from_numpy(labels[rec].copy()).cuda()
            ri, di=augment(rec,don,augment_rng,transform["train_transform"])
            with torch.no_grad():
                receiver=bb(ri.cuda()); donor=bb(di.cuda())
            if receiver.requires_grad or donor.requires_grad:
                raise RuntimeError("Backbone unexpectedly requires gradient")
            g=gumbel[epoch,lower:upper]
            trace=online_masks(receiver,labels[don],g)
            sig=signature(epoch,batch,rec,don,labels,ri,di,receiver,donor,g,trace,augment_rng)
            if reference is not None:
                previous=reference[step]
                wrong=[k for k in PAIR_KEYS if str(sig[k]) != str(previous[k])]
                if wrong:
                    raise RuntimeError(f"STRICT pairing mismatch seed={seed}, {arm}, epoch={epoch+1}, "
                                       f"batch={batch}, fields={wrong}; do NOT waive this gate")
            chosen=choose_masks(arm,trace,g)
            mask_records[epoch,lower:upper]=chosen
            with torch.no_grad():
                mixed,mask=fuse(receiver,donor,chosen)
                full=mask[:,:,None,None].expand_as(mixed)
                if not (torch.equal(mixed[full],donor[full]) and torch.equal(mixed[~full],receiver[~full])):
                    raise RuntimeError("Fusion operator is not exact channel replacement")
            selected=selected_statistics(trace,chosen)
            for k in composition_sum:
                composition_sum[k] += float(np.asarray(selected[k]).sum())
            logits=cls(mixed)
            loss=criterion(logits,target,extra_info=None)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite classifier loss")
            opt.zero_grad()
            loss.backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in cls.parameters()):
                raise RuntimeError("Invalid classifier gradients")
            opt.step()
            sig.update({"selected_mask_sha256":array_hash(chosen),
                        "learning_rate":float(opt.param_groups[0]["lr"]),
                        "loss":float(loss.detach())})
            batch_rows.append(sig)
            step+=1
        check_frozen(bb,snapshot)
        val,logits,pred=evaluate(cls,val_labels,feature_maps=val_maps)
        if val["Overall"] > best:
            best=val["Overall"];best_epoch=epoch+1
            torch.save({"epoch":epoch+1,"classifier":{k:v.detach().cpu().clone() for k,v in cls.state_dict().items()},
                        "stage1_sha256":input_sha["stage1"],"seed":seed,"arm":arm},
                       folder/"best_fc.pth")
        if epoch == EPOCHS-1:
            torch.save({"epoch":epoch+1,"classifier":{k:v.detach().cpu().clone() for k,v in cls.state_dict().items()},
                        "stage1_sha256":input_sha["stage1"],"seed":seed,"arm":arm},
                       folder/"final_fc.pth")
            with (folder/"validation_epoch10.npz").open("xb") as fh:
                np.savez_compressed(fh,labels=val_labels, predictions=pred, logits=logits)
        log_rows.append({"epoch":epoch+1,**{k:val[k] for k in METRICS},
                         "train_lr":float(opt.param_groups[0]["lr"])})
        print(f"{arm} seed={seed} epoch={epoch+1}: Overall={val['Overall']:.4f}, "
              f"Head={val['Head']:.4f}, Tail={val['Tail']:.4f}",flush=True)
    if step != 850 or _test_events:
        raise AssertionError("Incomplete run or attempted official TEST access")
    check_frozen(bb,snapshot)
    if state_hash(bb)!=bb_init:
        raise RuntimeError("BN/backbone changed")
    _, end_sha = input_manifest(seed)
    if end_sha != input_sha:
        raise RuntimeError("Historical input file changed during run")
    # Load the tiny checkpoints and recompute the same held-out metrics.
    reloads={}
    for label in ("best","final"):
        c=torch.load(folder/f"{label}_fc.pth",map_location="cpu",weights_only=False)
        cls.load_state_dict(c["classifier"],strict=True)
        score,_,_=evaluate(cls,val_labels,feature_maps=val_maps)
        expected = log_rows[c["epoch"]-1]
        if any(score[k] != expected[k] for k in METRICS):
            raise RuntimeError(f"{label} checkpoint evaluation mismatch")
        reloads[label]={"epoch":c["epoch"],"sha256":sha(folder/f"{label}_fc.pth")}
    write_csv(folder/"epochs.csv",log_rows)
    write_csv(folder/"batch_signatures.csv",batch_rows)
    with (folder/"actual_masks.npz").open("xb") as fh:
        np.savez_compressed(fh,masks=mask_records)
    summary={"status":"complete","seed":seed,"arm":arm,"epoch10":log_rows[-1],
             "best_epoch":best_epoch,"best_Overall":best,
             "updates":step,"epochs":EPOCHS,"input_sha256":input_sha,
             "initial_backbone_hash":bb_init,"initial_classifier_hash":fc_init,
             "backbone_bn_bitwise_frozen":True,"paired_batch_count":850,
             "baseline_control_fields":PAIR_KEYS,
             "policy":{"shared_rule":"batch-mean eligible VS scores, bottom 48, uniform 16",
                       "uniform_shared_rng":"first receiver's Gumbel64; broadcast to batch",
                       "original_official_rng_exact":False},
             "selected_stats_mean":{k:v/(N*EPOCHS) for k,v in composition_sum.items()},
             "checkpoints":reloads,"runtime_seconds":time.perf_counter()-start,
             "test_access_events":_test_events}
    dump(folder/"summary.json",summary)
    dump(folder/"status.json",{"state":"complete","seed":seed,"arm":arm,"updates":850},replace=True)
    print("COMPLETE",seed,arm,"10 epochs / 850 updates",flush=True)


def _group_mask(y, group):
    y=np.asarray(y,dtype=np.int64)
    if group=="Overall":return np.ones(y.shape,dtype=bool)
    if group=="Head":return y < 36
    if group=="OfficialMedium":return (y >= 36)&(y < 71)
    if group=="Tail":return y >= 71
    raise ValueError(group)


def report():
    results={}
    table=[]
    masks={}
    for seed in SEEDS:
        for arm in ARMS:
            folder=run_dir(seed,arm)
            if not (folder/"summary.json").is_file():
                raise FileNotFoundError(f"All 8 fresh runs required: missing {folder}")
            summary=json.loads((folder/"summary.json").read_text(encoding="utf-8"))
            if summary["status"]!="complete" or summary["updates"]!=850 or not summary["backbone_bn_bitwise_frozen"]:
                raise RuntimeError(f"Invalid run: {folder}")
            rows=read_csv(folder/"batch_signatures.csv")
            if len(rows)!=850: raise RuntimeError("Short batch signatures")
            if arm=="U-Shared":baseline=rows
            elif any(any(x[k]!=y[k] for k in PAIR_KEYS) for x,y in zip(rows,baseline)):
                raise RuntimeError(f"Failed pairing: {seed} {arm}")
            b=np.load(folder/"validation_epoch10.npz",allow_pickle=False)
            y,p=b["labels"],b["predictions"]
            if len(y)!=1980 or set(y.tolist())!=set(range(1,100)):
                raise RuntimeError("Unexpected evaluation split")
            if not np.isfinite(b["logits"]).all() or b["logits"].shape!=(1980,100):
                raise RuntimeError("Invalid frozen evaluation logits")
            if arm=="U-Shared":baseline_y=y
            elif not np.array_equal(y,baseline_y):raise RuntimeError("Changed validation labels/order")
            results[(seed,arm)]=(y,p,summary)
            table.append({"seed":seed,"arm":arm,**{k:summary["epoch10"][k] for k in METRICS},
                          "best_epoch":summary["best_epoch"],"best_Overall":summary["best_Overall"]})
    contrasts=(("VS-Shared","U-Shared"),("VS-Individual","U-Individual"),
               ("U-Individual","U-Shared"),("VS-Individual","VS-Shared"),
               ("VS-Individual","U-Shared"))
    rows=[];classes=[]
    for seed in SEEDS:
        for improved,baseline in contrasts:
            y,pb,_=results[(seed,baseline)];_,pa,_=results[(seed,improved)]
            for group in ("Overall","Head","OfficialMedium","Tail"):
                mask=_group_mask(y,group)
                gained=int(np.sum((pb[mask]!=y[mask])&(pa[mask]==y[mask])))
                lost=int(np.sum((pb[mask]==y[mask])&(pa[mask]!=y[mask])))
                difference=100*(gained-lost)/int(mask.sum())
                rows.append({"seed":seed,"contrast":f"{improved} - {baseline}",
                             "group":group,"images":int(mask.sum()),"gained":gained,
                             "lost":lost,"net":gained-lost,"delta_pp":difference})
            for class_id in range(1,100):
                mask=y==class_id
                gained=int(np.sum((pb[mask]!=y[mask])&(pa[mask]==y[mask])))
                lost=int(np.sum((pb[mask]==y[mask])&(pa[mask]!=y[mask])))
                classes.append({"seed":seed,"contrast":f"{improved} - {baseline}",
                                "class_id":class_id,"images":int(mask.sum()),
                                "gained":gained,"lost":lost,"net":gained-lost})
    interaction=[]
    for seed in SEEDS:
        def delta(group,a,b):
            matching=[r for r in rows if r["seed"]==seed and r["group"]==group
                      and r["contrast"]==f"{a} - {b}"]
            return matching[0]["delta_pp"]
        for group in ("Overall","Tail"):
            interaction.append({"seed":seed,"group":group,
                "VS_by_granularity_interaction_pp":
                    delta(group,"VS-Individual","U-Individual")-delta(group,"VS-Shared","U-Shared"),
                "interpretation_limit":"Shared VS uses batch-mean scoring; not a pure mask-granularity interaction"})
    primary = ("VS-Shared - U-Shared", "VS-Individual - U-Individual")
    primary_rows = [r for r in rows if r["contrast"] in primary]
    criteria = {f"{contrast}|{group}|{seed}":next(
        r["delta_pp"] for r in primary_rows if r["contrast"] == contrast
        and r["group"] == group and r["seed"] == seed)
        for contrast in primary for group in ("Overall","Tail") for seed in SEEDS}
    provisional_go = all(
        criteria[f"{contrast}|Tail|{seed}"] >= 0.5
        and criteria[f"{contrast}|Overall|{seed}"] >= 0.
        for contrast in primary for seed in SEEDS
    )
    decision = "PROVISIONAL_GO_FOR_INDEPENDENT_CONFIRMATION" if provisional_go else "NOT_GO_OR_INCONCLUSIVE"
    write_csv(RESULTS/"epoch10.csv",table)
    write_csv(RESULTS/"paired_contrasts.csv",rows)
    write_csv(RESULTS/"per_class_contrasts.csv",classes)
    write_csv(RESULTS/"interactions.csv",interaction)
    lines=["# Low-VS same-environment 2x2 scientific report","",
        "This report is mechanically generated from **eight newly trained arms**; no historical",
        "accuracy is reused as a contemporary arm. All outcomes are on the repeatedly used",
        "**TRAIN-held-out 99-class development split (1980 images, class 0 absent)**,",
        "not on the official TEST set. Results are exploratory, NOT statistically independent",
        "confirmation of a new method.","",
        "## Fixed epoch 10, percent (%)","",
        "| Seed | Arm | Overall | Head | Medium-ID | Tail | Few |",
        "|---|---|---:|---:|---:|---:|---:|"]
    for r in table:
        lines.append(f"|{r['seed']}|{r['arm']}|{r['Overall']:.4f}|{r['Head']:.4f}|"
                     f"{r['OfficialMedium']:.4f}|{r['Tail']:.4f}|{r['Few']:.4f}|")
    lines+=["","## Paired comparisons: positive delta means first method wins","",
           "| Seed | Comparison | Group | delta pp | net correct |",
           "|---|---|---|---:|---:|"]
    for r in rows:
        if r["group"] in ("Overall","Tail"):
            lines.append(f"|{r['seed']}|{r['contrast']}|{r['group']}|{r['delta_pp']:+.4f}|{r['net']:+d}|")
    lines+=["",f"## Predefined decision: {decision}","",\n           "This only decides whether independent confirmation is worth running, not publication readiness.","",\n           "## Interpretation rules","",
       "Primary: same-granularity VS-Shared vs U-Shared and VS-Individual vs U-Individual.",
       "Secondary: VS-Individual vs U-Shared is a **confounded** replacement comparison.",
       "Only fixed epoch10 is confirmatory *within this exploratory screen*. Best validation",
       "epoch is descriptive. No official-test or independent-data benefit can be claimed.",
       "Shared score aggregation differs from individual scores, so the numeric interaction",
       "also contains scoring-aggregation effects, not just mask-sharing effects.",
       "No parameter tuning, no new seeds, no new channel candidate count.",
       "",
       "Suggested provisional GO only if both within-granularity VS controls improve Tail",
       "by >=0.5 pp in BOTH seeds and have nonnegative Overall in BOTH seeds; even a GO",
       "requires a genuinely independent follow-up split / new Stage I / unseen dataset.",
       "Otherwise: NO-GO for blanket superiority or INCONCLUSIVE for an ambiguous tradeoff.",
       "",
       "## Source data","",
       "`epoch10.csv`, `paired_contrasts.csv`, `per_class_contrasts.csv`,",
       "`interactions.csv`, and each run's `batch_signatures.csv`,",
       "`actual_masks.npz`, `summary.json`, and saved classification head.",
       ""]
    report_path=RESULTS/"LOWVS_MASK_PARITY_REPORT.md"
    with report_path.open("x",encoding="utf-8") as handle:handle.write("\n".join(lines))
    print("\n".join(lines[:24]))
    print("Report:", report_path,"Decision:",decision)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("unit","preflight","train","report"))
    parser.add_argument("--seed",type=int,choices=SEEDS)
    parser.add_argument("--arm",choices=ARMS)
    args=parser.parse_args()
    if args.action=="train" and (args.seed is None or args.arm is None):
        parser.error("train needs --seed and --arm")
    RESULTS.mkdir(parents=True,exist_ok=True)
    install_test_guard()
    try:
        if args.action=="unit":
            import unittest
            r=unittest.TextTestRunner(verbosity=2).run(
                unittest.defaultTestLoader.discover(str(HERE),pattern="test_mask_policy.py"))
            if not r.wasSuccessful():raise RuntimeError("Unit tests failed")
        elif args.action=="preflight":preflight()
        elif args.action=="train":train(args.seed,args.arm)
        elif args.action=="report":report()
        if _test_events:raise RuntimeError(f"Official TEST events: {_test_events}")
    except BaseException:
        # A failure stops the current invocation. Never retry under a new model or
        # silently override a mismatched pairing contract.
        print("FAILED - STOP; no next arm should be run until this is explained.",file=sys.stderr)
        traceback.print_exc()
        raise


if __name__=="__main__":
    main()
