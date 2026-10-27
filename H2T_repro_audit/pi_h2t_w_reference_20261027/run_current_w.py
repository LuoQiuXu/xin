#!/usr/bin/env python3
"""ONE Stage-II arm for PI-H2T reference-W ambiguity, reusing completed Stage I.

The PI paper's W reference convention is not author-verified. This is an
exploratory paper-parity intervention, not a new method and NOT a 49%-tuning
sweep. Keep the exact old data/augmentation/BN/optimizer/checkpoint.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
SOURCE=REPO/"H2T_repro_audit/pi_h2t_strict_20261027"
sys.path.insert(0,str(SOURCE))
import run_reproduction as old  # noqa: E402
from ratio_reference import cosine_receiver_ratio,select_weights,fuse_features

SOURCE_RUN=SOURCE/"runs/pih2t_if100_seed123_42_v1"
EXPECTED_STAGE1_SHA="64f53e45a1b7f73f494f9c88374607e714b58dc6eb55118fa31154c4389f93ef"
RUN_ID="w_reference_current_seed42_01"
REFERENCE="current_stage2"
EXPECTED_EPOCHS=10


def read(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def dump(p: Path,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf-8")


def sha(p: Path) -> str:
    return old.hashfile(p)


def source_params() -> SimpleNamespace:
    plan=read(SOURCE_RUN/"protocol.json")
    return SimpleNamespace(
        data_root=plan["data_root"],download=False,run_id=plan["run_id"],
        stage1_seed=plan["stage1_seed"],stage2_seed=plan["stage2_seed"],
        bn_mode=plan["bn_running_buffers"],w_reference=plan["w_reference"],
    )


def checks() -> tuple[SimpleNamespace,dict,dict,dict]:
    if not SOURCE_RUN.is_dir():raise FileNotFoundError(f"Historic run missing: {SOURCE_RUN}")
    args=source_params()
    plan=read(SOURCE_RUN/"protocol.json")
    if plan!=old.check_plan(args):raise RuntimeError("Historic source protocol drift")
    if args.w_reference!="frozen_stage1" or args.bn_mode!="update":
        raise RuntimeError("Historical reference does not use expected frozen W / BN update")
    if args.stage2_seed!=42 or args.stage1_seed!=123:
        raise RuntimeError("Unexpected historical seeds")
    if sha(SOURCE_RUN/"stage1_final_epoch200.pth")!=EXPECTED_STAGE1_SHA:
        raise RuntimeError("Pinned Stage I checkpoint SHA mismatch")
    old.verify_checkpoint_digest(SOURCE_RUN)
    old.verify_raw_cache(args)
    old_metrics={}
    for arm in ("h2tf","no_fusion"):
        base=SOURCE_RUN/"stage2"/arm
        summary=read(base/"summary.json")
        history=list(csv.DictReader((base/"train_epochs.csv").open(encoding="utf-8-sig")))
        if len(history)!=EXPECTED_EPOCHS:
            raise RuntimeError(f"Historical arm incomplete: {arm}")
        if summary["source_stage1_sha256"]!=EXPECTED_STAGE1_SHA:
            raise RuntimeError(f"Historical Stage I SHA mismatch: {arm}")
        old_metrics[arm]={"summary":summary,"history":history}
    frozen=old_metrics["h2tf"]["summary"];none=old_metrics["no_fusion"]["summary"]
    for key in ("stage2_initial_classifier_hash","final_bn_buffers_hash"):
        if frozen[key]!=none[key]:raise RuntimeError(f"Original paired-check failed: {key}")
    if any(a["input_stream_sha256"]!=b["input_stream_sha256"] for a,b in
           zip(old_metrics["h2tf"]["history"],old_metrics["no_fusion"]["history"])):
        raise RuntimeError("Original Stage II paired streams mismatch")
    return args,plan,old_metrics,read(SOURCE_RUN/"dataset_counts.json")


def output_path() -> Path:
    return HERE/"runs"/RUN_ID


def prereg() -> dict:
    args,plan,old_metrics,counts=checks()
    return {
       "study":"one-factor StageII W-reference paper-parity fork",
       "goal":"determine whether Eq8 W=current StageII FC changes fused training vs W=frozen StageI FC",
       "evidence_state":"author W-reference convention still UNVERIFIED",
       "status":"PAPER_PARITY_DIAGNOSTIC_NOT_OFFICIAL_REPRODUCTION",
       "source_run":"H2T_repro_audit/pi_h2t_strict_20261027/runs/pih2t_if100_seed123_42_v1",
       "stage1_sha256":EXPECTED_STAGE1_SHA,
       "source_protocol":plan,
       "dataset_sha256":counts["lt_data_and_labels_sha256"],
       "intervention":"Eq8 reference classifier ONLY: frozen_stage1 -> current_stage2",
       "unchanged":["stage1 model and PIF","Eq8 formula and normalization","Eq7 mixture",
                    "CE labels","BS/IS sampling","BN update and two forwards",
                    "StageII head init","SGD and cosine schedule","epoch10 evaluation"],
       "stage2_seed":42,"epochs":10,"batch_size":128,"amp":False,
       "official_test_role":"repeated paper-reproduction diagnostic only, NEVER hyperparameter/model selection",
       "analysis_predeclared":"Compare epoch10 current W to frozen W and matched no_fusion; report r mean + class groups; do NOT sweep more W choices",
       "stop":"any source lineage, hash, batch stream, BN or frozen-backbone mismatch",
       "baseline_epochs":len(old_metrics["h2tf"]["history"]),
       "source_sha256":{str((HERE/n).relative_to(REPO)).replace("\\","/"):sha(HERE/n)
          for n in ("run_current_w.py","ratio_reference.py")},
       "known_limits":["author W semantics uncertain","one seed","same reused official TEST",
                       "PIF init and author checkpoint selection not resolved"],
    }


def preflight() -> None:
    p=prereg()
    if not torch.cuda.is_available():raise RuntimeError("CUDA unavailable")
    if output_path().exists():raise FileExistsError(output_path())
    print("PREFLIGHT PASS; no training started")
    print(json.dumps({k:v for k,v in p.items() if k not in ("source_protocol",)},indent=2,ensure_ascii=False))


def run() -> None:
    preflight()
    args,_,historical,counts=checks()
    old.seeding(args.stage2_seed)
    device=torch.device("cuda:0")
    net,head,stage1_sha=old.read_stage1(SOURCE_RUN,device)
    if stage1_sha!=EXPECTED_STAGE1_SHA:raise RuntimeError("Stage I lineage mismatch")
    for parameter in net.parameters():parameter.requires_grad_(False)
    fixed_stage1_w=head.fc.weight.detach().clone()
    initial_head_hash=old.state_hash(head.state_dict())
    initial_net_hash=old.state_hash(dict(net.named_parameters()))
    frozen_summary=historical["h2tf"]["summary"]
    if initial_head_hash!=frozen_summary["stage2_initial_classifier_hash"]:
        raise RuntimeError("Initial classifier weights differ from frozen-W reference arm")
    loader,test,_,lt_hash=old.build_loader(args,2)
    if lt_hash!=counts["lt_data_and_labels_sha256"]:
        raise RuntimeError("LT train subset drift")
    target=output_path()
    target.mkdir(parents=True,exist_ok=False)
    dump(target/"preregistered_protocol.json",prereg())
    opt=torch.optim.SGD(head.parameters(),lr=.02,momentum=.9,weight_decay=5e-4)
    logs=[]
    for epoch in range(EXPECTED_EPOCHS):
        net.train()    # same BN running-buffer update as prior Stage II
        head.train()
        lr=.2*old.cosine_lr(.1,epoch,EXPECTED_EPOCHS)
        for g in opt.param_groups:g["lr"]=lr
        stream=hashlib.sha256()
        train_loss=0.
        live_r=[];initial_r=[]
        live_y=[];donor_y=[]
        for images,labels,meta in loader:
            for value in (images,labels,meta["sample_image"],meta["sample_label"]):
                stream.update(value.contiguous().numpy().tobytes())
            live_y.append(labels.numpy().copy())
            donor_y.append(meta["sample_label"].numpy().copy())
            x=images.to(device,non_blocking=True)
            y=labels.to(device,non_blocking=True)
            d=meta["sample_image"].to(device,non_blocking=True)
            with torch.no_grad():
                a=old.pool_feature(net(x))
                b=old.pool_feature(net(d))
                r=cosine_receiver_ratio(a,y,select_weights(fixed_stage1_w,head.fc.weight,REFERENCE))
                r0=cosine_receiver_ratio(a,y,select_weights(fixed_stage1_w,head.fc.weight,"frozen_stage1"))
                mixed=fuse_features(a,b,r)
                if not bool(torch.isfinite(r).all() and torch.isfinite(mixed).all()):
                    raise FloatingPointError("Non-finite r/mixed features")
                live_r.append(r.cpu().numpy().copy())
                initial_r.append(r0.cpu().numpy().copy())
            logits=old.classify_vector(head,mixed)
            loss=F.cross_entropy(logits,y)
            if not bool(torch.isfinite(loss)):raise FloatingPointError("non-finite CE")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            train_loss+=float(loss.detach())*len(y)
        signature=stream.hexdigest()
        expected=historical["h2tf"]["history"][epoch]["input_stream_sha256"]
        if signature!=expected:
            raise RuntimeError(f"Paired source image/label stream mismatch at epoch {epoch+1}")
        live=np.concatenate(live_r);init=np.concatenate(initial_r)
        np.savez_compressed(target/f"ratio_epoch{epoch+1:02d}.npz",r_current=live,r_frozen=init,
                           receiver_labels=np.concatenate(live_y),donor_labels=np.concatenate(donor_y))
        line={
           "epoch":epoch+1,"lr":lr,"train_ce":train_loss/len(loader.dataset),
           "r_current_mean":float(live.mean(dtype=np.float64)),
           "r_frozen_on_current_features_mean":float(init.mean(dtype=np.float64)),
           "donor_current_mean":float((1-live).mean(dtype=np.float64)),
           "candidate_r_diff_abs_mean":float(np.abs(live-init).mean(dtype=np.float64)),
           "input_stream_sha256":signature,
        }
        logs.append(line)
        old.save_rows(target/"epochs.csv",logs)
        print(f"CURRENT_W stage2 {epoch+1}/10",line,flush=True)
    if old.state_hash(dict(net.named_parameters()))!=initial_net_hash:
        raise RuntimeError("Frozen PIF/backbone parameters mutated")
    bn=hashlib.sha256(b"".join(v.detach().cpu().numpy().tobytes()
        for k,v in net.state_dict().items()
        if "running_" in k or "num_batches_tracked" in k)).hexdigest()
    if bn!=frozen_summary["final_bn_buffers_hash"]:
        raise RuntimeError("Final BN running buffers differ from prior paired run")
    old.verify_raw_cache(args)
    eval_metrics,pred,labels,logits=old.evaluate(net,head,test,device)
    assert len(labels)==10000 and logits.shape==(10000,100)
    checkpoint=target/"classifier_current_W_epoch10.pth"
    torch.save({"arm":"h2tf_current_stage2_W","epoch":10,"stage1_sha":stage1_sha,
                "classifier":{k:v.detach().cpu() for k,v in head.state_dict().items()},
                "bn_mode":"update","w_reference":REFERENCE},checkpoint)
    np.savez_compressed(target/"test_epoch10.npz",
                        labels=labels.numpy(),pred=pred.numpy(),logits=logits.numpy())
    dump(target/"result.json",{
        "status":"COMPLETED_CONDITIONAL_PARITY_PROBE",
        "metrics_epoch10":eval_metrics,"initial_classifier_hash":initial_head_hash,
        "final_classifier_hash":old.state_hash(head.state_dict()),
        "source_stage1_sha256":stage1_sha,"final_bn_buffers_hash":bn,
        "epoch10_donor_mass_current":logs[-1]["donor_current_mean"],
        "average_donor_mass_over_training":float(np.mean([x["donor_current_mean"] for x in logs])),
        "classifier_checkpoint_sha256":sha(checkpoint),
        "test_logits_sha256":sha(target/"test_epoch10.npz"),
        "warning":"one seed; official TEST repeatedly used; author W convention unverified",
    })
    report()
    print("ONE-ARM CURRENT-W STUDY COMPLETE; see",target,flush=True)


def report() -> None:
    args,_,history,_=checks()
    target=output_path()
    stored=read(target/"preregistered_protocol.json")
    pinned=prereg()
    if stored!=pinned:
        raise RuntimeError("Current source/preregistered protocol drift")
    result=read(target/"result.json")
    frozen_summary=history["h2tf"]["summary"]
    for field,expected in (
        ("source_stage1_sha256",EXPECTED_STAGE1_SHA),
        ("initial_classifier_hash",frozen_summary["stage2_initial_classifier_hash"]),
        ("final_bn_buffers_hash",frozen_summary["final_bn_buffers_hash"])):
        if result.get(field)!=expected:
            raise RuntimeError(f"Invalid new source/FC/BN provenance: {field}")
    if sha(target/"classifier_current_W_epoch10.pth")!=result["classifier_checkpoint_sha256"]:
        raise RuntimeError("New FC checkpoint SHA drift")
    if sha(target/"test_epoch10.npz")!=result["test_logits_sha256"]:
        raise RuntimeError("New TEST NPZ SHA drift")
    with (target/"epochs.csv").open(encoding="utf-8-sig") as f:
        epoch_rows=list(csv.DictReader(f))
    if len(epoch_rows)!=EXPECTED_EPOCHS:
        raise RuntimeError("Incomplete current-W training history")
    for idx,entry in enumerate(epoch_rows):
        if entry["input_stream_sha256"]!=history["h2tf"]["history"][idx]["input_stream_sha256"]:
            raise RuntimeError(f"Replayed paired stream mismatch epoch {idx+1}")
    with np.load(target/"test_epoch10.npz",allow_pickle=False) as z:
        labels=z["labels"].copy();pred=z["pred"].copy();logits=z["logits"].copy()
    if not np.array_equal(pred,logits.argmax(axis=1)):
        raise RuntimeError("Current-W output does not match logit argmax")
    old_preds={}
    for arm in ("h2tf","no_fusion"):
        with np.load(SOURCE_RUN/"stage2"/arm/"test_epoch10_predictions.npz",allow_pickle=False) as z:
            if not np.array_equal(labels,z["labels"]):
                raise RuntimeError("Different ordered TEST labels")
            old_preds[arm]=z["pred"].copy()
    metrics=result["metrics_epoch10"]
    original=history["h2tf"]["summary"]["metrics"]
    nofusion=history["no_fusion"]["summary"]["metrics"]
    # One pre-specified science contrast, no best-epoch selection or parameter search.
    delta_current_frozen={k:metrics[k]-original[k] for k in ("Overall","Many","Medium","Few")}
    delta_current_nofusion={k:metrics[k]-nofusion[k] for k in ("Overall","Many","Medium","Few")}
    rows=[]
    for klass in range(100):
        mask=labels==klass
        if int(mask.sum())!=100:raise RuntimeError("Wrong TEST class count")
        cur=pred[mask]==klass
        oldcorrect=old_preds["h2tf"][mask]==klass
        rows.append({"class":klass,
                     "current_correct":int(cur.sum()),
                     "frozen_correct":int(oldcorrect.sum()),
                     "current_minus_frozen_pp":int(cur.sum()-oldcorrect.sum()),
                     "wins":int(np.count_nonzero(cur & ~oldcorrect)),
                     "losses":int(np.count_nonzero(oldcorrect & ~cur))})
    old.save_rows(target/"per_class_current_vs_frozen.csv",rows)
    dump(target/"comparison.json",{
        "current_w":metrics,"frozen_w":original,"matched_no_fusion":nofusion,
        "current_minus_frozen_pp":delta_current_frozen,
        "current_minus_no_fusion_pp":delta_current_nofusion,
        "warning":"For source protocol/one seed only, not official author implementation or independent TEST validation"
    })
    head=["|Arm (Stage II epoch10)|Overall %|Many %|Medium %|Few %|",
          "|---|---:|---:|---:|---:|"]
    for title,m in (("no fusion",nofusion),("frozen Stage I W",original),("current Stage II W",metrics)):
        head.append("|"+title+"|"+"|".join(f"{m[k]:.2f}" for k in ("Overall","Many","Medium","Few"))+"|")
    total=sum(float(z["donor_current_mean"]) for z in csv.DictReader((target/"epochs.csv").open(encoding="utf-8-sig")))/10
    text="\n".join([
        "# PI-H2T Stage-II W-reference parity diagnostic",
        "",
        "This is **one pre-specified Stage-II protocol fork, not a published-method fix**.",
        "Reuses the exact locked 2-param PIF Stage I epoch200 checkpoint; no new Stage I training.",
        "",
        "## Fixed epoch10 accuracy (existing source arms + one new current-W arm)",
        *head,
        "",
        f"Current W minus frozen W Overall = {delta_current_frozen['Overall']:+.4f} pp.",
        f"Current W mean donor mass over 10 epochs = {total:.6f}.",
        "",
        "Input/label streams matched all 10 epochs; initial classifier, backbone, dataset and BN buffers remain paired.",
        "Compare epoch-wise reference-r, current-r and CE in epochs.csv; do not infer ratio causality from aggregate accuracy.",
        "",
        "## Interpretation restrictions",
        "- The v2 paper's W-reference update convention is still unverified.",
        "- This is an exploratory single-seed reconstruction on a previously used CIFAR100 TEST set.",
        "- If current W is better, do NOT call the author implementation reproduced; other protocol gaps remain.",
        "- If current W is worse or unchanged, STOP this W-reference fork; no formula reversal or scale sweep.",
        "- Do not implement donor selection, TailOff, Low-VS, margin protection or any new module in this round.",
        "- If a local historical input/source/hash mismatch is detected, report BLOCKED instead of a metric.",
    ])+"\n"
    (target/"SCIENTIFIC_REPORT.md").write_text(text,encoding="utf-8")
    print("RESULTS:",json.dumps(read(target/"comparison.json"),ensure_ascii=False,indent=2))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command",choices=("preflight","run","report"))
    args=ap.parse_args()
    if args.command=="preflight":preflight()
    elif args.command=="run":run()
    elif args.command=="report":report()


if __name__=="__main__":
    main()
