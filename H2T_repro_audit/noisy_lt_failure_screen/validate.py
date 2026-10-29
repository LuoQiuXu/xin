"""Independent persisted-file validation; no training and no model selection."""
import csv
import json
from pathlib import Path

import numpy as np
import torch

from runtime import *
from train import check_lock, now
from compare_outcomes import compare
from evaluate import group_masks


def rows(path):
    with Path(path).open(newline="",encoding="utf-8") as f:return list(csv.DictReader(f))


def tensors(path):return torch.load(path,map_location="cpu",weights_only=True)


def main():
    lock=check_lock();g=install_guard("validation")
    assert read(OUT / "DRYRUN.json")["passed"]
    assert read(OUT / "TRAINING_COMPLETED.json")["completed_runs"]==10
    data=read(OUT / "protocol/DATA_AUDIT.json")
    assert data["empty_observed_classes"]==[] and data["stop_reason"] is None
    attest=read(OUT / "INPUT_ATTESTATION.json")
    assert attest["human_clean_matches_Python_TRAIN_all50000"]
    assert sha(ROOT / "H2T/data/cifar100/cifar-100-python/train")==attest["train_file_sha256"]
    assert sha(ROOT / "H2T_repro_audit/clean_vs_runs/clean_s1_seed42_vs_seed123_20261013/split_indices.npz")==attest["historical_split_sha256"]
    for name,h in data["manifest_sha256"].items():assert sha(OUT / "protocol" / name)==h
    with np.load(OUT / "protocol/clean_train.npz",allow_pickle=False) as z:clean_ids,clean_labels=z["ids"],z["labels"]
    with np.load(OUT / "protocol/noisy_train.npz",allow_pickle=False) as z:noisy_ids,noisy_labels=z["ids"],z["labels"]
    with np.load(OUT / "protocol/clean_validation.npz",allow_pickle=False) as z:val_ids,labels=z["ids"],z["labels"]
    assert np.array_equal(clean_ids,noisy_ids) and np.intersect1d(clean_ids,val_ids).size==0
    assert int((clean_labels!=noisy_labels).sum())==3902
    assert np.bincount(noisy_labels,minlength=100).tolist()==data["observed_noisy_counts"]
    initial_meta=read(OUT / "INITIALIZATION.json")["checkpoint"]
    assert sha(ROOT / initial_meta["file"])==initial_meta["sha256"]
    init=tensors(ROOT / initial_meta["file"]);assert init["epoch"]==0
    s1={};s2={};checks={}
    for r in ("clean","noisy"):
        proof=read(OUT / ("stage1_"+r) / "RUN.json");start=read(OUT / ("stage1_"+r) / "START.json")
        ep=rows(OUT / ("stage1_"+r) / "epochs.csv")
        assert len(ep)==200 and [int(e["epoch"]) for e in ep]==list(range(1,201))
        assert all(np.isfinite(float(e["Loss"])) and int(e["updates"])==85*int(e["epoch"]) for e in ep)
        assert proof["parent_sha256"]==initial_meta["sha256"]
        assert proof["optimizer_updates"]==proof["finite_gradient_checks"]==proof["BB_forwards"]==17000
        assert len(proof["BN_final_counters"])==31 and set(proof["BN_final_counters"].values())=={17000}
        assert sha(ROOT / proof["checkpoint"]["file"])==proof["checkpoint"]["sha256"]
        weight=tensors(ROOT / proof["checkpoint"]["file"])
        assert weight["epoch"]==200 and state_hash(weight["bb"])==proof["checkpoint"]["bb_state_sha"]
        assert state_hash(weight["bb"])!=state_hash(init["bb"]) and state_hash(weight["fc"])!=state_hash(init["fc"])
        assert all(torch.isfinite(v).all() for v in list(weight["bb"].values())+list(weight["fc"].values()))
        assert start["optimizer_initial_state_entries"]==0 and start["trainable_parameters"]==470004
        s1[r]=(proof,weight)
    assert s1["clean"][0]["initial"]==s1["noisy"][0]["initial"]
    for key in ("input_image_epoch_hashes","mixup_lambda_epoch_hashes"):
        assert s1["clean"][0][key]==s1["noisy"][0][key] and len(s1["clean"][0][key])==200
    checks["stage1_same_initialization_images_and_mixup_all200_epochs"]=True
    for r in ("clean","noisy"):
        for s in (124,125):
            pair=[]
            for m in ("no_fusion","h2t"):
                name=f"stage2_{r}_seed{s}_{m}";proof=read(OUT / name / "RUN.json");start=read(OUT / name / "START.json")
                ep=rows(OUT / name / "epochs.csv");control_path=ROOT / proof["controls_file"]
                assert len(ep)==10 and [int(e["epoch"]) for e in ep]==list(range(1,11))
                assert proof["optimizer_updates"]==proof["finite_gradient_checks"]==850 and proof["BB_forwards"]==1700
                assert set(proof["BN_final_counters"].values())=={18700} and len(proof["BN_final_counters"])==31
                assert proof["parent_sha256"]==s1[r][0]["checkpoint"]["sha256"]
                assert start["optimizer_initial_state_entries"]==0 and start["trainable_parameters"]==6500
                assert proof["H2T_replacements"]==(850 if m=="h2t" else 0)
                assert sha(control_path)==proof["controls_file_sha256"]
                controls=rows(control_path);assert len(controls)==850
                for epoch in range(10):
                    chunk=[{**v,"batch":int(v["batch"])} for v in controls[epoch*85:(epoch+1)*85]]
                    assert [v["batch"] for v in chunk]==list(range(85))
                    assert hashlib.sha256(json.dumps(chunk,sort_keys=True).encode()).hexdigest()==proof["all_batch_control_epoch_hashes"][epoch]
                weight=tensors(ROOT / proof["checkpoint"]["file"])
                assert sha(ROOT / proof["checkpoint"]["file"])==proof["checkpoint"]["sha256"] and weight["epoch"]==10
                assert all(torch.isfinite(v).all() for v in list(weight["bb"].values())+list(weight["fc"].values()))
                params={k:v for k,v in weight["bb"].items() if not any(x in k for x in ("running_mean","running_var","num_batches_tracked"))}
                assert all(torch.equal(v,s1[r][1]["bb"][k]) for k,v in params.items())
                assert state_hash(weight["fc"])!=state_hash(s1[r][1]["fc"])
                assert proof["final_BB_parameters_sha"]==proof["initial"]["bb_parameters_sha"]
                pair.append((proof,controls,weight));s2[name]=proof
            assert pair[0][0]["initial"]==pair[1][0]["initial"] and pair[0][1]==pair[1][1]
            assert pair[0][0]["BN_final_sha"]==pair[1][0]["BN_final_sha"] and pair[1][0]["pair_checked_before_all_updates"]
            for k in pair[0][2]["bb"]:assert torch.equal(pair[0][2]["bb"][k],pair[1][2]["bb"][k])
    checks["stage2_all3400_pairs_exact_controls_BN_and_frozen_BB"]=True
    for name,proof in [("stage1_"+r,s1[r][0]) for r in s1]+list(s2.items()):
        guard=proof["IO_guard"];assert_guard(guard)
        allowed={str(Path(OUT / "protocol" / (proof["regime"]+"_train.npz")).relative_to(ROOT)).replace("\\","/"),
                 str((CACHE / "inputs/lt_images.npy").relative_to(ROOT)).replace("\\","/"),
                 str((CACHE / "initial_seed42.pth").relative_to(ROOT)).replace("\\","/"),
                 proof["checkpoint"]["file"].replace("\\","/")}
        if proof["stage"]==2:
            allowed.add(s1[proof["regime"]][0]["checkpoint"]["file"].replace("\\","/"))
        assert set(guard["numeric_files_read"])<=allowed, "Unexpected training numerical input: "+name
    checks["no_training_validation_clean_oracle_or_official_TEST_access"]=True
    evaluation=OUT / "evaluation";eval_audit=read(evaluation / "EVALUATION_AUDIT.json");assert_guard(eval_audit["IO_guard"])
    eval_lock=read(evaluation / "EVALUATION_LOCK.json")
    metric_rows=rows(evaluation / "final_metrics.csv");assert len(metric_rows)==32
    masks=group_masks(labels,np.array(data["true_lt_counts"]));logit_files={}
    for name,entry in eval_audit["models"].items():
        assert entry["single_eval_passes"]==1 and entry["checkpoint_sha256"]==eval_lock["checkpoints_before_any_evaluation"][name]
        path=CACHE / "evaluation" / (name+".npz");assert sha(path)==entry["predictions_file_sha256"]
        with np.load(path,allow_pickle=False) as z:
            assert np.array_equal(z["raw_ids"],val_ids) and np.array_equal(z["labels"],labels)
            logits,pred=z["logits"],z["predictions"]
            assert logits.shape==(1980,100) and logits.dtype==np.float32 and np.isfinite(logits).all()
            assert np.array_equal(pred,logits.argmax(1))
        logit_files[name]=pred
        if name.startswith("stage2"):
            for row in [x for x in metric_rows if name==f"stage2_{x['regime']}_seed{x['stage2_seed']}_{x['method']}"]:
                mask=masks[row["group"]]
                assert int(row["n"])==int(mask.sum()) and abs(float(row["accuracy_pct"])-100*(pred[mask]==labels[mask]).mean())<1e-10
                assert row["val_ids_sha"]==ahash(val_ids) and int(row["epoch"])==10
    expected=compare(metric_rows);actual=rows(OUT / "analysis/paired_interactions.csv")
    assert [{k:str(v) for k,v in r.items()} for r in expected]==actual
    for row in rows(evaluation / "paired_reclassifications.csv"):
        prefix=f"stage2_{row['regime']}_seed{row['stage2_seed']}_"
        nf=logit_files[prefix+"no_fusion"]==labels;h=logit_files[prefix+"h2t"]==labels;mask=masks[row["group"]]
        assert int(row["wrong_to_right"])==int((~nf&h&mask).sum()) and int(row["right_to_wrong"])==int((nf&~h&mask).sum())
    class_rows=rows(evaluation / "per_class_accuracy.csv");class_changes=rows(evaluation / "per_class_reclassifications.csv")
    assert len(class_rows)==1000 and len(class_changes)==400
    for row in class_rows:
        name="stage1_"+row["regime"] if row["stage"]=="1" else f"stage2_{row['regime']}_seed{row['stage2_seed']}_{row['method']}"
        c=int(row["class_id"]);mask=labels==c;correct=int((logit_files[name][mask]==labels[mask]).sum())
        assert int(row["n"])==int(mask.sum()) and int(row["correct"])==correct
        assert (row["accuracy_pct"]=="") if c==0 else abs(float(row["accuracy_pct"])-100*correct/int(mask.sum()))<1e-10
    for row in class_changes:
        prefix=f"stage2_{row['regime']}_seed{row['stage2_seed']}_";mask=labels==int(row["class_id"])
        nf=logit_files[prefix+"no_fusion"]==labels;h=logit_files[prefix+"h2t"]==labels
        assert int(row["wrong_to_right"])==int((~nf&h&mask).sum()) and int(row["right_to_wrong"])==int((nf&~h&mask).sum())
    checks["all_final_logits_recompute_exact_metrics_interactions_and_paired_counts"]=True
    historical_before=read(CACHE / "historical_metadata_before.json")
    protected=[]
    for root in ("H2T_official_clean","H2T","PI_H2T_review","H2T_repro_audit","data"):
        for p in (ROOT / root).rglob("*"):
            if p.is_file() and not p.is_relative_to(TASK):
                st=p.stat();protected.append([p.relative_to(ROOT).as_posix(),st.st_size,str(st.st_mtime_ns)])
    before_map={r[0]:r[1:] for r in historical_before};after_map={r[0]:r[1:] for r in protected}
    changes=[p for p in before_map.keys()|after_map.keys() if before_map.get(p)!=after_map.get(p)]
    assert not changes,"Protected historical metadata changed: "+str(changes[:10])
    checks["historical20543_metadata_unchanged_and_locked_source_SHA256_unchanged"]=True
    assert_guard(g)
    def label_group(n):return "Many" if n>100 else ("Medium" if n>=20 else "Few")
    frequency=[]
    for c in range(100):
        true_n=data["true_lt_counts"][c];obs_n=data["observed_noisy_counts"][c]
        frequency.append(dict(class_id=c,benchmark_clean_LT_n=true_n,observed_noisy_n=obs_n,
            observed_count_delta=obs_n-true_n,benchmark_group=label_group(true_n),
            observed_group=label_group(obs_n),LT_noise_fraction_within_true_class=data["by_true_class_noise_rate"][c],
            validation_n=int((labels==c).sum())))
    with (OUT / "CLASS_FREQUENCY_AUDIT.csv").open("x",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(frequency[0]));writer.writeheader();writer.writerows(frequency)
    write_json(OUT / "VALIDATION.json",dict(passed=True,UTC=now(),checks=checks,
        protected_files_checked=len(historical_before),metadata_changes=changes,
        historical_check_scope="Full path/size/mtime_ns metadata census; cryptographic content checks additionally cover locked original source and used historical inputs. Not a full content hash of all20543 files.",IO_guard=g))
    local=[]
    for p in sorted(CACHE.rglob("*")):
        if p.is_file():local.append(dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=sha(p)))
    write_json(OUT / "RUN_MANIFEST.json",dict(UTC=now(),branch=lock["branch"],initial_head=read(OUT / "RUN_CONTEXT.json")["initial_head"],
        code_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in TASK.glob("*.py")},
        locked_author_source_and_training_code=lock["code_sha256"],official_label_source=attest,
        data_audit_sha256=sha(OUT / "protocol/DATA_AUDIT.json"),training_lock_sha256=sha(OUT / "TRAINING_LOCK.json"),
        checkpoints={**{"stage1_"+r:s1[r][0]["checkpoint"] for r in s1},**{k:v["checkpoint"] for k,v in s2.items()}},
        local_only_files=local,local_only_bytes=sum(x["bytes"] for x in local),
        official_TEST_used=False,selection="fixed epoch200/10 only; all final weights locked before evaluation",
        validation_passed=True))
    print(json.dumps(dict(passed=True,checks=checks,local_only_files=len(local),local_only_bytes=sum(x["bytes"] for x in local)),indent=2))


if __name__=="__main__":main()
