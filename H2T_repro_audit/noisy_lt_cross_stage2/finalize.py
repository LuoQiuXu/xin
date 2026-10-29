"""Validate saved crossed results and index local assets; never trains/evaluates."""
import csv
import hashlib
from pathlib import Path

import numpy as np
import torch

from cross_stage2 import (THIS, ROOT, SRC_OUT, OUT, LOCAL, GROUPS, name, read,
    sha, state_hash, safe_json, check_sources, frozen_runtime)
from analyze_factorial import csvrows, assess


def write_csv(path,rows):
    with path.open("x",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    check_sources();assert read(OUT/"DRYRUN.json")["passed"]
    lock=read(OUT/"EVAL_LOCK.json");assert lock["all_8_complete_before_eval"]
    audit=read(OUT/"EVALUATION_AUDIT.json")
    frozen_runtime.assert_guard(audit["IO_guard"])
    with np.load(SRC_OUT/"protocol/clean_validation.npz",allow_pickle=False) as z:ids,truth=z["ids"],z["labels"]
    counts=np.array(read(SRC_OUT/"protocol/DATA_AUDIT.json")["true_lt_counts"])
    masks=dict(Overall=np.ones(1980,bool),Many=counts[truth]>100,
        Medium=(counts[truth]>=20)&(counts[truth]<=100),Few=counts[truth]<20)
    source={r:torch.load(ROOT/p["file"],map_location="cpu",weights_only=True) for r,p in read(OUT/"SOURCE_AUDIT.json")["checkpoint_sources"].items()}
    raw=csvrows(OUT/"crossed_final_metrics.csv");assert len(raw)==32
    manifest=[];pair_checks=[]
    for r,l in (("clean","noisy"),("noisy","clean")):
        for seed in (124,125):
            pair=[]
            for method in ("no_fusion","h2t"):
                tag=name(r,l,seed,method);run=read(OUT/tag/"RUN.json")
                cp=ROOT/run["checkpoint"]["file"];assert sha(cp)==run["checkpoint"]["sha256"]==lock["checkpoints"][tag]
                assert run["checkpoint"]["epoch"]==10 and not run["dryrun"]
                weights=torch.load(cp,map_location="cpu",weights_only=True)
                assert weights["epoch"]==10 and state_hash(weights["bb"])==run["checkpoint"]["bb_state_sha"]
                assert state_hash(weights["fc"])==run["checkpoint"]["fc_state_sha"]
                assert run["initial"]["bb_state_sha"]==state_hash(source[r]["bb"])
                assert run["initial"]["fc_state_sha"]==state_hash(source[r]["fc"])
                assert all(torch.equal(v,source[r]["bb"][k]) for k,v in weights["bb"].items() if not any(x in k for x in ("running_mean","running_var","num_batches_tracked")))
                assert len(run["BN_final_counters"])==31 and set(run["BN_final_counters"].values())=={18700}
                assert run["optimizer_updates"]==run["finite_gradient_checks"]==850 and run["BB_forwards"]==1700
                assert run["optimizer_initial_entries"]==0 and run["BB_parameters_frozen"]
                assert run["H2T_replacements"]==(850 if method=="h2t" else 0)
                frozen_runtime.assert_guard(run["IO_guard"])
                controls=LOCAL/tag/"batch_controls.csv";assert sha(controls)==run["controls_file_sha"]
                batch=csvrows(controls);assert len(batch)==850
                epochs=csvrows(OUT/tag/"epochs.csv");assert [int(x["epoch"]) for x in epochs]==list(range(1,11))
                assert all(np.isfinite(float(x["Loss"])) for x in epochs)
                saved=LOCAL/tag/"evaluation.npz";assert sha(saved)==audit["models"][tag]["logits_sha256"]
                assert audit["models"][tag]["single_eval_passes"]==1
                with np.load(saved,allow_pickle=False) as z:
                    assert np.array_equal(z["raw_ids"],ids) and np.array_equal(z["labels"],truth)
                    assert z["logits"].shape==(1980,100) and np.isfinite(z["logits"]).all()
                    prediction=z["logits"].argmax(1);assert np.array_equal(prediction,z["predictions"])
                rows=[x for x in raw if x["representation_regime"]==r and x["label_regime"]==l and int(x["stage2_seed"])==seed and x["method"]==method]
                assert len(rows)==4
                for x in rows:
                    mask=masks[x["group"]]
                    assert int(x["epoch"])==10 and int(x["n"])==int(mask.sum())
                    assert abs(float(x["accuracy_pct"])-100*(prediction[mask]==truth[mask]).mean())<1e-10
                    assert x["val_ids_sha"]==audit["models"][tag]["val_ids_sha"]
                diagonal=csvrows(SRC_OUT/"primary_cache"/f"stage2_{l}_seed{seed}_{method}"/"batch_controls.csv")
                keys=[k for k in batch[0] if k not in ("receiver_features_sha","donor_features_sha")]
                assert [{k:x[k] for k in keys} for x in batch]==[{k:x[k] for k in keys} for x in diagonal]
                pair.append((run,batch,weights))
                manifest.append(dict(tag=tag,representation=r,labels=l,seed=seed,method=method,
                    source_sha=run["source_checkpoint_sha"],epoch=10,checkpoint_path=str(cp.relative_to(ROOT)),
                    checkpoint_sha256=sha(cp),elapsed_seconds=run["elapsed_seconds"],peak_GPU_MiB=run["peak_GPU_bytes"]/1024**2))
            a,b=pair;assert a[0]["initial"]==b[0]["initial"] and a[1]==b[1]
            assert a[0]["BN_final_sha"]==b[0]["BN_final_sha"] and b[0]["paired_all_steps"]
            assert all(torch.equal(a[2]["bb"][k],b[2]["bb"][k]) for k in a[2]["bb"])
            pair_checks.append(dict(representation=r,labels=l,seed=seed,exact_paired_batches=850,
                same_final_BN=True,diagonal_sampling_augmentation_and_mask_equal=True))
    diagonal=csvrows(SRC_OUT/"evaluation/final_metrics.csv")
    assert {x["val_ids_sha"] for x in raw}=={x["val_ids_sha"] for x in diagonal}
    details,effects,summary=assess(diagonal,raw)
    assert summary==read(OUT/"analysis/summary.json")
    means=[]
    for r in ("clean","noisy"):
        for l in ("clean","noisy"):
            for g in GROUPS:
                xs=[x for x in details if x["representation"]==r and x["observed_labels"]==l and x["group"]==g]
                assert len(xs)==2
                means.append(dict(representation=r,observed_labels=l,group=g,
                    no_fusion_accuracy_pct=float(np.mean([x["no_fusion_accuracy_pp"] for x in xs])),
                    h2t_accuracy_pct=float(np.mean([x["h2t_accuracy_pp"] for x in xs])),
                    h2t_gain_pp=float(np.mean([x["h2t_gain_pp"] for x in xs]))))
    write_csv(OUT/"analysis/factorial_cell_means.csv",means)
    write_csv(OUT/"CHECKPOINT_INDEX.csv",manifest)
    before=read(LOCAL/"historical_before.json");current=[]
    for dirname in ("H2T_official_clean","H2T","PI_H2T_review","H2T_repro_audit","data"):
        for p in (ROOT/dirname).rglob("*"):
            if p.is_file() and not p.is_relative_to(THIS):
                s=p.stat();current.append([p.relative_to(ROOT).as_posix(),s.st_size,str(s.st_mtime_ns)])
    assert sorted(before)==sorted(current),"Historical metadata changed"
    for p,h in read(OUT/"SOURCE_AUDIT.json")["data_sha256"].items():assert sha(ROOT/p)==h
    perclass=csvrows(OUT/"crossed_per_class.csv");assert len(perclass)==800
    for x in perclass:
        with np.load(LOCAL/x["tag"]/"evaluation.npz",allow_pickle=False) as z:
            mask=truth==int(x["class_id"])
            assert int(x["n"])==int(mask.sum()) and int(x["correct"])==int((z["predictions"][mask]==truth[mask]).sum())
    safe_json(OUT/"VALIDATION.json",dict(passed=True,paired_runs=4,paired_batches=3400,
        same_S_diagonal_stream_pairing_passed=True,all_final_logits_recomputed=True,
        old_validation_ids_and_labels_equal=True,historical_metadata_entries=len(before),historical_metadata_changed=0,
        stage1_epochs_trained=0,stage2_updates=6800,checkpoint_sources_unchanged=True,pair_details=pair_checks))
    files=[dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(OUT.rglob("*")) if p.is_file() and (p.is_relative_to(LOCAL) or p.suffix==".log")]
    write_csv(OUT/"LOCAL_ONLY_INDEX.csv",files)
    safe_json(OUT/"RUN_MANIFEST.json",dict(branch=read(OUT/"SOURCE_AUDIT.json")["branch"],
        code_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in THIS.glob("*.py")},
        old_training_code_sha256=read(OUT/"SOURCE_AUDIT.json")["code_sha256"],
        source_audit_sha256=sha(OUT/"SOURCE_AUDIT.json"),checkpoints=manifest,
        local_only_files=len(files),local_only_bytes=sum(x["bytes"] for x in files),
        stage1_epochs=0,stage2_epochs=80,official_TEST_used=False,best_epoch_selection=False))
    print("VALIDATION PASS; 8 crossed models, 3400 exact pairs, zero Stage-I epochs",flush=True)
    print(summary["groups"]["Few"])


if __name__=="__main__":main()
