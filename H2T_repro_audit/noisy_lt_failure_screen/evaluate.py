"""One fixed-final-epoch evaluation, only after all ten training runs complete."""
import csv
import subprocess
import sys

import numpy as np
import torch
from PIL import Image

from runtime import *
from train import check_lock, load_weights, environment, now

GROUPS=("Overall","Many","Medium","Few")


def write_csv(path,rows):
    with Path(path).open("x",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def group_masks(labels,counts):
    frequency=counts[labels]
    return dict(Overall=np.ones(len(labels),dtype=bool),Many=frequency>100,
                Medium=(frequency>=20)&(frequency<=100),Few=frequency<20)


def main():
    check_lock();assert read(OUT / "TRAINING_COMPLETED.json")["completed_runs"]==10
    g=install_guard("evaluation")
    models_to_eval=[(1,r,0,"CE_MixUp") for r in ("clean","noisy")]
    models_to_eval += [(2,r,s,m) for r in ("clean","noisy") for s in (124,125) for m in ("no_fusion","h2t")]
    paths={}
    for stage,r,s,m in models_to_eval:
        name="stage1_"+r if stage==1 else f"stage2_{r}_seed{s}_{m}"
        report=read(OUT / name / "RUN.json");path=ROOT / report["checkpoint"]["file"]
        assert report["checkpoint"]["epoch"]==(200 if stage==1 else 10) and sha(path)==report["checkpoint"]["sha256"]
        paths[name]=(path,report)
    evaluation=OUT / "evaluation";evaluation.mkdir(exist_ok=False)
    local=CACHE / "evaluation";local.mkdir(exist_ok=False)
    lock=dict(UTC=now(),training_completed_sha=sha(OUT / "TRAINING_COMPLETED.json"),
              checkpoints_before_any_evaluation={n:sha(p[0]) for n,p in paths.items()},
              validation_ids_and_labels_sha=sha(OUT / "protocol/clean_validation.npz"),
              validation_images_sha=sha(CACHE / "inputs/validation_images.npy"),
              evaluation_script_sha256=sha(Path(__file__)))
    write_json(evaluation / "EVALUATION_LOCK.json",lock)
    with np.load(OUT / "protocol/clean_validation.npz",allow_pickle=False) as z:ids,labels=z["ids"].copy(),z["labels"].copy()
    images=np.load(CACHE / "inputs/validation_images.npy",allow_pickle=False,mmap_mode="r")
    assert images.shape==(1980,32,32,3) and ids.shape==labels.shape==(1980,)
    counts=np.array(read(OUT / "protocol/DATA_AUDIT.json")["true_lt_counts"])
    masks=group_masks(labels,counts);val_hash=ahash(ids);target_hash=ahash(labels)
    assert [int(masks[k].sum()) for k in GROUPS]==[1980,680,700,600]
    assert np.bincount(labels,minlength=100).tolist()==[0]+[20]*99
    trans=transform("eval");main_rows=[];class_rows=[];s1_rows=[];correctness={};audit={}
    for stage,r,s,m in models_to_eval:
        name="stage1_"+r if stage==1 else f"stage2_{r}_seed{s}_{m}"
        path,run=paths[name];cfg=config(stage);seed(42 if stage==1 else s,cfg);bb,fc=models()
        data=load_weights(path,bb,fc);before=state_hash(cpu_state(bb));bb.eval();fc.eval()
        outputs=[]
        with torch.inference_mode():
            for start in range(0,len(images),128):
                batch=torch.stack([trans(Image.fromarray(im)) for im in images[start:start+128]]).cuda()
                logits=fc(bb(batch));assert torch.isfinite(logits).all() and logits.dtype==torch.float32
                outputs.append(logits.cpu().numpy())
        logits=np.concatenate(outputs);assert logits.shape==(1980,100) and state_hash(cpu_state(bb))==before
        predictions=logits.argmax(1);correct=predictions==labels
        with (local / (name+".npz")).open("xb") as f:np.savez_compressed(f,raw_ids=ids,labels=labels,logits=logits,predictions=predictions)
        audit[name]=dict(checkpoint_sha256=sha(path),epoch=data["epoch"],single_eval_passes=1,
                        logits_shape=list(logits.shape),logits_dtype=str(logits.dtype),logits_all_finite=True,
                        val_ids_sha=val_hash,val_labels_sha=target_hash,BB_state_unchanged=True,
                        predictions_file_sha256=sha(local / (name+".npz")),environment=environment(cfg))
        if stage==2:correctness[r,s,m]=correct
        for group,mask in masks.items():
            row=dict(stage1_seed=42,stage2_seed=s,regime=r,method=m,group=group,
                     accuracy_pct=100*int(correct[mask].sum())/int(mask.sum()),n=int(mask.sum()),
                     stage1_sha=sha(path) if stage==1 else run["parent_sha256"],val_ids_sha=val_hash,epoch=data["epoch"])
            (main_rows if stage==2 else s1_rows).append(row)
        for c in range(100):
            mask=labels==c;n=int(mask.sum())
            class_rows.append(dict(stage=stage,stage2_seed=s,regime=r,method=m,class_id=c,
                benchmark_clean_LT_count=int(counts[c]),n=n,correct=int(correct[mask].sum()),
                accuracy_pct=100*int(correct[mask].sum())/n if n else "",missing_validation_class=n==0))
        print(name+" final fixed evaluation " +str({k:round(100*correct[v].mean(),4) for k,v in masks.items()}),flush=True)
    paired=[];paired_classes=[]
    for r in ("clean","noisy"):
        for s in (124,125):
            nf,h=correctness[r,s,"no_fusion"],correctness[r,s,"h2t"]
            for group,mask in masks.items():
                gain=int((~nf&h&mask).sum());loss=int((nf&~h&mask).sum())
                assert gain-loss==int(h[mask].sum())-int(nf[mask].sum())
                paired.append(dict(regime=r,stage2_seed=s,group=group,n=int(mask.sum()),
                    wrong_to_right=gain,right_to_wrong=loss,net_correct=gain-loss,net_pp=100*(gain-loss)/int(mask.sum())))
            for c in range(100):
                mask=labels==c;gain=int((~nf&h&mask).sum());loss=int((nf&~h&mask).sum())
                paired_classes.append(dict(regime=r,stage2_seed=s,class_id=c,n=int(mask.sum()),wrong_to_right=gain,right_to_wrong=loss,net_correct=gain-loss))
    write_csv(evaluation / "final_metrics.csv",main_rows)
    write_csv(evaluation / "stage1_final_descriptive.csv",s1_rows)
    write_csv(evaluation / "per_class_accuracy.csv",class_rows)
    write_csv(evaluation / "paired_reclassifications.csv",paired)
    write_csv(evaluation / "per_class_reclassifications.csv",paired_classes)
    assert_guard(g)
    write_json(evaluation / "EVALUATION_AUDIT.json",dict(models=audit,IO_guard=g,
        evaluated_class_ids=list(range(1,100)),missing_class_ids=[0],
        groups_defined_from="clean true benchmark LT counts, not noisy observed counts",val_ids_sha=val_hash,val_labels_sha=target_hash))
    subprocess.run([sys.executable,"-B","-X","utf8",str(TASK / "compare_outcomes.py"),"--csv",str(evaluation / "final_metrics.csv"),"--output",str(OUT / "analysis")],check=True)


if __name__=="__main__":main()
