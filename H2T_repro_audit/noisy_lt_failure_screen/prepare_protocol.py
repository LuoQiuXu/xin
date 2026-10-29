"""Audit real CIFAR-100N human fine labels against an existing IF100 LT split.

Inputs: original_labels.npy [50000] original CIFAR-100 Python TRAIN fine labels;
human_labels.npz clean_label/noisy_label [50000]; split.npz train_ids/val_ids
raw Python-CIFAR TRAIN indices. Never load CIFAR TEST; never generate a new split.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

def sha256(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for part in iter(lambda:f.read(1024*1024),b""):h.update(part)
    return h.hexdigest()

def clean_labels(a,name):
    a=np.asarray(a)
    if a.shape!=(50000,) or a.dtype.kind not in "iu" or np.any((a<0)|(a>=100)):
        raise ValueError(f"{name} must contain 50000 integer fine labels 0..99")
    return a.astype(np.int64)

def check_ids(a,name):
    a=np.asarray(a)
    if a.ndim!=1 or a.dtype.kind not in "iu" or np.any((a<0)|(a>=50000)) or len(np.unique(a))!=len(a):
        raise ValueError(f"bad raw TRAIN index array {name}")
    return a.astype(np.int64)

def build(original,human_clean,human_noisy,train_ids,val_ids):
    orig=clean_labels(original,"original")
    hc=clean_labels(human_clean,"CIFAR100N clean_label")
    hn=clean_labels(human_noisy,"CIFAR100N noisy_label")
    if not np.array_equal(orig,hc):
        where=np.flatnonzero(orig!=hc)
        raise ValueError(f"CIFAR-N image order mismatch: {len(where)} indices, first {where[0]}")
    tr=check_ids(train_ids,"train_ids");va=check_ids(val_ids,"val_ids")
    if np.intersect1d(tr,va).size:raise ValueError("overlap between training and validation raw IDs")
    if len(tr)!=10847:raise ValueError(f"Expected 10847 original LT train images: found {len(tr)}")
    true_counts=np.bincount(orig[tr],minlength=100)
    if true_counts.max()!=500 or true_counts.min()!=5 or np.any(true_counts==0):
        raise ValueError("IF100 original training count range must be 500->5")
    noisy_counts=np.bincount(hn[tr],minlength=100)
    result={
        "total_train_images":50000,"lt_train_images":len(tr),"validation_images":len(va),
        "full_human_annotation_noise_rate":float(np.mean(hc!=hn)),
        "lt_human_annotation_noise_rate":float(np.mean(orig[tr]!=hn[tr])),
        "true_lt_counts":true_counts.tolist(),
        "observed_noisy_counts":noisy_counts.tolist(),
        "empty_observed_classes":np.flatnonzero(noisy_counts==0).tolist(),
        "label_count_L1_shift":int(np.abs(true_counts-noisy_counts).sum()),
        "by_true_class_noise_rate":[float(np.mean(hn[tr][orig[tr]==c]!=c)) for c in range(100)],
        "stop_reason":"EMPTY_OBSERVED_LABEL_CLASS" if np.any(noisy_counts==0) else None,
        "limits":["True labels may be used to CONSTRUCT the audited LT benchmark, not for noisy training, sampler or loss.",
                  "Observed labels may change class frequencies; record, do not silently normalize them using clean truth.",
                  "Validation uses only clean disjoint TRAIN-held-out raw IDs; no official TEST access."]}
    files={"clean_train":{"ids":tr,"labels":orig[tr]},
           "noisy_train":{"ids":tr,"labels":hn[tr]},
           "clean_validation":{"ids":va,"labels":orig[va]}}
    return files,result

def main():
    p=argparse.ArgumentParser()
    for key in ("original-labels","human-labels","split","output"):p.add_argument("--"+key,required=True)
    a=p.parse_args()
    original=np.load(a.original_labels,allow_pickle=False)
    with np.load(a.human_labels,allow_pickle=False) as f:
        hc=f["clean_label"];hn=f["noisy_label"]
    with np.load(a.split,allow_pickle=False) as f:
        tr=f["train_ids"];va=f["val_ids"]
    files,audit=build(original,hc,hn,tr,va)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    for name,data in files.items():
        np.savez_compressed(out/(name+".npz"),**data)
    inputs=(a.original_labels,a.human_labels,a.split)
    audit["source_sha256"]={Path(i).name:sha256(i) for i in inputs}
    audit["manifest_sha256"]={name+".npz":sha256(out/(name+".npz")) for name in files}
    (out/"DATA_AUDIT.json").write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding="utf8")
    print(json.dumps({k:v for k,v in audit.items() if k not in ("true_lt_counts","observed_noisy_counts","by_true_class_noise_rate")},ensure_ascii=False,indent=2))
    if audit["stop_reason"]:raise SystemExit("BLOCKED: absent observed noisy class, cannot run original class-balanced sampler unchanged")

if __name__=="__main__":main()
