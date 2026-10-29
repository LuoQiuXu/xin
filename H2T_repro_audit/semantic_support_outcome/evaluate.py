"""Outcome-first check: does original H2T gain vary by semantic head support?
No training and no semantic-causality claim. Python 3.10+, numpy.
npz arrays: ids, labels, logits. metadata JSON: class_counts (100 ints),
coarse_labels (100 ints), model_pair_note; compare exactly paired examples.
"""
import argparse, csv, json
from pathlib import Path
import numpy as np

def load(path, C):
    with np.load(path, allow_pickle=False) as z:
        ids, labels, logits=(z[k] for k in ("ids","labels","logits"))
    if ids.ndim!=1 or labels.shape!=ids.shape or logits.shape!=(len(ids),C):
        raise ValueError("bad prediction shapes")
    if ids.dtype.kind not in "iu" or labels.dtype.kind not in "iu": raise ValueError("IDs/labels must be integer")
    if len(np.unique(ids))!=len(ids) or np.any((labels<0)|(labels>=C)) or not np.isfinite(logits).all():
        raise ValueError("duplicate ids, bad labels, or invalid logits")
    ix=np.argsort(ids)
    return ids[ix],labels[ix],logits[ix].argmax(1)

def measure(data,mask):
    v=data[mask]
    return float(v.mean()) if len(v) else None

def main():
    a=argparse.ArgumentParser()
    a.add_argument("--no-fusion",required=True)
    a.add_argument("--h2t",required=True)
    a.add_argument("--metadata",required=True)
    a.add_argument("--output",required=True)
    a.add_argument("--seed",type=int,default=20261029)
    a=a.parse_args()
    meta=json.loads(Path(a.metadata).read_text(encoding="utf8"))
    counts=np.asarray(meta["class_counts"],dtype=int)
    coarse=np.asarray(meta["coarse_labels"],dtype=int)
    C=len(counts)
    if C!=100 or coarse.shape!=(C,) or np.any(counts<=0):raise ValueError("expected CIFAR-100 class counts and coarse mapping")
    i,y,p0=load(a.no_fusion,C);j,z,p1=load(a.h2t,C)
    if not np.array_equal(i,j) or not np.array_equal(y,z):raise ValueError("unpaired predictions")
    many=counts>100;few=counts<20
    support=np.array([bool(np.any(many & (coarse==coarse[c]) & (np.arange(C)!=c))) for c in range(C)])
    c0=p0==y;c1=p1==y
    rows=[]
    for c in range(C):
        if not few[c]:continue
        pos=np.flatnonzero(y==c)
        if len(pos)==0:
            rows.append({"class_id":c,"coarse_id":int(coarse[c]),"train_n":int(counts[c]),"eval_n":0,
                         "has_many_same_coarse":int(support[c]),"no_fusion_acc":None,"h2t_acc":None,
                         "gain_pp":None,"fixed_by_h2t":None,"broken_by_h2t":None})
            continue
        r0=c0[pos];r1=c1[pos]
        rows.append({"class_id":c,"coarse_id":int(coarse[c]),"train_n":int(counts[c]),"eval_n":int(len(pos)),
                     "has_many_same_coarse":int(support[c]),"no_fusion_acc":float(r0.mean()*100),
                     "h2t_acc":float(r1.mean()*100),"gain_pp":float((r1.mean()-r0.mean())*100),
                     "fixed_by_h2t":int((~r0&r1).sum()),"broken_by_h2t":int((r0&~r1).sum())})
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    with (out/"per_few_class.csv").open("w",encoding="utf8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    observed=[r for r in rows if r["eval_n"]]
    rng=np.random.default_rng(a.seed)
    res={}
    for group, flag in (("supported",1),("unsupported",0)):
        rr=[r for r in observed if r["has_many_same_coarse"]==flag]
        delta=np.array([r["gain_pp"] for r in rr],dtype=float)
        res[group]={"observed_classes":len(rr),"images":sum(r["eval_n"] for r in rr),
                    "mean_class_gain_pp":float(delta.mean()) if len(delta) else None,
                    "median_class_gain_pp":float(np.median(delta)) if len(delta) else None,
                    "improved":int((delta>0).sum()),"harmed":int((delta<0).sum()),"unchanged":int((delta==0).sum()),
                    "mean_baseline_acc":float(np.mean([r["no_fusion_acc"] for r in rr])) if rr else None}
    if res["supported"]["observed_classes"] and res["unsupported"]["observed_classes"]:
        A=np.array([r["gain_pp"] for r in observed if r["has_many_same_coarse"]])
        B=np.array([r["gain_pp"] for r in observed if not r["has_many_same_coarse"]])
        bs=np.empty(2000)
        for k in range(len(bs)):
            bs[k]=rng.choice(A,len(A),replace=True).mean()-rng.choice(B,len(B),replace=True).mean()
        res["contrast_supported_minus_unsupported_pp"]={"point":float(A.mean()-B.mean()),
            "exploratory_class_bootstrap_95pct":[float(x) for x in np.quantile(bs,[.025,.975])]}
    else:res["contrast_supported_minus_unsupported_pp"]=None
    res["protocol"]={"class_definition":"Few<20, Many>100; same official CIFAR-100 coarse ID",
                     "inference":"Identical test-time backbone and paired predictions required",
                     "scope":"Development holdout; association only, not semantic causality",
                     "cautions":["Superclass support is correlated with class identity, training count and difficulty.",
                                 "Missing class0 does not invalidate Few group, but overall metrics are not official TEST.",
                                 "One paired Stage-II seed is exploratory; avoid post-hoc seed/epoch selection.",
                                 "H2T fixed-BN checkpoints are not full official BN-shift H2T.",
                                 "Do not claim PI-H2T failure from this test."]}}
    (out/"summary.json").write_text(json.dumps(res,indent=2,ensure_ascii=False),encoding="utf8")
    print(json.dumps(res,indent=2,ensure_ascii=False))
if __name__=="__main__":main()
