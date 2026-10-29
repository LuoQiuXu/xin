"""No-training donor semantic compatibility screen. Python 3.10+, numpy only.

Input .npz (allow_pickle=False):
 rec_id [N] int; rec_label [N] int; donor_id [N] int; donor_label [N] int;
 rec_feat [N,D] float; donor_feat [N,D] float;
 coarse_label [C] int (CIFAR-100 published fine->coarse mapping, ordered by fine ID);
 weight [C,D] float; bias [C] float (frozen Stage-I classifier);
 lambda_donor [N] float OR --lambda-fixed constant.
One row per pair; each recipient should have multiple donor choices.
Outputs are diagnostics only: reference margin flip, not final test accuracy.
"""
import argparse, csv, json
from pathlib import Path
import numpy as np

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",required=True)
    ap.add_argument("--output",required=True)
    ap.add_argument("--lambda-fixed",type=float,default=0.25)
    ap.add_argument("--distance-bins",type=int,default=4)
    ap.add_argument("--minimum-matches",type=int,default=30)
    a=ap.parse_args()
    if not 0<=a.lambda_fixed<=1: raise ValueError("invalid donor mass")
    if not 2<=a.distance_bins<=20: raise ValueError("invalid number of bins")
    with np.load(a.input,allow_pickle=False) as f:
        required=("rec_id","rec_label","donor_id","donor_label","rec_feat","donor_feat","coarse_label","weight","bias")
        if any(k not in f for k in required): raise ValueError("missing keys: "+",".join(k for k in required if k not in f))
        x={k:f[k] for k in required}
        mass=f["lambda_donor"] if "lambda_donor" in f else np.full(len(x["rec_id"]),a.lambda_fixed)
    ri,y,di,dy=(x[k] for k in ("rec_id","rec_label","donor_id","donor_label"))
    fr,fd,coarse,w,b=(x[k] for k in ("rec_feat","donor_feat","coarse_label","weight","bias"))
    n=len(y);C=len(coarse)
    if n==0 or len(set(zip(ri.tolist(),di.tolist())))!=n: raise ValueError("empty or repeated pair")
    if any(v.shape!=(n,) for v in (ri,y,di,dy,mass)): raise ValueError("invalid vector length")
    if ri.dtype.kind not in "iu" or di.dtype.kind not in "iu": raise ValueError("IDs must be integers")
    if y.dtype.kind not in "iu" or dy.dtype.kind not in "iu": raise ValueError("labels must be integers")
    if fr.ndim!=2 or fd.shape!=fr.shape or w.shape!=(C,fr.shape[1]) or b.shape!=(C,): raise ValueError("shape mismatch")
    if coarse.ndim!=1 or C<2 or len(np.unique(coarse))<2: raise ValueError("invalid coarse mapping")
    if np.any(y<0) or np.any(y>=C) or np.any(dy<0) or np.any(dy>=C):raise ValueError("invalid labels")
    if not all(np.isfinite(t).all() for t in (fr,fd,w,b,mass)) or np.any(mass<0) or np.any(mass>1):
        raise ValueError("nonfinite tensors or invalid donor mass")
    for rid in np.unique(ri):
        loc=np.flatnonzero(ri==rid)
        if len(np.unique(y[loc]))!=1 or not np.allclose(fr[loc],fr[loc[0]],rtol=0,atol=1e-5):
            raise ValueError("recipient feature/label differs across candidate donors")
        if len(loc)<2: raise ValueError("every recipient must have >=2 candidates")
    same=(coarse[y]==coarse[dy]) & (y!=dy)
    different=(coarse[y]!=coarse[dy])
    valid=same|different
    distance=np.linalg.norm(fd-fr,axis=1)
    mixed=(1-mass[:,None])*fr+mass[:,None]*fd
    logits=mixed@w.T+b
    correct=(logits.argmax(1)==y)
    clean_logits=fr@w.T+b
    clean_correct=(clean_logits.argmax(1)==y)
    margin=logits[np.arange(n),y]-np.max(np.where(np.arange(C)[None,:]==y[:,None],-np.inf,logits),axis=1)
    clean_margin=clean_logits[np.arange(n),y]-np.max(np.where(np.arange(C)[None,:]==y[:,None],-np.inf,clean_logits),axis=1)
    margin_drop=margin-clean_margin
    # Within-recipient, within-distance-bin matching. Never compare raw semantic groups
    # pooled across different recipient distributions.
    matches=[]
    for rid in np.unique(ri):
        loc=np.flatnonzero((ri==rid)&valid)
        if not np.any(same[loc]) or not np.any(different[loc]):continue
        ds=distance[loc]
        # Equal-width? use rank-quantile bins; empty/duplicate quantiles permitted.
        cuts=np.quantile(ds,np.linspace(0,1,a.distance_bins+1)[1:-1])
        bins=np.searchsorted(cuts,ds,side="right")
        for bucket in np.unique(bins):
            idx=loc[bins==bucket]
            pos=idx[same[idx]];neg=idx[different[idx]]
            if not len(pos) or not len(neg):continue
            # deterministic greedy nearest-distance match without replacement
            pairs=[]
            candidates=[(abs(float(distance[p]-distance[q])),int(p),int(q)) for p in pos for q in neg]
            usedp=set();usedq=set()
            for delta,p,q in sorted(candidates):
                if p not in usedp and q not in usedq:
                    usedp.add(p);usedq.add(q);pairs.append((p,q))
            for p,q in pairs:
                matches.append({"recipient_id":int(rid),"same_donor_id":int(di[p]),"diff_donor_id":int(di[q]),
                                "distance_gap":float(abs(distance[p]-distance[q])),
                                "same_distance":float(distance[p]),"diff_distance":float(distance[q]),
                                "same_mass":float(mass[p]),"diff_mass":float(mass[q]),
                                "same_correct":int(correct[p]),"diff_correct":int(correct[q]),
                                "same_margin":float(margin[p]),"diff_margin":float(margin[q]),
                                "same_margin_drop":float(margin_drop[p]),"diff_margin_drop":float(margin_drop[q]),
                                "clean_correct":int(clean_correct[p])})
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    columns=["recipient_id","same_donor_id","diff_donor_id","distance_gap","same_distance","diff_distance","same_mass","diff_mass","same_correct","diff_correct","same_margin","diff_margin","same_margin_drop","diff_margin_drop","clean_correct"]
    with (out/"matched_pairs.csv").open("w",newline="",encoding="utf8") as f:
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader();writer.writerows(matches)
    if matches:
        m=np.array([[r["same_correct"]-r["diff_correct"],r["same_margin"]-r["diff_margin"],
                     r["same_margin_drop"]-r["diff_margin_drop"],r["distance_gap"],
                     r["same_mass"]-r["diff_mass"]] for r in matches])
        means=m.mean(axis=0).tolist()
    else:means=[None]*5
    summary={"n_candidate_pairs":n,"n_recipients":int(len(np.unique(ri))),
             "n_same_superclass_pairs":int(same.sum()),"n_different_superclass_pairs":int(different.sum()),
             "n_matched_pairs":len(matches),"n_matched_recipients":len(set(r["recipient_id"] for r in matches)),
             "matched_same_minus_diff_accuracy_pp":None if means[0] is None else 100*means[0],
             "matched_same_minus_diff_margin":means[1],"matched_same_minus_diff_margin_drop":means[2],
             "mean_absolute_distance_gap":means[3],"mean_mass_gap":means[4],
             "screen_sufficient_pairs":len(matches)>=a.minimum_matches,
             "interpretation":"Exploratory frozen-classifier diagnostic, NOT final Stage-II benefit or author PI-H2T failure.",
             "important_controls":["Same recipient; approximately matched feature distance; check mass gap before interpreting.",
                 "Superclass similarity is a coarse proxy, not semantic causality; image/content and donor-frequency confounding remain.",
                 "No checkpoint trained or modified; no evaluation data should tune a method.",
                 "Do not attribute failures to PI-H2T without author-faithful parity."]}
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf8")
    print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
