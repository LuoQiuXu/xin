"""Paired final-epoch H2T vs NoFusion gains under clean/noisy training labels.

CSV schema: stage1_seed,stage2_seed,regime,method,group,accuracy_pct,n,
stage1_sha,val_ids_sha,epoch. Methods h2t/no_fusion, regimes clean/noisy,
groups Overall/Many/Medium/Few. All evaluated on identical CLEAN held-out IDs.
"""
import argparse,csv,json
from pathlib import Path
import numpy as np
REGIMES=("clean","noisy");METHODS=("no_fusion","h2t")
GROUPS=("Overall","Many","Medium","Few")
def compare(rows):
    data={}
    for r in rows:
        key=((int(r["stage1_seed"]),int(r["stage2_seed"])),r["regime"],r["method"],r["group"])
        if key[1] not in REGIMES or key[2] not in METHODS or key[3] not in GROUPS:
            raise ValueError(f"unexpected arm {key}")
        if key in data or int(r["epoch"])!=10:raise ValueError("duplicate arm or wrong epoch (must be 10)")
        acc=float(r["accuracy_pct"]);n=int(r["n"])
        if not np.isfinite(acc) or not 0<=acc<=100 or n<=0 or not r["stage1_sha"] or not r["val_ids_sha"]:
            raise ValueError("invalid accuracy, denominator or source provenance")
        data[key]=(acc,n,r["stage1_sha"],r["val_ids_sha"])
    seeds=sorted(set(k[0] for k in data))
    if not seeds:raise ValueError("no results")
    out=[]
    for seed in seeds:
        for regime in REGIMES:
            for group in GROUPS:
                keys=[(seed,regime,method,group) for method in METHODS]
                if any(k not in data for k in keys):raise ValueError(f"missing matched arm {keys}")
                if data[keys[0]][1:]!=data[keys[1]][1:]:
                    raise ValueError(f"non-identical Stage-I, val, group count {seed}/{regime}/{group}")
        if len(set(data[k][3] for k in data if k[0]==seed))!=1:
            raise ValueError(f"validation IDs mismatch between regimes for {seed}")
        for group in GROUPS:
            n=set(data[seed,regime,method,group][1] for regime in REGIMES for method in METHODS)
            if len(n)!=1:raise ValueError(f"changed validation denominator {seed}/{group}")
            dclean=data[seed,"clean","h2t",group][0]-data[seed,"clean","no_fusion",group][0]
            dnoisy=data[seed,"noisy","h2t",group][0]-data[seed,"noisy","no_fusion",group][0]
            out.append({"stage1_seed":seed[0],"stage2_seed":seed[1],"group":group,
                        "clean_H2T_minus_DR_pp":round(dclean,8),
                        "noisy_H2T_minus_DR_pp":round(dnoisy,8),
                        "noise_interaction_pp":round(dnoisy-dclean,8),
                        "noisy_no_fusion_acc":data[seed,"noisy","no_fusion",group][0],
                        "noisy_h2t_acc":data[seed,"noisy","h2t",group][0],
                        "n_val":next(iter(n))})
    return out

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--csv",required=True);p.add_argument("--output",required=True)
    a=p.parse_args()
    with open(a.csv,newline="",encoding="utf-8-sig") as f:rows=compare(list(csv.DictReader(f)))
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    with (out/"paired_interactions.csv").open("w",newline="",encoding="utf8") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary={"by_group":{},"warnings":[
        "This compares human noisy training scenario against clean; observed class frequency changes as well as loss targets.",
        "One Stage-I seed and two Stage-II seeds form a PILOT, not independent Stage-I replication.",
        "Only valid for a clean TRAIN-held-out development validation; not official TEST.",
        "A negative interaction alone is not a demonstrated method failure; noisy H2T must be worse than same-source DR.",
        "This cannot establish a PI-H2T failure."]}
    for group in GROUPS:
        v=[r for r in rows if r["group"]==group]
        summary["by_group"][group]={
            "stage1_seeds":len(set(r["stage1_seed"] for r in v)),
            "paired_runs":len(v),"per_run_interaction_pp":[r["noise_interaction_pp"] for r in v],
            "mean_interaction_pp":float(np.mean([r["noise_interaction_pp"] for r in v])),
            "all_negative":all(r["noise_interaction_pp"]<0 for r in v)}
    few=[r for r in rows if r["group"]=="Few"]
    summary["pilot_gate"]={
        "status":"CANDIDATE_CONFIRM" if len(few)>=2 and len(set(r["stage2_seed"] for r in few))>=2
           and all(r["noise_interaction_pp"]<=-2 and r["noisy_H2T_minus_DR_pp"]<0 for r in few)
           else "NO_GO_OR_INCONCLUSIVE",
        "rule":"At least two Stage-II seeds: Few noise interaction <= -2pp AND noisy H2T Few < noisy DR Few in each; then independently reproduce with new Stage-I seed."}
    (out/"summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding="utf8")
    print(json.dumps(summary,indent=2,ensure_ascii=False))
if __name__=="__main__":main()
