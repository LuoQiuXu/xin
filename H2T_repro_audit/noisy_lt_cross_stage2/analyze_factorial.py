"""Read-only 2x2 factorial outcomes using 8 pre-existing diagonal + 8 crossed models.

Stage-I representation regime: clean/noisy; Stage-II observed-label regime: clean/noisy.
Unit: accuracy percentage points, not normalized relative percentage.
Each Stage-II seed shares the SAME Stage-I checkpoint, not independent Stage-I replication.
"""
import argparse
import csv
import json
from pathlib import Path

GROUPS=("Overall","Many","Medium","Few")
REGIMES=("clean","noisy")
METHODS=("no_fusion","h2t")


def csvrows(path):
    with Path(path).open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def assess(oldrows,crossrows):
    records={}
    valhash={x["val_ids_sha"] for x in oldrows}
    if len(valhash)!=1:raise ValueError("old validation IDs differ")
    for x in oldrows:
        if int(x["epoch"])!=10 or x["regime"] not in REGIMES or x["group"] not in GROUPS:
            raise ValueError("invalid original result")
        key=(int(x["stage2_seed"]), x["regime"], x["regime"], x["method"], x["group"])
        record=dict(accuracy=float(x["accuracy_pct"]),n=int(x["n"]),source=x["stage1_sha"])
        if key in records:raise ValueError("duplicate diagonal arm")
        records[key]=record
    if len(records)!=32: raise ValueError("expected 32 existing diagonal metrics")
    for x in crossrows:
        rep,lab=x["representation_regime"],x["label_regime"]
        if rep not in REGIMES or lab not in REGIMES or rep==lab:raise ValueError("not crossed")
        key=(int(x["stage2_seed"]),rep,lab,x["method"],x["group"])
        record=dict(accuracy=float(x["accuracy_pct"]),n=int(x["n"]),source=x["source_checkpoint_sha"])
        if key in records:raise ValueError("duplicate crossed arm")
        records[key]=record
    if len(records)!=64:raise ValueError("expected 64 metrics; incomplete eight crossed runs")
    seeds=sorted({k[0] for k in records})
    if seeds!=[124,125]:raise ValueError("unexpected Stage-II seeds")
    source_by_repr={}
    for seed in seeds:
        for rep in REGIMES:
            for lab in REGIMES:
                for group in GROUPS:
                    a,b=(records[seed,rep,lab,m,group] for m in METHODS)
                    if a["source"]!=b["source"] or a["n"]!=b["n"]:
                        raise ValueError("method arm not same-source or same evaluation count")
                    if not 0<=a["accuracy"]<=100 or not 0<=b["accuracy"]<=100:
                        raise ValueError("invalid accuracy")
                    if group=="Few" and a["n"]!=600:raise ValueError("Few n must be 600")
                    if group=="Overall" and a["n"]!=1980:raise ValueError("Overall n must be 1980")
                    if group=="Medium" and a["n"]!=700:raise ValueError("Medium n must be 700")
                    if group=="Many" and a["n"]!=680:raise ValueError("Many n must be 680")
                    source_by_repr.setdefault(rep, a["source"])
                    if a["source"]!=source_by_repr[rep]:
                        raise ValueError("representation source changed across labels/seeds")
    details=[]
    by_group={g:[] for g in GROUPS}
    for seed in seeds:
        for group in GROUPS:
            deltas={}
            for rep in REGIMES:
                for lab in REGIMES:
                    a=records[seed,rep,lab,"no_fusion",group]["accuracy"]
                    b=records[seed,rep,lab,"h2t",group]["accuracy"]
                    deltas[rep,lab]=b-a
                    details.append(dict(stage2_seed=seed,group=group,representation=rep,
                        observed_labels=lab,no_fusion_accuracy_pp=a,h2t_accuracy_pp=b,
                        h2t_gain_pp=deltas[rep,lab]))
            D=lambda r,l:deltas[r,l]
            # Balanced factorial contrasts; exact algebra in 2x2 design.
            repr_effect=((D("noisy","clean")-D("clean","clean"))+
                         (D("noisy","noisy")-D("clean","noisy")))/2
            label_effect=((D("clean","noisy")-D("clean","clean"))+
                          (D("noisy","noisy")-D("noisy","clean")))/2
            inter=D("noisy","noisy")-D("noisy","clean")-D("clean","noisy")+D("clean","clean")
            gap=D("noisy","noisy")-D("clean","clean")
            assert abs(repr_effect+label_effect-gap)<1e-8
            by_group[group].append(dict(stage2_seed=seed,representation_effect_pp=repr_effect,
                observed_stage2_label_sampling_effect_pp=label_effect,
                factorial_interaction_pp=inter, total_diagonal_change_pp=gap,
                clean_on_clean_gain_pp=D("clean","clean"),
                clean_repr_noisy_labels_gain_pp=D("clean","noisy"),
                noisy_repr_clean_labels_gain_pp=D("noisy","clean"),
                noisy_on_noisy_gain_pp=D("noisy","noisy")))
    def mean(xs,k):return sum(x[k] for x in xs)/len(xs)
    summary={"status":"DESCRIPTIVE_FACTORIAL_ONLY",
             "source_stage1_sha_by_repr":source_by_repr,
             "note":"Two Stage-II seeds, ONE clean and ONE noisy Stage-I checkpoint; do not infer Stage-I seed robustness.",
             "definitions":{"representation_effect":"Noisy minus clean Stage-I, averaged over both Stage-II label regimes",
                            "label_effect":"Noisy minus clean Stage-II observed-label/sampler regime, averaged over both Stage-I sources",
                            "interaction":"nonadditive 2x2 term; do NOT treat as causal label-only effect"},
             "groups":{}}
    for group,xs in by_group.items():
        summary["groups"][group]={
            "per_seed":xs,
            "mean_representation_effect_pp":mean(xs,"representation_effect_pp"),
            "mean_stage2_label_sampling_effect_pp":mean(xs,"observed_stage2_label_sampling_effect_pp"),
            "mean_factorial_interaction_pp":mean(xs,"factorial_interaction_pp"),
            "mean_total_diagonal_change_pp":mean(xs,"total_diagonal_change_pp")}
    return details,by_group,summary


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--diagonal",required=True)
    p.add_argument("--crossed",required=True)
    p.add_argument("--output",required=True)
    a=p.parse_args()
    d,e,s=assess(csvrows(a.diagonal),csvrows(a.crossed))
    out=Path(a.output)
    out.mkdir(parents=True,exist_ok=False)
    for filename,rows in (("factorial_accuracy.csv",d),("factorial_contrasts.csv",[r for g in GROUPS for r in e[g]])):
        with (out/filename).open("x",encoding="utf8",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]))
            w.writeheader();w.writerows(rows)
    (out/"summary.json").write_text(json.dumps(s,indent=2,ensure_ascii=False))
    print(json.dumps(s["groups"]["Few"],indent=2,ensure_ascii=False))


if __name__=="__main__":main()
