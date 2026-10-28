"""Paired, hypothesis-led failure-screen analysis. No training, no checkpoint changes.

Usage:
  python screen.py --manifest manifest.json --output results
NPZ contract: ids (N unique string/int), labels (N int), logits (N,C float).
Manifest: {"class_counts":[...], "methods":{"crt":{"clean":"...npz","noise_s1":"...npz"},...},
"reference":"crt","candidate":"pi_h2t","conditions":["clean","noise_s1",...]}
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np

def load(path, classes):
    with np.load(path, allow_pickle=False) as z:
        for k in ("ids", "labels", "logits"):
            if k not in z: raise ValueError(f"{path}: missing {k}")
        ids, labels, logits = z["ids"], z["labels"], z["logits"]
    if ids.ndim != 1 or labels.shape != ids.shape or logits.shape != (len(ids), classes):
        raise ValueError(f"{path}: invalid shapes")
    if len(np.unique(ids)) != len(ids) or np.any((labels < 0) | (labels >= classes)):
        raise ValueError(f"{path}: duplicate IDs or invalid labels")
    if not np.isfinite(logits).all(): raise ValueError(f"{path}: nonfinite logits")
    order = np.argsort(ids)
    return ids[order], labels[order].astype(int), logits[order].argmax(axis=1)

def metrics(labels, correct, counts):
    groups = {"all": np.ones(len(counts), dtype=bool),
              "many": counts > 100, "medium": (counts >= 20) & (counts <= 100),
              "few": counts < 20}
    result = {}
    for name, mask in groups.items():
        sample_mask = mask[labels]
        result[name] = float(correct[sample_mask].mean()*100) if sample_mask.any() else None
    per_class = np.array([correct[labels == c].mean() if np.any(labels == c) else np.nan
                           for c in range(len(counts))])
    result["macro"] = float(np.nanmean(per_class)*100)
    result["worst_class"] = float(np.nanmin(per_class)*100)
    result["classes_observed"] = int(np.isfinite(per_class).sum())
    return result

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    cfg = json.loads(Path(args.manifest).read_text(encoding="utf8"))
    methods, conditions = cfg["methods"], cfg["conditions"]
    counts = np.asarray(cfg["class_counts"], dtype=int)
    if counts.ndim != 1 or len(counts) < 2 or (counts <= 0).any(): raise ValueError("invalid class counts")
    ref, cand = cfg["reference"], cfg["candidate"]
    if ref == cand or ref not in methods or cand not in methods: raise ValueError("invalid comparison")
    if not conditions or conditions[0] != "clean" or len(set(conditions)) != len(conditions):
        raise ValueError("conditions must be unique, clean first")
    loaded = {}
    original_ids = original_labels = None
    for method in methods:
        for condition in conditions:
            path = methods[method].get(condition)
            if not path: raise ValueError(f"missing {method}/{condition}")
            ids, y, pred = load(Path(args.manifest).parent / path, len(counts))
            if original_ids is None: original_ids, original_labels = ids, y
            if not np.array_equal(ids, original_ids) or not np.array_equal(y, original_labels):
                raise ValueError(f"unpaired ids or labels in {method}/{condition}")
            loaded[method, condition] = pred == y
    y = original_labels
    rows = []
    for condition in conditions:
        for method in methods:
            c = loaded[method, condition]
            row = {"condition":condition,"method":method}
            row.update(metrics(y, c, counts))
            rows.append(row)
    contrasts = []
    for condition in conditions:
        a, b = loaded[ref,condition], loaded[cand,condition]
        delta = b.astype(int)-a.astype(int)
        group_defs = {"all":np.ones(len(y),bool),"many":counts[y]>100,
                      "medium":(counts[y]>=20)&(counts[y]<=100),"few":counts[y]<20}
        for group, mask in group_defs.items():
            if not mask.any(): continue
            d = delta[mask]
            contrasts.append({"condition":condition,"group":group,"n":int(mask.sum()),
                              "delta_pp":float(100*d.mean()),
                              "ref_right_cand_wrong":int((d==-1).sum()),
                              "ref_wrong_cand_right":int((d==1).sum())})
    changes = []
    for method in (ref,cand):
        for condition in conditions[1:]:
            clean = loaded[method,"clean"]; shifted = loaded[method,condition]
            for group, mask in {"all":np.ones(len(y),bool),"many":counts[y]>100,
                                "medium":(counts[y]>=20)&(counts[y]<=100),"few":counts[y]<20}.items():
                if not mask.any(): continue
                changes.append({"method":method,"condition":condition,"group":group,
                                "clean_to_shift_delta_pp":float(100*(shifted[mask].mean()-clean[mask].mean()))})
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    for name,data in (("metrics",rows),("paired_contrasts",contrasts),("shift_degradation",changes)):
        with (out/(name+".csv")).open("w",newline="",encoding="utf8") as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    report={"reference":ref,"candidate":cand,"conditions":conditions,
            "evaluated_images":len(y),"observed_classes":int(len(np.unique(y))),
            "caveats":["Screening only; no claims of paper-faithful PI-H2T without implementation audit.",
                       "Paired image counts are descriptive; no independent training seed inference.",
                       "Synthetic corruptions must be visually audited for label preservation.",
                       "If clean/shift datasets were used during method selection, this is development evidence.",
                       "Worst class can be unstable when class samples are scarce."]}
    (out/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf8")
    print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=="__main__": main()
