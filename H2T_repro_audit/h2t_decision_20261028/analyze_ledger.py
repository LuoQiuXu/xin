"""Historical evidence inventory only; no GPU, torch, models or TEST use."""
import argparse, csv, hashlib, json
from pathlib import Path
from collections import Counter
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
STATUSES={"SUPPORTED_CONDITIONAL","SUPPORTED_DIAGNOSTIC_ONLY","PARTIAL_CONFOUNDED","NOT_ESTABLISHED","NOT_SUPPORTED","NO_GO","BLOCKED_PARITY","PROTOCOL_ONLY","BLOCKED_REPLAY"}
FIELDS=("id","branch","path","claim","status","evidence","confound","next")
def check(root,registry):
    if registry.get("schema_version")!=1 or registry.get("no_gpu") is not True:raise ValueError("unsafe registry")
    records=registry.get("records")
    if not records:raise ValueError("no records")
    rows=[];seen=set()
    for row in records:
        if any(not isinstance(row.get(k),str) or not row[k].strip() for k in FIELDS):raise ValueError("missing fields")
        if row["id"] in seen or row["status"] not in STATUSES:raise ValueError("duplicate id or unknown status")
        seen.add(row["id"])
        p=(root/row["path"]).resolve()
        if not p.is_relative_to(root.resolve()):raise ValueError("path escaped repository")
        if not p.is_file():raise FileNotFoundError(row["path"])
        rows.append({**row,"sha256":hashlib.sha256(p.read_bytes()).hexdigest()})
    return rows
def summarize(rows):
    counts=Counter(r["status"] for r in rows)
    s=["# H2T historical hypothesis gate","","**Audit only: not a training experiment or novelty proof.**",""]
    for r in rows:s.append("- "+r["id"]+" ["+r["status"]+"] "+r["claim"])
    s+=["","## Status counts"]+["- "+k+": "+str(v) for k,v in sorted(counts.items())]
    s+=["","## Evidence limits","PROTOCOL_ONLY has no measured result. NO_GO rejects only its tested formulation.","BLOCKED_PARITY/REPLAY cannot establish a paper or method defect.","An offline local evidence-file hash cannot prove the truth of a scientific claim.","Independent paper overlap checking and author-code comparison are still required.","No new GPU trial is permitted by this report."]
    return "\n".join(s)+"\n"
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--output",required=True,type=Path)
    a=ap.parse_args()
    try:
        j=json.loads((HERE/"EVIDENCE_LEDGER.json").read_text(encoding="utf-8"))
        rows=check(ROOT,j)
        out=(ROOT/a.output).resolve()
        allowed=(HERE/"results").resolve()
        if not out.is_relative_to(allowed) or out.exists():raise ValueError("output not new or outside results/")
        out.mkdir(parents=True)
        with (out/"EVIDENCE_CHECK.csv").open("w",newline="",encoding="utf-8-sig") as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        (out/"HYPOTHESIS_GATE.md").write_text(summarize(rows),encoding="utf-8")
        (out/"STATUS.json").write_text(json.dumps({"rows":len(rows),"source_file_hashes_only":True,"training_performed":False,"new_hypothesis_authorized":False,"decision":"PENDING_PRIMARY_SOURCE_REVIEW"},indent=2)+"\n",encoding="utf-8")
        print("verified",len(rows),"existing local sources; no GPU work");return 0
    except (ValueError,OSError,KeyError,TypeError) as e:
        print("FAIL",repr(e));return 2
if __name__=="__main__":raise SystemExit(main())
