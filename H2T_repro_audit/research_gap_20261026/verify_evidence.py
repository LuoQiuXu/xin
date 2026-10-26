#!/usr/bin/env python3
"""Source-anchored H2T/PI-H2T evidence check. No model execution."""
import argparse, csv, hashlib, json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
KEYS=("id","source","anchor","observation","boundary","type")

def filehash(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def verify(root, doc):
    if doc.get("schema_version")!=1 or not isinstance(doc.get("claims"),list):
        raise ValueError("invalid ledger")
    rows=[]; seen=set()
    for entry in doc["claims"]:
        if any(not isinstance(entry.get(k),str) or not entry[k] for k in KEYS):
            raise ValueError("missing evidence field")
        if entry["id"] in seen: raise ValueError("duplicate id")
        seen.add(entry["id"])
        p=(root/entry["source"]).resolve()
        if not p.is_relative_to(root.resolve()): raise ValueError("path escape")
        if not p.is_file(): raise FileNotFoundError(p)
        content=p.read_text(encoding="utf8")
        if entry["anchor"] not in content: raise AssertionError(entry["id"]+" anchor missing")
        rows.append(dict(id=entry["id"],source=entry["source"],
          line=content[:content.index(entry["anchor"])].count("\n")+1,
          sha256=filehash(p),type=entry["type"],
          observation=entry["observation"],boundary=entry["boundary"],
          status="LOCAL_TEXT_MATCH_ONLY"))
    if not rows: raise ValueError("empty ledger")
    return rows

def main():
    a=argparse.ArgumentParser()
    a.add_argument("--output",required=True)
    a.add_argument("--root",default=str(ROOT))
    args=a.parse_args()
    try:
        root=Path(args.root).resolve()
        rows=verify(root,json.loads((HERE/"LOCAL_EVIDENCE.json").read_text(encoding="utf8")))
        out=(root/args.output).resolve()
        if not out.is_relative_to((root/"H2T_repro_audit/research_gap_20261026/results").resolve()):
            raise ValueError("output path outside study results")
        out.mkdir(parents=True,exist_ok=False)
        with (out/"EVIDENCE_INDEX.csv").open("w",newline="",encoding="utf-8-sig") as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        manifest={"ledger_sha256":filehash(HERE/"LOCAL_EVIDENCE.json"),
          "sources":{r["source"]:r["sha256"] for r in rows},
          "count":len(rows),"scope":"textual anchors only; not causal or paper verification"}
        (out/"SOURCE_MANIFEST.json").write_text(json.dumps(manifest,indent=2,ensure_ascii=False))
        print("PASS",len(rows),"local evidence anchors",out)
        return 0
    except (OSError,ValueError,AssertionError) as err:
        print("FAIL",err);return 2

if __name__=="__main__":raise SystemExit(main())
