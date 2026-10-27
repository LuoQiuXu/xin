#!/usr/bin/env python3
"""Offline, read-only H2T/PI-H2T source-provenance verifier.

Checks curated, quoted LOCAL evidence anchors. It cannot decide author code
availability or prove PDF semantics; those require independent primary-source
inspection by Codex. No models, checkpoints, downloads, or CUDA access.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
ALLOWED = {
    "PAPER_FORM_LOCAL_CANDIDATE",
    "UNVERIFIED_AUTHOR_PI",
    "INHERITED_ORIGINAL_H2T_NOT_PI_VERIFIED",
    "LOCAL_PROTOCOL_VARIANT",
    "NOT_LOCATED_IN_CHECKED_AUTHOR_REPO",
}
REQUIRED = ("id", "claim", "status", "paper_pointer", "path",
            "anchor", "author_pi_code", "question")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_source(root: Path, rel: str) -> Path:
    if not isinstance(rel, str) or not rel or rel.startswith(("/", "\\")):
        raise ValueError("Invalid relative source")
    path = (root / rel).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Evidence path escapes repository: " + rel)
    if not path.is_file():
        raise FileNotFoundError("Evidence source missing: " + rel)
    return path


def audit(root: Path, manifest: dict) -> list[dict]:
    if manifest.get("schema_version") != 1:
        raise ValueError("Unknown evidence schema version")
    fields = manifest.get("fields")
    if not isinstance(fields, list) or not fields:
        raise ValueError("Empty evidence fields")
    upstream = manifest.get("upstream", {})
    if upstream.get("repo") != "Keke921/H2T":
        raise ValueError("Expected pinned H2T upstream identity")
    if upstream.get("branch_names_observed") != ["main"]:
        raise ValueError("Unexpected pinned branch inventory: re-audit")
    if not upstream.get("complete_tree_observed"):
        raise ValueError("Incomplete pinned author repository tree")
    rows = []
    seen = set()
    for item in fields:
        if any(not isinstance(item.get(key), str) or not item[key].strip()
               for key in REQUIRED):
            raise ValueError("Missing or empty field in evidence registry")
        if item["id"] in seen:
            raise ValueError("Duplicate claim ID: " + item["id"])
        seen.add(item["id"])
        if item["status"] not in ALLOWED:
            raise ValueError("Unapproved claim status: " + item["status"])
        if item["author_pi_code"] == "VERIFIED":
            raise ValueError(
                "Cannot elevate author PI implementation from LOCAL text-anchor evidence"
            )
        path = safe_source(root, item["path"])
        content = path.read_text(encoding="utf-8")
        if item["anchor"] not in content:
            raise AssertionError("Missing exact source excerpt for " + item["id"])
        line = content[:content.index(item["anchor"])].count("\n") + 1
        rows.append({
            "id": item["id"],
            "status": item["status"],
            "author_pi_code": item["author_pi_code"],
            "source": item["path"],
            "line": line,
            "sha256": sha(path),
            "paper_pointer_unverified": item["paper_pointer"],
            "question": item["question"],
            "verification_scope": "LOCAL_TEXT_ANCHOR_ONLY",
        })
    return rows


def render_report(manifest: dict, rows: list[dict]) -> str:
    lines = [
        "# PI-H2T author-source provenance gate",
        "",
        "**Status: BLOCKED_FOR_CLAIM_OF_EXACT_AUTHOR_REPRODUCTION**.",
        "",
        "This audit validates quoted local files only. It does not verify PDF page",
        "numbers, current GitHub branch/tag availability or authors' unpublished code.",
        "Even if all anchors match, no GPU training is authorized by this report.",
        "",
        "## Upstream inventory (preliminary, recheck live)",
        "",
        "- Author-referenced repository: " + manifest["upstream"]["repo"],
        "- Checked commit: " + manifest["upstream"]["checked_head"],
        "- Observed branches: " + ", ".join(manifest["upstream"]["branch_names_observed"]),
        "- Observed tree entries: " + str(manifest["upstream"]["tree_size_observed"]),
        "- README identifies the AAAI 2024 H2T sample code, not a complete",
        "  PI-H2T implementation. This is NOT proof of global non-availability.",
        "",
        "## Line-anchored claims",
        "",
        "| Field | Present local evidence | Status | PI author code? |",
        "|---|---|---|---|",
    ]
    for row in rows:
        loc = row["source"] + ":" + str(row["line"])
        lines.append("| " + row["id"] + " | " + loc.replace("|", "/") +
                     " | " + row["status"] + " | " + row["author_pi_code"] + " |")
    lines.extend([
        "",
        "## Scientific gate",
        "",
        "The source README and old local reconstruction cannot establish which",
        "PIF initialization, Eq8 W-reference, detach rule, BN-running-buffer mode",
        "or checkpoint selection the PI-H2T authors actually used.",
        "",
        "Codex MUST independently read the full PI-H2T journal/arXiv paper,",
        "supplement, Code Availability statement, and the actual author GitHub",
        "refs and files. Record commit-qualified code paths or PDF page numbers.",
        "Only paper/author evidence may resolve an UNKNOWN field; NEVER choose",
        "frozen/current W or another parameter because its TEST result is larger.",
        "",
        "Outcomes: AUTHOR_PI_CODE_VERIFIED; PAPER_RECONSTRUCTION_WITH_EXPLICIT_GAPS;",
        "or BLOCKED_NEEDS_AUTHOR_CONFIRMATION. No training in this study.",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--manifest", type=Path, default=HERE / "SOURCE_EVIDENCE.json")
    parser.add_argument("--output", type=Path, required=True)
    a = parser.parse_args()
    try:
        root = a.root.resolve()
        manifest = json.loads(a.manifest.read_text(encoding="utf-8"))
        rows = audit(root, manifest)
        out = (root / a.output).resolve()
        permitted = (root / "H2T_repro_audit/pi_h2t_author_audit_20261027/results").resolve()
        if not out.is_relative_to(permitted):
            raise ValueError("Output must stay inside this study's results/")
        out.mkdir(parents=True, exist_ok=False)
        with (out / "LOCAL_ANCHORS.csv").open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        (out / "AUTHOR_SOURCE_GATE.md").write_text(
            render_report(manifest, rows), encoding="utf-8"
        )
        status = {
            "local_anchors_verified": len(rows),
            "exact_author_pi_reproduction_cleared": False,
            "author_code_directly_verified_by_script": False,
            "training_allowed": False,
            "next_action": "PRIMARY_AUTHOR_SOURCE_AND_PAPER_MANUAL_REVIEW",
            "manifest_sha256": sha(a.manifest),
            "evidence_sources_sha256": {r["source"]: r["sha256"] for r in rows},
        }
        (out / "GATE_STATUS.json").write_text(
            json.dumps(status, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except (OSError, ValueError, AssertionError, json.JSONDecodeError) as exc:
        print("[SOURCE GATE FAIL] " + str(exc))
        return 2
    print("[SOURCE GATE PASS] local anchors=", len(rows), "output=", out)
    print("Exact author PI code NOT verified. No training permitted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
