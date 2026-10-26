"""Verify one new static audit result using bytes/AST only, never H2T imports.

The explicit run directory must exist. New verification artifacts use exclusive
creation so an earlier run can never be silently replaced.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys


REPO = Path(__file__).resolve().parents[2]
SOURCE = REPO / "H2T_official_clean/H2T-main"
FIELDS = ["component_id", "component_name", "category", "definition_path", "definition_line",
          "callsite_path", "callsite_line", "selected_stage1_status", "selected_stage2_status",
          "paper_evidence", "code_evidence", "removal_effect", "evidence_class",
          "existing_ablation", "confounders", "unresolved_question"]


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def save_new(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def git(*args):
    process = subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True,
                             encoding="utf-8", check=False)
    assert process.returncode == 0, (args, process.stderr)
    return {"command": ["git", *args], "exit_code": process.returncode,
            "stdout": process.stdout, "stderr": process.stderr}


def verify(run):
    before = json.loads((run / "FROZEN_TREE_BEFORE.json").read_text(encoding="utf-8"))
    manifest = json.loads((run / "source_manifest.json").read_text(encoding="utf-8"))
    inventory = json.loads((run / "static_inventory.json").read_text(encoding="utf-8"))
    reviewed = json.loads((run / "REVIEWED_COMPONENTS.json").read_text(encoding="utf-8"))
    with (run / "COMPONENT_EVIDENCE_MATRIX.csv").open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == FIELDS
        rows = list(reader)
    assert len(rows) == 42 == len(reviewed)
    assert len({row["component_id"] for row in rows}) == len(rows)
    references = []
    for row, original in zip(rows, reviewed):
        assert {k: str(original[k]) for k in FIELDS} == row
        assert row["removal_effect"] in {"ALREADY_TESTED", "CODE_DERIVED_EXPECTATION", "UNKNOWN"}
        assert all(row[k] for k in ("code_evidence", "confounders", "unresolved_question"))
        for kind in ("definition", "callsite"):
            path, line = row[kind + "_path"], row[kind + "_line"]
            if not path:
                assert line == ""
                continue
            resolved = (REPO / path).resolve()
            resolved.relative_to(SOURCE.resolve())
            contents = resolved.read_text(encoding="utf-8").splitlines()
            number = int(line)
            assert 0 < number <= len(contents)
            references.append({"component_id": row["component_id"], "kind": kind,
                               "path": path, "line": number, "source_text": contents[number - 1]})

    report = (run / "COMPONENT_AUDIT_REPORT.md").read_text(encoding="utf-8")
    component_rows = [line for line in report.splitlines() if re.match(r"\|C\d\d ", line)]
    assert len(component_rows) == 42
    assert all(len(re.split(r"(?<!\\)\|", line)) - 2 == 15 for line in component_rows)
    for section in ("特征融合", "双采样", "归一化与 BN", "分类头、损失", "输入增强",
                    "可选/未激活", "已确定与仍未测试", "按优先级排序"):
        assert section in report
    assert "```mermaid" in report

    config = json.loads((run / "CONFIG_RESOLUTION.json").read_text(encoding="utf-8"))
    stages = config["stages"]
    expected = {"stage1": {"mixup": True, "H2T": -1.0, "shift_bn": False,
                           "sampler.type": "default", "sampler.dual_sample.enable": False,
                           "lr_scheduler.type": "multistep", "lr_scheduler.lr_step": [160, 180]},
                "stage2": {"mixup": False, "H2T": 0.25, "shift_bn": True,
                           "sampler.type": "weighted sampler", "sampler.dual_sample.enable": True,
                           "sampler.dual_sample.type": "long-tailed", "lr_scheduler.lr_factor": 0.2}}
    for stage, values in expected.items():
        effective = stages[stage]["effective_without_CLI"]
        assert all(effective[k] == v for k, v in values.items())
        assert effective["loss.type"] == "CrossEntropyLoss"
        assert effective["head_class_idx"] == [0, 36]
        assert effective["med_class_idx"] == [36, 71]
        assert effective["tail_class_idx"] == [71, 100]
    assert 64 * stages["stage2"]["effective_without_CLI"]["H2T"] == 16
    assert abs(stages["stage2"]["effective_without_CLI"]["lr"] * .2 - .02) < 1e-15

    syntax = []
    for path in sorted(Path(__file__).parent.glob("*.py")):
        ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        syntax.append(path.relative_to(REPO).as_posix())
    assert not inventory["parse_errors"]
    manifest_differences = []
    for item in manifest["source_files"]:
        if digest(REPO / item["path"]) != item["sha256"]:
            manifest_differences.append(item["path"])
    assert not manifest_differences

    after_files = [{"path": path.relative_to(REPO).as_posix(), "bytes": path.stat().st_size,
                    "sha256": digest(path)} for path in sorted(SOURCE.rglob("*")) if path.is_file()]
    earlier = {item["path"]: item for item in before["files"]}
    later = {item["path"]: item for item in after_files}
    added = sorted(later.keys() - earlier.keys())
    deleted = sorted(earlier.keys() - later.keys())
    changed = sorted(key for key in earlier.keys() & later.keys() if earlier[key] != later[key])
    assert not added and not deleted and not changed, (added, deleted, changed)

    branch = git("branch", "--show-current")
    head = git("rev-parse", "HEAD")
    status = git("status", "--short")
    frozen_status = git("status", "--short", "--", "H2T_official_clean/H2T-main/")
    frozen_diff = git("diff", "HEAD", "--", "H2T_official_clean/H2T-main/")
    whitespace = git("diff", "--check")
    assert branch["stdout"].strip() == before["branch"] == "audit/h2t-component-map-20261026"
    assert head["stdout"].strip() == before["HEAD"]
    assert not frozen_status["stdout"] and not frozen_diff["stdout"]
    original_lines = set(before["initial_status"].splitlines())
    current_lines = set(status["stdout"].splitlines())
    assert original_lines <= current_lines
    new_lines = sorted(current_lines - original_lines)
    assert all(line[3:].startswith("H2T_repro_audit/component_audit/") for line in new_lines), new_lines

    result = {"checks_passed": True, "component_rows": len(rows), "CSV_columns": len(FIELDS),
              "markdown_component_columns": 15, "source_references_checked": len(references),
              "config_checks": "selected scalar/list values; defaults -> YAML; CLI not executed",
              "audit_tool_AST_files": syntax, "scanner_source_parse_errors": inventory["parse_errors"],
              "manifest_source_files": len(manifest["source_files"]), "manifest_differences": manifest_differences,
              "frozen_tree_files": len(after_files), "frozen_tree_bytes": sum(x["bytes"] for x in after_files),
              "frozen_tree_added": added, "frozen_tree_deleted": deleted, "frozen_tree_changed": changed,
              "git_checks": [branch, head, status, frozen_status, frozen_diff, whitespace],
              "preexisting_status": before["initial_status"], "new_status_lines": new_lines,
              "runtime_evidence": "NONE; only stdlib audit execution and CPU document tools",
              "source_hash_read_mode": "raw bytes, no checkpoint/NPZ deserialization"}
    save_new(run / "SOURCE_REFERENCE_CHECKS.json", references)
    save_new(run / "FROZEN_TREE_AFTER.json", {"files": after_files, "count": len(after_files)})
    save_new(run / "FINAL_INTEGRITY.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.run_dir.resolve()), ensure_ascii=False, indent=2))
