"""Verify audit evidence and record exact source snapshots without training."""
import ast
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

from prepare_compatibility import write_diff

BASE = Path(__file__).resolve().parent
EVIDENCE = BASE / "evidence"
OFFICIAL = BASE / "official"
RUNTIME = BASE / "runtime"


def read(name):
    return json.loads((EVIDENCE / name).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(name, value):
    (EVIDENCE / name).write_text(json.dumps(value, indent=2, ensure_ascii=False),
                                 encoding="utf-8")


class NormalizeCheckpointLoad(ast.NodeTransformer):
    def visit_Call(self, node):
        self.generic_visit(node)
        if (isinstance(node.func, ast.Attribute) and node.func.attr == "load"
                and isinstance(node.func.value, ast.Name) and node.func.value.id == "torch"):
            node.keywords = [k for k in node.keywords if k.arg != "weights_only"]
        return node


def source_tree(path):
    return ast.parse(path.read_text(encoding="utf-8"))


def function_ast(path, name):
    tree = source_tree(path)
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    return ast.dump(function, include_attributes=False)


def main():
    sys.path.insert(0, str(RUNTIME))
    from utils import config, update_config
    stage1_cfg = config.clone()
    update_config(stage1_cfg, argparse.Namespace(
        cfg=str(RUNTIME / "config/cifar100/cifar100_imb001_stage1_ce_mixup.yaml"),
        opts=["resume", "''", "name", "cifar100_imb001_stage1_ce_mixup_formal",
              "data_path", str(BASE.parent / "H2T/data/cifar100")]))
    assert stage1_cfg.resume == "" and stage1_cfg.num_epochs == 200
    assert stage1_cfg.mixup and stage1_cfg.gpu is None
    stage2_cfg = config.clone()
    update_config(stage2_cfg, argparse.Namespace(
        cfg=str(RUNTIME / "config/cifar100/cifar100_imb001_stage2_ce_H2T.yaml"),
        opts=["resume", "D:/placeholder/model_best.pth.tar",
              "name", "cifar100_imb001_stage2_ce_h2t_formal"]))
    assert stage2_cfg.num_epochs == 10 and stage2_cfg.H2T == .25
    assert stage2_cfg.shift_bn and stage2_cfg.gpu is None
    write_diff()
    source = read("source_manifest.json")
    verified_source = all(sha(OFFICIAL / f["path"]) == f["sha256"] for f in source["files"])
    assert verified_source
    identical_files = ["methods.py", "datasets/cifar100.py", "loss.py", "utils/logger.py",
        "config/cifar100/cifar100_imb001_stage1_ce_mixup.yaml",
        "config/cifar100/cifar100_imb001_stage2_ce_H2T.yaml"]
    assert all((OFFICIAL/p).read_bytes() == (RUNTIME/p).read_bytes() for p in identical_files)
    assert all(function_ast(OFFICIAL / p, f) == function_ast(RUNTIME / p, f)
               for p in ["train_stage1.py", "train_stage2.py"]
               for f in ["train", "validate", "adjust_learning_rate"])
    for filename in ["train_stage1.py", "train_stage2.py", "eval-modifed.py"]:
        original = NormalizeCheckpointLoad().visit(source_tree(OFFICIAL / filename))
        runtime = NormalizeCheckpointLoad().visit(source_tree(RUNTIME / filename))
        assert ast.dump(original) == ast.dump(runtime)

    executions = {}
    log_evidence = {}
    for stage in ["stage1", "stage2", "eval"]:
        result = read(stage + ".json")
        assert result["status"] == "pass"
        text = (EVIDENCE / (stage + ".stdout.log")).read_text(encoding="utf-8", errors="replace")
        metrics = re.search(r"\* Acc@1 ([\d.]+)% Acc@5 ([\d.]+)% HAcc ([\d.]+)% MAcc ([\d.]+)% TAcc ([\d.]+)%", text)
        assert metrics
        printed = [float(metrics.group(i)) for i in [1, 3, 4, 5]]
        computed = [result["evaluation"][key] for key in ["overall", "head", "medium", "tail"]]
        assert all(abs(a-b) < .00051 for a, b in zip(printed, computed))
        log_evidence[stage] = dict(zip(["overall", "head", "medium", "tail"], printed))
        if stage != "eval":
            snapshot = Path(result["model_dir"]).parent / "codes"
            files = {p.relative_to(snapshot).as_posix(): sha(p) for p in sorted(snapshot.rglob("*"))
                     if p.is_file() and "__pycache__" not in p.parts}
            assert files
            assert sha(Path(result["checkpoint"]["path"])) == result["checkpoint"]["sha256"]
            executions[stage] = {"argv": result["argv"], "effective_config": result["effective_config"],
                "code_snapshot_directory": str(snapshot), "snapshot_sha256": files,
                "checkpoint": result["checkpoint"]}
    stage2 = read("stage2.json")
    evaluation = read("eval.json")
    assert stage2["evaluation"]["correct_per_class"] == evaluation["evaluation"]["correct_per_class"]
    assert stage2["loaded_checkpoint"]["sha256"] == read("stage1.json")["checkpoint"]["sha256"]
    assert stage2["loaded_checkpoint"]["strict_backbone_state_equality"]
    assert stage2["loaded_checkpoint"]["strict_classifier_state_equality"]
    executions["eval"] = {"argv": evaluation["argv"], "effective_config": evaluation["effective_config"],
        "loaded_checkpoint": evaluation["loaded_checkpoint"]}
    dump("execution_source_manifest.json", {"commit_from_archive": source["commit"],
        "archive_sha256": source["archive_sha256"], "executions": executions,
        "audit_runner_sha256": sha(BASE / "audit_original.py"),
        "compatibility_diff_sha256": sha(EVIDENCE / "compatibility.diff"),
        "evidence_sha256": {p.name: sha(p) for p in sorted(EVIDENCE.iterdir())
                            if p.is_file() and p.suffix in [".json", ".log", ".diff"]
                            and p.name not in ["execution_source_manifest.json", "final_verification.json"]}})
    dump("final_verification.json", {"status": "pass", "official_manifest_file_count": len(source["files"]),
        "official_manifest_all_match": verified_source, "identical_runtime_files": identical_files,
        "train_validate_scheduler_AST_unchanged": True,
        "training_and_eval_entrypoints_only_change_checkpoint_loading": True,
        "log_metrics_match_independent_count": log_evidence,
        "stage2_eval_correct_per_class_identical": True,
        "stage2_loaded_exact_stage1_audit_checkpoint": True,
        "formal_config_parse_pass": True,
        "formal_training_executed_during_command_check": False,
        "complete_training_epochs_executed": {"stage1": 1, "stage2": 1},
        "full_training_started": False})
    report = BASE.parent / "H2T_REPRO_AUDIT.md"
    text = report.read_text(encoding="utf-8").replace("(/D:/", "(D:/")
    report.write_text(text, encoding="utf-8", newline="\n")
    links = re.findall(r"\]\((D:/[^)]+)\)", text)
    assert all(Path(re.sub(r":\d+$", "", target)).is_file() for target in links)
    print("FINAL VERIFICATION PASS. Original files, algorithm, checkpoints and metrics verified.")


if __name__ == "__main__":
    main()
