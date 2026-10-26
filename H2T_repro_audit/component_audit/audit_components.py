#!/usr/bin/env python3
"""Static, read-only component audit for the clean official H2T source snapshot.

This tool does not import H2T training code, load datasets/checkpoints, or run
training. It records source hashes, AST-level definitions/call sites, selected
YAML settings, and evidence-backed candidate status for human review.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_ROOT = Path("H2T_official_clean/H2T-main")
DEFAULT_STAGE1 = Path("config/cifar100/cifar100_imb001_stage1_ce_mixup.yaml")
DEFAULT_STAGE2 = Path("config/cifar100/cifar100_imb001_stage2_ce_H2T.yaml")
TEXT_SUFFIXES = {".py", ".yaml", ".yml", ".md"}
TARGET_SYMBOLS = {
    "H2T", "H2Tv2", "mixup_data", "mixup_criterion",
    "LearnableWeightScaling", "NormedLinear", "Classifier",
    "ResNet_fe", "ResNet_s", "BasicBlock", "resnet32_fe",
    "CIFAR100_LT", "IMBALANCECIFAR100", "ClassAwareSampler",
    "BalancedDatasetSampler", "EffectNumSampler",
}
PATTERNS: dict[str, re.Pattern[str]] = {
    "feature_fusion": re.compile(r"(^|\.)(H2T|H2Tv2)(\b|\()|fusion|donor|sample_image", re.I),
    "input_mixup": re.compile(r"mixup", re.I),
    "normalization": re.compile(
        r"BatchNorm|LayerNorm|GroupNorm|InstanceNorm|SyncBatchNorm|RMSNorm|"
        r"NormedLinear|\.normalize$|normalize\(", re.I
    ),
    "attention_candidate": re.compile(
        r"attention|attn|CBAM|SELayer|SEBlock|SqueezeExcitation|ECA|NonLocal", re.I
    ),
    "sampling": re.compile(r"sampl|DataLoader|weighted", re.I),
    "loss": re.compile(r"loss|criterion|CrossEntropy|LDAM|BalancedSoftmax|Focal", re.I),
    "classifier_head": re.compile(r"classifier|NormedLinear|(^|\.)Linear(\b|\()", re.I),
    "pooling": re.compile(r"avg_pool|adaptive_avg_pool|global_pool", re.I),
    "gradient_control": re.compile(
        r"no_grad|inference_mode|requires_grad|detach|backward|zero_grad|optimizer|autocast", re.I
    ),
    "model_mode": re.compile(r"\.(train|eval)$", re.I),
}
CONFIG_KEYS_OF_INTEREST = (
    "name", "dataset", "num_classes", "imb_factor", "backbone", "classifier",
    "loss.type", "loss.add_extra_info", "mixup", "alpha", "H2T", "shift_bn",
    "sampler.type", "sampler.weighted_sampler.type",
    "sampler.dual_sample.enable", "sampler.dual_sample.type",
    "lr", "batch_size", "weight_decay", "num_epochs", "momentum",
    "lr_scheduler.type", "num_experts",
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def strip_yaml_comment(line: str) -> str:
    """Strip YAML comments without treating # inside quoted strings as comments."""
    single = double = escaped = False
    for index, char in enumerate(line):
        if escaped:
            escaped = False
            continue
        if char == "\\" and double:
            escaped = True
            continue
        if char == "'" and not double:
            single = not single
        elif char == '"' and not single:
            double = not double
        elif char == "#" and not single and not double:
            if index == 0 or line[index - 1].isspace():
                return line[:index]
    return line


def parse_scalar(raw: str) -> Any:
    value = raw.strip()
    if not value:
        return None
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    lowered = value.lower()
    if lowered in {"true", "yes", "on"}:
        return True
    if lowered in {"false", "no", "off"}:
        return False
    if lowered in {"null", "none", "~"}:
        return None
    if re.fullmatch(r"[+-]?\d+", value):
        try:
            return int(value)
        except ValueError:
            pass
    if re.fullmatch(r"[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?", value):
        try:
            return float(value)
        except ValueError:
            pass
    return value


def parse_config_text(text: str) -> dict[str, Any]:
    """Parse the indentation-based scalar mapping subset used by H2T YAML configs.

    This intentionally extracts scalar mappings only. YAML lists are ignored;
    this is sufficient for the audit switches and avoids a PyYAML dependency.
    It is not intended to replace a general YAML parser.
    """
    flat: dict[str, Any] = {}
    stack: list[tuple[int, str]] = []
    for raw_line in text.splitlines():
        line = strip_yaml_comment(raw_line).rstrip()
        if not line.strip():
            continue
        stripped = line.lstrip()
        if stripped.startswith("- ") or stripped == "-":
            continue
        match = re.match(r"^([^:#][^:]*?):(?:\s*(.*))?$", stripped)
        if not match:
            continue
        indent = len(line) - len(stripped)
        key = match.group(1).strip().strip("'\"")
        raw_value = (match.group(2) or "").strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        full_key = ".".join([part for _, part in stack] + [key])
        if not raw_value:
            stack.append((indent, key))
        else:
            flat[full_key] = parse_scalar(raw_value)
    return flat


def call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = call_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    if isinstance(node, ast.Call):
        return call_name(node.func)
    if isinstance(node, ast.Subscript):
        return call_name(node.value)
    return ""


def categories_for(name: str, source_segment: str = "") -> list[str]:
    probe = f"{name} {source_segment}".strip()
    return [category for category, pattern in PATTERNS.items() if pattern.search(probe)]


def is_relevant_symbol(name: str) -> bool:
    return name in TARGET_SYMBOLS or bool(categories_for(name))


class ComponentVisitor(ast.NodeVisitor):
    def __init__(self, source: str, relative_path: str):
        self.source = source
        self.relative_path = relative_path
        self.scope: list[str] = []
        self.symbols: list[dict[str, Any]] = []
        self.calls: list[dict[str, Any]] = []
        self.contexts: list[dict[str, Any]] = []

    @property
    def scope_name(self) -> str:
        return ".".join(self.scope) or "<module>"

    def _record_symbol(self, node: ast.AST, kind: str, name: str) -> None:
        # Preserve every definition as call-ownership context. Generic names
        # such as forward/train_step need not contain component keywords;
        # an empty category list does not make them active method components.
        self.symbols.append({
            "kind": kind,
            "name": name,
            "scope": self.scope_name,
            "path": self.relative_path,
            "line": getattr(node, "lineno", None),
            "categories": categories_for(name),
        })

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._record_symbol(node, "class", node.name)
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._record_symbol(node, "function", node.name)
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._record_symbol(node, "async_function", node.name)
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    def visit_Call(self, node: ast.Call) -> None:
        name = call_name(node.func)
        segment = ast.get_source_segment(self.source, node) or ""
        categories = categories_for(name, segment)
        if name == "eval" and "config.loss.type" in segment:
            categories = sorted(set(categories + ["loss", "dynamic_dispatch"]))
        if categories:
            self.calls.append({
                "name": name,
                "scope": self.scope_name,
                "path": self.relative_path,
                "line": node.lineno,
                "categories": categories,
                "source": " ".join(segment.split())[:220],
            })
        self.generic_visit(node)

    def visit_With(self, node: ast.With) -> None:
        for item in node.items:
            name = call_name(item.context_expr)
            if any(token in name.lower() for token in ("no_grad", "inference_mode", "autocast")):
                self.contexts.append({
                    "kind": "context_manager",
                    "name": name,
                    "scope": self.scope_name,
                    "path": self.relative_path,
                    "line": getattr(item.context_expr, "lineno", node.lineno),
                })
        self.generic_visit(node)

    def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
        self.visit_With(node)


def git_value(args: list[str]) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(REPO_ROOT), *args],
            check=True, capture_output=True, text=True, timeout=3,
        )
        return result.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def source_files(source_root: Path) -> list[Path]:
    return sorted(
        path for path in source_root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in TEXT_SUFFIXES
        and "__pycache__" not in path.parts
        and ".git" not in path.parts
    )


def make_source_manifest(source_root: Path, files: list[Path]) -> dict[str, Any]:
    entries = []
    for path in files:
        data = path.read_bytes()
        entries.append({
            "path": path.relative_to(REPO_ROOT).as_posix()
            if path.is_relative_to(REPO_ROOT) else path.as_posix(),
            "bytes": len(data),
            "sha256": sha256_bytes(data),
        })
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository_root": str(REPO_ROOT),
        "git_branch": git_value(["branch", "--show-current"]),
        "git_head": git_value(["rev-parse", "HEAD"]),
        "source_root": str(source_root),
        "source_file_count": len(entries),
        "source_files": entries,
    }


def active_config_view(config: dict[str, Any]) -> dict[str, Any]:
    return {key: config[key] for key in CONFIG_KEYS_OF_INTEREST if key in config}


def load_config(path: Path) -> dict[str, Any]:
    return parse_config_text(path.read_text(encoding="utf-8"))


def scan_source(source_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]],
                                            list[dict[str, Any]], list[dict[str, Any]]]:
    symbols: list[dict[str, Any]] = []
    calls: list[dict[str, Any]] = []
    contexts: list[dict[str, Any]] = []
    parse_errors: list[dict[str, Any]] = []
    for path in sorted(source_root.rglob("*.py")):
        if "__pycache__" in path.parts or ".git" in path.parts:
            continue
        relative = path.relative_to(REPO_ROOT).as_posix() if path.is_relative_to(REPO_ROOT) else path.as_posix()
        try:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=relative)
        except (OSError, UnicodeDecodeError, SyntaxError) as exc:
            parse_errors.append({
                "path": relative,
                "error_type": type(exc).__name__,
                "message": str(exc),
            })
            continue
        visitor = ComponentVisitor(source, relative)
        visitor.visit(tree)
        symbols.extend(visitor.symbols)
        calls.extend(visitor.calls)
        contexts.extend(visitor.contexts)
    symbols.sort(key=lambda x: (x["path"], x["line"] or 0, x["name"]))
    calls.sort(key=lambda x: (x["path"], x["line"], x["name"]))
    contexts.sort(key=lambda x: (x["path"], x["line"], x["name"]))
    return symbols, calls, contexts, parse_errors


def find_symbol(symbols: list[dict[str, Any]], name: str) -> list[dict[str, Any]]:
    return [item for item in symbols if item["name"] == name]


def find_calls(calls: list[dict[str, Any]], name: str) -> list[dict[str, Any]]:
    leaf = name.rsplit(".", 1)[-1]
    return [item for item in calls if item["name"] == name or item["name"].rsplit(".", 1)[-1] == leaf]


def build_component_status(
    symbols: list[dict[str, Any]],
    calls: list[dict[str, Any]],
    stage1: dict[str, Any],
    stage2: dict[str, Any],
) -> list[dict[str, Any]]:
    definitions = {name: find_symbol(symbols, name) for name in TARGET_SYMBOLS}
    result = []
    for name in sorted(TARGET_SYMBOLS):
        call_sites = find_calls(calls, name)
        status = "defined or referenced; active use not established"
        evidence = []
        if name == "H2T":
            if float(stage2.get("H2T", 0) or 0) > 0 and any("train_stage2.py" in x["path"] for x in call_sites):
                status = "called by Stage 2 and enabled by selected Stage 2 config"
                evidence.append("Stage 2 H2T scalar is greater than zero")
            else:
                status = "defined; selected-config activation not established"
        elif name in {"mixup_data", "mixup_criterion"}:
            s1 = stage1.get("mixup") is True
            s2 = stage2.get("mixup") is True
            if s1 and any("train_stage1.py" in x["path"] for x in call_sites):
                status = "Stage 1 path enabled by selected config"
                evidence.append("Stage 1 mixup=true")
            if s2:
                status = "Stage 2 path enabled by selected config"
                evidence.append("Stage 2 mixup=true")
            elif s1:
                evidence.append("Stage 2 mixup is false or absent")
        elif name == "H2Tv2":
            status = "alternative implementation; no executable call site found" if not call_sites else "alternative implementation with call site(s)"
        elif name == "LearnableWeightScaling":
            status = "optional class; no executable call site found" if not call_sites else "class has executable call site(s)"
        elif name == "NormedLinear":
            status = "alternative classifier path; do not assume active for selected backbone"
        elif name == "Classifier":
            status = "explicit classifier class; verify active factory/call path in source"
        elif name in {"ClassAwareSampler", "BalancedDatasetSampler", "EffectNumSampler"}:
            status = "sampler implementation exists; selected loader usage must be checked separately"
        elif name in {"ResNet_fe", "ResNet_s", "BasicBlock", "resnet32_fe"}:
            status = "backbone implementation; context only, not an H2T method innovation"
        if not definitions[name] and not call_sites:
            continue
        result.append({
            "name": name,
            "definitions": definitions[name],
            "call_sites": call_sites,
            "status": status,
            "evidence_notes": evidence,
        })
    return result


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        safe = [str(cell).replace("|", "\\|").replace("\n", " ") for cell in row]
        lines.append("| " + " | ".join(safe) + " |")
    return "\n".join(lines)


def render_report(inventory: dict[str, Any]) -> str:
    cfg = inventory["configs"]
    stage1 = cfg["stage1"]["selected"]
    stage2 = cfg["stage2"]["selected"]
    status_rows = []
    for item in inventory["component_status"]:
        definitions = item["definitions"]
        calls = item["call_sites"]
        def_text = ", ".join(f'{x["path"]}:{x["line"]}' for x in definitions[:3]) or "—"
        call_text = ", ".join(f'{x["path"]}:{x["line"]}' for x in calls[:4]) or "none found"
        status_rows.append([item["name"], item["status"], def_text, call_text])
    norm_rows = [
        [x["name"], x["path"], str(x["line"]), x["scope"]]
        for x in inventory["calls"] if "normalization" in x["categories"]
    ]
    attention_rows = [
        [x["kind"], x["name"], x["path"], str(x["line"]), ", ".join(x.get("categories", []))]
        for x in inventory["symbols"] if "attention_candidate" in x.get("categories", [])
    ]
    attention_rows += [
        ["call", x["name"], x["path"], str(x["line"]), ", ".join(x["categories"])]
        for x in inventory["calls"] if "attention_candidate" in x.get("categories", [])
    ]
    mode_rows = [
        [x["name"], x["path"], str(x["line"]), x["scope"]]
        for x in inventory["calls"] if "model_mode" in x["categories"] or "gradient_control" in x["categories"]
    ]
    lines = [
        "# H2T Static Component Map",
        "",
        "Generated by audit_components.py. Static evidence only: a definition or call site does not by itself prove runtime execution.",
        "",
        "## 1. Audit provenance",
        "",
        f'- Source root: {inventory["source_manifest"]["source_root"]}',
        f'- Git branch: {inventory["source_manifest"].get("git_branch") or "not available"}',
        f'- Git HEAD: {inventory["source_manifest"].get("git_head") or "not available"}',
        f'- Hashed source files: {inventory["source_manifest"]["source_file_count"]}',
        f'- Python parse errors: {len(inventory["parse_errors"])}',
        "",
        "The script did not import training modules, load data/checkpoints, or run training.",
        "",
        "## 2. Selected configuration values",
        "",
        "### Stage 1",
        "",
        markdown_table(["Key", "Value"], [[k, json.dumps(v, ensure_ascii=False)] for k, v in stage1.items()]),
        "",
        "### Stage 2",
        "",
        markdown_table(["Key", "Value"], [[k, json.dumps(v, ensure_ascii=False)] for k, v in stage2.items()]),
        "",
        "Configuration values are static YAML evidence. Defaults supplied by the config loader and runtime overrides must be checked separately.",
        "",
        "## 3. Core/optional symbol status",
        "",
        markdown_table(["Symbol", "Static status", "Definition(s)", "Call site(s)"], status_rows),
        "",
        "## 4. Normalization and normalization-like operations",
        "",
        markdown_table(["Call", "Path", "Line", "Enclosing scope"], norm_rows) if norm_rows else "No normalization-like calls matched the current heuristics.",
        "",
        "## 5. Attention-like candidates",
        "",
        markdown_table(["Kind", "Name", "Path", "Line", "Category"], attention_rows) if attention_rows else "No identifier-based attention candidates were detected. This is not proof that no attention-like operation exists; inspect module bodies and imports.",
        "",
        "## 6. Training-mode, gradient, and optimization evidence",
        "",
        markdown_table(["Call/context", "Path", "Line", "Enclosing scope"], mode_rows) if mode_rows else "No relevant training-control calls matched the current heuristics.",
        "",
        "## 7. Parse errors and limitations",
        "",
    ]
    if inventory["parse_errors"]:
        lines.extend([
            markdown_table(["Path", "Error type", "Message"], [
                [x["path"], x["error_type"], x["message"]] for x in inventory["parse_errors"]
            ]),
            "",
        ])
    else:
        lines.extend(["No Python syntax/decoding errors were detected by the AST pass.", ""])
    lines.extend([
        "- Static AST analysis cannot prove a call executes at runtime.",
        "- YAML settings may be overridden by command-line arguments or config defaults.",
        "- Identifier-based attention detection is a heuristic; inspect class bodies, tensor operations, and imports.",
        "- A component being defined in the repository does not mean it is active in the selected CIFAR-100-LT configuration.",
        "- Use this evidence map for the human-written COMPONENT_AUDIT_REPORT.md; do not treat it as a causal ablation result.",
        "",
    ])
    return "\n".join(lines)


def run_audit(args: argparse.Namespace) -> Path:
    source_root = Path(args.source_root)
    if not source_root.is_absolute():
        source_root = REPO_ROOT / source_root
    source_root = source_root.resolve()
    required = [source_root / "methods.py", source_root / "train_stage1.py", source_root / "train_stage2.py"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Source root is not the expected H2T source snapshot; missing: " + ", ".join(missing)
        )

    stage1_path = Path(args.stage1_config)
    stage2_path = Path(args.stage2_config)
    if not stage1_path.is_absolute():
        stage1_path = source_root / stage1_path
    if not stage2_path.is_absolute():
        stage2_path = source_root / stage2_path
    if not stage1_path.is_file() or not stage2_path.is_file():
        raise FileNotFoundError(f"Missing config(s): stage1={stage1_path}, stage2={stage2_path}")

    files = source_files(source_root)
    manifest = make_source_manifest(source_root, files)
    symbols, calls, contexts, parse_errors = scan_source(source_root)
    stage1 = load_config(stage1_path)
    stage2 = load_config(stage2_path)
    inventory = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_manifest": manifest,
        "configs": {
            "stage1": {
                "path": stage1_path.relative_to(REPO_ROOT).as_posix()
                if stage1_path.is_relative_to(REPO_ROOT) else str(stage1_path),
                "selected": active_config_view(stage1),
                "all_scalar_values": stage1,
            },
            "stage2": {
                "path": stage2_path.relative_to(REPO_ROOT).as_posix()
                if stage2_path.is_relative_to(REPO_ROOT) else str(stage2_path),
                "selected": active_config_view(stage2),
                "all_scalar_values": stage2,
            },
        },
        "symbols": symbols,
        "calls": calls,
        "contexts": contexts,
        "component_status": build_component_status(symbols, calls, stage1, stage2),
        "parse_errors": parse_errors,
        "limitations": [
            "Static analysis does not prove runtime execution.",
            "Config-loader defaults and CLI overrides are not resolved by this tool.",
            "Attention detection is heuristic and identifier-based.",
        ],
    }

    if args.output_dir:
        output_dir = Path(args.output_dir)
        if not output_dir.is_absolute():
            output_dir = REPO_ROOT / output_dir
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = REPO_ROOT / "H2T_repro_audit/component_audit/results" / f"static_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=False)

    (output_dir / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output_dir / "static_inventory.json").write_text(
        json.dumps(inventory, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output_dir / "STATIC_COMPONENT_MAP.md").write_text(render_report(inventory), encoding="utf-8")
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", default=str(DEFAULT_SOURCE_ROOT),
                        help="Path relative to repository root, or an absolute path.")
    parser.add_argument("--stage1-config", default=str(DEFAULT_STAGE1),
                        help="Config path relative to source root, or an absolute path.")
    parser.add_argument("--stage2-config", default=str(DEFAULT_STAGE2),
                        help="Config path relative to source root, or an absolute path.")
    parser.add_argument("--output-dir", default=None,
                        help="New output directory. It must not already exist.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        output_dir = run_audit(args)
    except (OSError, ValueError) as exc:
        print(f"[component-audit] ERROR: {exc}")
        return 2
    print(f"[component-audit] outputs: {output_dir}")
    print(f"[component-audit] manifest: {output_dir / 'source_manifest.json'}")
    print(f"[component-audit] inventory: {output_dir / 'static_inventory.json'}")
    print(f"[component-audit] map: {output_dir / 'STATIC_COMPONENT_MAP.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
