"""Run the frozen experiment in order and preserve partial results on failure."""
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

from analyze_results import ANALYSIS, analyze
from cross_dataset_smoke import SMOKE, load_smoke, run_smoke
from cuda_statistics import check_cuda_memory
from evaluate_final import FINAL, evaluate
from experiment_common import (
    ROOT, OUTPUT_ROOT, cuda_device, load_experiment, now, read_json, record_run, require_complete,
    run_log, run_path, save_status, sha256_file, write_csv, write_json,
)
from extract_features import extract_all
from prepare_experiment import PREFLIGHT, load_prepared, prepare
from train_stage1 import CHECKPOINT, train_all as train_stage1
from train_stage2 import HEAD, train_all as train_stage2

CUDA_CHECK = OUTPUT_ROOT / "cuda_self_check"
FORMAL = OUTPUT_ROOT / "formal_execution"
REPORT = OUTPUT_ROOT / "FORMAL_EXECUTION_REPORT.md"
COMPLETED = OUTPUT_ROOT / "formal_completed.json"
INDEX = OUTPUT_ROOT / "ARTIFACT_SHA256.csv"
PHASES = ("stage1", "features", "stage2", "final_evaluation", "analysis")
COMMANDS = ("preflight", "cuda-self-check", "smoke", "stage1", "extract", "stage2", "evaluate-final", "analyze", "run-formal")


def check_unused(paths) -> None:
    for path in paths:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite or resume existing output: {path}")


def status_at(directory: Path) -> dict:
    path = directory / "status.json"
    if path.is_file():
        return read_json(path)
    return {"state": "incomplete" if directory.exists() else "not started"}


def run_cuda_check(exp) -> None:
    with record_run(CUDA_CHECK, exp.config("cuda-self-check")) as status:
        status["cuda_diagnostics"] = {}
        result = check_cuda_memory(exp.protocol["resource_target"]["gpu"], status["cuda_diagnostics"])
        write_json(CUDA_CHECK / "cuda_statistics.json", result)
        status["passed"] = True
    print(f"CUDA allocation and peak statistics PASS: {CUDA_CHECK / 'cuda_statistics.json'}")


def require_cuda_check(exp, device) -> None:
    status = require_complete(CUDA_CHECK, phase="cuda-self-check", expected={"passed": True, "official_test_access": False})
    if "cuda_statistics.json" not in status["artifacts"]:
        raise RuntimeError("CUDA self-check lacks the memory statistics report")
    result = read_json(CUDA_CHECK / "cuda_statistics.json")
    config = read_json(CUDA_CHECK / "config.json")
    if (config["source_sha256"] != exp.source_hashes or result["passed"] is not True
        or result["cuda"]["device"] != str(device)
        or result["cuda"]["cuda_visible_devices"] != os.environ.get("CUDA_VISIBLE_DEVICES")):
        raise RuntimeError("CUDA self-check is for a different source snapshot or CUDA device")


def log_result(filename: str, pattern: str, exit_label: str) -> str:
    path = OUTPUT_ROOT / "validation" / filename
    if not path.is_file():
        return "No recorded validation log"
    content = path.read_text(encoding="utf-8-sig")
    summary = re.search(pattern, content)
    exit_code = re.search(rf"{exit_label}=(\d+)", content)
    return " · ".join(item for item in (summary.group(0) if summary else "",
                                       f"exit {exit_code.group(1)}" if exit_code else "") if item)


def execution_report(exp, error: str | None = None) -> None:
    preflight, smoke, evaluation = (status_at(directory) for directory in (PREFLIGHT, SMOKE, FINAL))
    runtime = preflight.get("environment", {})
    cuda_check = status_at(CUDA_CHECK)
    test_result = log_result("cpu_tests.log", r"Ran \d+ tests?", "CPU_TEST_EXIT_CODE")
    syntax_result = log_result("syntax_check.log", r"Compiled \d+ Python files", "SYNTAX_CHECK_EXIT_CODE")
    lines = [
        "# CC-SAM cross-dataset execution", "",
        f"Recorded {now()}.", "",
        f"- Branch: {exp.branch}", f"- Git HEAD: {exp.head}",
        f"- Implementation files with worktree changes: {len(exp.dirty_sources)}",
        "- The per-run configs contain exact SHA256 hashes for the implementation and protocol.",
        f"- Python: {runtime.get('python')}. PyTorch: {runtime.get('pytorch')}. torchvision: {runtime.get('torchvision')}.",
        f"- PyArrow: {runtime.get('pyarrow')}. CUDA: {runtime.get('cuda')}. cuDNN: {runtime.get('cudnn')}.",
        f"- GPU: {runtime.get('gpu')}. Device memory: {runtime.get('gpu_memory_bytes')} bytes.",
        f"- CPU unit tests: {test_result}",
        f"- Syntax check: {syntax_result}",
        f"- Preflight: {preflight['state']}. CUDA self-check: {cuda_check['state']}. "
        f"Smoke: {smoke['state']}. Final evaluation: {evaluation['state']}.",
        f"- Official test split access started: {evaluation.get('official_test_access', False)}.",
    ]
    if exp.dirty_sources:
        lines.extend(["", "Worktree source changes at execution time", ""])
        lines.extend(f"- {item.strip()}" for item in exp.dirty_sources)
    if error:
        lines.extend(["", "## Execution stopped", "", error])
    for item, directory in (("Preflight", PREFLIGHT), ("CUDA self-check", CUDA_CHECK),
                            ("Smoke", SMOKE), ("Final evaluation", FINAL)):
        details = status_at(directory)
        if details.get("error"):
            lines.extend(["", f"{item} error — {details['error']}"])

    for dataset in exp.datasets:
        details = status_at(SMOKE / dataset)
        if details["state"] != "not started":
            real = details.get("real_gradient_equivalence", {})
            lines.extend(["", f"- {dataset} Smoke {details['state']}, Stage-I batches completed "
                          f"{len(details.get('stage1_batches', []))}, real autograd equivalence {real.get('passed')}."])
    lines.extend(["", "## Stage I", "", "| Dataset | Seed | State | Completed epochs | Checkpoint SHA256 |",
                  "| --- | ---: | --- | ---: | --- |"])
    allocations = [smoke.get("peak_cuda_allocated_bytes")]
    for name, seed in exp.jobs():
        details = status_at(run_path("stage1", name, seed))
        allocations.append(details.get("peak_cuda_allocated_bytes"))
        lines.append(f"| {name} | {seed} | {details['state']} | {details.get('epochs_completed', 0)} | "
                     f"{details.get('artifacts', {}).get(CHECKPOINT, '—')} |")
    lines.extend(["", "## Stage II", "", "| Dataset | Seed | Arm | State | Completed epochs | Checkpoint SHA256 |",
                  "| --- | ---: | --- | --- | ---: | --- |"])
    for name, seed in exp.jobs():
        allocations.append(status_at(run_path("features", name, seed)).get("peak_cuda_allocated_bytes"))
        for arm in exp.arms:
            details = status_at(run_path("stage2", name, seed, arm))
            allocations.append(details.get("peak_cuda_allocated_bytes"))
            lines.append(f"| {name} | {seed} | {arm} | {details['state']} | {details.get('epochs_completed', 0)} | "
                         f"{details.get('artifacts', {}).get(HEAD, '—')} |")
    allocations.append(evaluation.get("peak_cuda_allocated_bytes"))
    peak = max((value for value in allocations if value is not None), default=None)
    memory = f"{peak:,} bytes" if peak is not None else "unavailable"
    lines.extend(["", f"Maximum recorded PyTorch CUDA allocation: {memory}."])
    test_access = evaluation.get("test_splits", {})
    if test_access:
        lines.extend(["", f"All formal checkpoint checks completed at {evaluation.get('formal_gate_verified_at')}."])
        for name, item in test_access.items():
            lines.append(f"- {name} access started {item['opened_at']}, completed {item.get('completed_at')}, "
                         f"evaluated rows {item.get('rows')}.")
    summary_path = ANALYSIS / "cross_dataset_summary.json"
    if summary_path.is_file() and status_at(ANALYSIS)["state"] == "completed":
        summary = read_json(summary_path)
        for name, result in summary["datasets"].items():
            lines.extend(["", f"## {name} fixed-epoch accuracy (%)", "",
                          "| Seed | Arm | Epoch | Overall | Many | Medium | Few |",
                          "| ---: | --- | ---: | ---: | ---: | ---: | ---: |"])
            for row in result["per_seed"]:
                values = " | ".join(f"{row[m]:.8g}" for m in ("Overall", "Many", "Medium", "Few"))
                lines.append(f"| {row['seed']} | {row['arm']} | {row['epoch']} | {values} |")
            lines.extend(["", "Paired differences in percentage points", "",
                          "| Seed | Contrast | Overall | Many | Medium | Few |",
                          "| ---: | --- | ---: | ---: | ---: | ---: |"])
            for row in result["paired_per_seed"]:
                values = " | ".join(f"{row[m]:+.8g}" for m in ("Overall", "Many", "Medium", "Few"))
                lines.append(f"| {row['seed']} | {row['contrast']} | {values} |")
        lines.extend(["", "The per-seed CSV files retain full numerical precision. The analysis is descriptive."])
    with REPORT.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines) + "\n")


def seal_completion(exp) -> None:
    require_complete(ANALYSIS, phase="analyze", expected={"datasets": list(exp.datasets)})
    evaluation = require_complete(FINAL, phase="evaluate-final", expected={
        "official_test_access": True,
        "stage1_verified": len(exp.datasets) * len(exp.seeds),
        "stage2_verified": len(exp.datasets) * len(exp.seeds) * len(exp.arms),
    })
    if any(evaluation["test_splits"].get(name, {}).get("rows") != spec["test_images"]
           for name, spec in exp.datasets.items()):
        raise RuntimeError("Formal completion requires the exact final test counts for both datasets")
    check_unused((REPORT, INDEX, COMPLETED))
    execution_report(exp)
    folders = (PREFLIGHT, CUDA_CHECK, SMOKE, FORMAL, *(OUTPUT_ROOT / phase for phase in PHASES))
    files = [REPORT, ROOT / "IMPLEMENTATION_REPORT.md"]
    files += [path for folder in folders if folder.exists() for path in folder.rglob("*") if path.is_file()]
    write_csv(INDEX, [
        {"file": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(files)
    ])
    write_json(COMPLETED, {
        "state": "completed", "completed_at": now(), "git_head": exp.head, "git_branch": exp.branch,
        "source_sha256": exp.source_hashes, "datasets": list(exp.datasets), "seeds": exp.seeds, "arms": list(exp.arms),
        "stage1_completed": evaluation["stage1_verified"],
        "stage2_completed": evaluation["stage2_verified"],
        "stage1_final_epoch": exp.protocol["stage1"]["epochs"], "stage2_final_epoch": exp.protocol["stage2"]["epochs"],
        "official_test_splits": evaluation["test_splits"], "artifact_index_sha256": sha256_file(INDEX),
    })


def run_formal(exp):
    check_unused((FORMAL, REPORT, INDEX, COMPLETED, *(OUTPUT_ROOT / phase for phase in PHASES)))
    prepared = load_prepared(exp)
    device = cuda_device()
    require_cuda_check(exp, device)
    load_smoke(exp)
    with record_run(FORMAL, exp.config("run-formal")) as status:
        status["completed_phases"] = ["preflight", "cuda-self-check", "smoke"]
        for label, fn in (
            ("stage1", lambda: train_stage1(exp, prepared, device)),
            ("extract", lambda: extract_all(exp, prepared, device)),
            ("stage2", lambda: train_stage2(exp, prepared, device)),
            ("evaluate-final", lambda: evaluate(exp, prepared, device)),
            ("analyze", lambda: analyze(exp, prepared)),
        ):
            status["current_phase"] = label
            save_status(FORMAL, status)
            run_log(FORMAL, f"Starting phase {label}")
            try:
                fn()
            finally:
                status["official_test_access"] = status_at(FINAL).get("official_test_access", False)
            status["completed_phases"].append(label)
            save_status(FORMAL, status)
    seal_completion(exp)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=COMMANDS)
    parser.add_argument("--all", action="store_true", help="Run both datasets and all frozen seeds")
    parser.add_argument("--workers", type=int, help="Defaults to 4; use 0 to debug Windows loader issues")
    args = parser.parse_args()
    exp = load_experiment(args.workers)
    try:
        if args.command == "preflight":
            prepare(exp)
        elif args.command == "cuda-self-check":
            run_cuda_check(exp)
        elif args.command == "run-formal":
            run_formal(exp)
        else:
            prepared = load_prepared(exp)
            if args.command == "smoke":
                device = cuda_device()
                require_cuda_check(exp, device)
                run_smoke(exp, prepared, device)
            elif args.command == "analyze":
                check_unused((ANALYSIS, REPORT, INDEX, COMPLETED))
                analyze(exp, prepared)
                seal_completion(exp)
            else:
                load_smoke(exp)
                device = cuda_device()
                require_cuda_check(exp, device)
                if args.command == "stage1":
                    check_unused(OUTPUT_ROOT / name for name in PHASES)
                    train_stage1(exp, prepared, device)
                elif args.command == "extract":
                    check_unused(OUTPUT_ROOT / name for name in PHASES[1:])
                    extract_all(exp, prepared, device)
                elif args.command == "stage2":
                    check_unused(OUTPUT_ROOT / name for name in PHASES[2:])
                    train_stage2(exp, prepared, device)
                elif args.command == "evaluate-final":
                    evaluate(exp, prepared, device)
    except (Exception, KeyboardInterrupt) as exc:
        if not REPORT.exists():
            execution_report(exp, f"{type(exc).__name__}: {exc}")
        print(f"Execution report: {REPORT}", flush=True)
        raise
    print(f"Git HEAD: {exp.head}")
    for path in (ROOT / "IMPLEMENTATION_REPORT.md", REPORT, ANALYSIS / "CROSS_DATASET_RESULTS.md", INDEX, COMPLETED):
        if path.is_file():
            print(path)


if __name__ == "__main__":
    main()
