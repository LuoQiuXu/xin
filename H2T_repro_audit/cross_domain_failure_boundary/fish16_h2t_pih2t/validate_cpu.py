"""One-shot, reproducible CPU validation gate for the Fish16 clean launch.

Runs syntax compilation and the complete CPU unittest suite, then records the
exact source fingerprints required by Smoke and the later formal runner.
No dataset image is opened and no CUDA/model training operation is performed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import traceback

from protocol_config import (
    BRANCH, CPU_VALIDATION_SOURCE_FILES, EXPECTED_CPU_TEST_METHODS,
    FORMAL_RUNNER_SOURCE_FILES, LOCKED_UPSTREAM_SOURCE_HASHES,
)

BASE = Path(__file__).resolve().parent
REPO = BASE.parents[2]


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_text(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir).resolve()
    if output.is_relative_to(REPO):
        raise ValueError("CPU validation output must be outside the Git repository")
    output.mkdir(parents=True, exist_ok=False)

    result = {
        "status": "RUNNING",
        "branch": git_text("branch", "--show-current"),
        "head": git_text("rev-parse", "HEAD"),
        "worktree_clean_before": not bool(git_text("status", "--porcelain")),
        "python_executable": sys.executable,
        "source_sha256_before": {},
        "source_sha256_after": {},
        "source_hashes": {},
        "upstream_source_sha256": {},
        "checks": [],
        "test_methods_expected": EXPECTED_CPU_TEST_METHODS,
        "test_methods_run": None,
        "gpu_started": False,
        "dataset_images_opened": 0,
    }
    try:
        if result["branch"] != BRANCH:
            raise RuntimeError(f"Wrong branch: {result['branch']} != {BRANCH}")
        if not result["worktree_clean_before"]:
            raise RuntimeError("Git worktree must be clean before CPU validation")

        result["source_sha256_before"] = {
            name: sha(BASE / name) for name in CPU_VALIDATION_SOURCE_FILES
        }
        result["upstream_source_sha256"] = {
            name: sha(REPO / name) for name in LOCKED_UPSTREAM_SOURCE_HASHES
        }
        if result["upstream_source_sha256"] != LOCKED_UPSTREAM_SOURCE_HASHES:
            raise RuntimeError("Locked upstream H2T/PI-H2T source fingerprint mismatch")

        compile_files = [str(BASE / name) for name in CPU_VALIDATION_SOURCE_FILES]
        compile_command = [
            sys.executable, "-B", "-X", "utf8", "-m", "py_compile", *compile_files
        ]
        compile_run = subprocess.run(
            compile_command, cwd=BASE, text=True, capture_output=True, encoding="utf-8"
        )
        (output / "SYNTAX.log").write_text(
            compile_run.stdout + compile_run.stderr, encoding="utf-8"
        )
        result["checks"].append({
            "name": "SYNTAX",
            "command": compile_command,
            "exit_code": compile_run.returncode,
            "files": len(compile_files),
            "log": "SYNTAX.log",
        })
        if compile_run.returncode != 0:
            raise RuntimeError("Python syntax compilation failed")

        test_command = [
            sys.executable, "-B", "-X", "utf8", "-m", "unittest",
            "test_amp_recovery", "test_contract", "test_formal_train",
            "diagnostics.test_diagnostic_stats", "-v",
        ]
        test_run = subprocess.run(
            test_command, cwd=BASE, text=True, capture_output=True, encoding="utf-8"
        )
        test_log = test_run.stdout + test_run.stderr
        (output / "CPU_ALL_SUITES.log").write_text(test_log, encoding="utf-8")
        match = re.search(r"Ran (\\d+) tests?", test_log)
        count = int(match.group(1)) if match else None
        result["test_methods_run"] = count
        result["checks"].append({
            "name": "CPU_ALL_SUITES",
            "command": test_command,
            "exit_code": test_run.returncode,
            "test_methods_run": count,
            "log": "CPU_ALL_SUITES.log",
        })
        if test_run.returncode != 0 or count != EXPECTED_CPU_TEST_METHODS or "OK" not in test_log:
            raise RuntimeError(
                f"CPU unittest gate failed: exit={test_run.returncode}, "
                f"methods={count}, expected={EXPECTED_CPU_TEST_METHODS}"
            )

        result["source_sha256_after"] = {
            name: sha(BASE / name) for name in CPU_VALIDATION_SOURCE_FILES
        }
        if result["source_sha256_before"] != result["source_sha256_after"]:
            raise RuntimeError("Source fingerprints changed during CPU validation")
        result["source_hashes"] = {
            name: result["source_sha256_after"][name]
            for name in FORMAL_RUNNER_SOURCE_FILES
        }
        result["worktree_clean_after"] = not bool(git_text("status", "--porcelain"))
        if not result["worktree_clean_after"]:
            raise RuntimeError("Git worktree became dirty during CPU validation")
        result["status"] = "PASS"
    except Exception as exc:
        result["status"] = "FAIL"
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)
        result["traceback"] = traceback.format_exc()

    (output / "CPU_VALIDATION.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
