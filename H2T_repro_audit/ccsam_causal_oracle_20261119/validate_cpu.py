"""Run syntax checks and the actual CPU test suite; save an exclusive receipt."""
from __future__ import annotations

import os
from pathlib import Path
import py_compile
import subprocess
import sys

from oracle_runtime import ROOT, PARENT, code_hashes, new_json, now


def main():
    receipt = ROOT / "validation" / "CPU_VALIDATION.json"
    if receipt.exists():
        raise FileExistsError("Do not overwrite a validation receipt")
    syntax = []
    cache = ROOT / "__pycache__"
    cache.mkdir(exist_ok=True)
    for path in sorted(ROOT.glob("*.py")):
        py_compile.compile(str(path), cfile=str(cache / (path.stem + ".pyc")), doraise=True)
        syntax.append(path.name)
    logs = ROOT / "logs"
    logs.mkdir(exist_ok=True)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    tests = []
    for name, directory, pattern in (("causal_oracle", ROOT, "test_*.py"),
                                      ("parent_ccmia", PARENT, "test_ccmia_core.py")):
        command = [sys.executable, "-B", "-m", "unittest", "discover", "-s", str(directory), "-p", pattern, "-v"]
        completed = subprocess.run(command, cwd=directory, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        log = logs / (name + "_cpu_tests.log")
        with log.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(completed.stdout)
        print(completed.stdout, flush=True)
        tests.append({"suite": name, "command": command, "exit_code": completed.returncode,
                      "test_log": str(log.relative_to(ROOT))})
        if completed.returncode != 0:
            break
    new_json(receipt, {"passed": all(t["exit_code"] == 0 for t in tests) and len(tests) == 2, "completed_at": now(),
                       "interpreter": sys.executable, "syntax_files": syntax,
                       "test_suites": tests,
                       "code_sha256": code_hashes(),
                       "GPU_started": False, "official_TEST_used": False})
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)
    print("CPU SYNTAX AND UNIT TESTS PASS", flush=True)


if __name__ == "__main__":
    main()
