from __future__ import annotations

import hashlib
import json
import py_compile
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
VALIDATION = ROOT / "validation"
HASHED_DOCS = (
    ROOT / "README.md",
    ROOT / "TPMAG_PROTOCOL.md",
    ROOT / "CODEX_TPMAG_EXECUTE_20261120.txt",
)


def now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def code_hashes():
    paths = sorted(ROOT.glob("*.py")) + list(HASHED_DOCS)
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing code/protocol files: " + ", ".join(missing))
    return {str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path) for path in paths}


def write_exclusive(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def main():
    VALIDATION.mkdir(parents=True, exist_ok=True)
    receipt = VALIDATION / "CPU_VALIDATION.json"
    failure = VALIDATION / "CPU_FAILURE.json"
    if receipt.exists() or failure.exists():
        raise FileExistsError("CPU validation output already exists; refusing overwrite")

    test = subprocess.run(
        [sys.executable, "-m", "unittest", "-v", "test_gate_core"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        encoding="utf-8",
    )
    compile_errors = []
    for path in sorted(ROOT.glob("*.py")):
        try:
            py_compile.compile(str(path), doraise=True)
        except Exception as exc:
            compile_errors.append({"path": str(path), "error": repr(exc)})

    report = {
        "completed_at": now(),
        "python": sys.executable,
        "python_version": sys.version,
        "test_returncode": int(test.returncode),
        "test_stdout": test.stdout,
        "test_stderr": test.stderr,
        "compile_errors": compile_errors,
        "code_sha256": code_hashes(),
        "passed": test.returncode == 0 and not compile_errors,
    }
    if not report["passed"]:
        report["traceback"] = "Unit tests or py_compile failed; formal phase is prohibited."
        write_exclusive(failure, report)
        raise RuntimeError("TP-MAG CPU validation failed; inspect validation/CPU_FAILURE.json")

    write_exclusive(receipt, report)
    print("TP-MAG CPU VALIDATION PASS")
    print(test.stdout)
    print(test.stderr)


if __name__ == "__main__":
    main()
