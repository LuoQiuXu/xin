"""Finish artifact bookkeeping after a fully completed original Stage I run.

The official save_code intentionally excludes PDFs/PPTX; our first post-run
check expected every archive file. Preserve the failure, supplement those
non-code documents, and copy the already saved epoch-200 checkpoint. No model
or training process is created, and no checkpoint tensors are changed.
"""
from pathlib import Path
import hashlib
import json
import shutil
import sys
import time

BASE = Path(__file__).resolve().parent
RUN = BASE / "formal_runs" / sys.argv[1]
status_path = RUN / "status.json"
status = json.loads(status_path.read_text(encoding="utf-8"))
preflight = json.loads((RUN / "preflight.json").read_text(encoding="utf-8"))
assert status["state"] == "failed" and status["completed_epochs"] == 200
assert len(status["epochs"]) == 200 and not status["checkpoint_load_calls"]
assert "FileNotFoundError" in status["error"] and "codes" in status["error"]
assert not (RUN / "status_before_finalize_recovery.json").exists()
shutil.copy2(status_path, RUN / "status_before_finalize_recovery.json")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


snapshot = Path(status["code_dir"])
supplemented = []
for relative, expected in preflight["runtime_files"].items():
    source = BASE / "runtime" / relative
    target = snapshot / relative
    assert sha(source) == expected, f"Runtime modified: {relative}"
    if not target.exists():
        assert target.suffix.lower() in {".pdf", ".pptx"}, f"Unexpected missing source file: {relative}"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        supplemented.append(relative)
    assert sha(target) == expected, f"Snapshot mismatch: {relative}"
ckps = Path(status["model_dir"])
assert (ckps / "current.pth.tar").is_file() and (ckps / "model_best.pth.tar").is_file()
assert not (ckps / "model_final.pth.tar").exists()
shutil.copy2(ckps / "current.pth.tar", ckps / "model_final.pth.tar")
status["checkpoints_sha256"] = {p.name: sha(p) for p in ckps.glob("*.pth.tar")}
status["source_snapshot_verified"] = True
status["state"] = "completed"
status["postprocessing_recovery"] = {
    "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    "original_launcher_exit_code": 1,
    "training_completed_before_error": True,
    "reason": "Official save_code excludes PDF/PPTX; external all-file snapshot assertion expected them",
    "supplemented_non_code_files": supplemented,
    "core_or_checkpoint_tensors_modified": False,
    "reran_training": False,
    "original_error_preserved_in": str(RUN / "status_before_finalize_recovery.json")}
status_path.write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")
shutil.copy2(__file__, RUN / "finalize_completed_stage1.py")
print("POSTPROCESSING COMPLETE; training remained at 200 epochs; supplemented:", supplemented)
