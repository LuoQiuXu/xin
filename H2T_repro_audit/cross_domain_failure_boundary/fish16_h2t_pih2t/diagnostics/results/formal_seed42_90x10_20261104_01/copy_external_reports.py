"""Copy only the finished small review archive to new, explicit external paths."""
import hashlib
import json
from pathlib import Path

SRC = Path(__file__).parent.resolve()
DEST = Path(r"D:\h2t_cross_domain_results\fish16_formal_seed42_90x10_20261104_01_POSTRUN_ANALYSIS")
REPORT = Path(r"D:\h2t_cross_domain_results\fish16_formal_seed42_90x10_20261104_01_POSTRUN_ANALYSIS.md")
assert not DEST.exists() and not REPORT.exists(), "Never overwrite external outputs"
files = sorted(p for p in SRC.rglob("*") if p.is_file())
assert files and sum(p.stat().st_size for p in files) < 10_000_000
assert all(p.suffix.lower() not in (".pt", ".pth", ".npz", ".jsonl") for p in files)
assert all(p.resolve().is_relative_to(SRC) for p in files)
DEST.mkdir()
copies = []
for source in files:
    target = DEST / source.relative_to(SRC)
    target.parent.mkdir(parents=True, exist_ok=True)
    content = source.read_bytes()
    with target.open("xb") as stream:
        stream.write(content)
    digest = hashlib.sha256(content).hexdigest()
    assert hashlib.sha256(target.read_bytes()).hexdigest() == digest
    copies.append({"relative_path": source.relative_to(SRC).as_posix(), "bytes": len(content), "sha256": digest})
text = (SRC / "POSTRUN_ANALYSIS.md").read_text(encoding="utf-8")
for prefix in ("analysis/", "raw/", "TASK_SPEC.txt", "VALIDATION_REPORT.md"):
    text = text.replace("(" + prefix, "(" + DEST.name + "/" + prefix)
with REPORT.open("x", encoding="utf-8", newline="\n") as stream:
    stream.write(text)
delivery = dict(status="COMPLETE_EXTERNAL_REPORT_DELIVERY", external_report=str(REPORT),
                external_archive=str(DEST), report_sha256=hashlib.sha256(REPORT.read_bytes()).hexdigest(),
                copied_files=copies, original_training_outputs_modified=0,
                earlier_external_analysis_launch_approval_timeouts=2,
                external_copy_is_safer_alternative_to_executing_analysis_outside_workspace=True)
for target in (SRC / "DELIVERY_STATUS.json", DEST / "DELIVERY_STATUS.json"):
    with target.open("x", encoding="utf-8") as stream:
        json.dump(delivery, stream, indent=2)
print(json.dumps({k: v for k, v in delivery.items() if k != "copied_files"}))
