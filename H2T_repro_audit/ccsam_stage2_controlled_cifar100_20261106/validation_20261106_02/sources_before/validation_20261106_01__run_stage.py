"""Execute only the requested original prepare/smoke entry points, preserving failures."""
import argparse
import contextlib
import hashlib
import json
import runpy
import sys
import traceback
from datetime import datetime
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
parser = argparse.ArgumentParser()
parser.add_argument("stage", choices=("prepare", "smoke"))
args = parser.parse_args()
target = ROOT / ("prepare_ccsam.py" if args.stage == "prepare" else "train_ccsam.py")
argv = [str(target)] + (["--smoke-only"] if args.stage == "smoke" else [])
record = dict(stage=args.stage, target_argv=argv, interpreter=sys.executable,
              started=datetime.now().astimezone().isoformat(), formal_training_started=False)
sys.path.insert(0, str(ROOT))
sys.argv = argv
failure = None
with (OUT / (args.stage + ".log")).open("x", encoding="utf-8") as log:
    with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        try:
            runpy.run_path(str(target), run_name="__main__")
        except BaseException as exc:
            failure = exc
            record.update(error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
            traceback.print_exc()
common = sys.modules.get("ccsam_common")
record["test_guard"] = getattr(common, "_TEST_GUARD", None)
torch = sys.modules.get("torch")
if torch is not None:
    record["environment"] = dict(
        torch=torch.__version__, cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
        matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        default_dtype=str(torch.get_default_dtype()))
    if torch.cuda.is_initialized():
        record["environment"]["gpu"] = torch.cuda.get_device_name(0)
record.update(ended=datetime.now().astimezone().isoformat(),
              status="FAILED_STOPPED" if failure else "PASS", exit_code=1 if failure else 0)
with (OUT / (args.stage + "_execution.json")).open("x", encoding="utf-8") as stream:
    json.dump(record, stream, indent=2, ensure_ascii=False)
print(json.dumps({k: v for k, v in record.items() if k != "traceback"}, ensure_ascii=False))
if failure:
    print((OUT / (args.stage + ".log")).read_text(encoding="utf-8")[-5000:])
    sys.exit(1)
