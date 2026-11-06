"""One prepare-completion or Smoke invocation in this fresh log directory; no formal mode."""
import argparse
import contextlib
import json
import runpy
import sys
import traceback
from datetime import datetime
from pathlib import Path

OUT=Path(__file__).resolve().parent
ROOT=OUT.parent
parser=argparse.ArgumentParser()
parser.add_argument("stage",choices=("prepare","smoke"))
args=parser.parse_args()
target=ROOT/("prepare_ccsam.py" if args.stage=="prepare" else "train_ccsam.py")
argv=[str(target)]+(["--complete-existing-preparation"] if args.stage=="prepare" else ["--smoke-only"])
record=dict(stage=args.stage,target_argv=argv,interpreter=sys.executable,
    started=datetime.now().astimezone().isoformat(),formal_training_started=False)
sys.path.insert(0,str(ROOT))
sys.argv=argv
failure=None
with (OUT/(args.stage+".log")).open("x",encoding="utf-8") as log:
    with contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
        try:
            runpy.run_path(str(target),run_name="__main__")
        except BaseException as exc:
            failure=exc
            record.update(error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc())
            traceback.print_exc()
common=sys.modules.get("ccsam_common")
record["test_guard"]=getattr(common,"_TEST_GUARD",None)
record.update(status="FAILED_STOPPED" if failure else "PASS",exit_code=1 if failure else 0,
    ended=datetime.now().astimezone().isoformat())
with (OUT/(args.stage+"_execution.json")).open("x",encoding="utf-8") as f:
    json.dump(record,f,indent=2,ensure_ascii=False,allow_nan=False)
print(json.dumps({k:v for k,v in record.items() if k!="traceback"},ensure_ascii=False))
if failure:
    print((OUT/(args.stage+".log")).read_text(encoding="utf-8")[-5000:])
    sys.exit(1)
