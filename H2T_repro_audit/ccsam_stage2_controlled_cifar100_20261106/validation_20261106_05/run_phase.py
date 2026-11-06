"""Execute a named audit phase once, recording complete stdout, stderr and exceptions."""
import contextlib
import json
import runpy
import sys
import traceback
from datetime import datetime
from pathlib import Path

OUT=Path(__file__).resolve().parent
phase=sys.argv[1]
allowed={"snapshot":("preflight.py","snapshot"),"snapshot_pathfix":("preflight.py","snapshot"),
         "structure":("preflight.py","structure"),
         "sweep":("numerical_audit.py","sweep"),"stress":("numerical_audit.py","stress"),
         "regression":("production_validation.py","regression"),
         "smoke":("production_validation.py","smoke"),"syntax":("production_validation.py","syntax")}
target,arg=allowed[phase]
record=dict(phase=phase,interpreter=sys.executable,argv=[str(OUT/target),arg],
            started=datetime.now().astimezone().isoformat(),formal_training_started=False)
assert not (OUT/(phase+"_execution.json")).exists()
failure=None
with (OUT/(phase+".log")).open("x",encoding="utf-8",newline="\n") as log:
    with contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
        try:
            sys.argv=record["argv"]
            runpy.run_path(str(OUT/target),run_name="__main__")
        except BaseException as exc:
            failure=exc
            record.update(error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc())
            traceback.print_exc()
    common=sys.modules.get("ccsam_common")
    record["test_guard"]=getattr(common,"_TEST_GUARD",None)
    record.update(status="FAILED_STOPPED" if failure else "PASS",exit_code=1 if failure else 0,
                  ended=datetime.now().astimezone().isoformat())
with (OUT/(phase+"_execution.json")).open("x",encoding="utf-8",newline="\n") as stream:
    json.dump(record,stream,indent=2,ensure_ascii=False,allow_nan=False)
print(json.dumps({k:v for k,v in record.items() if k!="traceback"},ensure_ascii=False))
if failure:
    print((OUT/(phase+".log")).read_text(encoding="utf-8")[-5000:])
    sys.exit(1)
