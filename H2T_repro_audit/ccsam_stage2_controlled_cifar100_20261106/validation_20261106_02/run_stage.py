"""Fresh-output prepare/smoke runner. Never accepts formal training."""
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
argv=[str(target)]+(["--smoke-only"] if args.stage=="smoke" else [])
record=dict(stage=args.stage,target_argv=argv,interpreter=sys.executable,
    started=datetime.now().astimezone().isoformat(),formal_training_started=False,
    feature_path="ccsam_common.clean_compatible_gap via original entry point")
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
            tb=exc.__traceback__
            facts=[]
            while tb is not None:
                frame=tb.tb_frame
                if Path(frame.f_code.co_filename).resolve().is_relative_to(ROOT):
                    allowed=("baseline_metrics","expected_overall","train_counts","validation_counts","lt_pixel_hash","overlap","epoch","per_class","seed")
                    values={}
                    for key in allowed:
                        if key in frame.f_locals:
                            value=frame.f_locals[key]
                            values[key]=value.tolist() if hasattr(value,"tolist") else value
                    if values:facts.append(dict(file=frame.f_code.co_filename,function=frame.f_code.co_name,line=tb.tb_lineno,values=values))
                tb=tb.tb_next
            record["captured_failure_context"]=facts
            traceback.print_exc()
common=sys.modules.get("ccsam_common")
record["test_guard"]=getattr(common,"_TEST_GUARD",None)
record.update(status="FAILED_STOPPED" if failure else "PASS",exit_code=1 if failure else 0,
    ended=datetime.now().astimezone().isoformat())
with (OUT/(args.stage+"_execution.json")).open("x",encoding="utf-8") as f:
    json.dump(record,f,indent=2,ensure_ascii=False,allow_nan=False)
print(json.dumps({k:v for k,v in record.items() if k not in ("traceback","captured_failure_context")},ensure_ascii=False))
if failure:
    print((OUT/(args.stage+".log")).read_text(encoding="utf-8")[-4000:])
    sys.exit(1)
