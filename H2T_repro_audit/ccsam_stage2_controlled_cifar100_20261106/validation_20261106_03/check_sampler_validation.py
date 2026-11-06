"""Verify production plans after the validator-only repair, against pre-fix sealed plans."""
import ast
import csv
import json
import random
import sys
from pathlib import Path
import numpy as np

OUT=Path(__file__).resolve().parent
ROOT=OUT.parent
sys.path.insert(0,str(ROOT))
import ccsam_common as c

c.install_test_guard(c.DEFAULT_DATA_ROOT)
labels=np.load(ROOT/"prepared/train_labels.npy",allow_pickle=False)
rows=[];results=[]
for seed in (123,124,125):
    state=random.getstate()
    plan,statistics=c.make_class_aware_plan(labels,seed,return_statistics=True)
    assert random.getstate()==state
    reference=np.load(OUT/"reference_plans"/f"seed{seed}.npy",allow_pickle=False)
    assert np.array_equal(plan,reference) and plan.dtype==reference.dtype==np.int32
    assert statistics["matches_official_sequence"] and statistics["mismatched_indices"]==0
    assert statistics["local_array_sha256"]==statistics["official_array_sha256"]==c.hash_arrays(reference)
    for e,counts in enumerate(statistics["per_epoch_class_counts"],1):
        rows.append(dict(seed=seed,epoch=e,mismatches=0,count_min=min(counts),count_max=max(counts),
            **{f"class_{i:03d}":n for i,n in enumerate(counts)}))
    results.append(dict(seed=seed,all_30_epochs_equal=True,indices_compared=int(plan.size),
        mismatches=0,array_sha256=c.hash_arrays(plan),ambient_python_rng_unchanged=True))
    if seed==123:
        reordered=plan.copy()
        reordered[0,0],reordered[0,1]=int(plan[0,1]),int(plan[0,0])
        assert not np.array_equal(reordered,plan)
        assert np.array_equal(np.bincount(labels[reordered[0]],minlength=100),np.bincount(labels[plan[0]],minlength=100))
        try:
            c.validate_class_aware_plan(labels,seed,reordered)
        except RuntimeError as exc:
            assert "differ from official CC-SAM" in str(exc)
            rejection=str(exc)
        else:
            raise AssertionError("Validator accepted reordered indices with identical class counts")

def functions(path):
    return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef)}
common_before=functions(OUT/"sources_before/ccsam_common.py")
common_after=functions(ROOT/"ccsam_common.py")
for name in ("class_weights_from_counts","alpha_for_epoch","augment_batch","build_models","clean_compatible_gap"):
    assert common_before[name]==common_after[name],name
assert (OUT/"sources_before/train_ccsam.py").read_bytes()==(ROOT/"train_ccsam.py").read_bytes()
assert (OUT/"sources_before/experiment_protocol.json").read_bytes()==(ROOT/"experiment_protocol.json").read_bytes()
with (OUT/"FULL_SAMPLER_EPOCH_COUNTS_AFTER.csv").open("x",encoding="utf-8",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
report=dict(status="PASS",seeds=results,total_indices_compared=sum(r["indices_compared"] for r in results),
    total_mismatches=0,exact_pre_fix_official_sequences_preserved=True,
    identical_counts_but_reordered_indices_rejected=rejection,
    class_weights_alpha_augment_models_GAP_AST_unchanged=True,
    train_script_bytes_unchanged_since_scope_approval=True,experiment_protocol_bytes_unchanged=True,
    test_guard=c.guard_pass(),formal_training_started=False)
with (OUT/"FULL_SAMPLER_AFTER_FIX.json").open("x",encoding="utf-8") as f:json.dump(report,f,indent=2)
print(json.dumps(report),flush=True)
