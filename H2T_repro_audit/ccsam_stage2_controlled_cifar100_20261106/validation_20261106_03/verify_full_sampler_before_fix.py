"""Full CPU parity proof before changing production validation; no image/GPU access."""
import ast
import copy
import csv
import hashlib
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

OUT=Path(__file__).resolve().parent
ROOT=OUT.parent
sys.path.insert(0,str(ROOT))
import numpy as np
import ccsam_common as c

c.install_test_guard(c.DEFAULT_DATA_ROOT)
labels=np.load(ROOT/"prepared/train_labels.npy",allow_pickle=False)
source=json.loads((ROOT/"validation_20261106_01/official_sources.json").read_text())["sources"]
source=next(s for s in source if s["path"]=="datasets/Samplers.py")
content=source["content"].encode("utf-8")
blob=hashlib.sha1(b"blob "+str(len(content)).encode()+b"\0"+content).hexdigest()
assert blob==source["sha"]=="9711d4d60ac41da5d01b3b2214d3f07d4ba4ac23"
official={}
exec(compile(source["content"],"pinned_official_CC_SAM_Samplers.py","exec"),official)
original=ast.parse((ROOT/"ccsam_common.py").read_text())
function=next(n for n in original.body if isinstance(n,ast.FunctionDef) and n.name=="make_class_aware_plan")
candidate=copy.deepcopy(function)

class OmitKnownFalseAssertion(ast.NodeTransformer):
    removed=0
    def visit_If(self,node):
        if (len(node.body)==1 and isinstance(node.body[0],ast.Raise)
            and isinstance(node.body[0].exc,ast.Call)
            and node.body[0].exc.args
            and isinstance(node.body[0].exc.args[0],ast.Constant)
            and node.body[0].exc.args[0].value=="Class-aware sampler did not produce a balanced epoch"):
            self.removed+=1
            return ast.copy_location(ast.Pass(),node)
        return self.generic_visit(node)

transform=OmitKnownFalseAssertion()
candidate=transform.visit(candidate)
assert transform.removed==1
module=ast.fix_missing_locations(ast.Module(body=[candidate],type_ignores=[]))
namespace=dict(c.__dict__)
exec(compile(module,"diagnostic_only_original_generator_without_false_assertion.py","exec"),namespace)
local_function=namespace["make_class_aware_plan"]
refs=OUT/"reference_plans"
refs.mkdir(exist_ok=False)
rows=[];seeds=[]
for seed in (123,124,125):
    local=local_function(labels,seed,epochs=30)
    rng_state=random.getstate()
    try:
        random.seed(seed)
        sampler=official["ClassAwareSampler"](SimpleNamespace(labels=labels))
        reference=np.stack([np.asarray(list(iter(sampler)),dtype=np.int32) for _ in range(30)])
    finally:
        random.setstate(rng_state)
    assert local.shape==reference.shape==(30,10847)
    assert local.dtype==reference.dtype==np.int32
    assert np.array_equal(local,reference)
    for e in range(30):
        counts=np.bincount(labels[local[e]],minlength=100)
        rows.append(dict(seed=seed,epoch=e+1,samples=len(local[e]),mismatched_indices=int((local[e]!=reference[e]).sum()),
            count_min=int(counts.min()),count_max=int(counts.max()),count_spread=int(counts.max()-counts.min()),
            **{f"class_{i:03d}":int(v) for i,v in enumerate(counts)}))
    path=refs/f"seed{seed}.npy"
    with path.open("xb") as f:np.save(f,reference,allow_pickle=False)
    seeds.append(dict(seed=seed,epochs=30,indices_compared=int(local.size),mismatches=0,
        shape=list(local.shape),dtype=str(local.dtype),all_indices_equal=True,
        local_array_sha256=c.hash_arrays(local),official_array_sha256=c.hash_arrays(reference),
        reference_file_sha256=c.sha256_file(path)))
with (OUT/"FULL_SAMPLER_EPOCH_COUNTS_BEFORE.csv").open("x",encoding="utf-8",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
result=dict(status="PASS_EXACT_ALL_SEEDS_ALL_EPOCHS",seeds=seeds,total_indices_compared=sum(s["indices_compared"] for s in seeds),
    total_mismatches=0,official_source_git_blob=blob,official_source_sha256=hashlib.sha256(content).hexdigest(),
    official_source_url=source["display_url"],source=(ROOT/"ccsam_common.py").as_posix(),
    production_source_sha256=c.sha256_file(ROOT/"ccsam_common.py"),
    diagnostic_method="In-memory AST copy omits only the known false validation If/Raise; producer RNG, iterators and index construction untouched. Production file remains unchanged.",
    diagnostic_validation_statements_omitted=1,production_validation_modified=False,
    actual_epoch_count_ranges=sorted({(r["count_min"],r["count_max"]) for r in rows}),
    formal_training_started=False,sampling_plan_files_written_to_prepared=0,test_guard=c.guard_pass())
with (OUT/"FULL_SAMPLER_BEFORE_FIX.json").open("x",encoding="utf-8") as f:json.dump(result,f,indent=2)
print(json.dumps(result),flush=True)
