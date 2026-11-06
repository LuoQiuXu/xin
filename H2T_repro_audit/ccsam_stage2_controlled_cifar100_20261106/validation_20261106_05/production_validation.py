"""Post-admission regression, compilation, and at most one original smoke invocation."""
import ast
import copy
import inspect
import py_compile
import sys

from audit_support import *


def admitted():
    matrix=json_read("CANDIDATE_MATRIX.json")
    assert matrix["admission_all_510_and_stress"] is True
    assert matrix["selected_candidate"]=="F"
    real=matrix["real_510_batches"]
    assert real["batches"]==510 and real["F_all_real_batches_passed"]
    assert real["aggregate"]["F"]["final_delta"]["violations"]==0
    assert all(case["F"]["passed"] and case["F"]["all_finite"] for case in matrix["stress"].values())
    return matrix


def source_scope():
    relative=(ROOT/"train_ccsam.py").relative_to(REPO).as_posix()
    original=git("show",HEAD+":"+relative)
    current=(ROOT/"train_ccsam.py").read_text(encoding="utf-8")
    old=ast.parse(original);new=ast.parse(current)
    old_fn=next(n for n in old.body if isinstance(n,ast.FunctionDef) and n.name=="analytic_classwise_feature_delta")
    new_fn=next(n for n in new.body if isinstance(n,ast.FunctionDef) and n.name=="analytic_classwise_feature_delta")
    assert ast.dump(old_fn)!=ast.dump(new_fn)
    restored=copy.deepcopy(new)
    for i,node in enumerate(restored.body):
        if isinstance(node,ast.FunctionDef) and node.name==old_fn.name:
            restored.body[i]=copy.deepcopy(old_fn)
    assert ast.dump(restored)==ast.dump(old),"Production edits extend beyond analytic_classwise_feature_delta"
    assert git("diff","--name-only").splitlines()==[relative]
    assert not git("diff","--cached","--name-only")
    return dict(only_analytic_function_changed=True,oracle_unchanged=True,all_other_production_AST_unchanged=True,
        before_sha256=json_read("PROTECTION_BEFORE.json")["production_before"]["sha256"],
        after_sha256=c.sha256_file(ROOT/"train_ccsam.py"),
        original_rtol_atol_unchanged=True)


def regression():
    admitted();check_protection(allow_production=True)
    scope=source_scope()
    capture=json_read("MEAN_BACKWARD_CAPTURE.json")
    paths=["REPLAY_BATCH0.npz",f"DUPLICATE_CAPTURE_seed{capture['seed']}_batch{capture['batch']}.npz","TAIL_seed123_batch169.npz"]
    manifest,_,_=load_inputs()
    c.seed_everything(123,deterministic=True)
    backbone,classifier,_=c.build_models(c.resolve_checkpoint(manifest["checkpoint_path"]),torch.device("cuda:0"))
    before_backbone=c.state_hash(backbone);before_classifier=c.state_hash(classifier)
    expected_stats={"num_classes_in_batch","zero_gradient_classes","mean_class_gradient_norm_before_normalization",
        "mean_delta_l2_per_sample","mean_relative_delta_l2","max_delta_l2_per_sample"}
    results=[]
    for name in paths:
        metadata=json_read(name+".json");assert c.sha256_file(OUT/name)==metadata["sha256"]
        captured=np.load(OUT/name,allow_pickle=False)
        gap=torch.from_numpy(captured["gap"].copy()).cuda()
        labels=torch.from_numpy(captured["labels"].copy()).cuda()
        weights=c.class_weights_from_counts(np.asarray(manifest["train_class_counts"]),torch.device("cuda:0"))
        head=classifier.module.fc
        assert np.array_equal(array(head.weight),captured["head_weight"])
        assert np.array_equal(array(head.bias),captured["head_bias"])
        assert np.array_equal(array(weights),captured["weights"])
        delta,stats=production.analytic_classwise_feature_delta(head(gap),gap,labels,weights,head.weight)
        reference=production.autograd_reference_delta(gap,labels,head,weights)
        assert torch.allclose(delta,reference,rtol=RTOL,atol=ATOL)
        assert np.array_equal(array(delta),captured["F_final_delta"]),"Production output must match admitted F candidate"
        assert expected_stats==set(stats)
        assert torch.isfinite(delta).all() and all(p.grad is None for p in classifier.parameters())
        err=np.abs(array(delta).astype(np.float64)-array(reference).astype(np.float64))
        allowed=ATOL+RTOL*np.abs(array(reference).astype(np.float64))
        assert not np.any(err>allowed)
        results.append(dict(source=name,source_sha256=metadata["sha256"],shape=list(gap.shape),dtype=str(gap.dtype),
            class_counts=np.bincount(array(labels),minlength=100).tolist(),passed=True,
            violations=0,max_abs=float(err.max()),max_normalized_error=float((err/allowed).max()),
            equals_admitted_F=True,stats_fields_compatible=True,stats=stats))
    assert c.state_hash(backbone)==before_backbone and c.state_hash(classifier)==before_classifier
    with (OUT/"PRODUCTION_DIFF.patch").open("x",encoding="utf-8",newline="\n") as stream:
        stream.write(git("diff","--",(ROOT/"train_ccsam.py").relative_to(REPO).as_posix()))
    json_write("PRODUCTION_REGRESSION.json",dict(status="PASS",cases=results,scope=scope,
        backbone_and_bn_bitwise_unchanged=True,classifier_bitwise_unchanged=True,optimizer_steps=0,
        test_guard=c.guard_pass(),protection=check_protection(allow_production=True)))
    print(json.dumps(dict(regression="PASS",real_cases=len(results),scope=scope)),flush=True)


def syntax():
    admitted();assert json_read("PRODUCTION_REGRESSION.json")["status"]=="PASS"
    cache=OUT/"syntax_cache";cache.mkdir(exist_ok=False)
    paths=sorted(ROOT.glob("*.py"))
    paths.extend(sorted(OUT.glob("*.py")))
    records=[]
    for i,path in enumerate(paths):
        py_compile.compile(str(path),cfile=str(cache/(str(i)+"_"+path.name+"c")),doraise=True)
        records.append(dict(path=path.relative_to(REPO).as_posix(),sha256=c.sha256_file(path),passed=True))
    json_write("SYNTAX_CHECK.json",dict(status="PASS",files=records,test_guard=c.guard_pass(),
        protection=check_protection(allow_production=True)))
    print(json.dumps(dict(py_compile="PASS",files=len(records))),flush=True)


def smoke():
    # Native main installs the original TEST guard before its prepared-array/model loads.
    admitted()
    assert json_read("PRODUCTION_REGRESSION.json")["status"]=="PASS"
    assert json_read("SYNTAX_CHECK.json")["status"]=="PASS"
    assert_branch();scope=source_scope()
    assert not (ROOT/"smoke_report.json").exists()
    assert not (OUT/"smoke_report.json").exists()
    argv=[str(ROOT/"train_ccsam.py"),"--smoke-only"]
    json_write("SMOKE_ATTEMPT.json",dict(attempt=1,started=now(),argv=argv,
        scope=scope,output_routing="Only native smoke_report.json is redirected to validation_20261106_05 with exclusive creation; payload is unchanged.",
        formal_training_started=False))
    def exclusive_smoke_report(path,payload):
        assert Path(path).resolve()==(ROOT/"smoke_report.json").resolve()
        assert payload["formal_training_started"] is False
        json_write("smoke_report.json",payload)
    production.write_json=exclusive_smoke_report
    sys.argv=argv
    production.main()
    report=json_read("smoke_report.json")
    assert report["passed"] and report["official_TEST_used"] is False
    assert set(report["seed_smoke"]["arms"])==set(c.ARMS)
    assert report["seed_smoke"]["paired_inputs_and_initialization_match"]
    assert all(arm["classifier_updated"] and arm["backbone_and_bn_bitwise_unchanged"]
               for arm in report["seed_smoke"]["arms"].values())
    assert report["analytic_gradient_equivalence_actual_batch"]["passed"]
    assert report["test_guard"]["official_test_file_open_attempts"]==0
    assert not (ROOT/"smoke_report.json").exists() and not (ROOT/"results").exists()
    json_write("POST_SMOKE_VERIFICATION.json",dict(status="PASS",original_smoke_main_invocations=1,
        disposable_classifier_optimizer_steps=3,checkpoint_written=False,formal_training_started=False,
        source_scope=source_scope(),test_guard=c.guard_pass(),protection=check_protection(allow_production=True)))


if __name__=="__main__":
    if sys.argv[1]!="smoke":install_guard()
    {"regression":regression,"syntax":syntax,"smoke":smoke}[sys.argv[1]]()
