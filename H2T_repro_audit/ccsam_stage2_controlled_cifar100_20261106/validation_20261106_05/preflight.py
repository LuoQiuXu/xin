"""Snapshot and complete 3x30-epoch official sampling structure audit; no model forward."""
import ast
import collections
import inspect
import json
import pickle
import sys

from audit_support import *


def snapshot():
    assert_branch()
    assert not git("diff", "--name-only") and not git("diff", "--cached", "--name-only")
    paths = set(git("ls-files", "-z").split("\0")[:-1])
    paths.update(p.relative_to(REPO).as_posix() for p in ROOT.rglob("*")
                 if p.is_file() and OUT not in p.parents)
    prior = c.read_json(ROOT / "validation_20261106_04/PROTECTION_BEFORE.json")
    for name, expected in prior["protected_experiment_files"].items():
        assert record(REPO / name) == expected, name
    for name, expected in prior["protected_external_inputs"].items():
        assert record(Path(name)) == expected, name
    with (ROOT / "validation_20261106_04/ARTIFACT_SHA256.csv").open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            path=ROOT/"validation_20261106_04"/row["path"]
            assert c.sha256_file(path)==row["sha256"] and path.stat().st_size==int(row["bytes"])
    paths.update(Path(p).resolve().relative_to(REPO).as_posix() for p in prior["protected_external_inputs"])
    assets = json_read("PRE_SWITCH_UNTRACKED.json")
    assert len(assets) == 140
    for name, expected in assets.items():
        assert record(REPO / name) == expected, name
    paths.update(assets)
    protected = {name:record(REPO / name) for name in sorted(paths)
                 if name != (ROOT / "train_ccsam.py").relative_to(REPO).as_posix()}
    initial_env = environment()
    c.seed_everything(123, deterministic=True)
    active_env = environment()
    previous = c.read_json(ROOT / "validation_20261106_04/gradient_diagnostic.json")["environment"]
    for flag in ("torch", "cuda", "cudnn", "gpu", "cudnn_deterministic", "cudnn_benchmark",
                 "cudnn_allow_tf32", "matmul_allow_tf32"):
        assert active_env[flag] == previous[flag], (flag, active_env[flag], previous[flag])
    assert torch.get_default_dtype() == torch.float32 and not torch.is_autocast_enabled("cuda")
    text = inspect.getsource(production.analytic_classwise_feature_delta)
    oracle = inspect.getsource(production.autograd_reference_delta)
    assert "logits_clean.detach().softmax(dim=1)" in text
    assert "per_sample_gradient[mask] / float(class_count)" in text
    assert "torch.linalg.vector_norm(class_gradient)" in text
    assert "torch.autograd.grad(losses[mask].mean(), z, retain_graph=True)[0]" in oracle
    assert "torch.linalg.vector_norm(grad)" in oracle
    json_write("PROTECTION_BEFORE.json", dict(branch=BRANCH, head=HEAD,
        protected_files=protected, production_before=record(ROOT / "train_ccsam.py"),
        pre_switch_untracked_assets=assets, initial_environment=initial_env,
        effective_seeded_environment=active_env, original_smoke_report_exists=(ROOT / "smoke_report.json").exists(),
        original_results_exists=(ROOT / "results").exists(),
        numerical_order_note="Actual production multiplies ADVER_REG*weight*gradient before division by norm; preserve this order.",
        test_guard=c.guard_pass()))
    with (OUT / "GIT_STATUS_BEFORE.txt").open("x", encoding="utf-8", newline="\n") as stream:
        for args in (("branch","--show-current"),("rev-parse","HEAD"),
                     ("status","--short","--branch","--untracked-files=all"),
                     ("status","--short","--ignored"),("diff","--name-status"),
                     ("diff","--cached","--name-status")):
            stream.write("$ git " + " ".join(args) + "\n" + git(*args) + "\n")
    with (OUT / "PRODUCTION_PATH_BEFORE.txt").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write("A (exact original source):\n" + text + "\nB (unchanged oracle):\n" + oracle)
    print(json.dumps(dict(snapshot="PASS", protected_files=len(protected), assets=140, environment=active_env)), flush=True)


def structure():
    check_protection()
    manifest, images, labels = load_inputs()
    with (c.DEFAULT_DATA_ROOT / "cifar-100-python/train").open("rb") as stream:
        raw = pickle.load(stream, encoding="bytes")
    split = np.load(c.STABILITY_RUN / "split_indices.npz", allow_pickle=False)
    train_ids = split["train_raw_indices"]
    assert np.array_equal(np.asarray(raw[b"fine_labels"],dtype=np.int64)[train_ids], labels)
    pixels = raw[b"data"].reshape(-1,3,32,32).transpose(0,2,3,1)
    assert np.array_equal(pixels[train_ids], images)
    del raw, pixels
    assert labels.shape == (10847,) and np.all(np.bincount(labels,minlength=100)>0)
    fields = ["seed","epoch","batch","batch_size","unique_classes","max_multiplicity",
              "repeated_classes","absent_class_count","absent_class_ids","class_counts",
              "indices_sha256","labels_sha256","is_last_batch","labels_match_raw_train"]
    stream, writer = csv_open("BATCH_STRUCTURE_30EPOCH.csv", fields)
    patterns = collections.Counter();examples={};summary={};first_epoch={};rows=0
    try:
        for seed in c.SEEDS:
            path=ROOT/"prepared"/f"sampling_plan_seed{seed}.npy"
            assert c.sha256_file(path)==manifest["sampling_plans"][str(seed)]["sha256"]
            plan=np.load(path,mmap_mode="r",allow_pickle=False)
            official=c.validate_class_aware_plan(labels,seed,plan)
            assert official["indices_compared"]==325410 and official["mismatched_indices"]==0
            per_seed=collections.Counter();per_first=collections.Counter()
            for epoch,order in enumerate(plan):
                assert len(order)==10847 and np.all((order>=0)&(order<len(labels)))
                for batch,start in enumerate(range(0,len(order),64)):
                    indices=np.asarray(order[start:start+64],dtype=np.int64)
                    target=labels[indices];counts=np.bincount(target,minlength=100)
                    repeated=int((counts>1).sum());multiplicity=int(counts.max())
                    is_last=batch==169
                    assert len(indices)==(31 if is_last else 64)
                    assert np.array_equal(target,np.asarray(labels)[indices])
                    pattern=tuple(sorted(counts[counts>0].tolist(),reverse=True))
                    if repeated:
                        patterns[pattern]+=1
                        examples.setdefault(pattern,dict(seed=seed,epoch=epoch,batch=batch,
                            labels=target.tolist(),indices=indices.tolist()))
                    row=dict(seed=seed,epoch=epoch,batch=batch,batch_size=len(indices),
                        unique_classes=int((counts>0).sum()),max_multiplicity=multiplicity,
                        repeated_classes=repeated,absent_class_count=int((counts==0).sum()),
                        absent_class_ids=json.dumps(np.flatnonzero(counts==0).tolist()),
                        class_counts=json.dumps(counts.tolist()),indices_sha256=c.hash_arrays(indices),
                        labels_sha256=c.hash_arrays(target),is_last_batch=is_last,labels_match_raw_train=True)
                    writer.writerow(row);rows+=1;per_seed["batches"]+=1
                    per_seed["repeated_class_batches"]+=bool(repeated)
                    per_seed["last_batches"]+=is_last
                    per_seed["max_multiplicity"]=max(per_seed["max_multiplicity"],multiplicity)
                    if epoch==0:
                        per_first["batches"]+=1;per_first["sample_occurrences"]+=len(indices)
                        per_first["multi_sample_classes"]+=repeated
                        per_first["max_multiplicity"]=max(per_first["max_multiplicity"],multiplicity)
            assert per_seed["batches"]==5100 and per_seed["last_batches"]==30
            summary[str(seed)]=dict(per_seed,official_indices_compared=official["indices_compared"],
                official_mismatched_indices=0,plan_sha256=c.sha256_file(path))
            first_epoch[str(seed)]=dict(per_first)
    finally:
        stream.close()
    assert rows==15300
    pattern,count=patterns.most_common(1)[0]
    json_write("STRUCTURE_SUMMARY.json",dict(status="PASS",rows=rows,indices_compared=976230,
        label_and_pixel_parity_with_sealed_raw_TRAIN=True,global_empty_classes=[],out_of_range_indices=0,
        index_label_mismatches=0,seeds=summary,first_epoch=first_epoch,
        most_common_repeated_pattern=dict(multiplicities=list(pattern),occurrences=count,**examples[pattern]),
        pattern_frequencies=[dict(multiplicities=list(k),occurrences=v) for k,v in patterns.most_common()],
        test_guard=c.guard_pass(),protection=check_protection()))
    print(json.dumps(dict(structure="PASS",rows=rows,seeds=summary,first_epoch=first_epoch)),flush=True)


if __name__=="__main__":
    install_guard()
    {"snapshot":snapshot,"structure":structure}[sys.argv[1]]()
