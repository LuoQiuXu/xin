"""FP32, read-only candidate sweep and deterministic duplicate-label stress tests."""
import collections
import json
import sys

import torch.nn.functional as functional

from audit_support import *

CANDIDATES = {
    "A": dict(probability="softmax", count="post_division", norm="compact"),
    "B": dict(probability="actual_CE_saved_logsoftmax_exp", count="actual_mean_backward", norm="full", oracle=True),
    "C": dict(probability="logsoftmax_exp", count="post_division", norm="compact"),
    "D": dict(probability="softmax", count="post_division", norm="full"),
    "E": dict(probability="softmax", count="pre_division", norm="compact"),
    "F": dict(probability="logsoftmax_exp", count="pre_division", norm="full"),
    "G": dict(probability="logsoftmax_exp", count="pre_division", norm="full", conditional=True),
    "H": dict(probability="logsoftmax_exp", count="pre_reciprocal_multiply", norm="full", conditional=True),
}
ACTIVE = ("A", "B", "C", "D", "E", "F")
STAGES = ("probability", "logits_gradient", "raw_feature_gradient", "class_full_feature_gradient",
          "class_norm", "normalized_direction", "final_delta")


def saved_logsoftmax(loss):
    queue=[loss.grad_fn];seen=set()
    while queue:
        node=queue.pop()
        if node is None or node in seen:
            continue
        seen.add(node)
        if "LogSoftmaxBackward" in type(node).__name__:
            return node._saved_result.detach().clone()
        queue.extend(n for n,_ in node.next_functions)
    raise RuntimeError("Cannot inspect actual CrossEntropy LogSoftmax saved result")


def reference_layers(gap,labels,head,weights):
    # B remains the unique oracle: direct call, with independent instrumentation checked exactly.
    oracle=production.autograd_reference_delta(gap,labels,head,weights)
    z=gap.detach().clone().requires_grad_(True)
    logits=head(z)
    losses=functional.cross_entropy(logits,labels,reduction="none")
    logp=saved_logsoftmax(losses)
    qraw,raw=torch.autograd.grad(losses.sum(),(logits,z),retain_graph=True)
    classes=torch.unique(labels,sorted=True)
    qs=torch.zeros_like(logits);direction=torch.zeros_like(z);delta=torch.zeros_like(z)
    fulls=[];qfulls=[];norms=[]
    for cls in classes:
        mask=labels==cls
        # Capture actual mean backward at logits, rather than inferring 1/n algebraically.
        q,g=torch.autograd.grad(losses[mask].mean(),(logits,z),retain_graph=True)
        assert torch.count_nonzero(g[~mask]).item()==0
        assert torch.count_nonzero(q[~mask]).item()==0
        norm=torch.linalg.vector_norm(g)
        qs[mask]=q[mask]
        if norm.item()>1e-12:
            direction[mask]=g[mask]/norm
            delta[mask]=c.ADVER_REG*weights[cls]*g[mask]/norm
        fulls.append(g.detach());qfulls.append(q.detach());norms.append(norm.detach())
    assert torch.equal(delta,oracle),"Instrumented B must equal unchanged production oracle"
    return dict(probability=logp.exp(),logits_gradient=qs.detach(),raw_feature_gradient=raw.detach(),
        class_full_feature_gradient=torch.stack(fulls),class_norm=torch.stack(norms),
        normalized_direction=direction.detach(),final_delta=oracle,
        actual_unscaled_logits_gradient=qraw.detach(),actual_full_logits_gradients=torch.stack(qfulls),
        logits=logits.detach(),saved_log_probability=logp)


def analytic_layers(tag,logits,gap,labels,head,weights):
    spec=CANDIDATES[tag]
    p=(logits.detach().softmax(dim=1) if spec["probability"]=="softmax"
       else functional.log_softmax(logits.detach(),dim=1).exp())
    q=p-functional.one_hot(labels,num_classes=c.NUM_CLASSES).to(p.dtype)
    counts=torch.bincount(labels,minlength=c.NUM_CLASSES)
    ns=counts[labels].to(q.dtype).unsqueeze(1)
    scaled=(q*ns.reciprocal() if spec["count"]=="pre_reciprocal_multiply" else q/ns)
    raw=q@head.weight.detach()
    averaged=None if spec["count"]=="post_division" else scaled@head.weight.detach()
    classes=torch.unique(labels,sorted=True)
    direction=torch.zeros_like(gap);delta=torch.zeros_like(gap);fulls=[];norms=[]
    for cls in classes:
        mask=labels==cls
        if spec["count"]=="post_division":
            compact=raw[mask]/float(int(mask.sum().item()))
            full=torch.zeros_like(gap)
            full[mask]=compact
        else:
            full=averaged*mask.unsqueeze(1).to(averaged.dtype)
            compact=averaged[mask]
        norm=torch.linalg.vector_norm(full if spec["norm"]=="full" else compact)
        if norm.item()>1e-12:
            direction[mask]=compact/norm
            delta[mask]=c.ADVER_REG*weights[cls]*compact/norm
        fulls.append(full);norms.append(norm)
    return dict(probability=p,logits_gradient=scaled,raw_feature_gradient=raw,
        class_full_feature_gradient=torch.stack(fulls),class_norm=torch.stack(norms),
        normalized_direction=direction,final_delta=delta)


def location(a,b,metric,stage,labels,indices,raw_ids,classes):
    flat=int(np.argmax(metric));pos=np.unravel_index(flat,metric.shape)
    if stage=="class_norm":
        cls=int(classes[pos[0]]);row=int(np.flatnonzero(labels==cls)[0]);dim=None
    elif stage=="class_full_feature_gradient":
        cls=int(classes[pos[0]]);row=int(pos[1]);dim=int(pos[2])
        if float(metric.max())==0:
            row=int(np.flatnonzero(labels==cls)[0]);pos=(pos[0],row,pos[2])
    else:
        row=int(pos[0]);cls=int(labels[row]);dim=int(pos[1]) if a.shape[-1]==64 else None
    av=float(a[pos]);bv=float(b[pos]);err=abs(av-bv);allowed=ATOL+RTOL*abs(bv)
    return dict(class_id=cls,batch_position=row,train_index=int(indices[row]) if indices is not None else None,
        raw_cifar_train_index=int(raw_ids[row]) if raw_ids is not None else None,
        feature_dimension=dim,logit_class=int(pos[-1]) if stage in ("probability","logits_gradient") else None,
        candidate=av,reference=bv,error=err,allowed_error=allowed,normalized_error=err/allowed)


def comparison(value,reference,stage,labels,indices,raw_ids,classes):
    a=array(value);b=array(reference);ad=a.astype(np.float64);bd=b.astype(np.float64)
    finite=bool(np.isfinite(ad).all() and np.isfinite(bd).all())
    assert finite,"Nonfinite stage tensors; stop and preserve execution log"
    error=np.abs(ad-bd);allowed=ATOL+RTOL*np.abs(bd);normalized=error/allowed
    return dict(finite=finite,candidate_nan=int(np.isnan(ad).sum()),candidate_inf=int(np.isinf(ad).sum()),
        reference_nan=int(np.isnan(bd).sum()),reference_inf=int(np.isinf(bd).sum()),
        max_abs=float(error.max()),max_relative=float((error/np.maximum(np.abs(bd),1e-12)).max()),
        max_normalized_error=float(normalized.max()),violations=int((error>allowed).sum()),
        torch_allclose=bool(torch.allclose(value,reference,rtol=RTOL,atol=ATOL)),
        byte_equal=bool(a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()),
        max_absolute_location=location(ad,bd,error,stage,labels,indices,raw_ids,classes),
        max_normalized_location=location(ad,bd,normalized,stage,labels,indices,raw_ids,classes))


def update_aggregate(aggregate,key,metrics,context):
    for stage,result in metrics.items():
        entry=aggregate.setdefault(key,{}).setdefault(stage,dict(comparisons=0,failed_comparisons=0,
            violations=0,max_abs=-1.0,max_relative=0.0,max_normalized_error=-1.0,
            all_finite=True,all_byte_equal=True))
        entry["comparisons"]+=1;entry["failed_comparisons"]+=not result["torch_allclose"]
        entry["violations"]+=result["violations"]
        entry["all_finite"] &= result["finite"];entry["all_byte_equal"] &= result["byte_equal"]
        entry["max_relative"]=max(entry["max_relative"],result["max_relative"])
        if result["max_abs"]>entry["max_abs"]:
            entry["max_abs"]=result["max_abs"]
            entry["worst_absolute"]=dict(**context,**result["max_absolute_location"])
        if result["max_normalized_error"]>entry["max_normalized_error"]:
            entry["max_normalized_error"]=result["max_normalized_error"]
            entry["worst_normalized"]=dict(**context,**result["max_normalized_location"])


WORST_FIELDS=["phase","seed","batch","case","candidate","stage","finite","candidate_nan","candidate_inf",
    "reference_nan","reference_inf","max_abs","max_relative","max_normalized_error","violations",
    "torch_allclose","byte_equal","max_absolute_location","max_normalized_location"]
NORM_FIELDS=["phase","seed","batch","case","candidate","class_id","n_c","class_weight","class_norm","reference_class_norm"]


def evaluate_batch(gap,labels,head,weights,context,indices,raw_ids,worst_writer,norm_writer,aggregate,
                   group_aggregate,tags=ACTIVE):
    with torch.amp.autocast("cuda",enabled=False):
        logits=head(gap.detach())
        B=reference_layers(gap,labels,head,weights)
        assert torch.equal(logits,B["logits"])
        values={"B":B}
        for tag in tags:
            if tag!="B":values[tag]=analytic_layers(tag,logits,gap,labels,head,weights)
        if "A" in values:
            actual_A,_=production.analytic_classwise_feature_delta(logits,gap,labels,weights,head.weight)
            assert torch.equal(actual_A,values["A"]["final_delta"]),"A instrumentation differs from production"
    target=array(labels);classes=array(torch.unique(labels,sorted=True));counts=np.bincount(target,minlength=100)
    results={}
    for tag in tags:
        stages={stage:comparison(values[tag][stage],B[stage],stage,target,indices,raw_ids,classes) for stage in STAGES}
        results[tag]=stages
        update_aggregate(aggregate,tag,stages,context)
        for stage,result in stages.items():
            row=dict(context,candidate=tag,stage=stage,**result)
            row["max_absolute_location"]=json.dumps(row["max_absolute_location"],separators=(",",":"))
            row["max_normalized_location"]=json.dumps(row["max_normalized_location"],separators=(",",":"))
            worst_writer.writerow(row)
        ns=array(values[tag]["class_norm"]);nr=array(B["class_norm"])
        for j,cls in enumerate(classes):
            norm_writer.writerow(dict(context,candidate=tag,class_id=int(cls),n_c=int(counts[cls]),
                class_weight=float(weights[int(cls)].item()),class_norm=float(ns[j]),reference_class_norm=float(nr[j])))
        for group in ("1","2","3","4+"):
            mask=(counts[target]>=4 if group=="4+" else counts[target]==int(group))
            if not mask.any():continue
            a=values[tag]["final_delta"][torch.from_numpy(mask).to(labels.device)]
            b=B["final_delta"][torch.from_numpy(mask).to(labels.device)]
            gr=comparison(a,b,"final_delta",target[mask],indices[mask] if indices is not None else None,
                          raw_ids[mask] if raw_ids is not None else None,np.unique(target[mask]))
            # Map sliced local positions back into the original batch for auditable worst locations.
            positions=np.flatnonzero(mask)
            for location_name in ("max_absolute_location","max_normalized_location"):
                gr[location_name]["batch_position"]=int(positions[gr[location_name]["batch_position"]])
            update_aggregate(group_aggregate,tag+"_nc_"+group,{"final_delta":gr},context)
    assert all(p.grad is None for p in head.parameters())
    return results,values


def packed(values,extra):
    output=dict(extra)
    for tag,stages in values.items():
        for stage,value in stages.items():output[tag+"_"+stage]=value
    return output


def sweep():
    check_protection()
    structure=json_read("STRUCTURE_SUMMARY.json");assert structure["status"]=="PASS"
    manifest,images,raw_labels=load_inputs()
    split=np.load(c.STABILITY_RUN/"split_indices.npz",allow_pickle=False)
    train_raw_ids=split["train_raw_indices"]
    checkpoint=c.resolve_checkpoint(manifest["checkpoint_path"])
    transform,_,transform_sha=c.build_transforms();assert transform_sha==manifest["transform_source_sha256"]
    baseline=json_read("PROTECTION_BEFORE.json")["effective_seeded_environment"]
    fields=["seed","epoch","batch","batch_size","unique_classes","multi_sample_classes","max_multiplicity",
        "indices_sha256","labels_sha256","augmented_images_sha256","gap_features_sha256","classifier_state_sha256",
        "augmentation_rng_before_sha256","augmentation_rng_after_sha256","augmentation_rng_continuous","last_batch"]
    for tag in ACTIVE:
        fields.extend(tag+"_"+k for k in ("passed","violations","max_abs","max_normalized_error","byte_equal","all_finite"))
    batch_stream,batch_writer=csv_open("REAL_BATCH_SWEEP.csv",fields)
    worst_stream,worst_writer=csv_open("WORST_CASES_REAL.csv",WORST_FIELDS)
    norm_stream,norm_writer=csv_open("CLASS_NORMS_REAL.csv",NORM_FIELDS)
    aggregate={};groups={};seed_results={};batch0_replay=None;captured_duplicate=False;first_f_failure=False
    try:
        for seed in c.SEEDS:
            # Exactly one seed_everything and one augmentation generator initialization per actual seed.
            c.seed_everything(seed,deterministic=True)
            env=environment()
            for key in ("torch","cuda","cudnn","gpu","cudnn_deterministic","cudnn_benchmark","cudnn_allow_tf32","matmul_allow_tf32"):
                assert env[key]==baseline[key],key
            backbone,classifier,_=c.build_models(checkpoint,torch.device("cuda:0"))
            before_backbone=c.state_hash(backbone);before_head=c.state_hash(classifier)
            frozen={k:v.detach().cpu().clone() for k,v in backbone.state_dict().items()}
            head=classifier.module.fc
            weights=c.class_weights_from_counts(np.asarray(manifest["train_class_counts"]),torch.device("cuda:0"))
            plan=np.load(ROOT/"prepared"/f"sampling_plan_seed{seed}.npy",mmap_mode="r",allow_pickle=False)
            generator=torch.Generator(device="cpu").manual_seed(seed+300009)
            previous_rng=None;cache=[];all_indices=[];all_labels=[];total=0;last={};repeated_classes=0
            for batch,start in enumerate(range(0,len(raw_labels),64)):
                indices=np.asarray(plan[0,start:start+64],dtype=np.int64)
                labels=torch.from_numpy(np.asarray(raw_labels[indices],dtype=np.int64).copy()).cuda()
                rng_before=c.hash_arrays(generator.get_state().numpy())
                assert previous_rng is None or rng_before==previous_rng
                augmented=c.augment_batch(images,indices,transform,generator)
                rng_after=c.hash_arrays(generator.get_state().numpy());previous_rng=rng_after
                with torch.no_grad(),torch.amp.autocast("cuda",enabled=False):
                    feature_map=backbone(augmented.cuda());gap=c.clean_compatible_gap(feature_map)
                assert gap.dtype==torch.float32 and torch.isfinite(gap).all()
                target=array(labels);counts=np.bincount(target,minlength=100)
                row=dict(seed=seed,epoch=0,batch=batch,batch_size=len(indices),unique_classes=int((counts>0).sum()),
                    multi_sample_classes=int((counts>1).sum()),max_multiplicity=int(counts.max()),
                    indices_sha256=c.hash_arrays(indices),labels_sha256=c.hash_arrays(target),
                    augmented_images_sha256=c.hash_arrays(array(augmented)),gap_features_sha256=c.hash_arrays(array(gap)),
                    classifier_state_sha256=before_head,augmentation_rng_before_sha256=rng_before,
                    augmentation_rng_after_sha256=rng_after,augmentation_rng_continuous=True,last_batch=batch==169)
                if seed==123 and batch==0:
                    expected=c.read_json(ROOT/"validation_20261106_04/BATCH_RECONSTRUCTION.json")["hashes"]
                    checked={k:dict(actual=row[k],expected=expected[k],matches=row[k]==expected[k]) for k in
                        ("indices_sha256","labels_sha256","augmented_images_sha256","gap_features_sha256","classifier_state_sha256")}
                    json_write("BATCH0_REPLAY_HASHES.json",checked)
                    assert all(v["matches"] for v in checked.values()),"Exact historic batch replay hash mismatch; STOP"
                    batch0_replay=checked
                context=dict(phase="real",seed=seed,batch=batch,case="")
                results,values=evaluate_batch(gap,labels,head,weights,context,indices,train_raw_ids[indices],
                    worst_writer,norm_writer,aggregate,groups)
                for tag,stages in results.items():
                    d=stages["final_delta"]
                    for name,value in dict(passed=d["torch_allclose"] and d["violations"]==0,
                        violations=d["violations"],max_abs=d["max_abs"],max_normalized_error=d["max_normalized_error"],
                        byte_equal=d["byte_equal"],all_finite=all(x["finite"] for x in stages.values())).items():row[tag+"_"+name]=value
                batch_writer.writerow(row);batch_stream.flush()
                extra=dict(gap=gap,indices=indices,labels=labels,head_weight=head.weight,head_bias=head.bias,weights=weights)
                if seed==123 and batch==0:
                    extra["augmented_images"]=augmented
                    save_tensors("REPLAY_BATCH0.npz",packed(values,extra))
                    assert results["A"]["final_delta"]["max_abs"]==5.364418029785156e-7
                if not captured_duplicate and counts.max()>1:
                    save_tensors(f"DUPLICATE_CAPTURE_seed{seed}_batch{batch}.npz",packed(values,extra))
                    json_write("MEAN_BACKWARD_CAPTURE.json",dict(seed=seed,batch=batch,counts=counts.tolist(),
                        actual_mean_backward_logits_captured=True,
                        full_logits_gradient_shape=list(values["B"]["actual_full_logits_gradients"].shape),
                        F_logits_gradient_comparison=results["F"]["logits_gradient"],
                        F_class_feature_gradient_comparison=results["F"]["class_full_feature_gradient"],
                        A_class_feature_gradient_comparison=results["A"]["class_full_feature_gradient"],
                        probability_reference="Actual CE saved LogSoftmax result; no proxy",
                        stage_note="For post-division A/C/D, logits_gradient is the implied q/n diagnostic; their actual feature scaling remains after GEMM."))
                    captured_duplicate=True
                if batch==169:
                    last={tag:row[tag+"_passed"] for tag in ACTIVE}
                    if seed==123:save_tensors("TAIL_seed123_batch169.npz",packed(values,extra))
                if not row["F_passed"] and not first_f_failure:
                    save_tensors(f"FIRST_F_FAILURE_seed{seed}_batch{batch}.npz",packed(values,extra))
                    json_write("FIRST_F_FAILURE.json",dict(context,stages=results["F"]))
                    first_f_failure=True
                cache.append(array(gap));all_indices.append(indices.copy());all_labels.append(target)
                total+=len(indices);repeated_classes+=int((counts>1).sum())
                if batch%10==0 or batch==169:
                    print(json.dumps(dict(seed=seed,batch=batch,batch_size=len(indices),samples_processed=total,
                        candidate_F_passed=row["F_passed"],candidate_F_violations=row["F_violations"],
                        candidate_F_max_abs=row["F_max_abs"])),flush=True)
            assert total==10847 and batch+1==170
            c.check_backbone_unchanged(backbone,frozen)
            assert c.state_hash(backbone)==before_backbone and c.state_hash(classifier)==before_head
            assert all(p.grad is None for p in classifier.parameters())
            save_tensors(f"FEATURE_CACHE_seed{seed}.npz",dict(gap=np.concatenate(cache),indices=np.concatenate(all_indices),
                labels=np.concatenate(all_labels),head_weight=head.weight,head_bias=head.bias,weights=weights))
            seed_results[str(seed)]=dict(batches=170,sample_occurrences=total,unique_train_indices=int(np.unique(np.concatenate(all_indices)).size),
                augmentation_generator_initialized_once=True,augmentation_seed=seed+300009,seed_everything_calls=1,
                multi_sample_classes=repeated_classes,last_batch_size=31,last_batch_passed=last,
                backbone_and_bn_bitwise_unchanged=True,classifier_bitwise_unchanged=True,
                initial_backbone_sha256=before_backbone,initial_classifier_sha256=before_head,classifier_grad_buffers_none=True)
            del backbone,classifier,head
    finally:
        batch_stream.close();worst_stream.close();norm_stream.close()
    json_write("REAL_SWEEP_SUMMARY.json",dict(status="COMPLETED",batches=510,candidate_definitions=CANDIDATES,
        active_candidates=list(ACTIVE),aggregate=aggregate,by_multiplicity=groups,seeds=seed_results,
        batch0_exact_hash_replayed=bool(batch0_replay),actual_mean_backward_capture=captured_duplicate,
        F_all_real_batches_passed=(aggregate["F"]["final_delta"]["failed_comparisons"]==0 and aggregate["F"]["final_delta"]["violations"]==0),
        candidate_G_H="Not executed unless F fails; G repeats F division, H changes division to reciprocal multiplication only",
        production_modified=False,optimizer_steps=0,formal_training_started=False,
        test_guard=c.guard_pass(),protection=check_protection()))


def stress():
    check_protection()
    structure=json_read("STRUCTURE_SUMMARY.json")
    cases={
        "unique_64":list(range(64)),
        "one_duplicate_2":[0,0]+list(range(1,63)),
        "several_duplicate_2":[0,0,1,1,2,2,3,3]+list(range(4,60)),
        "32_classes_x2":np.repeat(np.arange(32),2).tolist(),
        "16_classes_x4":np.repeat(np.arange(16),4).tolist(),
        "single_class_x64":[0]*64,
        "observed_modal_duplicate":structure["most_common_repeated_pattern"]["labels"],
        "tail_31":list(range(31)),
        "one_triplicate_3":[0]*3+list(range(1,62)),
        "nonpower_counts_5_7":[0]*5+[1]*7+list(range(2,54)),
    }
    stream,writer=csv_open("SYNTHETIC_DUPLICATE_STRESS.csv",["case","batch_size","candidate","max_multiplicity",
        "unique_classes","passed","violations","max_abs","max_normalized_error","all_finite","byte_equal"])
    ws,ww=csv_open("WORST_CASES_STRESS.csv",WORST_FIELDS)
    ns,nw=csv_open("CLASS_NORMS_STRESS.csv",NORM_FIELDS)
    aggregate={};groups={};case_results={}
    try:
        for case_index,(case,target) in enumerate(cases.items()):
            c.seed_everything(2026110600+case_index,deterministic=True)
            head=torch.nn.Linear(64,100).cuda()
            gap=torch.randn(len(target),64,device="cuda",dtype=torch.float32)
            labels=torch.tensor(target,device="cuda",dtype=torch.long)
            weights=c.class_weights_from_counts(np.linspace(5,500,100,dtype=np.float64),torch.device("cuda:0"))
            before=c.state_hash(head)
            context=dict(phase="synthetic",seed=2026110600+case_index,batch=0,case=case)
            results,values=evaluate_batch(gap,labels,head,weights,context,None,None,ww,nw,aggregate,groups)
            case_results[case]={}
            for tag,stages in results.items():
                d=stages["final_delta"]
                row=dict(case=case,batch_size=len(target),candidate=tag,max_multiplicity=max(collections.Counter(target).values()),
                    unique_classes=len(set(target)),passed=d["torch_allclose"] and d["violations"]==0,
                    violations=d["violations"],max_abs=d["max_abs"],max_normalized_error=d["max_normalized_error"],
                    all_finite=all(s["finite"] for s in stages.values()),byte_equal=d["byte_equal"])
                writer.writerow(row);case_results[case][tag]=row
            save_tensors("STRESS_"+case+".npz",packed(values,dict(gap=gap,labels=labels,head_weight=head.weight,head_bias=head.bias,weights=weights)))
            assert c.state_hash(head)==before and all(p.grad is None for p in head.parameters())
            print(json.dumps(dict(case=case,F=case_results[case]["F"])),flush=True)
    finally:
        stream.close();ws.close();ns.close()
    real=json_read("REAL_SWEEP_SUMMARY.json")
    stress_pass=all(case["F"]["passed"] and case["F"]["all_finite"] for case in case_results.values())
    admitted=real["F_all_real_batches_passed"] and stress_pass
    json_write("STRESS_SUMMARY.json",dict(status="COMPLETED",cases=case_results,aggregate=aggregate,
        by_multiplicity=groups,F_all_stress_cases_passed=stress_pass,actual_TRAIN_data_used=False,
        synthetic_seed_policy="Independent deterministic per-case seed; no effect on real sweep process",
        test_guard=c.guard_pass(),protection=check_protection()))
    matrix=dict(candidate_definitions=CANDIDATES,rtol=RTOL,atol=ATOL,production_dtype="FP32",AMP=False,
        real_510_batches=real,stress=case_results,stress_stage_aggregate=aggregate,
        stress_by_multiplicity=groups,admission_all_510_and_stress=admitted,
        selected_candidate="F" if admitted else None,
        G_H_status="NOT_RUN_F_PASSED_ALL_REQUIRED_CASES" if admitted else "REQUIRES_FIRST_FAILURE_LOCALIZATION_BEFORE_G_H",
        naming_note="Round05 E is count-pre-scaling only, unlike round04 E which combined probability and full norm.",
        logit_gradient_stage_note="For post-division A/C/D, the saved scaled logit gradient is an implied diagnostic; actual feature gradient scaling is after GEMM.",
        protection=check_protection(),test_guard=c.guard_pass(),production_code_modified=False)
    json_write("CANDIDATE_MATRIX.json",matrix)
    print(json.dumps(dict(real_batches=510,stress_cases=len(cases),candidate_F_admitted=admitted)),flush=True)


if __name__=="__main__":
    install_guard()
    {"sweep":sweep,"stress":stress}[sys.argv[1]]()
