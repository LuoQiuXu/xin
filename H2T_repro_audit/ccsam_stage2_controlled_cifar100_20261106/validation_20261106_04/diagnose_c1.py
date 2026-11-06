"""Read-only A/B/C/D/E comparison on the exact real C1 batch; no optimizer or Smoke."""
import csv
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

OUT=Path(__file__).resolve().parent
ROOT=OUT.parent
sys.path.insert(0,str(ROOT))
import ccsam_common as c
import train_ccsam as production

RTOL=1e-5
ATOL=1e-7
REL_EPS=1e-12
SEED=123
TENSORS={}
CSV_ROWS=[]

def save_json(name,value):
    with (OUT/name).open("x",encoding="utf-8",newline="\n") as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)

def array(tensor):
    return tensor.detach().cpu().contiguous().numpy().copy()

def retain(name,tensor):
    TENSORS[name]=array(tensor)

def stats(value,reference):
    a=np.asarray(value,dtype=np.float64)
    b=np.asarray(reference,dtype=np.float64)
    finite=bool(np.isfinite(a).all() and np.isfinite(b).all())
    base=dict(candidate_nan=int(np.isnan(a).sum()),candidate_inf=int(np.isinf(a).sum()),
        reference_nan=int(np.isnan(b).sum()),reference_inf=int(np.isinf(b).sum()),finite=finite)
    if not finite:
        return dict(base,max_abs=None,max_relative=None,relative_l2=None,cosine=None,
            violations=None,max_normalized_error=None,allclose_by_formula=False)
    error=np.abs(a-b)
    allowed=ATOL+RTOL*np.abs(b)
    norm_a=float(np.linalg.norm(a.ravel()))
    norm_b=float(np.linalg.norm(b.ravel()))
    cosine=float(np.dot(a.ravel(),b.ravel())/(norm_a*norm_b)) if norm_a and norm_b else None
    return dict(base,max_abs=float(error.max(initial=0)),
        max_relative=float((error/np.maximum(np.abs(b),REL_EPS)).max(initial=0)),
        relative_l2=float(np.linalg.norm((a-b).ravel())/max(norm_b,1e-30)),
        cosine=cosine,violations=int((error>allowed).sum()),
        max_normalized_error=float((error/allowed).max(initial=0)),
        allclose_by_formula=bool(np.all(error<=allowed)),
        elementwise_equal=bool(np.array_equal(np.asarray(value),np.asarray(reference))),
        byte_equal=bool(np.asarray(value).dtype==np.asarray(reference).dtype
            and np.asarray(value).shape==np.asarray(reference).shape
            and np.asarray(value).tobytes()==np.asarray(reference).tobytes()))

def error_location(value,reference,mode,labels,indices,class_ids,stage):
    a=np.asarray(value,dtype=np.float64);b=np.asarray(reference,dtype=np.float64)
    error=np.abs(a-b)
    metric={"absolute":error,"relative":error/np.maximum(np.abs(b),REL_EPS),
            "normalized":error/(ATOL+RTOL*np.abs(b))}[mode]
    pos=np.unravel_index(int(np.argmax(metric)),metric.shape)
    if stage=="class_norm":
        class_id=int(class_ids[pos[0]])
        batch_position=int(np.flatnonzero(labels==class_id)[0])
        feature_dim=None
    else:
        batch_position=int(pos[0]);class_id=int(labels[batch_position])
        feature_dim=int(pos[1]) if len(pos)>1 and stage!="probability" else None
    return dict(class_id=class_id,batch_position=batch_position,
        train_index=int(indices[batch_position]),feature_dimension=feature_dim,
        probability_class=int(pos[1]) if stage=="probability" else None,
        candidate=float(a[pos]),reference=float(b[pos]),absolute_error=float(error[pos]),
        relative_error=float(error[pos]/max(abs(b[pos]),REL_EPS)),
        allowed_error=float(ATOL+RTOL*abs(b[pos])),
        normalized_error=float(error[pos]/(ATOL+RTOL*abs(b[pos]))))

def saved_logsoftmax(losses,shape):
    queue=[losses.grad_fn];seen=set()
    while queue:
        node=queue.pop()
        if node is None or node in seen:
            continue
        seen.add(node)
        if "LogSoftmaxBackward" in type(node).__name__ and hasattr(node,"_saved_result"):
            result=node._saved_result.detach().clone()
            if tuple(result.shape)==tuple(shape):
                return result,type(node).__name__
        queue.extend(n for n,_ in node.next_functions)
    return None,None

def reference_layers(gap,labels,head,weights):
    z=gap.detach().clone().requires_grad_(True)
    logits=head(z)
    losses=F.cross_entropy(logits,labels,reduction="none")
    log_probability,node=saved_logsoftmax(losses,logits.shape)
    if log_probability is None:
        log_probability=F.log_softmax(logits.detach(),dim=1)
        probability_kind="explicit log_softmax proxy; internal saved output unavailable"
    else:
        probability_kind="exp(actual CrossEntropy-saved LogSoftmax output)"
    probability=log_probability.exp()
    q=torch.autograd.grad(losses.sum(),logits,retain_graph=True)[0].detach()
    raw=torch.autograd.grad(losses.sum(),z,retain_graph=True)[0].detach()
    class_ids=torch.unique(labels,sorted=True)
    class_gradient=torch.zeros_like(z)
    direction=torch.zeros_like(z)
    delta=torch.zeros_like(z)
    norms=[];full_gradients=[]
    for class_id in class_ids:
        mask=labels==class_id
        full=torch.autograd.grad(losses[mask].mean(),z,retain_graph=True)[0]
        norm=torch.linalg.vector_norm(full)
        assert int(torch.count_nonzero(full[~mask]).item())==0
        class_gradient[mask]=full[mask]
        if float(norm.detach().cpu())>1e-12:
            direction[mask]=full[mask]/norm
            delta[mask]=c.ADVER_REG*weights[class_id]*full[mask]/norm
        norms.append(norm.detach())
        full_gradients.append(full.detach())
    return dict(logits=logits.detach(),probability=probability.detach(),
        actual_logit_gradient=q,probability_recovered_from_backward=q+F.one_hot(labels,num_classes=100).to(q.dtype),
        saved_log_probability=log_probability,probability_kind=probability_kind,saved_node=node,
        raw_gradient=raw,class_gradient=class_gradient.detach(),class_norm=torch.stack(norms),
        direction=direction.detach(),delta=delta.detach(),full_class_gradients=torch.stack(full_gradients))

def analytic_layers(logits,gap,labels,head,weights,log_exp=False,full_norm=False):
    probability=(F.log_softmax(logits.detach(),dim=1).exp() if log_exp
                 else logits.detach().softmax(dim=1))
    q=probability-F.one_hot(labels,num_classes=100).to(probability.dtype)
    raw=q@head.weight.detach()
    class_gradient=torch.zeros_like(gap)
    direction=torch.zeros_like(gap)
    delta=torch.zeros_like(gap)
    norms=[];norm_inputs=[]
    for class_id in torch.unique(labels,sorted=True):
        mask=labels==class_id
        grad=raw[mask]/float(int(mask.sum().item()))
        class_gradient[mask]=grad
        if full_norm:
            padded=torch.zeros_like(gap)
            padded[mask]=grad
            norm=torch.linalg.vector_norm(padded)
            norm_inputs.append(padded)
        else:
            norm=torch.linalg.vector_norm(grad)
        if float(norm.detach().cpu())>1e-12:
            direction[mask]=grad/norm
            # Preserve the production multiplication-before-division operation order.
            delta[mask]=c.ADVER_REG*weights[class_id]*grad/norm
        norms.append(norm.detach())
    result=dict(probability=probability,raw_gradient=raw,class_gradient=class_gradient,
        class_norm=torch.stack(norms),direction=direction,delta=delta)
    if norm_inputs:
        result["full_norm_inputs"]=torch.stack(norm_inputs)
    return {k:v.detach() for k,v in result.items()}

class DoubleReadOnlyHead:
    def __init__(self,original):
        self.weight=original.weight.detach().to(torch.float64).clone().requires_grad_(True)
        self.bias=original.bias.detach().to(torch.float64).clone().requires_grad_(True)
    def __call__(self,x):
        return F.linear(x,self.weight,self.bias)

def run_precision(name,gap,labels,head,weights,indices):
    class_ids=array(torch.unique(labels,sorted=True))
    label_array=array(labels)
    with torch.amp.autocast("cuda",enabled=False):
        logits=head(gap.detach())
        actual_A,_=production.analytic_classwise_feature_delta(logits,gap,labels,weights,head.weight)
        actual_B=production.autograd_reference_delta(gap,labels,head,weights)
        B=reference_layers(gap,labels,head,weights)
        assert torch.equal(B["logits"],logits)
        assert torch.equal(B["delta"],actual_B),"Instrumented B differs from production reference"
        variants={"B":B}
        for variant,log_exp,padded in (("A",False,False),("C",True,False),("D",False,True),("E",True,True)):
            variants[variant]=analytic_layers(logits,gap,labels,head,weights,log_exp,padded)
        assert torch.equal(variants["A"]["delta"],actual_A),"Instrumented A differs from production A"
    retain(name+"_logits",logits)
    retain(name+"_head_weight",head.weight)
    retain(name+"_head_bias",head.bias)
    retain(name+"_class_weights",weights)
    retain(name+"_class_ids",torch.unique(labels,sorted=True))
    for variant,tensors in variants.items():
        for key,tensor in tensors.items():
            if isinstance(tensor,torch.Tensor):
                retain(name+"_"+variant+"_"+key,tensor)
    reference={k:array(v) for k,v in B.items() if isinstance(v,torch.Tensor)}
    summary={}
    stages=("probability","raw_gradient","class_gradient","class_norm","direction","delta")
    for variant in ("A","C","D","E"):
        values=variants[variant]
        stage_results={}
        for stage in stages:
            a=array(values[stage]);b=reference[stage]
            r=stats(a,b)
            r["torch_allclose"]=bool(torch.allclose(values[stage],B[stage],rtol=RTOL,atol=ATOL))
            r["max_absolute_location"]=error_location(a,b,"absolute",label_array,indices,class_ids,stage)
            r["max_normalized_location"]=error_location(a,b,"normalized",label_array,indices,class_ids,stage)
            stage_results[stage]=r
        summary[variant]=stage_results
        for j,class_id in enumerate(class_ids):
            mask=label_array==class_id
            row=dict(precision=name,variant=variant,class_id=int(class_id),batch_sample_count=int(mask.sum()),
                batch_positions=";".join(map(str,np.flatnonzero(mask).tolist())),
                train_indices=";".join(map(str,indices[mask].tolist())),
                class_weight=float(weights[int(class_id)].detach().cpu()),
                production_A_class_gradient_L2=float(variants["A"]["class_norm"][j].cpu()),
                analytic_class_gradient_L2=float(values["class_norm"][j].cpu()),
                autograd_class_gradient_L2=float(B["class_norm"][j].cpu()),
                raw_unaveraged_gradient_L2=float(np.linalg.norm(array(values["raw_gradient"])[mask].astype(np.float64))),
                autograd_raw_unaveraged_gradient_L2=float(np.linalg.norm(reference["raw_gradient"][mask].astype(np.float64))),
                norm_abs_error=abs(float(values["class_norm"][j].cpu())-float(B["class_norm"][j].cpu())))
            for stage in stages:
                a=array(values[stage])[j:j+1] if stage=="class_norm" else array(values[stage])[mask]
                b=reference[stage][j:j+1] if stage=="class_norm" else reference[stage][mask]
                for key,value in stats(a,b).items():
                    row[stage+"_"+key]=value
            row["any_nonfinite"]=not all(row[s+"_finite"] for s in stages)
            CSV_ROWS.append(row)
    probability_control=dict(
        softmax_vs_logsoftmax_exp=stats(array(variants["A"]["probability"]),array(variants["C"]["probability"])),
        explicit_logsoftmax_exp_vs_CE_saved_exp=stats(array(variants["C"]["probability"]),reference["probability"]),
        A_logit_gradient_vs_actual_CE_backward=stats(
            array(variants["A"]["probability"]-F.one_hot(labels,num_classes=100).to(gap.dtype)),
            reference["actual_logit_gradient"]),
        C_logit_gradient_vs_actual_CE_backward=stats(
            array(variants["C"]["probability"]-F.one_hot(labels,num_classes=100).to(gap.dtype)),
            reference["actual_logit_gradient"]),
        probability_reference_kind=B["probability_kind"],saved_autograd_node=B["saved_node"],
        probability_recovery_note="backward_gradient+one_hot retained separately; recovery can round target-class probabilities")
    replay=stats(array(actual_A),array(actual_B))
    replay["torch_allclose"]=bool(torch.allclose(actual_A,actual_B,rtol=RTOL,atol=ATOL))
    replay["production_A_equals_instrumented_A"]=True
    replay["production_B_equals_instrumented_B"]=True
    print(json.dumps(dict(phase=name,production_delta_comparison=replay,
        variants={k:{"delta_max_abs":v["delta"]["max_abs"],"delta_violations":v["delta"]["violations"],
            "delta_max_normalized_error":v["delta"]["max_normalized_error"],"torch_allclose":v["delta"]["torch_allclose"]}
            for k,v in summary.items()})),flush=True)
    return dict(production_replay=replay,probability_controls=probability_control,variants=summary,
        reference_norm_range=[float(B["class_norm"].min().cpu()),float(B["class_norm"].max().cpu())])

def main():
    started=datetime.now().astimezone().isoformat()
    c.install_test_guard(c.DEFAULT_DATA_ROOT)
    manifest=c.read_json(ROOT/"prepared/prepared_manifest.json")
    assert manifest["clean_stage1_sha256"]==c.EXPECTED_CLEAN_SHA256
    for name in ("train_images.npy","train_labels.npy"):
        assert c.sha256_file(ROOT/"prepared"/name)==manifest["prepared_array_actions"][name]["sha256"]
    plan_path=ROOT/"prepared/sampling_plan_seed123.npy"
    assert c.sha256_file(plan_path)==manifest["sampling_plans"]["123"]["sha256"]
    plan=np.load(plan_path,mmap_mode="r",allow_pickle=False)
    indices=np.asarray(plan[0,:64],dtype=np.int64)
    raw_images=np.load(ROOT/"prepared/train_images.npy",mmap_mode="r",allow_pickle=False)
    raw_labels=np.load(ROOT/"prepared/train_labels.npy",allow_pickle=False)
    labels=torch.from_numpy(np.asarray(raw_labels[indices],dtype=np.int64).copy()).cuda()
    weights=c.class_weights_from_counts(np.asarray(manifest["train_class_counts"]),torch.device("cuda:0"))
    c.seed_everything(SEED,deterministic=True)
    checkpoint=c.resolve_checkpoint(manifest["checkpoint_path"])
    backbone,classifier,metadata=c.build_models(checkpoint,torch.device("cuda:0"))
    frozen_backbone={k:v.detach().cpu().clone() for k,v in backbone.state_dict().items()}
    before_backbone=c.state_hash(backbone);before_classifier=c.state_hash(classifier)
    train_transform,_,transform_sha=c.build_transforms()
    assert transform_sha==manifest["transform_source_sha256"]
    generator=torch.Generator(device="cpu").manual_seed(SEED+300009)
    images=c.augment_batch(raw_images,indices,train_transform,generator)
    with torch.no_grad(),torch.amp.autocast("cuda",enabled=False):
        feature_map=backbone(images.cuda())
        gap=c.clean_compatible_gap(feature_map)
    assert gap.shape==(64,64) and gap.dtype==torch.float32 and torch.isfinite(gap).all()
    # Reproduce the original C1 forward before its verification; this performs no backward/update.
    read_only_loss,_,_,_,_,_,alpha=production.compute_arm_loss(
        "C1_CCSAM_FEATURE_GRAD",gap,labels,classifier.module.fc,weights,epoch_index=0)
    c1_forward_loss=float(read_only_loss.detach().cpu())
    del read_only_loss
    hashes=dict(indices_sha256=c.hash_arrays(indices),labels_sha256=c.hash_arrays(array(labels)),
        indices_labels_sha256=c.hash_arrays(indices,array(labels)),
        augmented_images_sha256=c.hash_arrays(array(images)),
        smoke_compatible_augmented_images_sha256=c.hash_arrays(indices,array(labels),array(images)),
        gap_features_sha256=c.hash_arrays(array(gap)),classifier_state_sha256=before_classifier,
        feature_map_sha256=c.hash_arrays(array(feature_map)))
    batch=dict(seed=SEED,batch_size=64,augmentation_seed=SEED+300009,hashes=hashes,
        feature_map_shape=list(feature_map.shape),gap_shape=list(gap.shape),
        gap_dtype=str(gap.dtype),class_counts={str(int(k)):int(v) for k,v in zip(*np.unique(array(labels),return_counts=True))},
        checkpoint_path=str(checkpoint),checkpoint_sha256=c.sha256_file(checkpoint),
        checkpoint_seed=metadata["seed"],checkpoint_epoch=metadata["epoch"],
        alpha=alpha,ADVER_REG=c.ADVER_REG,rtol=RTOL,atol=ATOL,readonly_C1_forward_loss=c1_forward_loss,
        historical_hashes_note="Previous failed Smoke did not persist its batch hashes; these hashes are newly recorded from the exact locked reconstruction.")
    save_json("BATCH_RECONSTRUCTION.json",batch)
    print(json.dumps(dict(phase="batch_reconstructed",batch=batch)),flush=True)
    TENSORS.update(indices=indices.copy(),labels=array(labels),augmented_images_fp32=array(images),
        feature_map_fp32=array(feature_map),gap_fp32=array(gap))
    results={}
    results["fp32"]=run_precision("fp32",gap,labels,classifier.module.fc,weights,indices)
    head64=DoubleReadOnlyHead(classifier.module.fc)
    gap64=gap.detach().to(torch.float64)
    weights64=weights.detach().to(torch.float64)
    retain("gap_fp64",gap64)
    results["fp64"]=run_precision("fp64",gap64,labels,head64,weights64,indices)
    cross={}
    for variant in ("A","B","C","D","E"):
        cross[variant]={stage:stats(TENSORS["fp32_"+variant+"_"+stage],
                                    TENSORS["fp64_B_"+stage])
            for stage in ("raw_gradient","class_norm","direction","delta")}
    c.check_backbone_unchanged(backbone,frozen_backbone)
    assert c.state_hash(backbone)==before_backbone and c.state_hash(classifier)==before_classifier
    assert all(p.grad is None for p in classifier.parameters())
    assert head64.weight.grad is None and head64.bias.grad is None
    assert c.sha256_file(checkpoint)==c.EXPECTED_CLEAN_SHA256
    baseline=json.loads((OUT/"PROTECTION_BEFORE.json").read_text())
    changed=[]
    for name,value in baseline["protected_experiment_files"].items():
        path=c.REPO_ROOT/name
        if not path.is_file() or c.sha256_file(path)!=value["sha256"]:
            changed.append(name)
    for name,value in baseline["protected_external_inputs"].items():
        if c.sha256_file(Path(name))!=value["sha256"]:
            changed.append(name)
    assert not changed,changed
    with (OUT/"per_class_gradient_errors.csv").open("x",encoding="utf-8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(CSV_ROWS[0]))
        writer.writeheader();writer.writerows(CSV_ROWS)
    with (OUT/"tensor_comparison.npz").open("xb") as stream:
        np.savez_compressed(stream,**TENSORS)
    payload=dict(status="DIAGNOSTIC_COMPLETED",started=started,ended=datetime.now().astimezone().isoformat(),
        batch=batch,results=results,fp32_variants_vs_fp64_autograd_reference=cross,
        historical_failure_exact_max_abs_reproduced=(
            results["fp32"]["production_replay"]["max_abs"]==5.364418029785156e-7
            and not results["fp32"]["production_replay"]["torch_allclose"]),
        relative_error_denominator="max(abs(reference),1e-12)",
        normalized_error_definition="abs(candidate-reference)/(1e-7+1e-5*abs(reference))",
        relative_l2_denominator="max(L2(reference),1e-30)",
        probability_reference_note="Actual saved CrossEntropy LogSoftmax output exponentiated; actual CE logit-gradient also retained",
        fp64_policy="Only the exact captured FP32 GAP, head weights/bias and class weights are cast to FP64; no FP64 backbone forward or weight recomputation",
        original_production_dtype=str(classifier.module.fc.weight.dtype),
        production_default_dtype=str(torch.get_default_dtype()),
        backbone_and_bn_bitwise_unchanged=True,classifier_bitwise_unchanged=True,
        classifier_parameter_grad_buffers_remain_none=True,optimizer_instances_created=0,optimizer_steps=0,
        smoke_entrypoint_invocations=0,formal_training_started=False,
        protected_experiment_files_verified=len(baseline["protected_experiment_files"]),
        protected_external_inputs_verified=len(baseline["protected_external_inputs"]),changed_protected_files=[],
        test_guard=c.guard_pass(),
        environment=dict(python=sys.version,executable=sys.executable,torch=torch.__version__,
            cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),gpu=torch.cuda.get_device_name(0),
            gpu_device_count=torch.cuda.device_count(),cudnn_deterministic=torch.backends.cudnn.deterministic,
            cudnn_benchmark=torch.backends.cudnn.benchmark,cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
            matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,autocast=False))
    save_json("gradient_diagnostic.json",payload)
    print(json.dumps(dict(status=payload["status"],historical_failure_reproduced=payload["historical_failure_exact_max_abs_reproduced"],
        output_tensors=len(TENSORS),per_class_rows=len(CSV_ROWS),optimizer_steps=0,test_guard=payload["test_guard"])),flush=True)

if __name__=="__main__":
    main()
