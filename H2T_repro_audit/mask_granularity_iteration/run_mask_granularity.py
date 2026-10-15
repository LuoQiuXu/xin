"""One-question iteration: Original-like BatchShared vs paired PerSample vs LowVS.

Run from anywhere: python run_mask_granularity.py preflight
                   python run_mask_granularity.py train --seed 124
                   python run_mask_granularity.py train --seed 125
                   python run_mask_granularity.py report
Requires the *locally complete* October 14 run, including omitted large files.
All historical paths are read-only. No Git operations. No official TEST access.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import os
import torchvision

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / "clean_vs_stability_runs" / "clean_s1_seed42_s2_124_125_20261014"
if not (OLD / "common.py").is_file():
    raise RuntimeError("Expected previous clean stability experiment adjacent to this folder")
sys.path.insert(0, str(OLD))
import numpy as np
import torch
from common import (N, BATCH, seed_all, array_hash, state_hash, sha, metrics, evaluate,
                    check_frozen)
from stability_models import (frozen_models, outdir, plan_dir, PAIR_KEYS, check_sources)
from reused_train_helpers import augment, fuse, tensor_hash, scheduler, optimizer_for
from online_policy import online_masks, selected_statistics
from mask_policy import choose, self_test

_TEST_EVENTS = []
def install_local_test_guard():
    forbidden = os.path.normcase(os.path.abspath(OLD.parent.parent.parent / "H2T/data/cifar100/cifar-100-python/test"))
    def audit(event, args):
        if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
            checked = os.path.normcase(os.path.abspath(os.fsdecode(args[0])))
            if checked == forbidden:
                _TEST_EVENTS.append(checked)
                raise RuntimeError("Official CIFAR100 TEST file access prohibited")
    sys.addaudithook(audit)
    previous = torchvision.datasets.CIFAR100.__init__
    def protected(self, root, train=True, *args, **kwargs):
        if not train:
            _TEST_EVENTS.append("CIFAR100(train=False)")
            raise RuntimeError("Official CIFAR100 TEST loading prohibited")
        return previous(self, root, train, *args, **kwargs)
    torchvision.datasets.CIFAR100.__init__ = protected

def guard_check():
    assert not _TEST_EVENTS, _TEST_EVENTS

def csv_write(path, rows):
    assert rows and path.resolve().is_relative_to(HERE), "Outputs must remain under new experiment directory"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

ARM = "Uniform-BatchShared"
SEEDS = (124, 125)
RESULTS = HERE / "results"
# Deterministically use first receiver's Uniform Gumbel mask for all samples in batch.
# This is batch-sharing semantics only, NOT torch.randperm RNG parity with official H2T.
PROTOCOL = {
    "purpose": "Separate mask granularity from VS filtering",
    "existing_controls": ["Uniform-PerSample", "Low-VS"],
    "new_arm": ARM,
    "batch_shared_rule": "tile Uniform-PerSample mask of first item of each actual minibatch",
    "nominal_original_H2T_4D_semantics": "one mask per minibatch",
    "not_equivalent_to_official_sampling_rng": True,
    "stage1_seed": 42, "stage2_seeds": list(SEEDS),
    "epoch_primary": 10, "epochs": 10, "rho": 0.25, "batch_size": 128,
    "backbone_bn": "frozen eval", "classifier": "SGD only",
    "dataset": "clean CIFAR100-LT train-only 99-class holdout, class0 missing; 100 logits",
    "test_access": False, "no_stage1_retraining": True,
    "run_only_new_arm": True,
}


def ensure_inputs(seed):
    assert seed in SEEDS
    needed = [
        OLD/"source_lock.json", OLD/"locked/clean_stage1_best.pth",
        OLD/"class_feature_statistics.npz", OLD/"train_images.npy",
        OLD/"train_labels.npy", OLD/"validation_labels.npy",
        OLD/"validation_feature_maps.npy", OLD/"split_indices.npz",
        OLD/"stage2_original_effective.yaml",
        OLD/"plans"/str(seed)/"sampling_plan.npz",
        OLD/"plans"/str(seed)/"gumbel_noise_float64.npy",
        outdir(seed, "Uniform-PerSample")/"batches.csv",
        outdir(seed, "Uniform-PerSample")/"epoch_validation_metrics.csv",
        outdir(seed, "Low-VS")/"epoch_validation_metrics.csv",
        outdir(seed, "Uniform-PerSample")/"validation_epoch10.npz",
        outdir(seed, "Low-VS")/"validation_epoch10.npz",
    ]
    missing = [str(p) for p in needed if not p.is_file()]
    if missing:
        raise FileNotFoundError("Local complete previous run is required; missing:\n" + "\n".join(missing))
    check_sources()
    return needed


def control_rows(seed):
    p = outdir(seed, "Uniform-PerSample")/"batches.csv"
    with p.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 850, len(rows)
    return rows


def preflight():
    from preflight_checks import preflight as strict_preflight
    return strict_preflight(sys.modules[__name__])


def train(seed):
    import time
    from PIL import Image
    from preflight_checks import dump,history_manifest
    ensure_inputs(seed)
    baseline=json.loads((RESULTS/'historical_readonly_baseline.json').read_text(encoding='utf-8'))
    assert history_manifest(OLD,sha)==baseline
    output=RESULTS/f'shared_seed{seed}'
    if output.exists():raise FileExistsError(f'Refusing to overwrite previous run: {output}')
    cfg,tr,bb,cls,CE,methods,s1,q=frozen_models(seed)
    init_bb,init_cls=state_hash(bb),state_hash(cls)
    reference_state=json.loads((outdir(seed,'Uniform-PerSample')/'status.json').read_text(encoding='utf-8'))['initial_state_hashes']
    assert {'backbone':init_bb,'classifier':init_cls}==reference_state
    output.mkdir(parents=True);(output/'checkpoints').mkdir()
    cfg.name=f'mask_granularity_shared_seed{seed}';cfg.model_dir=str(output/'checkpoints');cfg.shift_bn=False
    (output/'effective.yaml').write_text(cfg.dump(),encoding='utf-8')
    dump(output/'protocol.json',{**PROTOCOL,'seed':seed,'historical_hash_gate_override':True,
        'full_13_fields_recorded':True,'initial_state_matches_control':True,
        'user_authorization':'2026-10-15: continue despite preflight mismatch; fixed requested environment',
        'source_hashes':{name:sha(HERE/name) for name in ['run_mask_granularity.py','mask_policy.py','preflight_checks.py']},
        'Clean_StageI_sha256':sha(s1),'VS_statistics_sha256':sha(OLD/'class_feature_statistics.npz')})
    reference=control_rows(seed)
    frozen={k:v.detach().cpu().clone() for k,v in bb.state_dict().items()}
    initial_fc={k:v.detach().cpu().clone() for k,v in cls.state_dict().items()}
    optimizer=optimizer_for(cfg,cls);assert not optimizer.state
    labels=np.load(OLD/'train_labels.npy',allow_pickle=False)
    val_labels=np.load(OLD/'validation_labels.npy',allow_pickle=False)
    val_bank=np.load(OLD/'validation_feature_maps.npy',mmap_mode='r',allow_pickle=False)
    plan=np.load(plan_dir(seed)/'sampling_plan.npz',allow_pickle=False)
    noise=np.load(plan_dir(seed)/'gumbel_noise_float64.npy',mmap_mode='r',allow_pickle=False)
    original_masks=np.load(outdir(seed,'Uniform-PerSample')/'actual_masks.npz',allow_pickle=False)['masks']
    aug=torch.Generator().manual_seed(seed+300009)
    criterion=CE(cfg,np.bincount(labels,minlength=100)).cuda();lr_schedule=scheduler()
    images=np.load(OLD/'train_images.npy',mmap_mode='r',allow_pickle=False)[:BATCH]
    with torch.no_grad():
        warm=torch.stack([tr['eval_transform'](Image.fromarray(v)) for v in images]).cuda();bb(warm);bb(warm)
    torch.cuda.synchronize()
    records=[];masks=np.empty((10,N,16),dtype=np.uint8)
    composition={k:np.empty((10,N),dtype=np.float64) for k in ['V','B','ratio','score','common_with_uniform','fallback','GAP_L2','feature_map_L2']}
    equality={key:0 for key in PAIR_KEYS};mismatch_rows=[];batch_rows=[];best=-1.;counter=0
    initial_time=time.perf_counter();best_epoch=None
    log=(output/'training.log').open('w',encoding='utf-8',buffering=1)
    def message(text):log.write(text+'\n');print(text,flush=True)
    dump(output/'status.json',{'status':'running','seed':seed,'epochs_completed':0,'updates':0})
    for epoch in range(10):
        cls.train();lr_schedule(optimizer,epoch,cfg);torch.cuda.reset_peak_memory_stats()
        start_epoch=time.perf_counter();weighted_loss=0.
        for batch,start in enumerate(range(0,N,BATCH)):
            end=min(N,start+BATCH)
            rec,don=plan['receivers'][epoch,start:end],plan['donors'][epoch,start:end]
            target=torch.from_numpy(labels[rec].copy()).cuda()
            ri,di=augment(rec,don,aug,tr['train_transform'])
            with torch.no_grad():receiver,donor=bb(ri.cuda()),bb(di.cuda())
            assert receiver.shape==(end-start,64,8,8) and not receiver.requires_grad
            g=noise[epoch,start:end];trace=online_masks(receiver,labels[don],g)
            row={'epoch':epoch+1,'batch':batch,'size':end-start,
                'indices_sha256':array_hash(rec,don,labels[rec],labels[don]),
                'images_sha256':array_hash(ri.numpy(),di.numpy()),
                'features_sha256':array_hash(receiver.cpu().numpy(),donor.cpu().numpy()),
                'augmentation_state_after_sha256':tensor_hash(aug.get_state()),
                'ambient_cpu_rng_sha256':tensor_hash(torch.get_rng_state()),
                'ambient_cuda_rng_sha256':tensor_hash(torch.cuda.get_rng_state()),
                'gumbel_sha256':array_hash(g),'online_gap_sha256':trace['GAP_sha256'],
                'online_scores_sha256':trace['score_sha256'],'all_two_masks_sha256':trace['all_two_masks_sha256']}
            changed=[]
            for key in PAIR_KEYS:
                expected=reference[counter][key];actual=str(row[key]);matches=actual==expected
                equality[key]+=int(matches)
                if not matches:
                    changed.append(key);mismatch_rows.append({'seed':seed,'epoch':epoch+1,'batch':batch,'field':key,'expected':expected,'actual':actual})
            chosen=choose(ARM,trace)
            assert np.array_equal(chosen,np.broadcast_to(original_masks[epoch,start],chosen.shape)), 'Shared masks must equal sealed Uniform first mask'
            with torch.no_grad():
                mixed,mask=fuse(receiver,donor,chosen)
                full=mask[:,:,None,None].expand_as(mixed)
                assert torch.equal(mixed[full],donor[full]) and torch.equal(mixed[~full],receiver[~full])
                delta=mixed-receiver
                gap_delta=torch.nn.functional.avg_pool2d(delta,8).flatten(1)
                composition['GAP_L2'][epoch,start:end]=gap_delta.norm(dim=1).cpu().numpy()
                composition['feature_map_L2'][epoch,start:end]=delta.flatten(1).norm(dim=1).cpu().numpy()
            assert np.array_equal(target.cpu().numpy(),labels[rec]) and torch.isfinite(mixed).all()
            masks[epoch,start:end]=chosen
            selected=selected_statistics(trace,chosen)
            for key,value in selected.items():composition[key][epoch,start:end]=value
            logits=cls(mixed);loss=criterion(logits,target,extra_info=None)
            assert torch.isfinite(logits).all() and torch.isfinite(loss)
            optimizer.zero_grad();loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in cls.parameters())
            optimizer.step();weighted_loss+=float(loss.detach())*(end-start);counter+=1
            row.update({'loss':float(loss.detach()),'lr':optimizer.param_groups[0]['lr'],
                'shared_mask_sha256':array_hash(chosen),'mismatching_fields':';'.join(changed)})
            batch_rows.append(row)
            if batch%40==0:message(f'Shared seed={seed} epoch={epoch+1}/10 batch={batch}/85 loss={float(loss.detach()):.6f}; differing control fields={len(changed)}')
        torch.cuda.synchronize();train_seconds=time.perf_counter()-start_epoch
        check_frozen(bb,frozen)
        fc_change={k:float((v.detach().cpu()-initial_fc[k]).abs().max()) for k,v in cls.state_dict().items()}
        assert all(v>0 for v in fc_change.values())
        val,logits,preds=evaluate(cls,val_labels,feature_maps=val_bank)
        improved=val['Overall']>best
        if improved:best=val['Overall'];best_epoch=epoch+1
        checkpoint={'epoch':epoch+1,'state_dict_model':frozen,
            'state_dict_classifier':{k:v.detach().cpu().clone() for k,v in cls.state_dict().items()},
            'optimizer':optimizer.state_dict(),'validation':val,'arm':ARM,'seed':seed,'TEST_used':False,
            'Clean_StageI_sha256':sha(s1),'VS_statistics_sha256':sha(OLD/'class_feature_statistics.npz'),
            'split_sha256':sha(OLD/'split_indices.npz'),'strict_historical_feature_parity':False}
        if improved:torch.save(checkpoint,output/'checkpoints/model_best.pth.tar')
        if epoch==9:torch.save(checkpoint,output/'checkpoints/model_final.pth.tar')
        np.savez_compressed(output/f'validation_epoch{epoch+1:02d}.npz',logits=logits,predictions=preds,labels=val_labels)
        records.append({'arm':ARM,'seed':seed,'epoch':epoch+1,'train_loss':weighted_loss/N,
            'lr':optimizer.param_groups[0]['lr'],**{k:val[k] for k in ['Overall','Head','OfficialMedium','Tail','Many','Medium','Few','Macro','validation_CE']},
            'training_seconds':train_seconds,'peak_allocated_mib':torch.cuda.max_memory_allocated()/1024**2,
            'peak_reserved_mib':torch.cuda.max_memory_reserved()/1024**2})
        csv_write(output/'epoch_validation_metrics.csv',records);csv_write(output/'batches.csv',batch_rows)
        if mismatch_rows:csv_write(output/'control_mismatches.csv',mismatch_rows)
        dump(output/'status.json',{'status':'running','seed':seed,'epochs_completed':epoch+1,'updates':counter,
            'equal_fields':equality,'classifier_max_abs_changes':fc_change,'backbone_BN_bitwise_unchanged':True})
        guard_check()
        message(f'Shared seed={seed} epoch={epoch+1}/10 Overall={val["Overall"]:.4f} Head={val["Head"]:.4f} Medium={val["OfficialMedium"]:.4f} Tail={val["Tail"]:.4f} Few={val["Few"]:.4f}; {train_seconds:.2f}s; BB/BN frozen')
    assert counter==850
    np.savez_compressed(output/'actual_masks.npz',masks=masks)
    np.savez_compressed(output/'sampling_composition.npz',**composition)
    reloads={}
    for kind in ['best','final']:
        ck=torch.load(output/f'checkpoints/model_{kind}.pth.tar',map_location='cpu',weights_only=False)
        assert all(v.numpy().tobytes()==frozen[k].numpy().tobytes() for k,v in ck['state_dict_model'].items())
        cls.load_state_dict(ck['state_dict_classifier'],strict=True)
        loaded,_,_=evaluate(cls,val_labels,feature_maps=val_bank)
        assert loaded==ck['validation']
        reloads[kind]={'epoch':ck['epoch'],'metrics':loaded,'sha256':sha(output/f'checkpoints/model_{kind}.pth.tar'),
            'validation_reload_exact':True,'all_backbone_BN_state_entries_bitwise_unchanged':len(frozen)}
    check_frozen(bb,frozen);assert history_manifest(OLD,sha)==baseline
    summary={'status':'complete','seed':seed,'epoch10':records[-1],'best':max(records,key=lambda x:x['Overall']),
        'best_epoch':best_epoch,'initial_backbone_hash':init_bb,'initial_classifier_hash':init_cls,
        'paired_control_fields_verified':equality,'compared_batches':850,'all_13_fields_exact':all(x==850 for x in equality.values()),
        'historical_mismatch_authorized':True,'mismatching_field_records':len(mismatch_rows),
        'stage1_frozen':state_hash(bb)==init_bb,'classifier_changed':fc_change,'TEST_used':False,
        'updates':850,'epochs':10,'run_seconds':time.perf_counter()-initial_time,
        'checkpoints':reloads,'all_shared_masks_equal_sealed_Uniform_first_masks':True}
    dump(output/'summary.json',summary);dump(output/'status.json',{'status':'complete','seed':seed,'epochs_completed':10,'updates':850})
    log.close();message_text=f'TRAIN COMPLETE seed={seed}:850 updates,10epochs; historical 13-field equality={equality}'
    print(message_text,flush=True)



def csv_records(path):
    with path.open(encoding="utf-8",newline="") as f:
        return list(csv.DictReader(f))


def report_completed():
    for seed in SEEDS:
        ensure_inputs(seed)
        assert (RESULTS/f"shared_seed{seed}/summary.json").is_file(), f"train seed{seed} first"
    rows=[];flip_rows=[]
    for seed in SEEDS:
        paths={
            "Uniform-BatchShared":RESULTS/f"shared_seed{seed}",
            "Uniform-PerSample":outdir(seed,"Uniform-PerSample"),
            "Low-VS":outdir(seed,"Low-VS"),
        }
        metrics_by_arm={}
        pred_by_arm={}
        labels=None
        for arm,folder in paths.items():
            epochs=csv_records(folder/"epoch_validation_metrics.csv")
            at10=next(r for r in epochs if int(r["epoch"])==10)
            best=max(epochs,key=lambda r:float(r["Overall"]))
            metrics_by_arm[arm]={k:float(at10[k]) for k in
                                 ("Overall","Head","OfficialMedium","Tail","Many","Medium","Few","Macro")}
            bundle=np.load(folder/"validation_epoch10.npz",allow_pickle=False)
            y=bundle["labels"];p=bundle["predictions"]
            if labels is None: labels=y
            else: assert np.array_equal(labels,y)
            pred_by_arm[arm]=p
            rows.append({"seed":seed,"arm":arm,**metrics_by_arm[arm],
                         "best_epoch":int(best["epoch"]),"best_Overall":float(best["Overall"])})
        for arm in ("Uniform-PerSample","Low-VS"):
            a=pred_by_arm["Uniform-BatchShared"]
            b=pred_by_arm[arm]
            flip_rows.append({
                "seed":seed,"comparison":f"{arm} - Uniform-BatchShared",
                "gain_wrong_to_right":int(((a!=labels)&(b==labels)).sum()),
                "loss_right_to_wrong":int(((a==labels)&(b!=labels)).sum()),
                "changed_prediction":int((a!=b).sum()),
                "delta_Overall_pp":metrics_by_arm[arm]["Overall"]-metrics_by_arm["Uniform-BatchShared"]["Overall"],
                "delta_Tail_pp":metrics_by_arm[arm]["Tail"]-metrics_by_arm["Uniform-BatchShared"]["Tail"],
                "delta_Few_pp":metrics_by_arm[arm]["Few"]-metrics_by_arm["Uniform-BatchShared"]["Few"],
            })
    csv_write(RESULTS/"all_three_epoch10.csv",rows)
    csv_write(RESULTS/"paired_differences.csv",flip_rows)
    print("Report written:",RESULTS/"all_three_epoch10.csv")
    print("Paired comparisons:",RESULTS/"paired_differences.csv")
    for row in flip_rows:print(json.dumps(row,ensure_ascii=False))



def report():
    if all((RESULTS/f"shared_seed{s}/summary.json").is_file() for s in SEEDS):
        from scientific_report import report as full_report
        return full_report(sys.modules[__name__])
    from blocked_report import report as report_preflight_failure
    return report_preflight_failure(sys.modules[__name__])

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("action",choices=("preflight","train","report"))
    parser.add_argument("--seed",type=int,choices=SEEDS)
    args=parser.parse_args()
    if args.action=="train" and args.seed is None:
        parser.error("--seed 124 or --seed 125 required")
    # All status/exception/protection logs are scoped to the new results directory.
    from datetime import datetime, timezone, timedelta
    import traceback
    from preflight_checks import dump, protect_history
    global CURRENT
    stamp=datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%dT%H%M%S_%f")
    CURRENT=RESULTS/"execution"/(args.action+("_"+str(args.seed) if args.seed else "")+"_"+stamp)
    CURRENT.mkdir(parents=True,exist_ok=False)
    history_write_events=[]
    install_local_test_guard()
    protect_history(OLD,history_write_events)
    dump(CURRENT/"status.json",{"state":"running","action":args.action,"seed":args.seed})
    try:
        if args.action=="preflight":preflight()
        elif args.action=="train":
            gate=RESULTS/"continuation_authorization.json"
            if not gate.is_file():raise RuntimeError("Explicit user continuation authorization required")
            authorization=json.loads(gate.read_text(encoding="utf-8"))
            assert authorization["authorized_seeds"]==[124,125]
            assert sys.executable.casefold()==authorization["python"].casefold()
            assert torch.__version__==authorization["torch"] and torch.version.cuda==authorization["cuda"]
            print("USER-AUTHORIZED CONTINUATION: all 13 control fields recorded; exact historical hash gate waived",flush=True)
            train(args.seed)
        else:report()
        guard_check()
        assert not history_write_events
        dump(CURRENT/"status.json",{"state":"completed","action":args.action,"seed":args.seed})
    except BaseException:
        dump(CURRENT/"failure.json",{"action":args.action,"seed":args.seed,"traceback":traceback.format_exc(),
            "TEST_access_events":_TEST_EVENTS,"historical_write_events":history_write_events})
        dump(CURRENT/"status.json",{"state":"failed","action":args.action,"seed":args.seed})
        raise
    finally:
        dump(CURRENT/"protection.json",{"TEST_access_events":_TEST_EVENTS,
            "historical_write_events":history_write_events,"output_directory":str(CURRENT)})


if __name__=="__main__":
    main()
