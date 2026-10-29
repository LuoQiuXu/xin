"""Thin IO/observed-label adapter around unchanged author training functions."""
import ast
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from prepare_inputs import ROOT, TASK, OUT, CACHE, sha, write_json

OFFICIAL = ROOT / "H2T_official_clean/H2T-main"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def ahash(*values):
    h = hashlib.sha256()
    for a in values:
        if isinstance(a, torch.Tensor):
            a = a.detach().cpu().numpy()
        a = np.ascontiguousarray(a)
        h.update(str(a.shape).encode()); h.update(str(a.dtype).encode()); h.update(a.tobytes())
    return h.hexdigest()


def state_hash(state):
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        h.update(name.encode()); h.update(ahash(value).encode())
    return h.hexdigest()


def cpu_state(module):
    return {k:v.detach().cpu().clone() for k,v in module.module.state_dict().items()}


def install_guard(phase, regime=None):
    result = dict(phase=phase, regime=regime, official_TEST_attempts=[], historical_write_attempts=[],
                  prohibited_label_or_validation_attempts=[], numeric_files_read=[])
    training = phase in ("stage1", "stage2", "dryrun")
    forbidden = {"raw_clean_labels.npy", "human_labels.npz", "CIFAR-100_human.pt", "DATA_AUDIT.json", "INPUT_ATTESTATION.json",
                 "clean_validation.npz", "validation_images.npy"}
    if regime == "noisy": forbidden.add("clean_train.npz")
    def hook(event,args):
        if event != "open" or not isinstance(args[0],(str,bytes,os.PathLike)): return
        p = Path(os.fsdecode(args[0])).resolve()
        mode,flags = args[1] or "",args[2] or 0
        writing = any(c in str(mode) for c in "wax+") or flags & (os.O_WRONLY|os.O_RDWR)
        if p.name.lower() == "test" and "cifar" in p.as_posix().lower():
            result["official_TEST_attempts"].append(str(p)); raise PermissionError("Official TEST forbidden")
        if writing and p.is_relative_to(ROOT) and not p.is_relative_to(TASK):
            result["historical_write_attempts"].append(str(p)); raise PermissionError("Historical write forbidden")
        if not writing and training and ((p.name in forbidden) or (p.name == "train" and "cifar-100-python" in p.as_posix())):
            result["prohibited_label_or_validation_attempts"].append(str(p)); raise PermissionError("Training oracle/validation access forbidden: "+str(p))
        if not writing and p.is_relative_to(ROOT) and p.suffix.lower() in (".npy", ".npz", ".pth", ".pt"):
            rel = p.relative_to(ROOT).as_posix()
            if rel not in result["numeric_files_read"]: result["numeric_files_read"].append(rel)
    sys.addaudithook(hook)
    return result


def assert_guard(g):
    assert not g["official_TEST_attempts"] and not g["historical_write_attempts"] and not g["prohibited_label_or_validation_attempts"]


def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def function(path,name,glob,class_name=None):
    tree=ast.parse(Path(path).read_text(encoding="utf-8-sig"))
    if class_name: tree=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==class_name)
    node=next(n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),"exec"),glob)
    return glob[name]


def config(stage):
    m=module(OFFICIAL/"utils/logger.py","noisy_author_config")
    c=m._C.clone();c.defrost()
    c.merge_from_file(str(OFFICIAL/("config/cifar100/cifar100_imb001_stage1_ce_mixup.yaml" if stage==1 else "config/cifar100/cifar100_imb001_stage2_ce_H2T.yaml")))
    c.resume="";c.print_freq=40
    assert c.batch_size==128 and c.workers==0 and c.backbone=="resnet32_fe" and c.num_classes==100
    assert c.loss.type=="CrossEntropyLoss" and not c.loss.CE.reweight_CE and not c.loss.add_extra_info
    if stage==1: assert c.num_epochs==200 and c.mixup and c.alpha==1 and c.weight_decay==.0002
    else: assert c.num_epochs==10 and not c.mixup and c.shift_bn and c.H2T==.25 and c.lr_scheduler.lr_factor==.2 and c.weight_decay==.0005
    return c


def seed(seed,cfg):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=bool(cfg.deterministic)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.allow_tf32=True
    torch.backends.cuda.matmul.allow_tf32=False
    assert torch.cuda.is_available() and torch.cuda.device_count()==1


def models():
    m=module(OFFICIAL/"models/resnet_cifar.py","noisy_author_model")
    bb=torch.nn.DataParallel(m.resnet32_fe()).cuda()
    fc=torch.nn.DataParallel(m.Classifier(feat_in=64,num_classes=100)).cuda()
    return bb,fc


def transform(stage="train"):
    tree=ast.parse((OFFICIAL/"datasets/cifar100.py").read_text(encoding="utf-8-sig"))
    wanted="train_transform" if stage=="train" else "eval_transform"
    node=next(n for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==wanted for t in n.targets))
    return eval(compile(ast.Expression(node.value),"original_eval_transform","eval"),{"transforms":transforms})


class TrackedBank:
    def __init__(self,images):self.images=images;self.accesses=[]
    def __getitem__(self,index):self.accesses.append(int(index));return self.images[index]


class ObservedDataset(torch.utils.data.Dataset):
    def __init__(self,regime,cfg):
        path=OUT/"protocol"/(regime+"_train.npz")
        lock=read(OUT/"TRAINING_LOCK.json")
        assert sha(path)==lock["regime_label_sha256"][regime]
        images=CACHE/"inputs/lt_images.npy"
        assert sha(images)==lock["lt_image_sha256"]
        with np.load(path,allow_pickle=False) as z:self.ids=z["ids"].copy();labels=z["labels"].copy()
        assert self.ids.shape==labels.shape==(10847,) and labels.dtype.kind in "iu"
        self.targets=labels.tolist();self.config=cfg;self.train=True;self.cls_num=100
        self.data=TrackedBank(np.load(images,allow_pickle=False,mmap_mode="r"))
        self.transform=transform();self.target_transform=None
        self.cls_num_list=np.bincount(labels,minlength=100).tolist()
        assert min(self.cls_num_list)>0
        self.class_dict={c:np.flatnonzero(labels==c).tolist() for c in range(100)}
        self.class_weight=[max(self.cls_num_list)/n for n in self.cls_num_list];self.sum_weight=sum(self.class_weight)
        glob={"random":random,"Image":Image}
        self.original=function(OFFICIAL/"datasets/cifar100.py","__getitem__",glob,"IMBALANCECIFAR100")
    def __len__(self):return len(self.targets)
    def __getitem__(self,index):
        self.data.accesses=[]
        image,target,meta=self.original(self,index)
        if self.config.sampler.dual_sample.enable:
            donor,receiver=self.data.accesses
            meta["donor_id"]=int(self.ids[donor])
        else: receiver,=self.data.accesses
        meta["receiver_id"]=int(self.ids[receiver])
        return image,target,meta


class AuditedLoader:
    def __init__(self,dataset,stage,max_batches=None):
        self.dataset=dataset;self.stage=stage;self.batch_size=128;self.limit=max_batches
        self.base=torch.utils.data.DataLoader(dataset,batch_size=128,shuffle=True,num_workers=0,pin_memory=True)
    def __len__(self):return min(self.limit or len(self.base),len(self.base))
    def __iter__(self):
        it=iter(self.base)
        for i in range(len(self)):
            images,target,meta=next(it)
            self.current=dict(batch=i,receiver_ids_sha=ahash(meta["receiver_id"]),targets_sha=ahash(target),
                              receiver_images_sha=ahash(images),CPU_rng_ready=ahash(torch.get_rng_state()))
            if self.stage==2:
                self.current.update(donor_ids_sha=ahash(meta["donor_id"]),donor_labels_sha=ahash(meta["sample_label"]),donor_images_sha=ahash(meta["sample_image"]))
            self.rows.append(self.current)
            yield images,target,meta


class AuditedSGD(torch.optim.SGD):
    def __init__(self,*a,**kw):super().__init__(*a,**kw);self.updates=0;self.finite_gradient_checks=0
    def step(self,*a,**kw):
        grads=[p.grad for g in self.param_groups for p in g["params"] if p.grad is not None]
        assert grads and torch.isfinite(torch.stack(torch._foreach_norm(grads))).all(),"Nonfinite gradient"
        self.finite_gradient_checks+=1
        result=super().step(*a,**kw);self.updates+=1;return result


class Core:
    def __init__(self,stage,cfg,loader,method,logfile):
        self.stage,self.loader,self.method=stage,loader,method
        self.logfile=logfile;self.meters=[];self.replacements=0;self.mixups=[]
        self.methods=module(OFFICIAL/"methods.py","noisy_author_methods")
        meter=module(OFFICIAL/"utils/meter.py","noisy_author_meters")
        glob={"torch":torch,"np":np,"time":time,"math":math,"ProgressMeter":meter.ProgressMeter,
              "AverageMeter":self.meter,"mixup_data":self.mixup,"mixup_criterion":self.methods.mixup_criterion,
              "H2T":self.fusion}
        glob["accuracy"]=function(OFFICIAL/"utils/metric.py","accuracy",{"torch":torch})
        self.native_meter=meter.AverageMeter
        path=OFFICIAL/("train_stage1.py" if stage==1 else "train_stage2.py")
        self.train=function(path,"train",glob)
        self.schedule=function(path,"adjust_learning_rate",{"math":math})
        loss=module(OFFICIAL/"loss.py","noisy_author_losses")
        self.criterion=loss.CrossEntropyLoss(config=cfg,cls_num_list=loader.dataset.cls_num_list).to("cuda")
        self.criterion.register_forward_hook(lambda m,i,o: self.check_loss(o))
    def check_loss(self,value):assert torch.isfinite(value).all(),"Nonfinite original CE"
    def meter(self,*a,**kw):
        m=self.native_meter(*a,**kw);self.meters.append(m);return m
    def info(self,message):
        self.logfile.write(str(message)+"\n");self.logfile.flush()
    def mixup(self,*a,**kw):
        value=self.methods.mixup_data(*a,**kw)
        self.mixups.append(float(value[3]));return value
    def fusion(self,receiver,donor,rho):
        assert rho==.25 and receiver.shape[1:]==(64,8,8) and receiver.shape==donor.shape
        row=self.loader.current
        row.update(receiver_features_sha=ahash(receiver),donor_features_sha=ahash(donor))
        before=receiver.clone();rng=torch.get_rng_state()
        g=torch.Generator();g.set_state(rng)
        permutation=torch.randperm(64,generator=g);mask=permutation[:16]
        if self.method=="h2t":
            value=self.methods.H2T(receiver,donor,rho=rho);self.replacements+=1
            assert torch.equal(value[:,mask],donor[:,mask]) and torch.equal(value[:,permutation[16:]],before[:,permutation[16:]])
        else:
            torch.randperm(64).cuda()  # exact unused RNG draw; no replacement
            value=receiver;assert torch.equal(value,before)
        assert torch.equal(torch.get_rng_state(),g.get_state())
        row["mask_sha"]=ahash(mask)
        if self.baseline is not None:
            expected=self.baseline[self.epoch*85+row["batch"]]
            assert {k:str(v) for k,v in row.items()}==expected,"Pairing mismatch before classifier update"
        return value
    def epoch_run(self,optimizer,epoch,bb,fc,cfg,baseline=None):
        self.meters=[];self.loader.rows=[];self.mixups=[];self.baseline=baseline;self.epoch=epoch
        self.schedule(optimizer,epoch,cfg)
        start=time.perf_counter()
        self.train(self.loader,bb,fc,self.criterion,optimizer,epoch,cfg,self)
        stats={m.name:float(m.avg) for m in self.meters}
        stats.update(seconds=time.perf_counter()-start,mixup_lambdas_sha=ahash(np.array(self.mixups)),updates=optimizer.updates,
            input_images_epoch_sha=hashlib.sha256("".join(r["receiver_images_sha"] for r in self.loader.rows).encode()).hexdigest(),
            control_epoch_sha=hashlib.sha256(json.dumps(self.loader.rows,sort_keys=True).encode()).hexdigest())
        return stats
