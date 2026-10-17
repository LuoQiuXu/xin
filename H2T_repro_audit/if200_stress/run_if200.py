"""One clean IF=200 Stage-I plus 4 paired Stage-II arms x 2 seeds.

Commands: preflight, stage1, prepare, train --seed 124 --arm ..., report.
Run from repository root. Only writes under if200_stress/results/.
No official CIFAR TEST access, no model search, no resumed or old IF100 checkpoints.
This is a FIXED-BN, frozen-backbone mechanistic stress test, NOT the author's
shift_bn=True full recipe. Original-H2T is a batch-shared-mask analog; the
BS/IS receiver/donor distributions match the CIFAR author's semantics, but
the random number stream is separately locked and is not claimed bit-identical.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,math,os,random,sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
from PIL import Image

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
WORK=HERE.parents[1]
OLD=WORK/"H2T_repro_audit/clean_vs_stability_runs/clean_s1_seed42_s2_124_125_20261014"
RUNTIME=OLD/"runtime"
CONFIG=WORK/"H2T/config/cifar100"
RESULTS=HERE/"results"
SOURCE=OLD
sys.path.insert(0,str(RUNTIME))
from models.resnet_cifar import resnet32_fe,Classifier
from utils import config as base_config
from methods import mixup_data
from policy import ARMS,masks,fuse,self_test

N=9502
BATCH=128
S1_EPOCHS=200
S2_EPOCHS=10
SEEDS=(124,125)
STEPS=math.ceil(N/BATCH)
S1_SEED=42
GROUPS={"Head":np.arange(0,36),"OfficialMedium":np.arange(36,71),"Tail":np.arange(71,100)}
PROTOCOL={
    "setting":"CIFAR100-LT IF200 (.005), 9502 LT TRAIN, same 1980-image TRAIN-only 99-class holdout",
    "stage1":"fresh ResNet32 CE+MixUp alpha1, SGD, fixed epoch200, seed42, no best selection",
    "stage2":"frozen backbone and BN; CE classifier-only SGD, 10 epochs, seeds124/125",
    "stage2_lr":"author IF200 YAML lr=0.2, lr_factor=0.1 => FC actual initial lr=0.02",
    "mask_arms":list(ARMS),
    "sampling":"class-uniform independent BS receiver, permutation IS donor each epoch",
    "original_mask":"one 16/64 random mask shared within batch; analogue, not identical PRNG to released implementation",
    "VS":"TRAIN-only IF200 statistics; donor var/(var+mean-shift^2+epsilon); bottom48 random16, donor count<20 fallback",
    "primary_comparisons":["Original-BatchShared - DR-NoFusion","LowVS-PerSample - Uniform-PerSample"],
    "secondary_comparison":"Uniform-PerSample - Original-BatchShared distinguishes mask granularity",
    "evaluation":"fixed epoch10 on re-used 1980-image TRAIN holdout; class0 absent; no official TEST",
    "do_not_claim":"not original shift_bn=True reproduction, not cross-dataset transfer proof, not independent statistical significance",
}
def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
    return h.hexdigest()
def hash_data(*xs):
    h=hashlib.sha256()
    for x in xs:
        x=np.asarray(x)
        h.update(str(x.shape).encode());h.update(str(x.dtype).encode());h.update(x.tobytes())
    return h.hexdigest()
def js(p,obj):
    p=Path(p)
    assert p.resolve().is_relative_to(RESULTS.resolve())
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,ensure_ascii=False,allow_nan=False,indent=2),encoding="utf-8")
def writecsv(p,rows):
    if not rows:raise AssertionError("No rows")
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def readcsv(p):
    with p.open(encoding="utf-8",newline="") as f:return list(csv.DictReader(f))
def seedall(s):
    random.seed(s);np.random.seed(s);torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False

_GUARDED=False
def guard():
    global _GUARDED
    if _GUARDED:return
    _GUARDED=True
    # Since all input arrays come from a locked TRAIN-only experiment, the
    # official test file should never be opened or instantiated.
    forbidden=os.path.normcase(os.path.abspath(WORK/"H2T/data/cifar100/cifar-100-python/test"))
    def audit(event,args):
        if event=="open" and args and isinstance(args[0],(str,bytes,os.PathLike)):
            if os.path.normcase(os.path.abspath(os.fsdecode(args[0])))==forbidden:
                raise RuntimeError("Forbidden official CIFAR100 TEST read")
    sys.addaudithook(audit)
    import torchvision
    original=torchvision.datasets.CIFAR100.__init__
    def protected(self,root,train=True,*a,**kw):
        if not train:raise RuntimeError("Official TEST forbidden")
        return original(self,root,train,*a,**kw)
    torchvision.datasets.CIFAR100.__init__=protected

def counts200():
    return np.array([int(500*(0.005**(c/99.0))) for c in range(100)],dtype=np.int64)
def data():
    # IF200 is nested in the frozen IF100 per-class shuffled train list.
    required=["train_images.npy","train_labels.npy","validation_images.npy",
              "validation_labels.npy","split_indices.npz"]
    for s in required:
        if not (SOURCE/s).is_file():raise FileNotFoundError(f"Restore TRAIN-only locked {SOURCE/s}")
    lock=RESULTS/"preflight.json"
    if lock.is_file():
        locked=json.loads(lock.read_text(encoding="utf-8"))["source_sha256"]
        for name,expected in locked.items():
            if sha(SOURCE/name)!=expected:
                raise RuntimeError(f"Immutable TRAIN source changed since preflight: {name}")
    y100=np.load(SOURCE/"train_labels.npy",allow_pickle=False)
    x100=np.load(SOURCE/"train_images.npy",mmap_mode="r",allow_pickle=False)
    valx=np.load(SOURCE/"validation_images.npy",mmap_mode="r",allow_pickle=False)
    valy=np.load(SOURCE/"validation_labels.npy",allow_pickle=False)
    split=np.load(SOURCE/"split_indices.npz",allow_pickle=False)
    assert y100.shape==(10847,) and x100.shape==(10847,32,32,3)
    assert valx.shape==(1980,32,32,3) and np.array_equal(np.bincount(valy,minlength=100),np.r_[0,np.repeat(20,99)])
    index=np.concatenate([np.flatnonzero(y100==c)[:counts200()[c]] for c in range(100)])
    y=y100[index]
    assert len(index)==N and np.array_equal(np.bincount(y,minlength=100),counts200())
    raw=split["train_raw_indices"][index]
    assert not np.intersect1d(raw,split["validation_raw_indices"]).size
    assert not np.unique(raw).size<N
    return x100[index].copy(),y.astype(np.int64),np.asarray(valx).copy(),valy.astype(np.int64),index,raw

def tfms():
    train=transforms.Compose([transforms.RandomCrop(32,padding=4),
          transforms.RandomHorizontalFlip(),transforms.ToTensor(),
          transforms.Normalize((.4914,.4822,.4465),(.2023,.1994,.2010))])
    ev=transforms.Compose([transforms.ToTensor(),
          transforms.Normalize((.4914,.4822,.4465),(.2023,.1994,.2010))])
    return train,ev

class Images(Dataset):
    def __init__(self,images,labels,transform):
        self.images=images;self.labels=labels;self.transform=transform
    def __len__(self):return len(self.labels)
    def __getitem__(self,i):
        return self.transform(Image.fromarray(self.images[i])),int(self.labels[i])

def cfg(which):
    p=CONFIG/f"cifar100_imb0005_{which}_ce_"+("mixup" if which=="stage1" else "H2T")+".yaml"
    q=base_config.clone();q.defrost();q.merge_from_file(str(p))
    assert float(q.imb_factor)==0.005 and q.batch_size==128 and q.workers==0
    if which=="stage1":
        assert q.num_epochs==200 and q.mixup and q.alpha==1.0
        assert q.lr==.1 and q.weight_decay==.0002 and q.momentum==.9
    else:
        assert q.num_epochs==10 and q.H2T==.25
        assert q.lr==.2 and q.weight_decay==.0002 and q.momentum==.9
        assert q.lr_scheduler.lr_factor==.1 and q.shift_bn is True
    return q

def model():
    bb=torch.nn.DataParallel(resnet32_fe()).cuda()
    fc=torch.nn.DataParallel(Classifier(feat_in=64,num_classes=100)).cuda()
    assert sum(p.numel() for p in bb.parameters())==463504
    assert sum(p.numel() for p in fc.parameters())==6500
    return bb,fc

def statehash(m):
    h=hashlib.sha256()
    for k,v in m.state_dict().items():
        h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()

def vals(fc,labels,bb=None,images=None,ev=None,maps=None):
    fc.eval()
    if bb is not None:bb.eval()
    out=[]
    with torch.no_grad():
        for j in range(0,len(labels),BATCH):
            if maps is None:
                x=torch.stack([ev(Image.fromarray(v)) for v in images[j:j+BATCH]]).cuda()
                logits=fc(bb(x))
            else:
                x=torch.from_numpy(np.asarray(maps[j:j+BATCH]).copy()).cuda()
                logits=fc(x)
            out.append(logits.cpu().numpy())
    logits=np.concatenate(out);pred=logits.argmax(1)
    correct=np.bincount(labels[pred==labels],minlength=100)
    denom=np.bincount(labels,minlength=100)
    per=100*correct/np.maximum(denom,1)
    counts=counts200()
    many=np.flatnonzero((counts>100)&(denom>0))
    med=np.flatnonzero((counts>=20)&(counts<=100)&(denom>0))
    few=np.flatnonzero((counts<20)&(denom>0))
    ans={"Overall":float(100*np.mean(pred==labels)),
         "Head":float(per[1:36].mean()), # class0 absent from TRAIN dev
         "OfficialMedium":float(per[36:71].mean()),"Tail":float(per[71:100].mean()),
         "Many":float(per[many].mean()),"Medium":float(per[med].mean()),
         "Few":float(per[few].mean()),"WorstClass":float(per[1:100].min()),
         "Macro":float(per[1:100].mean()),
         "CE":float(np.mean(np.logaddexp.reduce(logits,axis=1)-logits[np.arange(len(labels)),labels]))}
    return ans,logits,pred

def preflight():
    guard();self_test();cfg("stage1");cfg("stage2")
    if not torch.cuda.is_available():raise RuntimeError("CUDA is needed for full pairwise CUDA replay")
    assert not (RESULTS/"preflight.json").exists(),"Do not overwrite"
    xx,y,vx,vy,idx,raw=data()
    _,ev=tfms()
    # Check the nested subset, LT counts and model dimensions, without any training.
    seedall(42);bb,fc=model()
    with torch.no_grad():
        z=bb(torch.stack([ev(Image.fromarray(q)) for q in xx[:8]]).cuda())
        assert z.shape==(8,64,8,8)
        assert fc(z).shape==(8,100)
    source_files={s:sha(SOURCE/s) for s in ("train_images.npy","train_labels.npy","validation_images.npy",
                                         "validation_labels.npy","split_indices.npz")}
    js(RESULTS/"preflight.json",{"passed":True,"num_train":N,"num_val":len(vy),
        "train_counts":counts200().tolist(),"source_sha256":source_files,
        "nested_index_sha256":hash_data(idx),"raw_id_sha256":hash_data(raw),
        "validation_source":"same untouched 1980-image TRAIN holdout, class0 absent",
        "GPU":torch.cuda.get_device_name(0),"training_done":False,"official_TEST_used":False,
        "protocol":PROTOCOL})
    print("PREFLIGHT PASS: IF200 nested 9502 images, TEST not touched; GPU forward")

def s1lr(epoch):
    # Original multistep: epoch+1 <= 5 warmup; drops after 160 and 180.
    e=epoch+1
    return .1*e/5 if e<=5 else (.001 if e>180 else (.01 if e>160 else .1))

def stage1():
    guard()
    pre=RESULTS/"preflight.json"
    if not pre.is_file():raise FileNotFoundError("Preflight required")
    assert json.loads(pre.read_text(encoding="utf-8"))["passed"]
    folder=RESULTS/"stage1"
    if folder.exists():raise FileExistsError("Refuse StageI resume/overwrite")
    q=cfg("stage1");seedall(42)
    xx,y,vx,vy,idx,raw=data()
    tr,ev=tfms()
    loader=DataLoader(Images(xx,y,tr),batch_size=BATCH,shuffle=True,num_workers=0,pin_memory=True)
    bb,fc=model();bb.train();fc.train()
    opt=torch.optim.SGD([{"params":bb.parameters()},{"params":fc.parameters()}],
                        lr=.1,momentum=.9,weight_decay=.0002)
    folder.mkdir(parents=True)
    hist=[]
    for epoch in range(S1_EPOCHS):
        bb.train();fc.train()
        for pg in opt.param_groups:pg["lr"]=s1lr(epoch)
        total=0.;seen=0;steps=0
        for images,labels in loader:
            images=images.cuda(non_blocking=True);labels=labels.cuda(non_blocking=True)
            mixed,ya,yb,lam=mixup_data(images,labels,alpha=1.)
            logits=fc(bb(mixed))
            loss=lam*nn.functional.cross_entropy(logits,ya)+(1-lam)*nn.functional.cross_entropy(logits,yb)
            assert torch.isfinite(loss)
            opt.zero_grad(set_to_none=True);loss.backward();opt.step()
            total+=float(loss.detach())*len(labels);seen+=len(labels);steps+=1
        assert seen==N and steps==STEPS
        hist.append({"epoch":epoch+1,"lr":s1lr(epoch),"steps":steps,"train_CE_mixup":total/seen})
        if (epoch+1)%10==0:
            writecsv(folder/"epochs.csv",hist)
            print(f"StageI {epoch+1}/200 train_mixup_CE={total/seen:.4f}",flush=True)
    summary,lg,pred=vals(fc,vy,bb=bb,images=vx,ev=ev)
    ck=folder/"stage1_fixed_epoch200.pth"
    torch.save({"seed":42,"epoch":200,"IF":200,"class_counts":counts200().tolist(),
                "backbone":{k:v.detach().cpu().clone() for k,v in bb.state_dict().items()},
                "classifier":{k:v.detach().cpu().clone() for k,v in fc.state_dict().items()},
                "selection":"fixed epoch200, not selected using validation",
                "source_nested_raw_ids_sha256":hash_data(raw),
                "official_TEST_used":False},ck)
    js(folder/"summary.json",{"complete":True,"fixed_epoch":200,"checkpoint_sha256":sha(ck),
                              "validation_epoch200":summary,"source_nested_raw_ids_sha256":hash_data(raw),
                              "backbone_hash":statehash(bb),"classifier_hash":statehash(fc),
                              "updates":STEPS*S1_EPOCHS,"official_TEST_used":False})
    print("StageI PASS fixed epoch200",summary["Overall"],summary["Tail"])

def load_s1():
    f=RESULTS/"stage1/stage1_fixed_epoch200.pth"
    p=RESULTS/"stage1/summary.json"
    if not (p.exists() and f.exists()):raise FileNotFoundError("Finish StageI first")
    assert sha(f)==json.loads(p.read_text(encoding="utf-8"))["checkpoint_sha256"]
    ck=torch.load(f,map_location="cpu",weights_only=False)
    assert ck["seed"]==42 and ck["epoch"]==200 and ck["IF"]==200 and not ck["official_TEST_used"]
    return ck

def prepare():
    guard()
    ck=load_s1()
    out=RESULTS/"stats.npz"
    if out.exists() or (RESULTS/"validation_maps.npy").exists():
        raise FileExistsError("No stats overwrite")
    xx,y,vx,vy,idx,raw=data()
    _,ev=tfms();seedall(42);bb,fc=model()
    bb.load_state_dict(ck["backbone"],strict=True);fc.load_state_dict(ck["classifier"],strict=True)
    bb.eval().requires_grad_(False);fc.eval().requires_grad_(False)
    baseline_b=statehash(bb);baseline_c=statehash(fc)
    def extract(images,store=None):
        gaps=[]
        mem=np.lib.format.open_memmap(store,mode="w+",dtype="float32",
                                      shape=(len(images),64,8,8)) if store else None
        with torch.no_grad():
            for j in range(0,len(images),BATCH):
                bx=torch.stack([ev(Image.fromarray(x)) for x in images[j:j+BATCH]]).cuda()
                feats=bb(bx)
                if mem is not None:mem[j:j+len(bx)]=feats.cpu().numpy()
                gaps.append(nn.functional.avg_pool2d(feats,8).flatten(1).cpu().numpy())
        if mem is not None:mem.flush()
        return np.concatenate(gaps).astype(np.float64)
    z=extract(xx)
    extract(vx,RESULTS/"validation_maps.npy")
    count=counts200();assert count.min()>=2
    means=np.stack([z[y==c].mean(0) for c in range(100)])
    variances=np.stack([np.var(z[y==c],axis=0,ddof=0) for c in range(100)])
    pooled=(variances*count[:,None]).sum(0)/(len(y)-100)
    lam=5/(count+5)
    reg=(1-lam[:,None])*variances+lam[:,None]*pooled
    eps=1e-6*np.var(z,axis=0,ddof=1).mean()
    assert np.isfinite(reg).all() and eps>0
    np.savez_compressed(out,counts=count,mean_eval=means,
                        variance_eval_regularized=reg,variance_eval_ddof0=variances,
                        shrinkage_lambda=lam,pooled=pooled,epsilon=eps)
    assert statehash(bb)==baseline_b and statehash(fc)==baseline_c
    js(RESULTS/"stats_provenance.json",{"complete":True,
        "stats_sha256":sha(out),"validation_maps_sha256":sha(RESULTS/"validation_maps.npy"),
        "source":"IF200 TRAIN subset only, class statistics not fitted on TRAIN holdout",
        "backbone_unchanged":True,"classes_with_n_less20":int((count<20).sum()),
        "checkpoint_sha256":sha(RESULTS/"stage1/stage1_fixed_epoch200.pth"),
        "official_TEST_used":False})
    print("STATS PASS eps",eps,"fallback classes",sum(count<20))

def plans(seed,y):
    # Locked, seed-specific donor permutation and class-balanced receiver index
    # for every epoch. Does not touch global PyTorch RNG.
    classids=[np.flatnonzero(y==c) for c in range(100)]
    out=[]
    for epoch in range(S2_EPOCHS):
        rng=np.random.default_rng(seed+10000*epoch+31)
        rr=random.Random(seed+10000*epoch+37)
        donor=rng.permutation(N)
        rec=np.array([int(rr.choice(classids[rr.randrange(100)])) for _ in range(N)],dtype=np.int32)
        gn=np.random.default_rng(seed+10000*epoch+41).gumbel(size=(N,64))
        sn=np.random.default_rng(seed+10000*epoch+43).gumbel(size=(STEPS,64))
        assert len(np.unique(donor))==N
        out.append((rec,donor,gn,sn))
    return out

def augment(images,rec,don,generator,transform):
    prior=torch.get_rng_state().clone()
    with torch.random.fork_rng(devices=[]):
        torch.set_rng_state(generator.get_state())
        a,b=[],[]
        for r,d in zip(rec,don):
            b.append(transform(Image.fromarray(images[int(d)])))
            a.append(transform(Image.fromarray(images[int(r)])))
        generator.set_state(torch.get_rng_state())
    assert torch.equal(torch.get_rng_state(),prior)
    return torch.stack(a),torch.stack(b)

def train(seed,arm):
    guard()
    if seed not in SEEDS or arm not in ARMS:raise ValueError("Seed/arm outside protocol")
    ck=load_s1()
    pv=RESULTS/"stats_provenance.json"
    if not pv.is_file():raise FileNotFoundError("Run prepare first")
    prov=json.loads(pv.read_text(encoding="utf-8"))
    assert sha(RESULTS/"stats.npz")==prov["stats_sha256"]
    assert sha(RESULTS/"validation_maps.npy")==prov["validation_maps_sha256"]
    folder=RESULTS/f"seed{seed}"/arm
    if folder.exists():raise FileExistsError(f"Refuse overwrite: {folder}")
    base=RESULTS/f"seed{seed}"/"DR-NoFusion"/"paired_batches.csv"
    control=readcsv(base) if arm!="DR-NoFusion" else None
    if arm!="DR-NoFusion" and (not base.exists() or len(control)!=STEPS*S2_EPOCHS):
        raise FileNotFoundError("DR-NoFusion baseline per-seed required first")
    xx,y,vx,vy,idx,raw=data()
    stat=np.load(RESULTS/"stats.npz",allow_pickle=False)
    maps=np.load(RESULTS/"validation_maps.npy",mmap_mode="r",allow_pickle=False)
    seedall(seed);bb,fc=model()
    bb.load_state_dict(ck["backbone"],strict=True)
    fc.load_state_dict(ck["classifier"],strict=True)
    bb.eval().requires_grad_(False);fc.train()
    initial_b=statehash(bb);initial_c=statehash(fc)
    q=cfg("stage2")
    opt=torch.optim.SGD(fc.parameters(),lr=q.lr,momentum=q.momentum,weight_decay=q.weight_decay)
    train_transform,_=tfms()
    aug=torch.Generator().manual_seed(seed+300009)
    schedule=plans(seed,y)
    folder.mkdir(parents=True)
    writecsv(folder/"protocol.csv",[{"seed":seed,"arm":arm,"N":N,"stage1_epoch":200,
             "stage1_sha":sha(RESULTS/"stage1/stage1_fixed_epoch200.pth"),
             "stats_sha":sha(RESULTS/"stats.npz"),
             "initial_classifier_hash":initial_c,
             "BN_mode":"frozen_eval","IF":200,"author_shift_bn":True,
             "initial_stage2_fc_lr":.2*.1}])
    history=[];batch_records=[];step=0
    group_totals={"Tail":{"u":0.,"l":0.,"n":0},"Head":{"u":0.,"l":0.,"n":0}}
    for epoch,(rec,don,gn,sn) in enumerate(schedule):
        lr=.2*.1*.5*(1+math.cos(epoch*math.pi/10))
        for pg in opt.param_groups:pg["lr"]=lr
        fc.train()
        total=0.
        for bi,start in enumerate(range(0,N,BATCH)):
            end=min(start+BATCH,N);r,d=rec[start:end],don[start:end]
            xa,xb=augment(xx,r,d,aug,train_transform)
            with torch.no_grad():
                fa=bb(xa.cuda());fb=bb(xb.cuda())
                gap=nn.functional.avg_pool2d(fa,8).flatten(1).cpu().numpy()
                choices,diag=masks(gap,y[d],stat["counts"],stat["mean_eval"],
                    stat["variance_eval_regularized"],float(stat["epsilon"]),
                    gn[start:end],sn[bi])
                mixed=fa if arm=="DR-NoFusion" else fuse(fa,fb,choices[arm])
            h=hash_data(r,d,xa.numpy(),xb.numpy(),fa.cpu().numpy(),fb.cpu().numpy(),
                        gn[start:end],sn[bi],
                        choices["Original-BatchShared"],
                        choices["Uniform-PerSample"],choices["LowVS-PerSample"])
            if control is not None and h!=control[step]["hash"]:
                raise RuntimeError(f"Paired input/masks mismatch seed{seed} arm{arm} step{step}")
            batch_records.append({"step":step,"epoch":epoch+1,"batch":bi,"hash":h})
            for label,flag in (("Tail",y[r]>=71),("Head",y[r]<=35)):
                n=int(flag.sum())
                if n:
                    group_totals[label]["u"]+=float(diag["per_sample_uniform_selected_VS"][flag].sum())
                    group_totals[label]["l"]+=float(diag["per_sample_lowVS_selected_VS"][flag].sum())
                    group_totals[label]["n"]+=n
            tgt=torch.as_tensor(y[r],dtype=torch.long,device="cuda")
            output=fc(mixed);loss=nn.functional.cross_entropy(output,tgt)
            assert torch.isfinite(loss)
            opt.zero_grad(set_to_none=True);loss.backward();opt.step()
            step+=1;total+=float(loss.detach())*len(r)
        assert statehash(bb)==initial_b
        cur,logits,preds=vals(fc,vy,maps=maps)
        history.append({"seed":seed,"arm":arm,"epoch":epoch+1,"lr":lr,"train_CE":total/N,
                        **cur})
        writecsv(folder/"epochs.csv",history)
        print(f"IF200 seed{seed} {arm} epoch{epoch+1}: Overall={cur['Overall']:.3f} Tail={cur['Tail']:.3f}",flush=True)
    assert step==STEPS*S2_EPOCHS
    writecsv(folder/"paired_batches.csv",batch_records)
    torch.save({"seed":seed,"arm":arm,"epoch":10,
         "classifier":{k:v.detach().cpu().clone() for k,v in fc.state_dict().items()},
         "stage1_sha":sha(RESULTS/"stage1/stage1_fixed_epoch200.pth"),
         "official_TEST_used":False},folder/"final_fc.pth")
    np.savez_compressed(folder/"validation_epoch10.npz",labels=vy,logits=logits,predictions=preds)
    # Replay checkpoint validation before declaring the result complete.
    pred0=preds.copy();fc.load_state_dict(torch.load(folder/"final_fc.pth",map_location="cpu",weights_only=False)["classifier"])
    check,lg2,p2=vals(fc,vy,maps=maps)
    assert np.array_equal(pred0,p2) and max(abs(check[k]-cur[k]) for k in cur)<1e-8
    assert statehash(bb)==initial_b and statehash(fc)!=initial_c
    js(folder/"summary.json",{"completed":True,"seed":seed,"arm":arm,
        "stage1_sha":sha(RESULTS/"stage1/stage1_fixed_epoch200.pth"),
        "steps":step,"epochs":10,"final":cur,"classifier_initial_hash":initial_c,
        "backbone_unchanged":True,"BN_frozen":True,"mask_VS_trace_by_group":group_totals,
        "paired_all_batches":arm!="DR-NoFusion","TEST_used":False})
    print("TRAIN COMPLETE",seed,arm,flush=True)

def report():
    guard()
    rows=[];paired=[];perclass=[]
    for seed in SEEDS:
        found={};base=None;truth=None
        for arm in ARMS:
            folder=RESULTS/f"seed{seed}"/arm
            summ=json.loads((folder/"summary.json").read_text(encoding="utf-8"))
            assert summ["completed"] and summ["steps"]==STEPS*S2_EPOCHS
            v=np.load(folder/"validation_epoch10.npz",allow_pickle=False)
            y=v["labels"];p=v["predictions"];lg=v["logits"]
            assert np.array_equal(p,lg.argmax(1))
            if truth is None:truth=y.copy()
            else:assert np.array_equal(truth,y)
            sig=readcsv(folder/"paired_batches.csv")
            assert len(sig)==STEPS*S2_EPOCHS
            if base is None:base=sig
            else:assert all(a["hash"]==b["hash"] for a,b in zip(base,sig))
            found[arm]=p
            rows.append({"seed":seed,"arm":arm,**summ["final"]})
        contrasts=(("Original-BatchShared","DR-NoFusion"),
                   ("Uniform-PerSample","Original-BatchShared"),
                   ("LowVS-PerSample","Uniform-PerSample"),
                   ("LowVS-PerSample","Original-BatchShared"))
        for a,b in contrasts:
            pa,pb=found[a],found[b]
            wins=(pa==truth)&(pb!=truth);losses=(pa!=truth)&(pb==truth)
            c={"seed":seed,"comparison":a+" - "+b,
               "wins":int(wins.sum()),"losses":int(losses.sum()),
               "net":int(wins.sum()-losses.sum())}
            for name,group in GROUPS.items():
                sel=np.isin(truth,group)
                c[name+"_net"]=int(wins[sel].sum()-losses[sel].sum())
                c[name+"_delta_pp"]=100*float((pa[sel]==truth[sel]).mean()-
                                                 (pb[sel]==truth[sel]).mean())
            c["head_true_to_tail_pred_delta"]=int(((truth<=35)&(pa>=71)).sum()-int(((truth<=35)&(pb>=71)).sum())
            c["tail_true_to_head_pred_delta"]=int(((truth>=71)&(pa<=35)).sum()-int(((truth>=71)&(pb<=35)).sum())
            paired.append(c)
            for k in range(1,100):
                sel=truth==k
                perclass.append({"seed":seed,"comparison":c["comparison"],"class":k,
                    "n":int(sel.sum()),"wins":int(wins[sel].sum()),
                    "losses":int(losses[sel].sum()),"net":int(wins[sel].sum()-losses[sel].sum())})
    writecsv(RESULTS/"epoch10.csv",rows)
    writecsv(RESULTS/"paired.csv",paired)
    writecsv(RESULTS/"per_class.csv",perclass)
    js(RESULTS/"scientific_summary.json",{"protocol":PROTOCOL,"results":rows,
        "paired":paired,
        "scope":"This tests within-IF200 causal intervention; historical IF100 used as qualitative reference only.",
        "warnings":["single StageI seed42, two dependent StageII seeds, repeatedly-used 1980-image TRAIN holdout",
                    "original author shift_bn=True NOT run; frozen-BN analog only",
                    "LowVS vs Original confounds mask granularity with VS policy",
                    "no significance claims or official TEST access"]})
    lines=["# H2T IF200 controlled stress | fixed-BN, fixed-epoch results","",
           "IF200 nested 9502 LT TRAIN images; original 1980-image TRAIN holdout (class0 absent).",
           "StageI fresh seed42, 200 epochs fixed checkpoint. StageII 10epochs, seeds124/125; single frozen StageI.",
           "Four arms: DR-NoFusion, Original-BatchShared, Uniform-PerSample, LowVS-PerSample.",
           "Original denotes the mask-and-sampling analogue; NOT author's shift_bn=True full implementation.",
           "","|Seed|Arm|Overall|Head|Tail|WorstClass|",
           "|---|---|---:|---:|---:|---:|"]
    for r in rows:lines.append(f"|{r['seed']}|{r['arm']}|{r['Overall']:.3f}|{r['Head']:.3f}|{r['Tail']:.3f}|{r['WorstClass']:.3f}|")
    lines+=["","## Paired changes, fixed epoch10",
            "|Seed|A - B|Wins|Losses|Net|Head net|Tail net|Tail true→Head predicted delta|",
            "|---|---|---:|---:|---:|---:|---:|---:|"]
    for c in paired:lines.append(f"|{c['seed']}|{c['comparison']}|{c['wins']}|{c['losses']}|{c['net']:+d}|{c['Head_net']:+d}|{c['Tail_net']:+d}|{c['tail_true_to_head_pred_delta']:+d}|")
    lines+=["","### Decision and limits",
        "If OriginalShared consistently harms Tail vs DR, this is an exploratory IF200 negative-transfer candidate, not proof of dataset-invariant failure.",
        "If LowVS improves only against Uniform and not Original, do not claim a meaningful original H2T replacement.",
        "Do not infer IF interaction without a harmonized IF100 within-setting replication (historical data are protocol-distinct).",
        "No alpha/K/hardness/optimizer tuning. Compare per_class.csv and paired.csv before interpreting worst-class zeros.",
        "No public CIFAR100 TEST was used; do not call the reused 99-class TRAIN holdout an independent final test."]
    (RESULTS/"IF200_STRESS_RESULTS.md").write_text("\n".join(lines),encoding="utf-8")
    print("REPORT COMPLETE",RESULTS/"IF200_STRESS_RESULTS.md")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("action",choices=("policy-test","preflight","stage1","prepare","train","report"))
    p.add_argument("--seed",type=int,choices=SEEDS)
    p.add_argument("--arm",choices=ARMS)
    a=p.parse_args()
    if a.action=="policy-test":self_test()
    elif a.action=="preflight":preflight()
    elif a.action=="stage1":stage1()
    elif a.action=="prepare":prepare()
    elif a.action=="train":
        if a.seed is None or a.arm is None:p.error("train needs --seed and --arm")
        train(a.seed,a.arm)
    else:report()
if __name__=="__main__":main()
