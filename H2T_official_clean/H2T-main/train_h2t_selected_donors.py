"""Seed42 Stage-II donor-selection pilot; clean inference only, fixed final epoch."""
import hashlib
import json
import random
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
from PIL import Image
import audit_h2t_calibration as recipe
import audit_h2t_transfer_oracle as audit
from methods import H2T


class DirectImages(Dataset):
    def __init__(self,ds,indices,transform): self.ds,self.indices,self.transform=ds,indices,transform
    def __len__(self): return len(self.indices)
    def __getitem__(self,i):
        j=int(self.indices[i])
        return self.transform(Image.fromarray(self.ds.data[j])),int(self.ds.targets[j])


def banks(model,ds,epoch):
    """No bank extraction may change training RNG or BN buffers."""
    py,npstate=random.getstate(),np.random.get_state()
    mode=model.training
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
        recipe.set_paired_seed(420000+epoch)
        model.eval()
        protos=[]; labs=[]; head_features=[]
        with torch.no_grad():
            loader=DataLoader(DirectImages(ds,np.arange(len(ds)),audit.eval_transform()),batch_size=256,shuffle=False)
            for x,y in loader:
                protos.append(model(x.cuda()).mean((2,3))); labs.append(y.cuda())
            v=torch.cat(protos); labels=torch.cat(labs)
            proto=torch.stack([v[labels==c].mean(0) for c in range(100)])
            head_ids=np.flatnonzero(np.asarray(ds.targets)<36)
            # Standard training crop/flip, refreshed each epoch, independent RNG stream.
            loader=DataLoader(DirectImages(ds,head_ids,ds.transform),batch_size=256,shuffle=False)
            for x,_ in loader: head_features.append(model(x.cuda()).detach())
        bank=torch.cat(head_features)
        model.train(mode)
    random.setstate(py); np.random.set_state(npstate)
    return bank,proto


def train_arm(name,args,out):
    recipe.set_paired_seed(42)
    _,loader,test=recipe.build_loaders(args)
    model,cls,_=recipe.load_released_stage1(args.stage1_checkpoint)
    opt=torch.optim.SGD(cls.parameters(),lr=args.lr,momentum=.9,weight_decay=5e-4)
    rng=np.random.default_rng(424242)
    sampling_generator=torch.Generator(device='cuda').manual_seed(420001)
    stream=hashlib.sha256(); candidate_hash=hashlib.sha256()
    history=[]; start=time.time()
    arm=out/name; arm.mkdir(parents=True,exist_ok=True)
    for epoch in range(10):
        bank,proto=banks(model,loader.dataset,epoch)
        bank_v=bank.mean((2,3)); norm_h=F.normalize(bank_v,dim=1); norm_p=F.normalize(proto,dim=1)
        opt.param_groups[0]['lr']=recipe.stage2_lr(epoch,args)
        model.train(); cls.train(); loss_sum=0.; count=0; changed=0
        entropy_sum=0.; effective_sum=0.; maxprob_sum=0.; sampled_top1=0
        for images,target,meta in loader:
            original_donor=meta['sample_image']
            stream.update(images.numpy().tobytes()); stream.update(original_donor.numpy().tobytes()); stream.update(target.numpy().tobytes())
            images=images.cuda(); target=target.cuda(); original_donor=original_donor.cuda()
            with torch.no_grad():
                feat=model(images); df=model(original_donor)
                tail=torch.nonzero(target>=71).flatten()
                if len(tail):
                    ids=np.stack([rng.choice(len(bank),50,replace=False) for _ in range(len(tail))])
                    candidate_hash.update(ids.tobytes())
                    cand=torch.as_tensor(ids,device='cuda')
                    class_score=(norm_h[cand]*norm_p[target[tail],None,:]).sum(-1)
                    top=class_score.topk(10,dim=1).indices
                    if name=='head_bank_random':
                        selected=cand[:,0]
                    elif name=='class_aware':
                        selected=cand.gather(1,top[:,:1]).squeeze(1)
                    elif name=='class_top10_uniform':
                        topids=cand.gather(1,top)
                        probs=torch.full((len(tail),10),.1,device='cuda')
                        draw=torch.multinomial(probs,1,generator=sampling_generator)
                        selected=topids.gather(1,draw).squeeze(1)
                    else:
                        topids=cand.gather(1,top)
                        f=feat[tail].mean((2,3))
                        z=cls(feat[tail]); z.scatter_(1,target[tail,None],float('-inf')); rival=z.argmax(1)
                        w=cls.module.fc.weight
                        direction=w[target[tail]]-w[rival]
                        score=F.cosine_similarity(bank_v[topids]-f[:,None,:],direction[:,None,:],dim=-1)
                        if name=='boundary_soft_t1':
                            standardized=(score-score.mean(1,keepdim=True))/(score.std(1,keepdim=True,unbiased=True)+1e-6)
                            probs=torch.softmax(standardized,dim=1)
                            draw=torch.multinomial(probs,1,generator=sampling_generator)
                            selected=topids.gather(1,draw).squeeze(1)
                            sampled_top1+=int((draw.squeeze(1)==score.argmax(1)).sum())
                        else:
                            selected=topids.gather(1,score.argmax(1,keepdim=True)).squeeze(1)
                    if name in ('class_top10_uniform','boundary_soft_t1'):
                        entropy=-(probs*probs.clamp_min(1e-30).log()).sum(1)
                        entropy_sum+=float(entropy.sum()); effective_sum+=float(entropy.exp().sum())
                        maxprob_sum+=float(probs.max(1).values.sum())
                    df[tail]=bank[selected]
                    changed+=len(tail)
                # Official operator, untouched: replace 16/64 channels at rho=.25.
                fused=H2T(feat,df,rho=.25)
            logits=cls(fused); loss=F.cross_entropy(logits,target)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            loss_sum+=loss.item()*len(target); count+=len(target)
        r=dict(epoch=epoch+1,lr=opt.param_groups[0]['lr'],loss=loss_sum/count,n=count,few_selected=changed,elapsed=time.time()-start)
        if name in ('class_top10_uniform','boundary_soft_t1'):
            r.update(selection_entropy_nats=entropy_sum/changed,normalized_entropy=entropy_sum/changed/np.log(10),
                mean_effective_donors=effective_sum/changed,mean_max_probability=maxprob_sum/changed,
                sampled_boundary_top1_fraction=sampled_top1/changed if name=='boundary_soft_t1' else None)
        history.append(r); print(f'{name} {r}',flush=True)
        del bank,proto
    old=torch.load(out.parent/'calibration_audit_official_h2t_seed42/h2t/last_epoch_fixed.pth.tar',map_location='cpu',weights_only=False)
    max_bn_diff=max(float((t.cpu()-old['state_dict_model'][k]).abs().max()) for k,t in model.state_dict().items())
    assert max_bn_diff==0, f'Original H2T backbone/BN mismatch: {max_bn_diff}'
    # No test evaluation before the predeclared final endpoint.
    metrics,raw=recipe.evaluate_clean(model,cls,test,args)
    torch.save(dict(epoch=10,rho=.25,arm=name,state_dict_model=model.state_dict(),state_dict_classifier=cls.state_dict(),
        optimizer=opt.state_dict(),history=history,stage1_checkpoint_sha256=recipe.sha256(args.stage1_checkpoint)),arm/'last_epoch_fixed.pth.tar')
    np.savez_compressed(arm/'clean_test_predictions.npz',**raw)
    result=dict(arm=name,clean_test=metrics,history=history,training_stream_sha256=stream.hexdigest(),
        candidate_stream_sha256=candidate_hash.hexdigest(),original_backbone_max_abs_difference=max_bn_diff,
        duration_seconds=time.time()-start)
    (arm/'metrics.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(name+' CLEAN '+json.dumps({k:v['accuracy_pct'] for k,v in metrics.items()}),flush=True)
    return result


def main():
    here=Path(__file__).resolve().parent; out=here/'outputs/h2t_donor_training_seed42'; out.mkdir(parents=True,exist_ok=True)
    args=SimpleNamespace(data_root=recipe.default_workspace_root()/'H2T/data/cifar100',
        stage1_checkpoint=here/'pretrained/cifar100_imb001_stage1_official.pth.tar',seed=42,epochs=10,
        batch_size=128,workers=0,rho=.25,lr=.1,lr_factor=.2,momentum=.9,weight_decay=5e-4,ece_bins=15)
    protocol=dict(seed=42,epochs=10,rho=.25,batch_size=128,lr=.1,lr_factor=.2,
        stage1_checkpoint_sha256=recipe.sha256(args.stage1_checkpoint),candidate_count=50,topk=10,
        scope='Few recipients only; Many donors only; others retain original donor',
        bank='epoch-refreshed train-augmented features in eval mode; current backbone BN snapshot',
        prototype='epoch-refreshed deterministic train-only class means',
        bn='all arms preserve original recipient + original donor forwards; bank eval does not update BN',
        selection='online current Stage-II classifier, no Oracle; no test fusion')
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    results={}
    original=json.loads((here/'outputs/calibration_audit_official_h2t_seed42/h2t/metrics.json').read_text(encoding='utf-8'))
    results['original_h2t']=original
    # Exact official original arm reused; no training changes to its existing checkpoint.
    for name in ('head_bank_random','class_aware','boundary_aware'):
        results[name]=train_arm(name,args,out)
    assert len({results[k]['training_stream_sha256'] for k in results if k!='original_h2t'})==1
    assert len({results[k]['candidate_stream_sha256'] for k in results if k!='original_h2t'})==1
    rows=[]
    for name,r in results.items():
        rows.append(dict(method=name,**{g:r['clean_test'][g]['accuracy_pct'] for g in ('Overall','Many','Medium','Few')}))
    audit.save_summary_csv(rows,out/'clean_inference_summary.csv')
    lines=['# H2T donor-selection Stage-II 训练验证：seed42','',
        '最终固定 epoch10，正常 clean inference；测试时无 donor selection、无 H2T fusion。官方 Stage-I checkpoint 未重训。',
        '主组 Original/Class/Boundary；另加同一 Head feature-bank 随机选择对照，以识别 Head 池限制和缓存路径本身的影响。',
        '仅 Few recipient 使用新选择器，Many/Medium 保持原 H2T donor；Head 类为 0–35，Few 类为 71–99。50 个均匀随机 Head 候选，class-aware 取 prototype cosine 最高者；boundary-aware 先取 Top10，再按当前 classifier 的 W_y-W_j* 方向 cosine 精排。',
        '每 epoch 用当前 backbone BN 状态刷新 bank；donor bank 使用训练 crop/flip，prototype 使用训练集 eval transform。bank 提取在 eval/no-grad 下独立 RNG，不更新 BN；每步仍执行原 recipient/donor 的两次 forward，完全匹配原 BN 路径。',
        'bank 特征在当前 epoch 内缓存，因此与实时训练 batch-stat 特征并不完全同分布；Head-bank Random 控制相同实现，避免把缓存与候选域变化误认为选择器收益。',
        '三个新组 recipient/donor 增强流与候选池哈希完全一致；最终 backbone 全部参数及 BN buffers 与复用原 H2T checkpoint 逐项完全相同。','',
        '| 方法 | Overall | Many | Medium | Few |','|---|---:|---:|---:|---:|']
    for r in rows: lines.append('| '+r['method']+' | '+' | '.join(f'{r[g]:.2f}' for g in ('Overall','Many','Medium','Few'))+' |')
    lookup={r['method']:r for r in rows}
    lines+=['','| 对比 | ΔOverall | ΔMany | ΔMedium | ΔFew |','|---|---:|---:|---:|---:|']
    for a,b in [('class_aware','original_h2t'),('boundary_aware','class_aware'),('boundary_aware','original_h2t'),('class_aware','head_bank_random'),('boundary_aware','head_bank_random')]:
        lines.append('| '+a+' − '+b+' | '+' | '.join(f'{lookup[a][g]-lookup[b][g]:+.2f}' for g in ('Overall','Many','Medium','Few'))+' |')
    lines+=['','这是一轮固定 seed42 训练 pilot，尚非多 seed 方法结论。全部新组保存最后 epoch checkpoint、optimizer、训练日志和逐样本 clean predictions。']
    (out/'TRAINING_CLEAN_INFERENCE_ZH.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(rows,indent=2),flush=True)


if __name__=='__main__': main()
