"""Offline evidence verification and Chinese norm/direction report; no training."""
from pathlib import Path
import sys,json,csv
import numpy as np
import run_experiment as run
from run_safeguards import protect,stamp

OUT=run.RESULTS
GROUPS=("Overall","Head","OfficialMedium","Tail","Many","Medium","Few","Macro")
CONTRASTS=(("LowVS-Full","Uniform-Full"),("LowVS-Matched","Uniform-Matched"),("LowVS-Matched","LowVS-Full"),("Uniform-Matched","Uniform-Full"))

def read(p):return json.loads(Path(p).read_text(encoding="utf-8"))
def table(headers,rows):return ["|"+"|".join(headers)+"|","|"+"|".join(["---"]*len(headers))+"|"]+["|"+"|".join(map(str,r))+"|" for r in rows]
def dist(a):
    x=np.asarray(a,dtype=np.float64).reshape(-1)
    assert len(x)>0 and np.isfinite(x).all()
    q=np.quantile(x,[0,.05,.25,.5,.75,.95,1])
    return {"count":len(x),"mean":float(x.mean()),"std":float(x.std()),**{k:float(v) for k,v in zip(("min","q05","q25","median","q75","q95","max"),q)}}

def main():
    run.old.forbid_official_test();protect(run.HERE.parent,OUT,run.HISTORY_EVENTS)
    pre=read(OUT/"preflight.json")
    assert pre["status"]=="PASS" and pre["new_source_sha256"]==run.source_hashes()
    assert (run.HERE/"norm_policy.py").read_bytes()==(OUT/"provided_source/norm_policy.py").read_bytes()
    protected=run.verify_history()
    rows=[];epoch_rows=[];pairs=[];per_class=[];integrity=[];checkpoints=[];data={};norm_summary=[];norm_checks=[];alpha_epochs=[];hybrid_checks=[];hybrid_effects=[]
    valbank=np.load(run.old.OLD/"validation_feature_maps.npy",mmap_mode="r",allow_pickle=False)
    gap=np.asarray(valbank.mean(axis=(2,3)),dtype=np.float64)
    all_labels=None
    for seed in run.SEEDS:
        base_sig=None
        for arm in run.ARMS:
            folder=OUT/f"seed{seed}"/arm
            summ=read(folder/"summary.json");ep=run.read_csv(folder/"epochs.csv");sig=run.read_csv(folder/"batch_signatures.csv")
            assert summ["status"]=="completed" and summ["steps"]==850 and len(sig)==850 and len(ep)==10
            assert summ["backbone_frozen"] and summ["backbone_state_tensors_bitwise_unchanged"]==186 and summ["BN_counters"]==[15640]
            assert summ["checkpoint_reload_exact"] and not summ["TEST_used"]
            assert not summ["official_TEST_events"] and not summ["historical_write_events"]
            assert all(value>0 for value in summ["classifier_parameter_max_abs_changes"].values())
            if base_sig is None:base_sig=sig
            keys=(*run.PAIR_KEYS,"uniform_mask_sha256","lowvs_mask_sha256","norm_diagnostics_sha256")
            matched={k:sum(a[k]==b[k] for a,b in zip(base_sig,sig)) for k in keys}
            assert all(n==850 for n in matched.values())
            vals=np.load(folder/"validation_epoch10.npz",allow_pickle=False)
            y,p,z=vals["labels"],vals["predictions"],vals["logits"]
            assert z.shape==(1980,100) and np.isfinite(z).all() and np.array_equal(p,z.argmax(1))
            assert np.array_equal(np.bincount(y,minlength=100),np.r_[0,np.repeat(20,99)])
            if all_labels is None:all_labels=y.copy()
            else:assert np.array_equal(all_labels,y)
            m=run.metrics(p,y)
            assert all(abs(m[k]-summ["last"][k])<1e-10 for k in GROUPS)
            for kind in ("best","final"):
                cp=folder/f"checkpoints/{kind}.pth"
                assert run.sha(cp)==summ["checkpoint_sha256"][kind]
                ck=run.torch.load(cp,map_location="cpu",weights_only=False)
                assert ck["seed"]==seed and ck["arm"]==arm and not ck["TEST_used"]
                assert ck["stage1_sha256"]==pre["checks"][str(seed)]["stage1_sha256"]
                checkpoints.append({"seed":seed,"arm":arm,"kind":kind,"epoch":ck["epoch"],"path":cp.relative_to(run.HERE).as_posix(),"sha256":run.sha(cp)})
            w,b=run.weights(folder/"checkpoints/final.pth")
            rows.append({"seed":seed,"arm":arm,**{k:m[k] for k in GROUPS},"best_epoch":summ["best_epoch"],"best_Overall":summ["best_Overall"],"elapsed_seconds":summ["elapsed_seconds"]})
            epoch_rows.extend(ep)
            integrity.append({"seed":seed,"arm":arm,"steps":850,"matched":matched,"frozen_state_tensors":186,"BN_counters":[15640],"checkpoint_reload_exact":True,"classifier_parameter_changes":summ["classifier_parameter_max_abs_changes"]})
            batch_norm=run.read_csv(folder/"batch_norm_diagnostics.csv")
            assert len(batch_norm)==850 and all(int(r["norm_tolerance_failures"])==0 for r in batch_norm)
            data[(seed,arm)]={"summary":summ,"epochs":ep,"pred":p.copy(),"metrics":m,"w":w,"b":b,"norm_rows":batch_norm}
        y=all_labels
        raw_path=OUT/f"seed{seed}/Uniform-Full/norm_diagnostics.npz"
        with np.load(raw_path,allow_pickle=False) as archive:
            raw={key:archive[key] for key in archive.files}
        norm_keys=sorted(k for k in raw if k not in ("receiver_indices","donor_indices","receiver_labels"))
        assert all(raw[k].shape==(10,run.N) for k in raw)
        plan=np.load(run.old.OLD/f"plans/{seed}/sampling_plan.npz",allow_pickle=False)
        train_y=np.load(run.old.OLD/"train_labels.npy",allow_pickle=False)
        assert np.array_equal(raw["receiver_indices"],plan["receivers"])
        assert np.array_equal(raw["donor_indices"],plan["donors"])
        assert np.array_equal(raw["receiver_labels"],train_y[plan["receivers"]])
        target=raw["target_norm"];nu=raw["original_norm_uniform"];nl=raw["original_norm_lowvs"]
        au=raw["alpha_uniform"];al=raw["alpha_lowvs"];mu=raw["matched_norm_uniform"];ml=raw["matched_norm_lowvs"]
        assert np.array_equal(target,np.minimum(nu,nl))
        assert ((au>=0)&(au<=1)&(al>=0)&(al<=1)).all()
        tol=1e-5+1e-4*target
        error=np.abs(mu-ml);eu=np.abs(mu-target);el=np.abs(ml-target)
        assert (error<=tol).all() and (eu<=tol).all() and (el<=tol).all()
        assert np.array_equal(error,raw["gap_norm_abs_error"])
        count=0
        for epoch in range(10):
            for batch,start in enumerate(range(0,run.N,run.BATCH)):
                end=min(start+run.BATCH,run.N)
                signature=run.array_hash(*[raw[k][epoch,start:end] for k in norm_keys])
                assert signature==base_sig[count]["norm_diagnostics_sha256"]
                for arm in run.ARMS:
                    r=data[(seed,arm)]["norm_rows"][count]
                    assert abs(float(r["max_pair_gap_error"])-float(error[epoch,start:end].max()))<1e-12
                count+=1
        assert count==850
        norm_checks.append({"seed":seed,"paired_training_draws":10*run.N,"batches":850,"max_pair_gap_error":float(error.max()),"max_target_error_U":float(eu.max()),"max_target_error_L":float(el.max()),"max_relative_to_tolerance":float((error/tol).max()),"norm_tolerance_failures":0,"n_exact_zero_U":int((nu==0).sum()),"n_exact_zero_L":int((nl==0).sum()),"n_either_below_epsilon":int(raw["zero_norm_pairs"].sum()),"nU_smaller":int((nu<nl).sum()),"nL_smaller":int((nl<nu).sum()),"equal_original_norms":int((nu==nl).sum())})
        recipient=raw["receiver_labels"]
        selectors={"All":np.ones_like(recipient,dtype=bool),"Head":recipient<=35,"OfficialMedium":(recipient>=36)&(recipient<=70),"Tail":recipient>=71}
        for group,mask in selectors.items():
            for field in norm_keys:
                d=dist(raw[field][mask]);norm_summary.append({"seed":seed,"receiver_group":group,"field":field,**d,"fraction_zero":float((raw[field][mask]==0).mean()),"fraction_one":float((raw[field][mask]==1).mean())})
            for epoch in range(10):
                for key in ("alpha_uniform","alpha_lowvs"):
                    value=raw[key][epoch][mask[epoch]]
                    alpha_epochs.append({"seed":seed,"epoch":epoch+1,"receiver_group":group,"alpha":key,**dist(value),"fraction_zero":float((value==0).mean()),"fraction_one":float((value==1).mean())})
        for a,b in CONTRASTS:
            pa=data[(seed,a)]["pred"];pb=data[(seed,b)]["pred"]
            win=(pa==y)&(pb!=y);lose=(pa!=y)&(pb==y)
            item={"seed":seed,"comparison":a+" - "+b,"wins":int(win.sum()),"losses":int(lose.sum()),"net_correct":int(win.sum()-lose.sum())}
            for group in GROUPS:
                active=range(1,100) if group in ("Overall","Macro") else data[(seed,a)]["metrics"]["effective_classes"][group]
                mask=np.isin(y,list(active));n=int(mask.sum());net=int(win[mask].sum()-lose[mask].sum())
                delta=data[(seed,a)]["metrics"][group]-data[(seed,b)]["metrics"][group]
                assert abs(delta-net/n*100)<1e-10
                item["delta_"+group+"_pp"]=delta;item[group+"_net"]=net;item[group+"_wins"]=int(win[mask].sum());item[group+"_losses"]=int(lose[mask].sum())
            pairs.append(item)
            for c in range(1,100):
                mask=y==c;per_class.append({"seed":seed,"comparison":item["comparison"],"class_id":c,"wins":int(win[mask].sum()),"losses":int(lose[mask].sum()),"net_correct":int(win[mask].sum()-lose[mask].sum())})
        for a,b in CONTRASTS[:2]:
            uw,ub=data[(seed,b)]["w"],data[(seed,b)]["b"]
            lw,lb=data[(seed,a)]["w"],data[(seed,a)]["b"]
            effects={};mp={}
            for name,w,bias,native in (("U_W+U_b",uw,ub,b),("L_W+L_b",lw,lb,a),("U_W+L_b",uw,lb,None),("L_W+U_b",lw,ub,None)):
                pred=(gap@w.T+bias).argmax(1);m=run.metrics(pred,y);mp[name]=m
                row={"seed":seed,"comparison":a+" vs "+b,"hybrid":name,**{k:m[k] for k in GROUPS},"native_float32_prediction_disagreements":int((pred!=data[(seed,native)]["pred"]).sum()) if native else None}
                hybrid_checks.append(row)
            for group in ("Overall","Head","Tail","Few"):
                original=mp["U_W+U_b"][group];total=mp["L_W+L_b"][group]-original;bias=mp["U_W+L_b"][group]-original;weight=mp["L_W+U_b"][group]-original
                hybrid_effects.append({"seed":seed,"comparison":a+" vs "+b,"group":group,"total_change_pp":total,"change_b_at_UW_pp":bias,"change_W_at_Ub_pp":weight,"nonadditive_residual_pp":total-bias-weight,"W_difference_frobenius":float(np.linalg.norm(lw-uw)),"b_difference_L2":float(np.linalg.norm(lb-ub))})
    original_hybrids=run.read_csv(OUT/"classifier_hybrids.csv")
    assert len(original_hybrids)==16
    for a,b in zip(hybrid_checks,original_hybrids):
        assert a["seed"]==int(b["seed"]) and a["comparison"]==b["comparison"] and a["hybrid"]==b["hybrid"]
        assert all(abs(a[k]-float(b[k]))<1e-10 for k in ("Overall","Head","Tail","Few"))
    for name,records in (("epoch10_verified.csv",rows),("epochs_all.csv",epoch_rows),("paired_verified.csv",pairs),("per_class.csv",per_class),("norm_distribution.csv",norm_summary),("norm_matching_verification.csv",norm_checks),("alpha_by_epoch_group.csv",alpha_epochs),("classifier_hybrids_verified.csv",hybrid_checks),("hybrid_parameter_effects.csv",hybrid_effects),("checkpoint_manifest.csv",checkpoints)):
        run.write_csv(OUT/name,records)
    def contrast(a,b):return [r for r in pairs if r["comparison"]==a+" - "+b]
    def mean(a,b,key):return float(np.mean([r[key] for r in contrast(a,b)]))
    full=contrast(*CONTRASTS[0]);matched=contrast(*CONTRASTS[1])
    sign_checks={"Full_Tail_both_positive":all(r["delta_Tail_pp"]>0 for r in full),"Matched_Tail_both_positive":all(r["delta_Tail_pp"]>0 for r in matched),"Full_Overall_both_nonnegative":all(r["delta_Overall_pp"]>=0 for r in full),"Matched_Overall_both_nonnegative":all(r["delta_Overall_pp"]>=0 for r in matched)}
    all_positive=all(sign_checks.values())
    mixed_tail=any(r["delta_Tail_pp"]>0 for r in matched) and any(r["delta_Tail_pp"]<0 for r in matched)
    mixed_overall=any(r["delta_Overall_pp"]>0 for r in matched) and any(r["delta_Overall_pp"]<0 for r in matched)
    if all_positive:
        case="A方向条件成立；仍按净图数和限制判断证据强度"
    elif mixed_tail or mixed_overall:
        case="C：Matched跨seed方向矛盾，暂停VS方法开发；若Full优势而Matched减弱，也仅有B的幅度线索，不能证明唯一范数机制"
    elif sign_checks["Full_Tail_both_positive"] and not sign_checks["Matched_Tail_both_positive"]:
        case="B：Full正向Tail、Matched未保留双seed正向；幅度解释更可信但非唯一原因，停止本轮不追加训练"
    else:
        case="C/未充分满足A：缺少一致方向，先按逐图效应大小审视，不追加VS训练"
    groups={k:len(v) for k,v in data[(124,"Uniform-Full")]["metrics"]["effective_classes"].items()}
    lines=["# Low-VS 扰动范数—方向归因：科学实验报告","","日期：2026-10-16（Asia/Shanghai）","","## 1. 问题与完成情况","","检验在相同样本、IS供体、增强、Gumbel和候选通道下，逐样本配平GAP扰动范数后，Low-VS相对Uniform-PerSample是否仍有分类收益。本轮不是新方法，也不直接对照原H2T的batch共享随机掩码。","",f"完成8组Stage II，每组10 epochs/850更新，共6800更新；没有追加训练。方向诊断：**{case}**。下文同时列全部原始净图片数，方向成立不等于实质或显著收益。","","## 2. 模型、训练与数据协议","",f"Clean Stage I seed42 epoch184，SHA256 `{pre['checks']['124']['stage1_sha256']}`。Backbone463504参数、31层BN完全eval冻结；同一初始6500参数100类FC用于全部独立训练。未加载其他Stage II权重。","","原CIFAR-100-LT IF100训练子集10847张，真实官方TRAIN像素/标签按raw indices再次核对一致；开发验证1980张来自官方TRAIN，class1..99各20张，缺class0，索引与LT重叠0，输出保留100 logits。训练供体/接收者及扰动都不依赖开发验证。官方TEST没有读取或评估；W/b互换属于8组训练结束后的开发验证诊断。", "",f"验证组有效类别数：{groups}。Head=0..35（验证有效1..35），OfficialMedium=36..70，Tail=71..99；Many按LT n>100，Medium按20≤n≤100，Few按n<20。训练alpha分组含Head class0（36类），与验证的有效35个Head类区别。","","BS接收者、IS供体、增强与两种Gumbel采样通道均复用封存历史实现；16/64整通道；Low-VS Bottom48及小供体n<20等fallback不变。CE硬接收标签，batch128（每epoch最后batch95），原SGD momentum.9/wd.0005与原cos调度。配置lr=.1；原函数对唯一FC参数组乘lr_factor=.2，实际epoch1为.02，再cos衰减。8组相同，具体effective.yaml和epochs.csv已保存。","","## 3. 干预与正确性","","先形成两个完整硬替换δU/δL，用相同feature-map差先GAP再L2求nU/nL；公共r=min(nU,nL)。各自alpha=r/n，n≤1e−12安全取0，不放大。Matched是整个选中通道统一连续缩放，不再是原始硬替换；其余通道保持接收者。提供的norm_policy.py逐字节未变，增加的代码只保存缺失证据、保护旧文件并核查状态，Diff见evidence_recording.patch。","",f"CPU数学测试及真实GPU预检PASS。全部8组均核对850 batch的5项输入签名、两种掩码哈希和额外范数诊断哈希；参考原始NPZ还能逐batch重算诊断hash。186个Backbone状态张量逐位不变，31个BN计数器全为15640；只有FC weight/bias实际变化。16个best/final均重载评估一致；{protected}个历史文件hash不变，历史写保护及TEST guard无事件。","","## 4. 固定epoch10主结果（%）",""]
    lines+=table(["seed","arm",*GROUPS,"best epoch","best Overall"],[[r["seed"],r["arm"],*[f"{r[k]:.4f}" for k in GROUPS],r["best_epoch"],f"{r['best_Overall']:.4f}"] for r in rows])
    lines+=["","## 5. 逐图配对与两个关键比较","","wrong→right与right→wrong在相同1980张图像逐图计数；每一组的准确率差严格等于对应净图数/组样本数×100。",""]
    lines+=table(["seed","比较","wins","losses","net","ΔOverall pp","Tail net","ΔTail pp","Head net","ΔHead pp","ΔFew pp"],[[r["seed"],r["comparison"],r["wins"],r["losses"],r["net_correct"],f"{r['delta_Overall_pp']:+.4f}",r["Tail_net"],f"{r['delta_Tail_pp']:+.4f}",r["Head_net"],f"{r['delta_Head_pp']:+.4f}",f"{r['delta_Few_pp']:+.4f}"] for r in pairs])
    lines+=["","按seed等权的描述性均值：",""]
    lines+=table(["比较","ΔOverall pp","ΔHead pp","ΔTail pp","ΔFew pp"],[[a+" - "+b,*[f"{mean(a,b,'delta_'+k+'_pp'):+.4f}" for k in ("Overall","Head","Tail","Few")]] for a,b in CONTRASTS])
    lines+=["","Full与Matched是各自同环境、同预算新训练。Full差值减Matched差值不是可识别的“范数贡献百分比”：两次缩放同时改变训练增强分布、完整图范数和头部优化轨迹。","","## 6. 逐样本配平，而非均值配平","","每个seed包含108470个带增强的BS配对训练draw；相同原图会多次出现，不是108470个独立原图。浮点比较沿用提供策略容差 `1e-5+1e-4*r`，不宣称逐位完全相等。",""]
    lines+=table(["seed","draws","batch","max abs(U*-L*)","max abs(U*-r)","max abs(L*-r)","max误差/容差","失败数","U精确零","L精确零","任一≤epsilon"],[[r["seed"],r["paired_training_draws"],r["batches"],f"{r['max_pair_gap_error']:.3g}",f"{r['max_target_error_U']:.3g}",f"{r['max_target_error_L']:.3g}",f"{r['max_relative_to_tolerance']:.4f}",0,r["n_exact_zero_U"],r["n_exact_zero_L"],r["n_either_below_epsilon"]] for r in norm_checks])
    fields=("original_norm_uniform","original_norm_lowvs","matched_norm_uniform","matched_norm_lowvs","alpha_uniform","alpha_lowvs","full_map_norm_uniform_matched","full_map_norm_lowvs_matched")
    overall=[r for r in norm_summary if r["receiver_group"]=="All" and r["field"] in fields]
    lines+=["","所有epoch训练draw的分布：",""]
    lines+=table(["seed","量","均值","标准差","5%","中位数","95%","min","max"],[[r["seed"],r["field"],*[f"{r[k]:.5f}" for k in ("mean","std","q05","median","q95","min","max")]] for r in overall])
    lines+=["","Head/Tail接收者的alpha分布（训练含class0；逐组频次不是验证集有效类数）：",""]
    grouped=[r for r in norm_summary if r["receiver_group"] in ("Head","Tail") and r["field"].startswith("alpha_")]
    lines+=table(["seed","接收组","alpha","draws","均值","5%","中位数","95%","alpha=0%","alpha=1%"],[[r["seed"],r["receiver_group"],r["field"],r["count"],*[f"{r[k]:.5f}" for k in ("mean","q05","median","q95")],f"{100*r['fraction_zero']:.3f}",f"{100*r['fraction_one']:.3f}"] for r in grouped])
    lines+=["","完整分位数、每epoch及分组alpha见norm_distribution.csv、alpha_by_epoch_group.csv；逐batch最大误差见各组batch_norm_diagnostics.csv；逐样本原值及接收者/供体索引在各seed/Uniform-Full/norm_diagnostics.npz。另三组的诊断哈希逐batch一致，因此共用这份范数原始证据。完整8×8特征图L2并未被等化，已额外测量而未把它当作调参目标。","","## 7. 分类头W/b交叉组合","","以下为训练结束后，用固定开发验证特征将两套FC的W/b交叉组合。U代表相应Uniform组，L代表相应Low-VS组；Full和Matched分开比较。与原classifier_hybrids.csv逐行复核，底层float64线性运算也检查了相对训练时float32原模型的预测变化。",""]
    lines+=table(["seed","比较","组合","Overall","Head","Tail","Few","原生预测差异图数"],[[r["seed"],r["comparison"],r["hybrid"],*[f"{r[k]:.4f}" for k in ("Overall","Head","Tail","Few")],r["native_float32_prediction_disagreements"] if r["native_float32_prediction_disagreements"] is not None else "不适用"] for r in hybrid_checks])
    tail_effects=[r for r in hybrid_effects if r["group"]=="Tail"]
    lines+=["","Tail的条件参数干预（相对U_W+U_b，单位pp）：",""]
    lines+=table(["seed","比较","完整L变化","只换b@U_W","只换W@U_b","非可加剩余"],[[r["seed"],r["comparison"],*[f"{r[k]:+.4f}" for k in ("total_change_pp","change_b_at_UW_pp","change_W_at_Ub_pp","nonadditive_residual_pp")]] for r in tail_effects])
    lines+=["","若保持U_W仅换L_b主要复制Tail差值，分类器偏置需优先审视；但另一路径和非可加剩余必须同时看。W和b共同训练，分类边界及argmax是非线性的，互换模型可能不处于其训练轨迹，不可把条件替换效果分解成唯一训练因果贡献。完整Overall/Head/Few和参数L2差见hybrid_parameter_effects.csv。","","## 8. 预先情形与机制边界",""]
    lines+=table(["严格方向描述（不替代效应大小）","是否"],[[k,v] for k,v in sign_checks.items()])
    lines+=["",case+"。未事后为“明显反向”或“很弱”新增数值阈值；请同时检查两个seed净图数、Head损失和best敏感性。", "", "- 实际干预证据：相同训练draw下逐样本GAP范数在预声明容差内相同，alpha仅衰减；这一点可由原始NPZ和850个batch核查。", "- 模型性能证据：只支持本表固定epoch10、99类开发验证协议下的差异；n=2不建立发表级显著性。", "- 机制解释：若配平后正向差异保留，GAP幅度一个标量不足以完全解释；剩余还包括通道身份、完整空间结构、扰动方向、类别条件缩放分布和分类头优化差异，不能直接称“尾类语义通道有效”。", "- 若Matched优势消失/反向，幅度解释更可信，但缩放改变增强分布，不能证明其为唯一原因。若仅少数图像或跨seed反向，应暂停VS算法开发。", "- Backbone和BN始终未变，本轮不存在训练得到更强Backbone表征的证据。", "", "## 9. best敏感性与运行开销", ""]
    for a,b in CONTRASTS[:2]:
        for seed in run.SEEDS:
            sa=data[(seed,a)]["summary"];sb=data[(seed,b)]["summary"]
            main=sa["last"]["Overall"]-sb["last"]["Overall"];best=sa["best_Overall"]-sb["best_Overall"]
            lines.append(f"- seed{seed} {a}−{b}：epoch10 {main:+.4f} pp，各自best {best:+.4f} pp，方向反转={main*best<0}。")
    lines+=[""]+table(["seed","arm","总工作秒","训练循环秒","allocated峰值MiB","reserved峰值MiB"],[[seed,arm,f"{data[(seed,arm)]['summary']['elapsed_seconds']:.2f}",f"{sum(float(e['training_seconds']) for e in data[(seed,arm)]['epochs']):.2f}",f"{max(float(e['peak_allocated_mib']) for e in data[(seed,arm)]['epochs']):.1f}",f"{max(float(e['peak_reserved_mib']) for e in data[(seed,arm)]['epochs']):.1f}"] for seed in run.SEEDS for arm in run.ARMS])
    lines+=["","总工作秒含加载、校验、保存和重载。训练循环秒包括增强、双分支冻结前向、四候选构建/诊断及分类器更新，不含epoch验证；所有组使用同样运行引擎和诊断，不表示生产优化后的纯算法耗时。显存为PyTorch分配器统计，不含Windows/驱动/CUDA context。effective.yaml保留历史model_dir/name元数据但resume清空；实际输出都显式指向本轮results并有运行时写保护。","","## 10. 局限与异常","","验证只有99类且缺class0，保留100 logits不能补足class0召回；不能直接和论文官方100类TEST准确率比较。相同1980张官方TRAIN留出已经多轮用于开发，Clean Stage I的epoch184也由该留出选择，存在开发选择偏差；没有独立终检。两个Stage II种子共用一个Stage I和同一验证图像，不把3960次重复预测当作独立观测。Overall一张图0.050505 pp、Tail一张0.172414 pp、Few一张0.166667 pp；弱差值不夸大。","","训练失败=0；配对失败=0；范数容差失败=0；TEST访问尝试=0；历史写入尝试=0；历史文件变化=0。训练无重跑、无续训、无删除失败目录。run_experiment.py和norm_policy.py在预检之后保持锁定；模型权重和原始数据未被改写。完整stdout在console/，执行状态/错误记录在run_registry.json。","","## 11. 环境、复现与文件","","```json",json.dumps(pre["environment"],ensure_ascii=False,indent=2),"```","","从实际本地项目根目录运行；已有结果不可覆盖：","","```powershell",f"$py = '{sys.executable}'","& $py -B H2T_repro_audit/lowvs_norm_direction/norm_policy.py","& $py -B H2T_repro_audit/lowvs_norm_direction/run_experiment.py preflight"]
    for seed in run.SEEDS:
        for arm in run.ARMS:lines.append(f"& $py -B H2T_repro_audit/lowvs_norm_direction/run_experiment.py train --seed {seed} --arm {arm}")
    lines+=["& $py -B H2T_repro_audit/lowvs_norm_direction/run_experiment.py report","& $py -B H2T_repro_audit/lowvs_norm_direction/analyze_results.py","```","","本报告所在results目录保留：preflight.json、dataset_protocol.json、preregistered_decision_rules.json、historical_manifest.json、初始源码provided_source/和evidence_recording.patch；epoch10.csv/epoch10_verified.csv、paired.csv/paired_verified.csv、per_class.csv、classifier_hybrids.csv/classifier_hybrids_verified.csv、hybrid_parameter_effects.csv、scientific_summary.json、norm_distribution.csv、norm_matching_verification.csv、alpha_by_epoch_group.csv、epochs_all.csv、checkpoint_manifest.csv、final_integrity.json。每个seed/arm下epochs.csv、batch_signatures.csv、batch_norm_diagnostics.csv、validation_epoch10.npz、config.json、effective.yaml、summary.json和best/final.pth；每seed Uniform-Full下另存逐样本norm_diagnostics.npz。","","## 12. 停止","","一次协议已完整执行；无新α、K、供体策略、网络、损失、Stage I、额外seed或官方TEST。当前仅保守记录范数控制与分类头诊断，不认定发现语义机制。结果由用户提供给ChatGPT复核后，再决定研究方向；本轮到此停止。",""]
    lines.insert(lines.index("## 10. 局限与异常")+2,"报告脚本准备有两次仅文本匹配的AssertionError，两次均未写入或影响训练。AST检查确认原Markdown换行正确，无需修复；之后仅完善了新报告的条件说明。详见analysis_preparation_notes.json。")
    lines.insert(lines.index("## 10. 局限与异常")+2,"第一次离线分析已完成全部核验并写出报告，但外层日志进程在中文stdout处遇到GBK/UTF-8编码冲突（UnicodeEncodeError）。保留失败日志，统一PYTHONIOENCODING=utf-8后仅重跑一次离线汇总；8组训练无重跑。NPZ诊断一次读入内存，避免重复解压，数值与判据不变。详见analysis_stdout_error.json。")
    full_gain=mean("LowVS-Full","Uniform-Full","delta_Overall_pp")
    full_tail=mean("LowVS-Full","Uniform-Full","delta_Tail_pp")
    matched_gain=mean("LowVS-Matched","Uniform-Matched","delta_Overall_pp")
    matched_tail=mean("LowVS-Matched","Uniform-Matched","delta_Tail_pp")
    summary_line=(f"固定epoch10均值：Full的Low-VS−Uniform为Overall {full_gain:+.4f} pp、Tail {full_tail:+.4f} pp；Matched为Overall {matched_gain:+.4f} pp、Tail {matched_tail:+.4f} pp。Matched逐seedOverall净图数={[r['net_correct'] for r in matched]}、Tail净图数={[r['Tail_net'] for r in matched]}，必须保留这种变号，不用均值掩盖不稳定性。")
    lines.insert(lines.index("## 2. 模型、训练与数据协议"),summary_line+"\n")
    if all(r["change_b_at_UW_pp"]==0 for r in tail_effects) and all(abs(r["change_W_at_Ub_pp"]-r["total_change_pp"])<1e-10 for r in tail_effects):
        message="实际W/b诊断：两个seed×Full/Matched共四个对比中，保持U_W只换L_b的Tail变化均为0；保持U_b只换L_W均复制对应完整L的Tail差值。本轮不支持“最终bias单独替换解释Tail差异”。变化在这项条件干预中跟随W，但W可编码类别尺度/边界等，不等于语义方向被证明；W/b共同训练，也不能排除bias在训练路径中的间接作用。"
        lines.insert(lines.index("## 8. 预先情形与机制边界"),message+"\n")
    lines.insert(lines.index("```powershell")+1, "$env:PYTHONIOENCODING = 'utf-8'")
    (OUT/"LOWVS_NORM_DIRECTION_SCIENTIFIC_REPORT.md").write_text("\n".join(lines),encoding="utf-8")
    original=read(OUT/"scientific_summary.json")
    original.update({"direction_sign_checks":sign_checks,"interpretation_case":case,"per_example_norm_verification":norm_checks,"verified_pairwise":pairs,"hybrid_conditional_effects":hybrid_effects})
    run.write_json(OUT/"scientific_summary.json",original)
    registry=read(OUT/"run_registry.json")
    assert len(registry)==10 and all(r["status"]=="completed" for r in registry)
    assert run.verify_history()==protected
    run.old.check_test_guard();assert not run.HISTORY_EVENTS
    final={"passed":True,"verified_at":stamp(),"runs":8,"updates":6800,"protected_historical_files":protected,"norm_tolerance_failures":0,"per_seed_norm_verification":norm_checks,"paired_batch_count_each":850,"paired_fields":8,"checkpoint_files_verified":16,"frozen_backbone_and_BN":True,"policy_bytewise_unchanged":True,"training_core_unchanged_after_preflight":True,"TEST_events":run.old.AUDIT_EVENTS,"historical_write_events":run.HISTORY_EVENTS,"details":integrity,"source_sha256":{p.name:run.sha(p) for p in run.HERE.glob("*.py")}}
    run.write_json(OUT/"final_integrity.json",final)
    run.write_json(OUT/"STOP_COMPLETED.json",{"completed_at":stamp(),"runs":8,"updates":6800,"further_training_started":False,"interpretation":case})
    print('FINAL INTEGRITY PASS: 8 runs, 6800 updates, 850/850 pairing, per-example GAP matching, 16 checkpoints',flush=True)
    print('DIRECTION CHECKS',json.dumps(sign_checks,ensure_ascii=False),flush=True)
    print('CASE',case,flush=True)
    for r in pairs[:2]+pairs[4:6]:print(json.dumps(r,ensure_ascii=False),flush=True)
    print('REPORT',OUT/'LOWVS_NORM_DIRECTION_SCIENTIFIC_REPORT.md',flush=True)

if __name__=="__main__":main()
