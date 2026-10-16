"""Post-training integrity checks and Chinese scientific report; no training."""
from pathlib import Path
import csv,json,itertools,sys
import numpy as np
import run_gu_lowvs as run
from run_safeguards import protect,stamp

OUT=run.RESULTS
GROUPS=("Overall","Head","OfficialMedium","Tail","Many","Medium","Few","Macro")

def read(path):return json.loads(Path(path).read_text(encoding="utf-8"))
def mdtable(headers,rows):
    return ["|"+"|".join(headers)+"|","|"+"|".join(["---"]*len(headers))+"|"]+["|"+"|".join(map(str,r))+"|" for r in rows]

def main():
    run.base.forbid_official_test()
    protect(run.PARENT,OUT,run.HISTORY_EVENTS)
    assert read(OUT/"preflight.json")["status"]=="PASS"
    assert run.source_hashes()==read(OUT/"preflight.json")["new_source_sha256"]
    assert (run.HERE/"utility_policy.py").read_bytes()==(OUT/"provided_source/utility_policy.py").read_bytes()
    reference=read(OUT/"reference_bank_provenance.json")
    assert run.sha(run.BANK)==reference["bank_sha256"]
    protected=run.verify_history()
    epoch10=[];epochs=[];counts=[];pair_rows=[];class_rows=[];checkpoints=[];arms={};integrity=[];shuffle_rows=[]
    labels=None
    for seed in run.SEEDS:
        baseline=None
        first_ref=None
        for arm in run.ARMS:
            folder=OUT/f"seed{seed}"/arm
            summ=read(folder/"summary.json")
            assert summ["status"]=="complete" and summ["updates"]==850
            assert summ["frozen_backbone"] and summ["backbone_state_tensors_bitwise_unchanged"]==186
            assert summ["BN_counters"]==[15640]
            assert summ["validation_reload_exact"] and not summ["official_TEST_used"]
            assert not summ["official_TEST_events"] and not summ["historical_write_events"]
            assert all(v>0 for v in summ["classifier_parameter_max_abs_changes"].values())
            sig=run.read_csv(folder/"batch_signatures.csv")
            ep=run.read_csv(folder/"epochs.csv")
            assert len(ep)==10 and len(sig)==850
            assert [int(e["epoch"]) for e in ep]==list(range(1,11))
            if baseline is None:baseline=sig
            matched={k:sum(a[k]==b[k] for a,b in zip(baseline,sig)) for k in run.KEYS}
            matched_masks=sum(a["lowvs_mask_sha256"]==b["lowvs_mask_sha256"] for a,b in zip(baseline,sig))
            assert all(v==850 for v in matched.values()) and matched_masks==850
            if first_ref is None:first_ref=ep[0]["reference_gradient_sha256"]
            assert ep[0]["reference_gradient_sha256"]==first_ref
            act=np.load(folder/"actions.npz",allow_pickle=False)
            selected,applied=act["selected"],act["applied"]
            assert selected.shape==applied.shape==(10,run.N)
            assert np.isin(selected,[0,1,2]).all() and np.isin(applied,[0,1,2]).all()
            for i,e in enumerate(ep):
                nr=np.bincount(applied[i],minlength=3)
                assert np.array_equal(nr,[int(e[k]) for k in ("alpha0_count","alphaHalf_count","alphaFull_count")])
                assert nr.sum()==run.N
                if arm=="GU-Shuffled":
                    for start in range(0,run.N,run.BATCH):
                        stop=min(start+run.BATCH,run.N)
                        assert np.array_equal(np.bincount(selected[i,start:stop],minlength=3),np.bincount(applied[i,start:stop],minlength=3))
                else:assert np.array_equal(selected[i],applied[i])
                counts.append({"seed":seed,"arm":arm,"epoch":i+1,**{f"alpha_{a}_count":int(nr[k]) for k,a in enumerate((0,0.5,1))},"mean_alpha":float(applied[i].mean()/2),"action_changed_by_permutation_pct":float((selected[i]!=applied[i]).mean()*100)})
            if arm=="GU-Shuffled":assert summ["shuffle_histogram_checks"]==850
            assert summ["lowvs_masks_paired"]==(0 if arm=="LowVS-Full" else 850)
            vals=np.load(folder/"validation_epoch10.npz",allow_pickle=False)
            y,p,z=vals["labels"],vals["predictions"],vals["logits"]
            assert z.shape==(1980,100) and np.array_equal(p,z.argmax(1))
            assert np.array_equal(np.bincount(y,minlength=100),np.r_[0,np.repeat(20,99)])
            if labels is None:labels=y.copy()
            else:assert np.array_equal(y,labels)
            metrics=run.base.sys.modules["common"].metrics(p,y)
            for key in GROUPS:assert abs(metrics[key]-summ["last"][key])<1e-10
            for kind in ("best","final"):
                cp=folder/f"checkpoints/{kind}.pth"
                assert run.sha(cp)==summ["checkpoint_sha256"][kind]
                checkpoints.append({"seed":seed,"arm":arm,"kind":kind,"path":cp.relative_to(run.HERE).as_posix(),"sha256":run.sha(cp)})
            epoch10.append({"seed":seed,"arm":arm,**{k:summ["last"][k] for k in GROUPS},"best_epoch":summ["best_epoch"],"best_Overall":summ["best_Overall"],"elapsed_seconds":summ["elapsed_seconds"]})
            epochs.extend(ep)
            arms[(seed,arm)]={"summary":summ,"pred":p.copy(),"actions":applied.copy(),"selected":selected.copy(),"metrics":metrics,"epochs":ep}
            integrity.append({"seed":seed,"arm":arm,"updates":850,"matched_fields":matched,"matched_LowVS_masks":matched_masks,"frozen_state_tensors":186,"BN_counters":[15640],"checkpoint_reload_exact":True,"classifier_deltas":summ["classifier_parameter_max_abs_changes"]})
        for a,b in itertools.combinations(run.ARMS,2):
            if a=="GU-LowVS":compare,control=a,b
            elif b=="GU-LowVS":compare,control=b,a
            else:compare,control=b,a
            pa=arms[(seed,compare)]["pred"];pb=arms[(seed,control)]["pred"]
            win=(pb!=labels)&(pa==labels);lose=(pb==labels)&(pa!=labels)
            row={"seed":seed,"comparison":f"{compare} - {control}","wins":int(win.sum()),"losses":int(lose.sum()),"net_correct":int(win.sum()-lose.sum())}
            for group in GROUPS:
                active=range(1,100) if group in ("Overall","Macro") else arms[(seed,compare)]["metrics"]["effective_classes"][group]
                mask=np.isin(labels,list(active));n=int(mask.sum())
                net=int(win[mask].sum()-lose[mask].sum())
                delta=arms[(seed,compare)]["metrics"][group]-arms[(seed,control)]["metrics"][group]
                assert abs(delta-net/n*100)<1e-10
                row[f"d_{group}_pp"]=delta;row[f"{group}_net"]=net;row[f"{group}_samples"]=n
            pair_rows.append(row)
            for c in range(1,100):
                mask=labels==c
                class_rows.append({"seed":seed,"comparison":row["comparison"],"class_id":c,"samples":20,"wins":int(win[mask].sum()),"losses":int(lose[mask].sum()),"net_correct":int(win[mask].sum()-lose[mask].sum())})
        direct=arms[(seed,"GU-LowVS")]["actions"];shuffled=arms[(seed,"GU-Shuffled")]["actions"]
        for i in range(10):
            a=np.bincount(direct[i],minlength=3);b=np.bincount(shuffled[i],minlength=3)
            shuffle_rows.append({"seed":seed,"epoch":i+1,**{f"GU_alpha{k}_count":int(a[k]) for k in range(3)},**{f"Shuffled_alpha{k}_count":int(b[k]) for k in range(3)},"cross_arm_histogram_exact":bool(np.array_equal(a,b))})
    def pairs(control):return [r for r in pair_rows if r["comparison"]==f"GU-LowVS - {control}"]
    def mean(control,metric):return float(np.mean([r[metric] for r in pairs(control)]))
    full=pairs("LowVS-Full");shuf=pairs("GU-Shuffled")
    endpoint_ok=all(max(arms[(seed,"GU-LowVS")]["summary"]["last"][k] for k in ("alpha0_pct","alphaFull_pct"))<95 for seed in run.SEEDS)
    text_conditions={"each_seed_Overall_vs_Full_nonnegative":all(r["d_Overall_pp"]>=0 for r in full),"mean_Tail_vs_Full_nonnegative":mean("LowVS-Full","d_Tail_pp")>=0,"mean_Overall_vs_Shuffled_positive":mean("GU-Shuffled","d_Overall_pp")>0,"mean_Head_vs_Shuffled_at_least_minus_0_15pp":mean("GU-Shuffled","d_Head_pp")>=-0.15,"no_endpoint_collapse_final_epoch":endpoint_ok}
    source=read(OUT/"scientific_summary.json")
    text_go=all(text_conditions.values())
    expected_source=(text_go and mean("LowVS-Full","d_Overall_pp")>0 and all(max(arms[(seed,"GU-LowVS")]["summary"]["last"][k] for k in ("alpha0_pct","alphaHalf_pct","alphaFull_pct"))<95 for seed in run.SEEDS))
    assert bool(source["go_exploratory"])==expected_source
    epochs_all=counts
    gu_counts=[r for r in counts if r["arm"]=="GU-LowVS"]
    all_epoch_endpoint_collapse=[r for r in gu_counts if max(r["alpha_0_count"],r["alpha_1_count"])/run.N>=0.95]
    for filename,data in (("epoch10_verified.csv",epoch10),("epochs_all.csv",epochs),("paired_verified.csv",pair_rows),("per_class_verified.csv",class_rows),("alpha_schedule.csv",counts),("cross_arm_action_histograms.csv",shuffle_rows),("checkpoint_manifest.csv",checkpoints)):
        run.save_csv(OUT/filename,data)
    group_counts={k:len(v) for k,v in arms[(124,"GU-LowVS")]["metrics"]["effective_classes"].items()}
    lines=["# GU-LowVS V1 科学实验报告","", "日期：2026-10-16（Asia/Shanghai）", "", "## 1. 命题、范围与预注册结论", "", "检验：固定 Low-VS Bottom48→随机16个完整8×8通道后，按干净 TRAIN 类别均衡风险梯度的对齐度，逐样本选择 α∈{0,0.5,1}，能否在本轮开发验证协议下优于固定完全替换、固定半强度和动作打乱对照。", "",f"**任务文字判据：{'GO' if text_go else 'NO-GO'}；提供的执行器数值判据：{'GO' if source['go_exploratory'] else 'NO-GO'}。** 判据来自训练前的 `preregistered_decision_rules.json`；无事后调整。", "", "数值执行器额外要求对 Full 平均 Overall 严格为正，并将最终 epoch 任意单一动作≥95%视为塌缩，比任务文字略严格；此处并列报告两者。Head 的“明显下降”按提供代码预先使用的 −0.15 pp 解释，同时展示各 seed 的实际值。", "", "## 2. 锁定模型、数据与算法", "",f"Stage I：Clean seed42 epoch184，SHA256 `{reference['stage1_sha256']}`；参考库 SHA256 `{reference['bank_sha256']}`。同一 Stage I 的 Backbone、BN 和初始100类分类头用于8组独立初始化，没有加载其他 Stage II 权重。", "", "真实 CIFAR-100 官方 TRAIN：原 LT 子集10847张、100类、IF100。逐个 raw index 核对缓存像素与标签和官方 TRAIN 完全一致。开发验证是官方 TRAIN 留出1980张，class1..99各20张，缺class0；与LT索引重叠0；始终输出100维logits。参考库仅为LT训练图像的官方 eval transform 特征，含class0的500张训练图像，没有加入开发验证图像。详见 `dataset_protocol.json` 和 `reference_bank_provenance.json`。", "",f"有效类别数：{group_counts}。Head=0..35（有效1..35），OfficialMedium=36..70，Tail=71..99；Many按LT训练n>100，Medium按20≤n≤100，Few按n<20。OfficialMedium与频次Medium不是同一集合。验证各类等量，因此Overall=99类Macro。", "", "原始BS/IS、增强、Gumbel和VS/小样本回退均复用冻结历史实现。ρ=.25对应预选16/64完整通道；α控制这些选中通道替换的幅度，α=0时实际不改变特征。每组10 epochs、850更新、batch128（每epoch最后一batch95），SGD lr=.1/momentum=.9/wd=.0005及原cos调度，硬接收者标签与CE。仅训练6500参数分类器，Backbone463504参数与全部BN固定。", "", "每个epoch开头，以当前分类器在全部训练参考特征上计算100类别等权CE解析梯度G_ref；每batch各α的解析梯度gα取当前分类器值，路由no_grad，argmax平分时由固定顺序优先α0。score=<G_ref,gα>，仅选定特征的原CE参与反向传播。α0/1端点逐位一致。GU-Shuffled在自己当前分类器计算动作后，按独立确定性置换打乱样本分配；每batch打乱前后动作频数严格相同。", "", "## 3. 正确性与执行证据", "",f"CPU解析梯度与autograd对照、GPU预检、TRAIN参考库检查全部PASS。8组共6800次optimizer.step；相对各seed新训Full基准，索引、增强图像、冻结特征、VS分数、Gumbel五字段各850/850一致；额外检查Low-VS实际掩码850/850一致。186个Backbone状态张量逐位不变，31个BN计数器均保持15640；分类器weight/bias均有实际更新；16个best/final checkpoint均重载验证并与原记录一致。", "",f"{protected}个历史文件逐字节hash核对不变；运行时历史写保护及官方TEST guard均无事件。未重新训练Stage I、未运行官方TEST、未执行任何版本控制命令。输出只在本轮results目录。", "", "`utility_policy.py`与提供版本逐字节不变。新增代码只补充结果路径约束、原始TRAIN核对、完整配对/掩码/动作频数断言、状态/耗时/配置/错误持久化，Diff见 `minimal_interface_and_evidence.patch`。没有改变参考风险、α候选、效用评分、VS或优化器。", "", "## 4. 固定第10epoch主结果（%）", ""]
    lines+=mdtable(["seed","arm",*GROUPS,"best epoch","best Overall"],[[r["seed"],r["arm"],*[f"{r[k]:.4f}" for k in GROUPS],r["best_epoch"],f"{r['best_Overall']:.4f}"] for r in epoch10])
    lines+=["", "## 5. GU与三个控制的逐图配对得失", "", "差值=GU−控制；wrong→right与right→wrong按完全相同1980张验证图像配对。Overall/Tail/Head净数与准确率差已逐项严格核验（Tail580张、Head700张、Few600张）。", ""]
    wanted=[r for r in pair_rows if r["comparison"].startswith("GU-LowVS - ")]
    lines+=mdtable(["seed","比较","wrong→right","right→wrong","净图数","ΔOverall pp","Tail净数","ΔTail pp","Head净数","ΔHead pp","ΔFew pp"],[[r["seed"],r["comparison"],r["wins"],r["losses"],r["net_correct"],f"{r['d_Overall_pp']:+.4f}",r["Tail_net"],f"{r['d_Tail_pp']:+.4f}",r["Head_net"],f"{r['d_Head_pp']:+.4f}",f"{r['d_Few_pp']:+.4f}"] for r in wanted])
    lines+=["", "两seed均值（每seed等权，不将3960个重复验证观测当独立样本）：", ""]
    lines+=mdtable(["控制","ΔOverall pp","ΔHead pp","ΔTail pp","ΔFew pp","两个seed Overall方向"],[[c,*[f"{mean(c,'d_'+k+'_pp'):+.4f}" for k in ("Overall","Head","Tail","Few")],'/'.join(f"{r['d_Overall_pp']:+.4f}" for r in pairs(c))] for c in ("LowVS-Full","LowVS-Half","GU-Shuffled")])
    lines+=["", "## 6. 动作比例与退化诊断", "", "下表逐epoch报告选择α0/0.5/1的比例。Full恒为0/0/100%，Half恒为0/100/0%；原始整数频数见 `alpha_schedule.csv`、各组`epochs.csv`和`actions.npz`。", ""]
    lines+=mdtable(["seed","epoch","GU α0%","GU α.5%","GU α1%","Shuffled α0%","Shuffled α.5%","Shuffled α1%","Shuffled分配实际改变%"],[[seed,i+1,*[f"{100*np.mean(arms[(seed,a)]['actions'][i]==k):.3f}" for a in ("GU-LowVS","GU-Shuffled") for k in range(3)],f"{100*np.mean(arms[(seed,'GU-Shuffled')]['selected'][i]!=arms[(seed,'GU-Shuffled')]['actions'][i]):.3f}"] for seed in run.SEEDS for i in range(10)])
    lines+=["",f"按端点动作≥95%的预设诊断，GU任意epoch端点塌缩记录数量={len(all_epoch_endpoint_collapse)}。最终epoch文字判据：{endpoint_ok}。统计显示是否退化见实际比例，不能把α0当作仍执行完全替换。", "", "GU-Shuffled每batch在自身计算的动作和打乱后的动作之间850/850频数一致；但GU与Shuffled各自训练使分类器、参考梯度和动作直方图随epoch分离，跨组不构成完全强度匹配。逐epoch跨组直方图见 `cross_arm_action_histograms.csv`；这是该对照本身的限制。日志`mean_predicted_gain`是置换前argmax相对α0的预测分数优势，在Shuffled组不能当成置换后实际分配的效用。", "", "## 7. 预注册筛选与解释", ""]
    lines+=mdtable(["文字判据数值化","通过"],[[k,v] for k,v in text_conditions.items()])
    if mean("GU-Shuffled","d_Overall_pp")>0:
        lines+=["", "GU相对Shuffled的平均Overall方向为正，提供按样本分配的有限开发集线索；仍须同时看每seed方向、Head/Tail损失和跨组强度直方图不同，不能据此确认因果泛化收益。"]
    else:lines+=["", "GU未优于Shuffled的平均Overall：本轮缺少支持“按样本效用匹配”带来收益的证据，即使相对某个固定强度控制有变化，也不足以证明该效用路由有效。"]
    lines+=["",f"GU相对Half平均Overall {mean('LowVS-Half','d_Overall_pp'):+.4f} pp。Half控制显示固定柔化的效果；GU与Half不同并不能排除强度分布改变的解释，且本轮Shuffled也不是跨组严格直方图匹配。", "", "## 8. best敏感性、耗时与显存", "", "best只展示开发验证选出的辅助结果，不替代epoch10，不筛seed或epoch挽救规则。", ""]
    for c in ("LowVS-Full","LowVS-Half","GU-Shuffled"):
        for seed in run.SEEDS:
            a=arms[(seed,"GU-LowVS")]["summary"];b=arms[(seed,c)]["summary"]
            d=a["last"]["Overall"]-b["last"]["Overall"];bd=a["best_Overall"]-b["best_Overall"]
            lines.append(f"- seed{seed} GU−{c}：epoch10 {d:+.4f} pp，双方各自best {bd:+.4f} pp；方向反转={d*bd<0}。")
    lines+=[""]+mdtable(["seed","arm","总进程工作秒","训练循环秒","相对Full额外秒","峰值allocated MiB","峰值reserved MiB"],[[seed,arm,f"{arms[(seed,arm)]['summary']['elapsed_seconds']:.2f}",f"{sum(float(e['training_seconds']) for e in arms[(seed,arm)]['epochs']):.2f}",f"{arms[(seed,arm)]['summary']['elapsed_seconds']-arms[(seed,'LowVS-Full')]['summary']['elapsed_seconds']:+.2f}",f"{max(float(e['peak_allocated_mib']) for e in arms[(seed,arm)]['epochs']):.1f}",f"{max(float(e['peak_reserved_mib']) for e in arms[(seed,arm)]['epochs']):.1f}"] for seed in run.SEEDS for arm in run.ARMS])
    lines+=["", "总进程工作秒从train入口起计，包含加载、校验、验证、保存和重载；训练循环秒含增强、冻结双分支前向、效用/签名检查和分类器更新，不含epoch验证。Full/Half也计算每epoch参考梯度以使用同一运行引擎，因此差值不是相对于完全删掉参考梯度计算的生产基线；少量耗时差可能是系统噪声。", "", "## 9. 历史背景与证据限制", "", "历史No-Fusion、Uniform和RAVS不是本轮新增的严格配对对照；不把它们混入本轮门槛。可只读追溯`../ravs_v1/results/all_arms_epoch10.csv`和历史科学报告。", "", "1. G_ref是有限LT训练参考库的类别等权经验风险梯度，参考样本也参与分类器训练；不是独立验证梯度或未见总体梯度。", "2. 一阶泰勒式需同一点梯度及小步长；本实现G_ref仅epoch开头刷新、gα逐batch更新，epoch内参考方向逐渐陈旧。真实SGD还有momentum和weight decay，均不在效用评分里；逐样本贪心也不等价于最优batch或长期训练。", "3. 评分可全负，此时argmax只是选择相对较高的代理值，不保证降低真实风险。即时分数不等于未来测试效果。", "4. 开发验证1980张被多轮研究反复使用，且Clean Stage I本身用此TRAIN留出选择epoch184，存在开发集方法选择偏差；并非独立最终测试。没有使用官方TEST规避不了该局限。", "5. class0没有验证样本，100维输出中预测成class0仍算错，但无法衡量class0召回；99类Overall不可直接对照H2T论文官方100类TEST准确率。", "6. 两个Stage II种子共享同一Stage I及同一批验证图像；不宣称n=2统计显著，不把3960条预测当独立样本。Overall一张图对应0.050505 pp，Tail一张对应0.172414 pp，Few一张对应0.166667 pp。", "7. 只训练分类头，不能声称学习到更好Backbone表征。自适应α与已有比例自适应思想可能有概念重叠，本轮不作方法新颖性结论。", "", "## 10. 失败、异常、环境和复现", ""]
    registry=read(OUT/"run_registry.json")
    assert len(registry)==11 and all(x["status"]=="completed" for x in registry)
    lines+=["训练失败=0；配对失败=0；TEST访问尝试=0；历史写入尝试=0；历史文件hash变化=0；Backbone/BN变化=0。没有重试、删除失败目录、自动续训、追加种子或更改α。首次静态检查尚未运行即补充的接口/证据修改有完整提供源码快照和Diff。", "", "环境：", "", "```json",json.dumps(read(OUT/"preflight.json")["environment"],ensure_ascii=False,indent=2),"```", "", "在本地项目根目录使用以下已实际执行的命令；输出目录存在时不得直接重复训练：", "", "```powershell", "$py = 'D:\\app\\minicondaaa\\envs\\xiaoju\\python.exe'", "& $py -B H2T_repro_audit/gu_lowvs/test_utility_policy.py", "& $py -B H2T_repro_audit/gu_lowvs/run_gu_lowvs.py preflight", "& $py -B H2T_repro_audit/gu_lowvs/run_gu_lowvs.py prepare"]
    for seed in run.SEEDS:
        for arm in run.ARMS:lines.append(f"& $py -B H2T_repro_audit/gu_lowvs/run_gu_lowvs.py train --seed {seed} --arm {arm}")
    lines+=["& $py -B H2T_repro_audit/gu_lowvs/run_gu_lowvs.py report", "& $py -B H2T_repro_audit/gu_lowvs/analyze_gu_results.py", "```", "", "## 11. 原始文件与停止", "", "本报告所在results目录：`epoch10.csv`/`epoch10_verified.csv`、`paired.csv`/`paired_verified.csv`、`per_class.csv`/`per_class_verified.csv`、`epochs_all.csv`、`alpha_schedule.csv`、`cross_arm_action_histograms.csv`、`scientific_summary.json`、`final_integrity.json`、`checkpoint_manifest.csv`；参考库`balanced_clean_train_bank.npz`及来源JSON；`console/`完整stdout/错误输出；`provided_source/`初始源码和补丁。各seed/arm目录有effective.yaml、config.json、epochs.csv、batch_signatures.csv、actions.npz、validation_epoch10.npz、train.log、summary.json和checkpoints/best.pth/final.pth。", "", f"{'达到探索门槛仅表示可以由用户评议下一步，不证明效用规则已有效。' if text_go else '未达到预注册探索门槛，停止GU-LowVS V1，不搜索α、参考风险、epoch或更多种子。'}本轮至此停止，后续由用户将产物提供给ChatGPT复核；没有自动启动其他训练。", ""]
    historical_rows=run.read_csv(run.RAVS/"results/all_arms_epoch10.csv")
    historical_no_fusion=[r for r in historical_rows if r["arm"]=="No-Fusion"]
    assert len(historical_no_fusion)==2
    background=["", "以下历史No-Fusion仅作背景，不是本轮严格配对控制，不计算跨轮逐图因果差值：", ""]
    background+=mdtable(["历史seed","Overall","Head","Tail","Many","Medium","Few"],[[r["seed"],*[f"{float(r[k]):.4f}" for k in ("Overall","Head","Tail","Many","Medium","Few")]] for r in historical_no_fusion])
    lr_values=[float(e["lr"]) for e in arms[(124,"LowVS-Full")]["epochs"]]
    assert all([float(e["lr"]) for e in item["epochs"]]==lr_values for item in arms.values())
    background+=["",f"8组实际epoch1..10学习率完全相同：{lr_values}。有效配置保留历史YAML的model_dir/name描述字段，但resume已清空；所有实际保存路径由本轮执行器显式指定在results/seed/arm，历史写保护同时生效。", "", "显存列为本进程PyTorch分配器allocated/reserved峰值，不含Windows桌面、驱动及CUDA context等整机占用。", ""]
    position=lines.index("## 9. 历史背景与证据限制")
    lines[position:position]=background
    note="报告脚本准备时有一次纯文本替换锚点不匹配的AssertionError；当次没有写入源文件或修改训练，随后仅修正报告生成脚本，训练无失败、无重跑。记录见analysis_preparation_notes.json。"
    lines.insert(lines.index("## 10. 失败、异常、环境和复现")+2,note)
    report="\n".join(lines)
    # Emit normal Windows PowerShell interpreter path, without JSON escaping.
    report=report.replace("D:\\app\\minicondaaa\\envs\\xiaoju\\python.exe",str(Path(sys.executable)))
    report=report.replace("## 2. 锁定模型、数据与算法",'本轮固定epoch10：GU相对Full平均Overall −0.7323 pp、Head +3.7143 pp、Tail −4.0517 pp；相对Shuffled平均Overall −0.4798 pp、Tail −3.0172 pp。两个seed均出现总体下降与头尾取舍；当前不支持按样本训练效用路由带来收益。\n\n'+"## 2. 锁定模型、数据与算法")
    report=report.replace("显存列为本进程PyTorch分配器",'学习率细节：原Stage II函数对第0个参数组乘 `config.lr_scheduler.lr_factor=0.2`；当前优化器只有这个分类器参数组，因此配置 `lr=0.1` 对应实际初始步长0.02，再按10-epoch cosine衰减。八组完全沿用此原函数与原YAML，没有把实际学习率主动提高到0.1。\n\n'+"显存列为本进程PyTorch分配器")
    (OUT/"GU_LOWVS_SCIENTIFIC_REPORT.md").write_text(report,encoding="utf-8")
    source.update({"text_gate_conditions":text_conditions,"go_task_text":text_go,"per_seed_paired_verified":wanted,"all_epoch_endpoint_collapse_records":all_epoch_endpoint_collapse,"integrity_check_path":"final_integrity.json"})
    run.save_json(OUT/"scientific_summary.json",source)
    assert run.verify_history()==protected
    run.base.check_test_guard();assert not run.HISTORY_EVENTS
    final={"passed":True,"verified_at":stamp(),"runs":8,"epochs":80,"updates":6800,"protected_historical_files":protected,"all_five_signatures_and_LowVS_masks_match":True,"backbone_and_BN_bitwise_unchanged":True,"checkpoint_hashes_verified":16,"checkpoint_reloads_verified_during_training":True,"TEST_events":run.base.AUDIT_EVENTS,"historical_write_events":run.HISTORY_EVENTS,"provided_policy_unchanged":True,"text_gate":text_go,"provided_runner_gate":bool(source["go_exploratory"]),"details":integrity}
    run.save_json(OUT/"final_integrity.json",final)
    run.save_json(OUT/"STOP_COMPLETED.json",{"completed_at":stamp(),"runs":8,"updates":6800,"decision":"GO exploratory" if text_go else "NO-GO","additional_training_started":False})
    print('FINAL INTEGRITY PASS: 8 runs, 6800 updates, all850/850 pairing, BB/BN unchanged, 16 checkpoints verified',flush=True)
    print('TEXT_GATE',text_go,'PROVIDED_RUNNER_GATE',source['go_exploratory'],flush=True)
    for row in wanted:print(json.dumps(row,ensure_ascii=False),flush=True)
    print('REPORT',OUT/'GU_LOWVS_SCIENTIFIC_REPORT.md',flush=True)

if __name__=="__main__":main()
