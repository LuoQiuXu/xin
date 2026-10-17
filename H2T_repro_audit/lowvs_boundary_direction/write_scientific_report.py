"""Read-only scientific interpretation of the fixed boundary probe; no training."""
from pathlib import Path
import json,csv,sys
import numpy as np
import run_probe as run

OUT=run.RESULTS

def read(p):return json.loads(Path(p).read_text(encoding="utf-8"))
def table(headers,rows):return ["|"+"|".join(headers)+"|","|"+"|".join(["---"]*len(headers))+"|"]+["|"+"|".join(map(str,r))+"|" for r in rows]
def records(p):
    with Path(p).open(newline="",encoding="utf-8") as f:return list(csv.DictReader(f))
def distribution(x):
    x=np.asarray(x,dtype=np.float64);x=x[np.isfinite(x)]
    q=np.quantile(x,[.05,.25,.5,.75,.95])
    return {"mean":float(x.mean()),"std":float(x.std()),**{k:float(v) for k,v in zip(("q05","q25","median","q75","q95"),q)}}

def main():
    base,*_=run.get_old()
    from common import sha,array_hash
    from run_safeguards import protect,stamp
    base.forbid_official_test();protect(run.ROOT.parent,OUT,run.HISTORY_EVENTS)
    protected=run.verify_history()
    source_lock=read(OUT/"source_lock.json")
    assert source_lock=={name:sha(run.HERE/name) for name in source_lock}
    assert (run.HERE/"geometry.py").read_bytes()==(OUT/"provided_source/geometry.py").read_bytes()
    group_rows=records(OUT/"group_summary.csv")
    by={(int(r["seed"]),r["group"]):r for r in group_rows}
    populations=[];checks=[];distributions=[];rates=[];all_data={}
    for seed in run.SEEDS:
        folder=OUT/f"seed{seed}";prop=read(folder/"provenance.json")
        assert prop["status"]=="complete" and prop["optimizer_steps"]==prop["backward_calls"]==0
        assert prop["paired_historical_batches"]==85 and all(v==85 for v in prop["pairing_counts"].values())
        assert prop["classifier_eval_and_requires_grad_false"] and prop["backbone_state_tensors_unchanged"]==186 and prop["classifier_state_tensors_unchanged"]==2
        assert prop["BN_counters"]==[15640] and not prop["official_TEST_events"] and not prop["historical_write_events"]
        assert prop["source_sha256"]==source_lock
        file=folder/"train_epoch1_boundary_probe.npz";assert sha(file)==prop["probe_file_sha256"]
        with np.load(file,allow_pickle=False) as archive:d={k:archive[k] for k in archive.files}
        all_data[seed]=d
        assert len(d["receiver_class"])==10847
        assert d["receiver_gap"].shape==d["donor_gap"].shape==(10847,64)
        assert d["uniform_channels"].shape==d["lowvs_channels"].shape==(10847,16)
        same=np.array([np.array_equal(np.sort(a),np.sort(b)) for a,b in zip(d["uniform_channels"],d["lowvs_channels"])])
        assert np.array_equal(same,d["same_mask"])
        assert np.all(d["same_mask"][d["fallback"]>0])
        assert np.all(d["competitor_class"]!=d["receiver_class"])
        assert np.isfinite(d["q_u"]).all() and np.isfinite(d["q_l"]).all()
        direct=d["direct_full_margin_contrast"]
        right=d["margin_diff_direction_at_r"]+d["margin_diff_extra_magnitude"]
        assert np.allclose(direct,right,atol=1e-9,rtol=1e-9)
        assert np.allclose(d["direct_matched_margin_contrast"],d["margin_diff_direction_at_r"],atol=1e-9,rtol=1e-9)
        assert np.allclose(d["matched_norm_actual_u"],d["r"],atol=1e-9,rtol=1e-9)
        assert np.allclose(d["matched_norm_actual_l"],d["r"],atol=1e-9,rtol=1e-9)
        assert np.array_equal(d["q_u"][same],d["q_l"][same])
        assert np.array_equal(d["full_margin_delta_u"][same],d["full_margin_delta_l"][same])
        assert np.array_equal(d["loss_delta_u"][same],d["loss_delta_l"][same])
        historical=records(run.OLD_NORM/f"seed{seed}/Uniform-Full/batch_signatures.csv")[:85]
        replay=records(folder/"replay_signatures.csv")
        assert len(replay)==85
        for a,b in zip(replay,historical):assert all(a[k]==b[k] for k in run.PAIR_KEYS)
        y=d["receiver_class"];h=d["donor_class"]
        tail=y>=71;eligible=(h!=y)&(d["fallback"]==0)&(~same)
        groups={"All":np.ones(len(y),dtype=bool),"AllEligible":eligible,"Head":y<=35,"Medium":(y>=36)&(y<=70),"Tail":tail,"HeadEligible":(y<=35)&eligible,"TailEligible":tail&eligible,"TailEligibleDonorHead":tail&eligible&(h<=35),"TailEligibleCleanCorrect":tail&eligible&d["clean_correct"],"TailEligibleCleanWrong":tail&eligible&(~d["clean_correct"])}
        for name,mask in groups.items():
            expected=run.summaries(seed,d,mask,name)
            source=by[(seed,name)]
            assert int(source["n"])==int(mask.sum())
            for key,value in expected.items():
                if isinstance(value,(float,np.floating)):assert abs(value-float(source[key]))<1e-12,(seed,name,key)
            populations.append({"seed":seed,"group":name,"draws":int(mask.sum()),"unique_receiver_images":len(np.unique(d["receiver_raw_index"][mask])),"fraction_of_all_draws":float(mask.mean()),"fraction_of_Tail_draws":float(mask.sum()/tail.sum()) if name.startswith('Tail') else None})
            diff=d["q_l"][mask]-d["q_u"][mask]
            distributions.append({"seed":seed,"group":name,"draws":int(mask.sum()),**distribution(diff)})
            rates.append({"seed":seed,"group":name,"draws":int(mask.sum()),**{key:float(d[key][mask].mean()) for key in run.BOOL_KEYS}})
        overlap=np.array([len(set(a.tolist())&set(b.tolist())) for a,b in zip(d["uniform_channels"],d["lowvs_channels"])])
        checks.append({"seed":seed,"draws":10847,"paired_batches":85,"unique_receivers":len(np.unique(d["receiver_raw_index"])),"Tail_draws":int(tail.sum()),"Tail_unique_receivers":len(np.unique(d["receiver_raw_index"][tail])),"fallback_draws":int((d["fallback"]>0).sum()),"same_mask_draws":int(same.sum()),"different_mask_fraction":float((~same).mean()),"same_label_draws":int((h==y).sum()),"eligible_draws":int(eligible.sum()),"TailEligible_draws":int((tail&eligible).sum()),"TailEligible_unique_receivers":len(np.unique(d["receiver_raw_index"][tail&eligible])),"TailEligible_fraction_of_Tail":float((tail&eligible).sum()/tail.sum()),"mean_mask_overlap_channels":float(overlap.mean()),"max_independent_identity_residual":float(np.abs(direct-right).max()),"max_matched_norm_error":float(np.abs(d["matched_norm_actual_u"]-d["matched_norm_actual_l"]).max()),"nonzero_sub_epsilon_norms":int(((d["nu"]>0)&(d["nu"]<=1e-12)).sum()+((d["nl"]>0)&(d["nl"]<=1e-12)).sum()),"zero_norms":int((d["nu"]==0).sum()+(d["nl"]==0).sum()),"sign_mismatch_U":int(d["sign_mismatch_u"].sum()),"sign_mismatch_L":int(d["sign_mismatch_l"].sum()),"native_FC_logit_error_max":prop["native_fc_float32_vs_matrix_float64_max_logit_error"],"official_pool_vs_mean_max_difference":prop["official_pool_vs_mean_max_difference"]})
    registry=read(OUT/"run_registry.json");assert len(registry)==3 and all(r["status"]=="completed" for r in registry)
    for name,data in (("population_diagnostics.csv",populations),("replay_verification.csv",checks),("q_difference_distribution.csv",distributions),("instantaneous_rates.csv",rates)):run.save_csv(OUT/name,data)
    primary=[by[(seed,"TailEligible")] for seed in run.SEEDS]
    same_negative=all(float(r["mean_qL_minus_qU"])<0 and float(r["mean_cosL_minus_cosU"])<0 and float(r["mean_matched_margin_L_minus_U"])<0 for r in primary)
    judgement="更明显的即时判别边界干扰（单位长度方向在两增强视图中同向）；未证明训练收益" if same_negative else "方向指标未获得同向支持或依赖组别；停止方向模块研发，不扫描参数"
    def f(row,key):return float(row[key])
    lines=["# Low-VS 判别边界方向机制：科学报告","","日期：2026-10-17（Asia/Shanghai）","","## 1. 结论与预声明主指标","",f"**本轮判断：{judgement}。** 只对既有几何方向作描述，不恢复或启动Low-VS新方法训练。前轮Matched准确率跨seed变号的事实没有被本轮静态探针推翻。","","主群体TailEligible预先固定：接收类71..99、供体异类、非fallback、实际掩码不同。每行是10847个TRAIN采样draw中的一个子集，原图会重复。",""]
    lines+=table(["seed","TailEligible draws","不同接收原图","mean(qL-qU)","P(qL<qU)","mean(cosL-cosU)","matched固定margin L-U"],[[r["seed"],r["n"],checks[i]["TailEligible_unique_receivers"],f"{f(r,'mean_qL_minus_qU'):+.8f}",f"{100*f(r,'fraction_qL_less_qU'):.3f}%",f"{f(r,'mean_cosL_minus_cosU'):+.8f}",f"{f(r,'mean_matched_margin_L_minus_U'):+.8f}"] for i,r in enumerate(primary)])
    lines+=["","归一化后差值仍为负，不能把这项固定边界差异全部解释为nL较大。q<0仅表示当前true-vs固定竞争类margin下降；它可能是训练压力，也可能是标签不一致。本轮没有训练，不能在这两种解释中作选择。P约59%也意味着存在大量相反方向的draw，不能把Low-VS视为每个样本都更强或更好。","","## 2. 冻结模型、输入与一致性","", "使用Clean Stage I seed42 epoch184，SHA256 `7ced9d07f71335357c08623573f038acd0dfdcdeadb55d95d57b332a14f83a88`；没有加载任何Stage II参数。ResNet-32、31层BN和Stage I 100类线性FC全部eval、requires_grad=False，186+2个状态张量逐位不变；BN计数器均15640；backward=0、optimizer.step=0。", "", f"只重放seed124/125封存计划第1epoch，共170个历史minibatch、21694次receiver/donor draw，每seed85批（84×128+95）。每批7项indices/images/features/scores/Gumbel/U-mask/L-mask对旧norm-direction基准全部85/85一致。保留历史准备warmup，不额外抽样epoch。{protected}个历史文件hash不变，TEST与历史写保护无事件。", "", "数值输入仅为原10847张LT TRAIN及其BS/IS增强。未加载验证图像、标签或logits进入本轮计算；源完整性检查会对历史文件（包括旧验证缓存）做hash，但其数值不参与探针。参考Stage I在历史上由99类TRAIN留出选中epoch184，因此不能声称整个研究完全没有开发集选择影响。训练源图也已被Stage I见过。", "", "本轮new-directory修复：显式冻结/eval FC；GAP改为原classifier的avg_pool2d并与在线VS GAP逐位核对；修正主群体metadata遗漏的mask差异条件；新增直接logits恒等式核验及原始GAP/掩码、签名、状态证据。geometry.py与提供版本逐字节相同。更改Diff为interface_and_verification.patch。", "", "主任务及实际geometry采用max(norm,epsilon)，epsilon=1e-12；本地旧README/长指令出现norm+epsilon写法，README已按实际实现更正。此处不改变评分公式。当前真实draw没有非零sub-epsilon范数，避免把该退化区间的安全定义误称严格归一化。原GPU FC使用float32，几何矩阵运算使用float64；最大logit差及pool差在下表记录，不影响历史feature-map哈希。", "", "## 3. 样本、掩码有效率与未筛选对照", ""]
    lines+=table(["seed","全部draws","不同接收原图","Tail draws","TailEligible","Tail有效率","全局fallback","相同掩码","掩码不同率","同类供体","平均交集通道"],[[c["seed"],c["draws"],c["unique_receivers"],c["Tail_draws"],c["TailEligible_draws"],f"{100*c['TailEligible_fraction_of_Tail']:.3f}%",c["fallback_draws"],c["same_mask_draws"],f"{100*c['different_mask_fraction']:.3f}%",c["same_label_draws"],f"{c['mean_mask_overlap_channels']:.3f}"] for c in checks])
    lines+=["","所有draw均保存在NPZ，未删除同类别、fallback或相同掩码记录。只有预声明Eligible统计排除这些情况；fallback必然回退Uniform，所有相同掩码draw的q、margin和CE对比严格相同。两个seed共享源图，不将21694次采样当成独立图像或独立训练模型，不给样本级显著性。", "", "## 4. Head、供体Head和初始分对/分错", ""]
    names=("AllEligible","HeadEligible","TailEligibleDonorHead","TailEligibleCleanCorrect","TailEligibleCleanWrong")
    side=[by[(seed,name)] for seed in run.SEEDS for name in names]
    lines+=table(["seed","预声明群体","draws","mean(qL-qU)","P(qL<qU)","cos差","matched固定margin差","供体q_yh差"],[[r["seed"],r["group"],r["n"],f"{f(r,'mean_qL_minus_qU'):+.6f}",f"{100*f(r,'fraction_qL_less_qU'):.2f}%",f"{f(r,'mean_cosL_minus_cosU'):+.6f}",f"{f(r,'mean_matched_margin_L_minus_U'):+.6f}",f"{f(r,'mean_donor_qL_minus_qU'):+.6f}"] for r in side])
    lines+=["","HeadEligible同样出现负方向差，绝不能据此称为尾类特异语义机制；TailEligibleDonorHead与主群体方向一致。初始分类正确Tail的干扰较大，初始错误Tail也为同号但较小，这是固定模型的条件关联，不是供体筛选规则。q_yh的y/h同类时保留NaN，不将其混成零；Eligible群体均为异类。", "", "主群体qL−qU的连续分布：", ""]
    dist=[r for r in distributions if r["group"]=="TailEligible"]
    lines+=table(["seed","均值","标准差","5%","25%","中位数","75%","95%"],[[r["seed"],*[f"{r[k]:+.6f}" for k in ("mean","std","q05","q25","median","q75","q95")]] for r in dist])
    lines+=["","## 5. 真实固定margin恒等式、符号与即时效应","","j*只由clean z和固定Stage I W,b在真实类之外取argmax，四种扰动始终沿用同一个j*。以独立的(z+delta)W^T+b计算真实margin变化，核对：Full(L−U)=r(qL−qU)+(nL−r)qL−(nU−r)qU。下表是线性margin的代数项，不是准确率的因果贡献。", ""]
    lines+=table(["seed","mean nU","mean nL","mean alphaU","mean alphaL","matched方向项","额外幅度项","Full真实margin差"],[[r["seed"],*[f"{f(r,k):.6f}" for k in ("mean_nU","mean_nL","mean_alpha_U","mean_alpha_L","mean_matched_margin_L_minus_U","mean_extra_magnitude_term","mean_full_margin_L_minus_U")]] for r in primary])
    lines+=['']+table(["seed","独立恒等式最大残差","配平范数最大差","q符号差U/L","零范数","非零sub-eps范数","GPU FC vs矩阵最大logit差","pool vs旧mean最大差"],[[c["seed"],f"{c['max_independent_identity_residual']:.3g}",f"{c['max_matched_norm_error']:.3g}",f"{c['sign_mismatch_U']}/{c['sign_mismatch_L']}",c["zero_norms"],c["nonzero_sub_epsilon_norms"],f"{c['native_FC_logit_error_max']:.3g}",f"{c['official_pool_vs_mean_max_difference']:.3g}"] for c in checks])
    lines+=["","TailEligible上的即时效应（L−U；概率差乘100为pp，均为当前冻结模型在TRAIN增强draw上的结果）：", ""]
    lines+=table(["seed","Full CE差","Matched CE差","Full预测改变率差pp","Matched预测改变率差pp","Full即时判对率差pp","Matched即时判对率差pp"],[[r["seed"],f"{f(r,'mean_loss_full_L_minus_U'):+.6f}",f"{f(r,'mean_loss_matched_L_minus_U'):+.6f}",*[f"{100*f(r,k):+.4f}" for k in ("flip_full_L_minus_U","flip_matched_L_minus_U","correct_full_L_minus_U","correct_matched_L_minus_U")]] for r in primary])
    lines+=["","CE、预测改变、初始j* margin跨越是不同事件。完整单组绝对率与margin跨越率见instantaneous_rates.csv；跨越只统计clean margin>0且干预后<0的严格穿越，不等价于所有argmax变化。即时判对率下降不能直接称破坏性训练，也不能把负q当作有益难例。", "", "## 6. 与上一轮结果的关系及停止","","上一轮固定epoch10、1980张99类TRAIN开发验证中，Full Tail差平均+0.9483 pp，Matched为+0.1724 pp且两seed变号。本轮不读取这些验证logits作选择，只将用户已给出的历史结果作背景。几何层发现稳定方向差，并未建立该差异能改善分类器训练；它与前轮泛化代理不稳定同时成立。方向项与幅度项在此固定margin都为负，方向并非可以忽略，但不能把代数项换算成最终Tail增益比例。", "", "当前不提出q门控、不扫描K/alpha/困难度、不训练任何新模块。保持方法开发暂停与审慎评议；若未来继续，应先以数据/条件压力测试把即时边界挑战与实际负迁移联系起来，再提出方法，而非把本探针直接变成筛选器。", "", "## 7. 未来跨设置与外部数据计划（仅规划）","","1. 先核对作者本地锁定配置：IF50→imb002(.02)、IF100→imb001(.01)、IF200→imb0005(.005)，Stage I各200 epochs、CE+MixUp、batch128；Stage II均10 epochs、batch128。注意IF200的Stage II YAML lr=.2，其余=.1；在每个IF内部三臂用同一预先锁定设置，不能把IF200悄悄统一成.1。保存数据索引、类频次、配置与各自独立Stage I来源。每个dataset/IF内DR、H2T、Low-VS共享该设置的同一Stage I，不能把当前IF100模型跨IF复用后仍称从对应数据同协议训练。", "", "2. BN需按比较面板统一：作者Stage II YAML shift_bn=True；当前Low-VS证据来自固定BN。原始复现面板与固定BN机制面板分开标明，不混算准确率或将BN变化归于FC。原H2T为batch共享随机掩码、Low-VS为逐样本，三臂比较会同时含粒度与评分差异；仅保留三臂时须披露此混杂，不把差值全归于VS。若要分离，另行预注册Uniform-PerSample对照，当前不追加。", "", "3. 统一Overall、Head/Tail、Many/Medium/Few、Macro、worst-class（附各类分母）、头→尾/尾→头混淆与每类净得失；固定epoch结果为主，best为辅；同一配置内用配对seed和相同训练预算。先预定义与本轮相同的donorHead、clean-correct/wrong条件，再检验边界干扰是否伴随实际负迁移。跨数据集或Backbone/预算改变的绝对精度下降不能单独指认H2T机制失效。", "", "4. 外部候选可先评估Tiny ImageNet-200再构造锁定LT训练子集；它有200类、64×64图像、每类500训练样本和50验证样本，见[Stanford CS231n原始实验报告](https://cs231n.stanford.edu/reports/2017/pdfs/12.pdf)。这是候选建议，未下载数据、未适配、未实测8GB。先在独立目录核实文件/标签/类映射和划分完整，再验证同一模型、batch128的Stage I前反向与冻结双分支峰值显存；不够则先重新登记资源协议，不能静默改batch比较。", "", "5. 若后续采用Tiny，TR-only估计统计和选择阶段，官方带标签VAL可作为预先封存的独立评估集（应明确它名义上是VAL，不能伪称官方TEST）；不要假设公开TEST标签可用。200类与64×64输入要求新配置与空间尺寸审计，不能强套本轮100类和8×8断言；同数据内三臂保持同构Backbone和预算。当前CIFAR官方TEST在历史研究中已被查看，不再宣称它是全新未见终检；外部封存评估集可在方法锁定后一次评估。本轮不读取任何这些评估数据。", "", "## 8. 异常、环境、原始文件和命令","","无缺文件、重放失配、数学校验失败或运行重试；没有更新Stage I/II。仅在本轮新目录修正接口/说明并增加核验，geometry公式未改。环境与完整stdout在environment.json、console/；源码锁source_lock.json、修改Diff和初始源码快照全部保留。", "", "```powershell", "$env:PYTHONIOENCODING = 'utf-8'",f"$py = '{sys.executable}'","& $py -B H2T_repro_audit/lowvs_boundary_direction/geometry.py","& $py -B H2T_repro_audit/lowvs_boundary_direction/run_probe.py run --seed 124","& $py -B H2T_repro_audit/lowvs_boundary_direction/run_probe.py run --seed 125","& $py -B H2T_repro_audit/lowvs_boundary_direction/run_probe.py report","& $py -B H2T_repro_audit/lowvs_boundary_direction/write_scientific_report.py","```","","已有目录不可直接重跑覆盖。每seed的train_epoch1_boundary_probe.npz包含全部draw、GAP、实际掩码、原/本地索引、j*、q/cos、损失和flip以及独立margin核验量，provenance.json与replay_signatures.csv记录模型与85批签名。统一group_summary.csv、per_class_summary.csv、scientific_summary.json、population_diagnostics.csv、q_difference_distribution.csv、instantaneous_rates.csv、replay_verification.csv和final_integrity.json可复查。", "", "本轮已完成并停止，由用户将结果提供给ChatGPT复核。没有执行任何版本控制操作，也没有启动跨IF、外部数据或新模块实验。", ""]
    text=chr(10).join(lines)
    (OUT/"BOUNDARY_DIRECTION_SCIENTIFIC_REPORT.md").write_text(text,encoding="utf-8")
    original=read(OUT/"scientific_summary.json")
    original.update({"judgement":judgement,"primary_direction_consistent_negative":same_negative,"population_diagnostics":populations,"replay_verification":checks,"future_stress_tests_executed":False,"external_candidate":"Tiny ImageNet-200, conditional on data and 8GB feasibility verification; not downloaded"})
    run.save_json(OUT/"scientific_summary.json",original)
    assert run.verify_history()==protected;base.check_test_guard();assert not run.HISTORY_EVENTS
    run.save_json(OUT/"final_integrity.json",{"passed":True,"verified_at":stamp(),"historical_batches":170,"draws":21694,"optimizer_steps":0,"backward_calls":0,"historical_files_unchanged":protected,"geometry_bytewise_unchanged":True,"run_source_locked":True,"TEST_events":base.AUDIT_EVENTS,"historical_write_events":run.HISTORY_EVENTS,"checks":checks,"judgement":judgement,"source_sha256":{p.name:sha(p) for p in run.HERE.glob("*.py")}})
    run.save_json(OUT/"STOP_COMPLETED.json",{"completed_at":stamp(),"optimizer_steps":0,"future_stress_tests_executed":False,"new_direction_module_started":False,"judgement":judgement})
    print('SCIENTIFIC REPORT PASS: 170 matched batches, no training; frozen states and direct margin identity verified.',flush=True)
    print('JUDGEMENT',judgement,flush=True)
    print('REPORT',OUT/'BOUNDARY_DIRECTION_SCIENTIFIC_REPORT.md',flush=True)

if __name__=="__main__":main()
