"""Descriptive analysis of frozen-reference offline donor perturbations."""
from __future__ import annotations
import csv
import hashlib
import itertools
import math
from pathlib import Path
import pickle

import numpy as np
from scipy import stats

from audit_donors import ROOT,BASE,DATA,N,K,GROUP_BATCH,read,dump,sha,now,historical_hashes

GROUPS = ['All','Head','Medium','Tail']
PAIR_A,PAIR_B = np.array(list(itertools.combinations(range(K),2))).T


def distribution(values):
    a = np.asarray(values,dtype=np.float64).ravel()
    assert len(a) and np.isfinite(a).all()
    q = np.percentile(a,[0,1,5,10,25,50,75,90,95,99,100])
    return {'n':len(a),'mean':float(a.mean()),'std_population':float(a.std()),'mean_abs':float(np.abs(a).mean()),
        **dict(zip(['min','p01','p05','p10','p25','p50','p75','p90','p95','p99','max'],map(float,q)))}


def pearson(x,y):
    x,y = np.asarray(x,dtype=np.float64).ravel(),np.asarray(y,dtype=np.float64).ravel()
    if x.std() == 0 or y.std() == 0:
        return None
    return float(np.corrcoef(x,y)[0,1])


def correlations(x,y):
    return {'pearson':pearson(x,y),'spearman':float(stats.spearmanr(x.ravel(),y.ravel()).statistic),
        'within_receiver_centered_pearson':pearson(x-x.mean(1,keepdims=True),y-y.mean(1,keepdims=True))}


def rate(condition,eligible=None):
    if eligible is None:
        eligible = np.ones_like(condition,dtype=bool)
    num,den = int(np.count_nonzero(condition & eligible)),int(np.count_nonzero(eligible))
    return {'count':num,'denominator':den,'percent':100*num/den if den else None}


def group_labels(classes):
    return np.where(classes < 36,'Head',np.where(classes < 71,'Medium','Tail'))


def load_mode(mode):
    with np.load(ROOT/(mode+'_statistics.npz')) as z:
        return {k:z[k] for k in z.files}


def summarize_mode(data,plan):
    classes = plan['receiver_classes']
    groups = group_labels(classes)
    y = classes[:,None]
    d = data['D']
    order = np.argsort(d,axis=1,kind='stable')
    rank = np.empty_like(order)
    np.put_along_axis(rank,order,np.broadcast_to(np.arange(1,K+1),(N,K)),axis=1)
    indices = np.arange(N)
    choices = {'Random':plan['random_positions'],'Low-D':order[:,0],
        'Mid-D':order[indices,plan['mid_sorted_positions']],'High-D':order[:,-1]}
    ranges,stds = np.ptp(d,axis=1),d.std(1)
    pair_cos = np.abs(data['cosine_spatial'][:,PAIR_A]-data['cosine_spatial'][:,PAIR_B])
    pair_d = np.abs(d[:,PAIR_A]-d[:,PAIR_B])
    closest = pair_cos.argmin(1)
    summaries = {}
    for group in GROUPS:
        selected = np.ones(N,dtype=bool) if group == 'All' else groups == group
        s = {key:value[selected] for key,value in data.items()}
        before_correct = s['pred_before'] == y[selected]
        after_correct = s['pred_after'] == y[selected]
        summaries[group] = {'receiver_groups':int(selected.sum()),'candidate_rows':int(selected.sum()*K),
            'unique_receiver_instances':int(len(np.unique(plan['receiver_indices'][selected]))),
            'D':distribution(s['D']),'D_min':distribution(s['D'].min(1)),
            'D_max':distribution(s['D'].max(1)),'D_range':distribution(ranges[selected]),
            'D_std':distribution(stds[selected]),'baseline_margin':distribution(s['baseline_margin'][:,0]),
            'baseline_abs_margin':distribution(np.abs(s['baseline_margin'][:,0])),
            'fused_margin':distribution(s['fused_margin']),
            'D_positive':rate(s['D'] > 0),'D_negative':rate(s['D'] < 0),'D_exact_zero':rate(s['D'] == 0),
            'mixed_D_sign_groups':rate((s['D'].min(1) < 0) & (s['D'].max(1) > 0)),
            'nonzero_range_groups':rate(ranges[selected] > 0),
            'mixed_fused_correctness_groups':rate(after_correct.any(1) & (~after_correct).any(1)),
            'different_fused_predictions_groups':rate(np.any(s['pred_after'] != s['pred_after'][:,0:1],axis=1)),
            'baseline_correct':rate(before_correct[:,0]),'fused_correct':rate(after_correct),
            'prediction_boundary_cross':rate(s['pred_after'] != s['pred_before']),
            'correct_to_incorrect':rate(~after_correct,before_correct),
            'incorrect_to_correct':rate(after_correct,~before_correct),
            'true_margin_positive_to_nonpositive':rate(s['fused_margin'] <= 0,s['baseline_margin'] > 0),
            'true_margin_nonpositive_to_positive':rate(s['fused_margin'] > 0,s['baseline_margin'] <= 0),
            'strongest_competitor_changed':rate(s['rival_after'] != s['rival_before']),
            'zero_baseline_margins':int(np.count_nonzero(s['baseline_margin'][:,0] == 0)),
            'correlations':{metric:correlations(s[metric],s['D']) for metric in ['cosine_spatial','cosine_gap','l2_spatial','l2_gap']},
            'cosine_spatial':distribution(s['cosine_spatial']),'cosine_gap':distribution(s['cosine_gap']),
            'l2_spatial':distribution(s['l2_spatial']),'l2_gap':distribution(s['l2_gap']),
            'nearest_cosine_pair_abs_cosine_difference':distribution(pair_cos[selected][np.arange(selected.sum()),closest[selected]]),
            'nearest_cosine_pair_abs_D_difference':distribution(pair_d[selected][np.arange(selected.sum()),closest[selected]]),
            'all_within_group_pair_abs_cosine_difference':distribution(pair_cos[selected]),
            'all_within_group_pair_abs_D_difference':distribution(pair_d[selected]),
            'choices':{name:{'D':distribution(s['D'][np.arange(selected.sum()),choice[selected]]),
                'fused_correct':rate(after_correct[np.arange(selected.sum()),choice[selected]]),
                'boundary_cross':rate((s['pred_after'] != s['pred_before'])[np.arange(selected.sum()),choice[selected]])}
                for name,choice in choices.items()}}
    same = plan['donor_classes'] == classes
    same_stats = {}
    for name,selected in [('same_class',same),('different_class',~same)]:
        same_stats[name] = {'receiver_groups':int(selected.sum()),'D':distribution(d[selected]),
            'D_range':distribution(ranges[selected]),'boundary_cross':rate(data['pred_after'][selected] != data['pred_before'][selected]),
            'strongest_competitor_changed':rate(data['rival_after'][selected] != data['rival_before'][selected])}
    cos_order = np.argsort(pair_cos.ravel(),kind='stable')
    deciles = []
    for bin_id,index in enumerate(np.array_split(cos_order,10)):
        deciles.append({'cosine_delta_rank_decile':bin_id+1,'pairs':len(index),
            'abs_cosine_difference':distribution(pair_cos.ravel()[index]),
            'abs_D_difference':distribution(pair_d.ravel()[index])})
    examples = []
    for group in ['Head','Medium','Tail']:
        index = np.flatnonzero(groups == group)
        ordered = index[np.argsort(ranges[index],kind='stable')]
        for quantile in [.1,.5,.9]:
            i = int(ordered[int(round(quantile*(len(ordered)-1)))])
            examples.append({'receiver_group':group,'range_quantile_target':quantile,'group_id':i,
                'receiver_index':int(plan['receiver_indices'][i]),'receiver_class':int(classes[i]),
                'donor_class':int(plan['donor_classes'][i]),'D_range':float(ranges[i]),
                'baseline_margin':float(data['baseline_margin'][i,0]),'closest_cosine_pair_positions':
                    [int(PAIR_A[closest[i]]),int(PAIR_B[closest[i]])],
                'closest_cosine_pair_abs_cosine_difference':float(pair_cos[i,closest[i]]),
                'closest_cosine_pair_abs_D_difference':float(pair_d[i,closest[i]]),
                'donor_indices':plan['donor_indices'][i].tolist(),'D':d[i].tolist(),
                'cosine_spatial':data['cosine_spatial'][i].tolist(),'fused_margin':data['fused_margin'][i].tolist(),
                'ascending_D_order_zero_based':order[i].tolist(),'prediction_after':data['pred_after'][i].tolist()})
    return {'groups':summaries,'same_class_analysis':same_stats,'similarity_delta_deciles':deciles,'examples':examples},order,rank,choices


def sampling_summary(plan,counts):
    bs = np.bincount(plan['receiver_classes'],minlength=100)
    donors = np.bincount(plan['donor_classes'],minlength=100)
    probabilities = counts/counts.sum()
    bs_chi = stats.chisquare(bs,np.full(100,N/100))
    is_chi = stats.chisquare(donors,N*probabilities)
    unique = np.array([len(set(row)) for row in plan['donor_indices']])
    duplicate_probability = 1-np.prod(1-np.arange(K)[None,:]/counts[:,None],axis=1)
    same = plan['receiver_classes'] == plan['donor_classes']
    self_candidates = plan['donor_indices'] == plan['receiver_indices'][:,None]
    mask_array = np.load(ROOT/'channel_masks.npy')
    assert mask_array.shape == (N//GROUP_BATCH,16) and all(len(set(row)) == 16 for row in mask_array)
    result = {'bs_class_counts':bs.tolist(),'is_class_counts':donors.tolist(),
        'expected_is_probabilities':probabilities.tolist(),
        'bs_chi_square':float(bs_chi.statistic),'bs_chi_square_df':99,'bs_chi_square_p':float(bs_chi.pvalue),
        'is_chi_square':float(is_chi.statistic),'is_chi_square_df':99,'is_chi_square_p':float(is_chi.pvalue),
        'bs_total_variation':float(.5*np.abs(bs/N-.01).sum()),
        'is_total_variation':float(.5*np.abs(donors/N-probabilities).sum()),
        'duplicate_candidate_groups':rate(unique < K),
        'repeated_candidate_slots':{'count':int((K-unique).sum()),'denominator':N*K,'percent':float((K-unique).sum()/(N*K)*100)},
        'candidate_unique_counts_histogram':{str(i):int((unique == i).sum()) for i in range(1,K+1)},
        'expected_duplicate_group_probability_percent':float(np.dot(probabilities,duplicate_probability)*100),
        'same_class_groups':rate(same),'self_donor_groups':rate(self_candidates.any(1)),
        'self_donor_candidates':rate(self_candidates),'receiver_group_counts':{},'donor_group_counts':{},
        'unique_receiver_instances':int(len(np.unique(plan['receiver_indices']))),
        'random_candidate_position_counts':np.bincount(plan['random_positions'],minlength=K).tolist(),
        'channel_selection_counts':np.bincount(mask_array.ravel(),minlength=64).tolist(),
        'same_class_by_receiver_group':{}}
    for group in ['Head','Medium','Tail']:
        r = group_labels(plan['receiver_classes']) == group
        h = group_labels(plan['donor_classes']) == group
        result['receiver_group_counts'][group] = int(r.sum())
        result['donor_group_counts'][group] = int(h.sum())
        result['same_class_by_receiver_group'][group] = rate(same,r)
    with (ROOT/'sampling_distribution.csv').open('w',encoding='utf-8',newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['class_id','train_count','expected_bs_probability','observed_bs_count','expected_is_probability','observed_is_class_count'])
        writer.writerows(zip(range(100),counts,np.full(100,.01),bs,probabilities,donors))
    return result


def save_csv(mode,data,plan,order,rank,choices):
    groups = group_labels(plan['receiver_classes'])
    fields = list(data)
    with (ROOT/(mode+'_candidates.csv')).open('w',encoding='utf-8',newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['group_id','mask_batch_id','receiver_index','receiver_class','receiver_group','donor_class',
            'candidate_position','donor_index','same_class','self_donor','ascending_D_rank',*['chosen_'+name for name in choices],*fields])
        for i in range(N):
            for k in range(K):
                writer.writerow([i,i//GROUP_BATCH,plan['receiver_indices'][i],plan['receiver_classes'][i],groups[i],
                    plan['donor_classes'][i],k,plan['donor_indices'][i,k],int(plan['donor_classes'][i] == plan['receiver_classes'][i]),
                    int(plan['donor_indices'][i,k] == plan['receiver_indices'][i]),rank[i,k],*[int(choice[i] == k) for choice in choices.values()],
                    *[data[key][i,k] for key in fields]])
    with (ROOT/(mode+'_receiver_groups.csv')).open('w',encoding='utf-8',newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['group_id','receiver_index','receiver_class','receiver_group','donor_class','baseline_margin',
            'D_min','D_max','D_range','D_std_population','D_ascending_order_zero_based','unique_candidate_instances',
            'mixed_D_sign','mixed_fused_correctness','different_fused_predictions',*['position_'+name for name in choices]])
        correct = data['pred_after'] == plan['receiver_classes'][:,None]
        for i in range(N):
            d = data['D'][i]
            writer.writerow([i,plan['receiver_indices'][i],plan['receiver_classes'][i],groups[i],plan['donor_classes'][i],
                data['baseline_margin'][i,0],d.min(),d.max(),np.ptp(d),d.std(),';'.join(map(str,order[i])),
                len(set(plan['donor_indices'][i])),int(d.min() < 0 < d.max()),int(correct[i].any() and not correct[i].all()),
                int(np.any(data['pred_after'][i] != data['pred_after'][i,0])),*[v[i] for v in choices.values()]])
    np.savez_compressed(ROOT/(mode+'_ranked_groups.npz'),D_ascending_order=order,D_rank=rank,
        **{name.replace('-','_')+'_position':pos for name,pos in choices.items()})


def main():
    assert read(ROOT/'run_status.json')['state'] == 'completed'
    assert historical_hashes() == read(ROOT/'historical_before.json')
    with np.load(ROOT/'sampling_plan.npz') as z:
        plan = {k:z[k] for k in z.files}
    counts = np.bincount(np.load(ROOT/'train_subset_labels.npy'),minlength=100)
    sampling = sampling_summary(plan,counts)
    analyses,data_modes = {},{}
    for mode in ['eval_preprocess','fixed_train_aug']:
        data = load_mode(mode)
        assert all(v.shape == (N,K) and np.isfinite(v).all() for v in data.values())
        analysis,order,rank,choices = summarize_mode(data,plan)
        save_csv(mode,data,plan,order,rank,choices)
        analyses[mode] = analysis
        data_modes[mode] = (data,choices)
    main_data,aug_data = data_modes['eval_preprocess'][0],data_modes['fixed_train_aug'][0]
    sensitivity = {'D_pearson_between_preprocessing_modes':pearson(main_data['D'],aug_data['D']),
        'range_pearson_between_preprocessing_modes':pearson(np.ptp(main_data['D'],1),np.ptp(aug_data['D'],1)),
        'low_D_candidate_identity_agreement':rate(data_modes['eval_preprocess'][1]['Low-D'] == data_modes['fixed_train_aug'][1]['Low-D']),
        'high_D_candidate_identity_agreement':rate(data_modes['eval_preprocess'][1]['High-D'] == data_modes['fixed_train_aug'][1]['High-D'])}
    result = {'completed_at':now(),'sampling':sampling,'analyses':analyses,'augmentation_sensitivity':sensitivity,
        'no_ad_hoc_difference_threshold':True,'training_performed':False,'correlations_are_descriptive_not_causal':True}
    dump(ROOT/'analysis_summary.json',result)
    for mode,analysis in analyses.items():
        a = analysis['groups']['All']
        t = analysis['groups']['Tail']
        print(mode,{'D_range_median':a['D_range']['p50'],'D_range_p95':a['D_range']['p95'],
            'mixed_candidate_correctness_percent':a['mixed_fused_correctness_groups']['percent'],
            'Tail_range_median':t['D_range']['p50'],'Tail_mixed_correctness_percent':t['mixed_fused_correctness_groups']['percent']},flush=True)
    assert historical_hashes() == read(ROOT/'historical_before.json')
    print('DESCRIPTIVE ANALYSIS PASS; all CSV/NPZ statistics saved; no training',flush=True)


if __name__ == '__main__':
    main()
