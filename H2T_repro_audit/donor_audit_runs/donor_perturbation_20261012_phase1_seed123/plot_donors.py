"""Static research figures for the offline donor audit (English figure labels)."""
import pickle
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from audit_donors import ROOT,DATA,read
from analyze_donors import GROUPS,PAIR_A,PAIR_B,group_labels,load_mode

COLORS = {'All':'#202c39','Head':'#2367a4','Medium':'#d58b25','Tail':'#a63b61',
          'Random':'#555f6b','Low-D':'#2367a4','Mid-D':'#258777','High-D':'#a63b61'}
FIGURES = ROOT/'figures'


def ecdf(ax,values,label,color,style='-'):
    values = np.sort(np.asarray(values).ravel())
    ax.plot(values,np.arange(1,len(values)+1)/len(values),label=label,color=color,lw=1.9,ls=style)


def save(fig,name):
    fig.savefig(FIGURES/(name+'.png'),dpi=170,bbox_inches='tight',facecolor='white')
    fig.savefig(FIGURES/(name+'.svg'),bbox_inches='tight',facecolor='white')
    plt.close(fig)


def main():
    FIGURES.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':12,
        'axes.labelsize':11,'axes.spines.top':False,'axes.spines.right':False,
        'axes.grid':True,'grid.alpha':.18,'svg.fonttype':'none'})
    summary = read(ROOT/'analysis_summary.json')
    with np.load(ROOT/'sampling_plan.npz') as z:
        plan = {k:z[k] for k in z.files}
    main_data = load_mode('eval_preprocess')
    aug_data = load_mode('fixed_train_aug')
    labels = group_labels(plan['receiver_classes'])
    sampling = summary['sampling']

    fig,axes = plt.subplots(1,2,figsize=(11.4,3.7),constrained_layout=True)
    classes = np.arange(100)
    axes[0].bar(classes,np.array(sampling['bs_class_counts'])/20000*100,color=COLORS['Head'],alpha=.8)
    axes[0].axhline(1,color=COLORS['All'],ls='--',label='Uniform BS expectation')
    axes[0].set(title='Recipient class sampling (BS)',xlabel='Class ID',ylabel='Observed probability (%)')
    axes[1].bar(classes,np.array(sampling['is_class_counts'])/20000*100,color=COLORS['Medium'],alpha=.75,label='Observed')
    axes[1].plot(classes,np.array(sampling['expected_is_probabilities'])*100,color=COLORS['All'],lw=1.5,label='IS expectation')
    axes[1].set(title='Donor class sampling (IS)',xlabel='Class ID',ylabel='Probability (%)')
    for ax in axes:
        ax.axvline(35.5,color='#777777',ls=':',lw=1)
        ax.axvline(70.5,color='#777777',ls=':',lw=1)
        ax.legend(frameon=False)
    save(fig,'01_sampling_distributions')

    fig,axes = plt.subplots(1,2,figsize=(11.4,4),constrained_layout=True)
    for group in GROUPS:
        take = np.ones(20000,dtype=bool) if group == 'All' else labels == group
        ecdf(axes[0],main_data['D'][take],group,COLORS[group])
        ecdf(axes[1],np.ptp(main_data['D'][take],axis=1),group,COLORS[group])
    axes[0].axvline(0,color='#777777',ls=':',lw=1)
    axes[0].set(title='All candidate perturbations',xlabel='D: true-class margin drop (logit units)',ylabel='Empirical CDF')
    axes[1].set(title='Within-recipient spread (K=4, same mask)',xlabel='D max - D min (logit units)',ylabel='Empirical CDF')
    for ax in axes: ax.legend(frameon=False)
    save(fig,'02_margin_and_within_receiver_range')

    fig,axes = plt.subplots(2,2,figsize=(11.4,8),constrained_layout=True)
    for ax,metric,title in zip(axes.ravel(),['cosine_spatial','l2_spatial','cosine_gap','l2_gap'],
        ['Cosine of full feature maps','L2 of full-map fusion change','Cosine after global average pooling','L2 of pooled fusion change']):
        counts = ax.hexbin(main_data[metric].ravel(),main_data['D'].ravel(),gridsize=55,mincnt=1,bins='log',cmap='viridis',linewidths=0)
        corr = summary['analyses']['eval_preprocess']['groups']['All']['correlations'][metric]
        ax.set(title=title,xlabel=metric.replace('_',' '),ylabel='Margin drop D')
        ax.text(.025,.97,f'Pearson r={corr["pearson"]:.3f}\nWithin-recipient r={corr["within_receiver_centered_pearson"]:.3f}',
            transform=ax.transAxes,va='top',bbox={'facecolor':'white','alpha':.85,'edgecolor':'none'})
        fig.colorbar(counts,ax=ax,label='Candidate count (log scale)')
    save(fig,'03_similarity_and_distance_vs_D')

    cosine_delta = np.abs(main_data['cosine_spatial'][:,PAIR_A]-main_data['cosine_spatial'][:,PAIR_B])
    d_delta = np.abs(main_data['D'][:,PAIR_A]-main_data['D'][:,PAIR_B])
    fig,axes = plt.subplots(1,2,figsize=(11.4,4.1),constrained_layout=True)
    density = axes[0].hexbin(cosine_delta.ravel(),d_delta.ravel(),gridsize=60,mincnt=1,bins='log',cmap='viridis',linewidths=0)
    axes[0].set(title='Candidate pairs within the same recipient',xlabel='Absolute cosine difference',ylabel='Absolute D difference')
    fig.colorbar(density,ax=axes[0],label='Pair count (log scale)')
    deciles = summary['analyses']['eval_preprocess']['similarity_delta_deciles']
    x = np.array([v['abs_cosine_difference']['p50'] for v in deciles])
    med = np.array([v['abs_D_difference']['p50'] for v in deciles])
    p25 = np.array([v['abs_D_difference']['p25'] for v in deciles])
    p75 = np.array([v['abs_D_difference']['p75'] for v in deciles])
    p95 = np.array([v['abs_D_difference']['p95'] for v in deciles])
    axes[1].plot(x,med,'o-',color=COLORS['All'],label='Median |D difference|')
    axes[1].fill_between(x,p25,p75,color=COLORS['Head'],alpha=.2,label='25th-75th percentiles')
    axes[1].plot(x,p95,'--',color=COLORS['Tail'],label='95th percentile')
    axes[1].set(title='All 10 cosine-difference rank bins',xlabel='Median absolute cosine difference in bin',ylabel='Absolute D difference')
    axes[1].legend(frameon=False)
    save(fig,'04_similar_cosine_different_D')

    with np.load(ROOT/'eval_preprocess_ranked_groups.npz') as z:
        chosen = {label:z[label.replace('-','_')+'_position'] for label in ['Random','Low-D','Mid-D','High-D']}
    fig,axes = plt.subplots(1,2,figsize=(11.4,4),constrained_layout=True)
    for ax,group in zip(axes,['All','Tail']):
        take = np.ones(20000,dtype=bool) if group == 'All' else labels == group
        idx = np.flatnonzero(take)
        for label,position in chosen.items():
            ecdf(ax,main_data['D'][idx,position[idx]],label,COLORS[label])
        ax.axvline(0,color='#777777',ls=':',lw=1)
        ax.set(title=f'{group}: descriptive choices from K=4',xlabel='Margin drop D',ylabel='Empirical CDF')
        ax.legend(frameon=False)
    save(fig,'05_random_low_mid_high_D')

    fig,axes = plt.subplots(1,2,figsize=(11.4,4),constrained_layout=True)
    events = ['prediction_boundary_cross','strongest_competitor_changed','mixed_fused_correctness_groups']
    event_names = ['Predicted class\nchanged','Strongest rival\nchanged','K=4 mixed\ncorrectness']
    x = np.arange(3)
    for i,group in enumerate(['Head','Medium','Tail']):
        values = [summary['analyses']['eval_preprocess']['groups'][group][event]['percent'] for event in events]
        axes[0].bar(x+(i-1)*.24,values,width=.24,label=group,color=COLORS[group])
    axes[0].set_xticks(x,event_names)
    axes[0].set(title='Reference classifier decision differences',ylabel='Observed proportion (%)')
    axes[0].legend(frameon=False)
    for group,style in [('All','-'),('Tail','--')]:
        take = np.ones(20000,dtype=bool) if group == 'All' else labels == group
        ecdf(axes[1],np.ptp(main_data['D'][take],1),f'{group}: eval preprocessing',COLORS[group],style)
        ecdf(axes[1],np.ptp(aug_data['D'][take],1),f'{group}: fixed train augmentation',COLORS['Head' if group == 'All' else 'Medium'],style)
    axes[1].set(title='Preprocessing sensitivity, paired samples/masks',xlabel='Within-recipient D range',ylabel='Empirical CDF')
    axes[1].legend(frameon=False,fontsize=8)
    save(fig,'06_decisions_and_augmentation_sensitivity')

    images = np.load(ROOT/'train_subset_images.npy',mmap_mode='r')
    with (DATA/'cifar-100-python/meta').open('rb') as stream:
        meta = pickle.load(stream,encoding='latin1')
    class_names = meta['fine_label_names']
    examples = summary['analyses']['eval_preprocess']['examples']
    fig,axes = plt.subplots(len(examples),5,figsize=(14,2.3*len(examples)),constrained_layout=True)
    for row,example in enumerate(examples):
        i = example['group_id']
        indices = [example['receiver_index'],*example['donor_indices']]
        for column,index in enumerate(indices):
            ax = axes[row,column]
            ax.imshow(images[index],interpolation='nearest')
            ax.grid(False)
            ax.set_xticks([]); ax.set_yticks([])
            if column == 0:
                ax.set_title(f'{example["receiver_group"]} range q{int(example["range_quantile_target"]*100)}\n'
                    f'Receiver {index}: {class_names[example["receiver_class"]]}\n'
                    f'm0={example["baseline_margin"]:.3f}; range={example["D_range"]:.3f}',fontsize=8)
            else:
                k = column-1
                rank = example['ascending_D_order_zero_based'].index(k)+1
                ax.set_title(f'Candidate {k}: {class_names[example["donor_class"]]} (idx {index})\n'
                    f'D={example["D"][k]:+.3f}; cosine={example["cosine_spatial"][k]:.3f}\n'
                    f'm1={example["fused_margin"][k]:+.3f}; D rank={rank}',fontsize=8)
    fig.suptitle('Examples selected mechanically at 10th / 50th / 90th within-group D-range ranks\n'
        'Not selected by decision outcome or candidate benefit; official eval preprocessing on training images',fontsize=12)
    save(fig,'07_systematic_candidate_examples')
    print(f'FIGURES COMPLETE: {len(list(FIGURES.glob("*.png")))} PNG and SVG pairs',flush=True)


if __name__ == '__main__':
    main()
