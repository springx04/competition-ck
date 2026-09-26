"""Plot the supplied Q1 Stage 2/3 observations; no inference or production writeback."""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
import build_paper_figures as base

RUN = base.ROOT / 'Q1/q1_features/runs'
S2 = RUN / 'q1_alignment_repair_stage2/reports'
S3 = RUN / 'q1_alignment_repair_stage3/reports'
FORMAL = RUN / 'q1_full_20260924'
STAGES = ['Formal', 'Stage 2', 'Stage 3']
STATE_COLORS = ['#277DA8', '#45A990', '#E9AE49', '#98A3AF', '#D46D82']


def build(m=base):
    def save(fig, name, title, data, paths, caption):
        m.save(fig, name, title, data, [p.relative_to(m.ROOT).as_posix() for p in paths],
               caption + ' Stage 2/3为随新版提供的自动候选；Stage 4仅有报告，未据此伪造最终数组。')

    cov = pd.read_csv(S3/'coverage_comparison_stage3.csv')
    groups = ['all100', 'original_51_isolated', 'original_37_visual_risk']
    labels = ['All 100', 'Isolated 51', 'Visual risk 37']
    fig, axes = plt.subplots(1, 4, figsize=(12.8, 3.4), layout='constrained')
    for ax, col, title in zip(axes, ['text', 'audio', 'vision', 'trimodal'], ['Text', 'Audio', 'Vision', 'Trimodal']):
        for group, label, color in zip(groups, labels, m.COLORS):
            values = cov[cov.group.eq(group)].set_index('version').loc[['formal','stage2','stage3'], col+'_observed_fraction']
            ax.plot(range(3), values*100, 'o-', label=label, color=color, lw=2, ms=5)
        ax.set(xticks=range(3), xticklabels=STAGES, ylim=(-3,104), title=title, ylabel='Usable / observed bins (%)')
        m.nice(ax)
    axes[0].legend(fontsize=8, loc='lower left')
    save(fig, 'Q1_11_stage_coverage', '正式提取与两阶段候选覆盖变化', cov, [S3/'coverage_comparison_stage3.csv'],
         '各组按50-bin的observed_mask计数；Formal是原始保留观测，修复后是质量控制后的可用观测。全100条Vision为57.46%→70.10%→76.04%，音频99.50%→78.28%→78.28%；音频下降来自主动隔离，不是提取器退化。分组重叠，不能相加。')

    index = pd.read_csv(FORMAL/'sample_index.csv').sort_values('sample_index')
    arrays = [FORMAL/'features/q1_compact50.npz', S2/'q1_compact50_repair_candidate.npz', S3/'q1_compact50_stage3_candidate.npz']
    frames=[]
    for stage, path in zip(STAGES, arrays):
        with np.load(path, allow_pickle=False) as z:
            assert np.array_equal(z['sample_index'], index.sample_index)
            rates=z['observed_mask'].mean(axis=1)
            for j, mod in enumerate('TAV'):
                frames.append(pd.DataFrame(dict(sample_id=index.sample_id, sample_index=index.sample_index, stage=stage, modality=mod, rate=rates[:,j])))
    rates=pd.concat(frames, ignore_index=True)
    paired=rates.pivot(index=['sample_id','sample_index','modality'], columns='stage', values='rate').reset_index()
    paired['change_pp']=(paired['Stage 3']-paired['Formal'])*100
    paired['stage3_increment_pp']=(paired['Stage 3']-paired['Stage 2'])*100
    fig, ax=m.canvas(2)
    sns.violinplot(data=paired, x='modality', y='change_pp', hue='modality', palette=m.MOD, cut=0, inner='quart', legend=False, ax=ax[0])
    ax[0].axhline(0,color='#657488',lw=1); ax[0].set(xlabel='',ylabel='Stage 3 − Formal (pp)',title='All 100 clips / modality')
    sns.ecdfplot(data=paired[paired.modality.eq('V')],x='stage3_increment_pp',color=m.MOD['V'],ax=ax[1])
    ax[1].set(xlabel='Stage 3 − Stage 2 vision (pp)',ylabel='Cumulative fraction',title='Increment within the repair pipeline')
    save(fig,'Q1_12_paired_coverage','全部样本覆盖变化分布',paired,arrays,
         '同一sample_index逐样本配对，保留零变化和负变化。左图跨质量口径描述变化，不能解释为准确率增益；右图仅比较同属候选流程的Stage 2/3视觉mask。pp为百分点。')

    visual=pd.read_csv(S3/'vision_summary_stage3.csv')
    visual=visual[visual.original_identity_risk.eq(1)].copy()
    status=pd.read_csv(S3/'final_resolution_status_stage3.csv')
    visual=visual.merge(status[['sample_id','sample_index','stage2_visual_status']],on='sample_id',validate='one_to_one')
    visual=visual.sort_values(['total_usable_visual_fraction','sample_index'],ascending=[False,True])
    fields=['direct_talknet_verified_fraction','continuity_recovered_fraction','ambiguous_fraction','missing_fraction','unresolved_fraction']
    names=['Anchor','Continuity','Ambiguous','Missing','Unresolved']
    fig, ax=plt.subplots(1,2,figsize=(12.5,5),gridspec_kw={'width_ratios':[2.2,1]},layout='constrained')
    left=np.zeros(len(visual))
    for field,label,color in zip(fields,names,STATE_COLORS):
        ax[0].barh(range(len(visual)),visual[field]*100,left=left,color=color,label=label,height=.9); left+=visual[field].to_numpy()*100
    ax[0].set(yticks=range(len(visual)),yticklabels=[f'S{i:03}' for i in visual.sample_index],xlim=(0,100),xlabel='Clip duration (%)',title='All 37 original visual-risk clips')
    ax[0].tick_params(axis='y',labelsize=6);ax[0].invert_yaxis()
    means=visual[fields].mean()*100
    ax[1].barh(names,means,color=STATE_COLORS)
    for j,v in enumerate(means):ax[1].text(v+.5,j,f'{v:.1f}%',va='center',fontsize=9)
    ax[1].set(xlim=(0,max(means)*1.25),xlabel='Mean clip duration (%)',title='Sample-weighted composition');ax[1].invert_yaxis()
    save(fig,'Q1_13_visual_composition','视觉身份风险样本的可用与隔离成分',visual,[S3/'vision_summary_stage3.csv',S3/'final_resolution_status_stage3.csv'],
         '37条原视觉风险全部保留。直接TalkNet确认平均33.55%，ArcFace连续性新增17.89%，合计51.44%；歧义、自然缺失与未决均显示。S编号对应CSV的零基sample_index；右图按样本等权，不是按总时长加权。')

    stage2_order=['fully_verified','partially_verified','unresolved']
    stage3_order=['FULLY_VERIFIED','PARTIALLY_VERIFIED','AMBIGUOUS','UNRESOLVED']
    trans=pd.crosstab(visual.stage2_visual_status,visual.stage3_visual_status).reindex(index=stage2_order,columns=stage3_order,fill_value=0)
    fig,ax=m.canvas(size=(7,3.6))
    sns.heatmap(trans,annot=True,fmt='d',cmap='YlGnBu',linewidths=1,cbar_kws={'label':'Clips'},ax=ax)
    ax.set(xticklabels=['Full','Partial','Ambiguous','Unresolved'],yticklabels=['Full','Partial','Unresolved'],xlabel='Stage 3',ylabel='Stage 2',title='Visual-state transitions / 37 clips');ax.tick_params(axis='y',rotation=0)
    save(fig,'Q1_14_visual_transitions','视觉风险样本状态转移',visual[['sample_id','sample_index','stage2_visual_status','stage3_visual_status']],[S3/'vision_summary_stage3.csv',S3/'final_resolution_status_stage3.csv'],
         'Stage 2为8/14/15条full/partial/unresolved；Stage 3为16/6/5/10条full/partial/ambiguous/unresolved。状态来自自动规则，full不代表人工Gold，ambiguous不并入成功。')

    sens=pd.read_csv(S3/'visual_gap_sensitivity_stage3.csv');fig,ax=m.canvas(2)
    for col,label,color in zip(['direct_fraction','continuity_fraction','usable_fraction'],['Anchor','Continuity','Total'],m.COLORS):
        ax[0].plot(sens.max_identity_gap_s,sens[col]*100,'o-',label=label,color=color)
    ax[0].set(xlabel='Maximum identity gap (s)',ylabel='Mean usable duration (%)',ylim=(0,58),xticks=sens.max_identity_gap_s,title='Fixed gap sensitivity');ax[0].legend()
    bottom=np.zeros(3)
    for col,label,color in zip(['fully_verified','partially_verified','ambiguous','unresolved'],['Full','Partial','Ambiguous','Unresolved'],STATE_COLORS[:4]):
        ax[1].bar(range(3),sens[col],bottom=bottom,color=color,label=label,width=.6);bottom+=sens[col]
    ax[1].set(xticks=range(3),xticklabels=['0.5','1.0*','1.5'],xlabel='Gap (s); * primary configuration',ylabel='Clips',ylim=(0,40),title='Same sample states across settings');ax[1].legend(fontsize=8,loc='upper center',ncol=2)
    save(fig,'Q1_15_gap_sensitivity','视觉身份连续间隔敏感性',sens,[S3/'visual_gap_sensitivity_stage3.csv'],
         '0.5/1.0/1.5秒平均可用比例50.25%/51.44%/51.44%，37条状态计数相同。主配置1.0秒固定，不据此宣称最优；增加间隔并不必然增加恢复率。')

    mapping=pd.read_csv(S3/'mapping_validation_stage3.csv');fig,ax=m.canvas(3,size=(12,3.6))
    short=['iRB / 3','iRB / 6','mJ2 / 1','ri04 / 2']
    for a,prefix,title in zip(ax[:2],['whisper_token','ctc_token'],['Whisper token overlap','CTC token overlap']):
        for j,r in mapping.iterrows():
            a.plot([r[prefix+'_original_score'],r[prefix+'_candidate_score']],[j,j],c='#B0BCC7',lw=2)
        a.scatter(mapping[prefix+'_original_score'],range(4),c=m.COLORS[0],label='Official',s=40)
        a.scatter(mapping[prefix+'_candidate_score'],range(4),c=m.COLORS[1],label='Candidate',s=40)
        a.set(yticks=range(4),yticklabels=short,xlim=(-.02,.56),xlabel='Token overlap score',title=title);a.invert_yaxis();a.legend(fontsize=8,loc='lower right')
    ax[2].barh(short,mapping.cross_metric_agreement*100,color=[m.COLORS[1],m.COLORS[2],m.COLORS[4],m.COLORS[4]])
    ax[2].set(xlim=(0,105),xlabel='Assignment agreement (%)',title='Four metrics / no clip swaps');ax[2].invert_yaxis()
    save(fig,'Q1_16_mapping_evidence','四条映射疑点的交叉证据',mapping,[S3/'mapping_validation_stage3.csv'],
         '显示原官方与候选的文本重合得分及四指标分配一致率。只有iRB/6达到MAPPING_STRONG；ri04/2虽100%一致但绝对分数低，仍未决。所有官方mapping保留，没有自动交换clip。')

    timing=pd.read_csv(S2/'ctc_vs_official_forced_alignment.csv');fig,ax=m.canvas(2)
    for col,label,color in zip(['midpoint_median','midpoint_p90'],['Clip median','Clip P90'],m.COLORS):
        sns.ecdfplot(data=timing.dropna(subset=[col]),x=col,label=label,color=color,ax=ax[0])
    ax[0].set(xlabel='CTC / forced-alignment difference (s)',ylabel='Cumulative fraction',title='Agreement on matched words');ax[0].legend()
    sns.scatterplot(data=timing,x='common_word_fraction',y='midpoint_median',hue='temporal_alignment_status',palette=m.COLORS[:3],s=35,alpha=.8,ax=ax[1])
    ax[1].set(xlabel='Common-word fraction',ylabel='Median time difference (s)',title='Coverage and timing are separate');ax[1].legend(title='',fontsize=8)
    save(fig,'Q1_17_alignment_agreement','CTC与官方文本强制对齐的一致性',timing,[S2/'ctc_vs_official_forced_alignment.csv'],
         '100条原始记录随CSV保留；ECDF和散点只使用对应字段非空的样本，不用零填补未适用记录。两个自动对齐器的一致性不是人工时间准确率，低共同词比例也不能因小时间差被忽略。')

    threshold=pd.read_csv(S2/'alignment_threshold_sensitivity.csv');fig,ax=m.canvas(2)
    for a,(group,g) in zip(ax,threshold.groupby('group',sort=False)):
        table=g.pivot(index='midpoint_median_max_s',columns='midpoint_p90_max_s',values='verified')
        sns.heatmap(table,annot=True,fmt='d',cmap='YlGnBu',vmin=0,vmax=int(g.sample_count.iloc[0]),ax=a,cbar_kws={'label':'Verified clips'},linewidths=1)
        a.set(xlabel='P90 threshold (s)',ylabel='Median threshold (s)',title=f'{"All clips" if group=="all100" else "Original text/audio risk"} / n={g.sample_count.iloc[0]}');a.tick_params(axis='y',rotation=0)
        a.add_patch(plt.Rectangle((1,1),1,1,fill=False,edgecolor='#D15D78',lw=2.5))
    save(fig,'Q1_18_alignment_thresholds','时间一致性阈值敏感性',threshold,[S2/'alignment_threshold_sensitivity.csv'],
         '红框为固定主阈值median≤0.3秒、P90≤0.6秒。全100条71 verified、10 partial、19 not-applicable；原23条风险仅1 verified。分面色标分别以各自总数为上限，不能跨面板按深浅比较比例，也不根据最大通过数选阈值。')

    resolution=['RESOLVED_VALID','RESOLVED_PARTIAL','RESOLVED_CONFLICT','RESOLVED_MISSING','UNRESOLVED']
    state_labels=['Valid','Partial','Conflict','Missing','Unresolved']
    fig,ax=m.canvas(2,size=(12,4))
    counts=status.resolution_status.value_counts().reindex(resolution,fill_value=0)
    ax[0].barh(state_labels,counts,color=STATE_COLORS)
    for j,v in enumerate(counts):ax[0].text(v+.7,j,str(v),va='center')
    ax[0].set(xlim=(0,65),xlabel='Clips',title='Five resolution states / all 100');ax[0].invert_yaxis()
    table=pd.crosstab(status.semantic_status,status.resolution_status).reindex(columns=resolution,fill_value=0)
    sns.heatmap(table,annot=True,fmt='d',cmap='YlGnBu',ax=ax[1],linewidths=1,cbar_kws={'label':'Clips'})
    ax[1].set(xticklabels=state_labels,yticklabels=[s.replace('MATCH_','').title() for s in table.index],xlabel='Resolution state',ylabel='Semantic state',title='Independent axes remain visible');ax[1].tick_params(axis='y',rotation=0)
    save(fig,'Q1_19_resolution_axes','最终候选状态与独立语义轴',status,[S3/'final_resolution_status_stage3.csv'],
         '五状态计数57/19/5/4/15。交叉表保留总状态可能遮蔽的语义冲突：UNRESOLVED并不等于只有一种问题；Conflict表示安全mask方案明确，而非原内容被修好。')

    audio=paired[paired.modality.eq('A')].merge(status[['sample_id','semantic_status','resolution_status']],on='sample_id',validate='one_to_one')
    audio_long=audio.melt(id_vars=['sample_id','semantic_status'],value_vars=STAGES,var_name='stage',value_name='fraction')
    fig,ax=m.canvas(size=(10,4))
    sns.boxplot(data=audio_long,x='semantic_status',y='fraction',hue='stage',palette=m.COLORS[:3],showfliers=True,fliersize=2,ax=ax)
    ax.set(xlabel='Semantic pairing',ylabel='Audio observed / usable fraction',ylim=(-.04,1.06),title='Audio is deliberately masked when pairing is unsafe');ax.legend(title='')
    save(fig,'Q1_20_semantic_masking','语义冲突与未决音频的安全屏蔽',audio_long,[*arrays,S3/'final_resolution_status_stage3.csv'],
         '逐样本音频50-bin观测率按最终语义状态分组；Formal与候选的质量口径不同。Conflict/Unresolved组主动屏蔽音频，表明覆盖率下降可能是安全处理结果，不能把它包装成提取失败或隐去。')

    with np.load(S3/'q1_word_aligned_stage3_candidate.npz',allow_pickle=False) as z:
        wordframes=[]
        for mod,stem in [('A','audio'),('V','vision')]:
            counts=np.diff(z[stem+'_indptr'])
            weight_sums=np.array([z[stem+'_weights'][a:b].sum() for a,b in zip(z[stem+'_indptr'][:-1],z[stem+'_indptr'][1:])])
            wordframes.append(pd.DataFrame(dict(sample_id=z['sample_id'],word_id=z['word_id'],modality=mod,observed=z[stem+'_mask'],coverage=z[stem+'_coverage'],source_count=counts,weight_sum=weight_sums)))
    words=pd.concat(wordframes,ignore_index=True);fig,ax=m.canvas(2)
    sns.violinplot(data=words,x='modality',y='source_count',hue='modality',palette=m.MOD,cut=0,inner='quart',legend=False,ax=ax[0])
    ax[0].set(xlabel='',ylabel='Source rows / word',title='CSR references / all 1932 words')
    for mod in 'AV':
        sns.ecdfplot(data=words[(words.modality==mod)&(words.observed==1)],x='coverage',label=mod,color=m.MOD[mod],ax=ax[1])
    ax[1].set(xlabel='Coverage among usable words',ylabel='Cumulative fraction',title='Traceability is separate from coverage');ax[1].legend()
    save(fig,'Q1_21_word_sources','词级候选CSR来源与覆盖分布',words,[S3/'q1_word_aligned_stage3_candidate.npz'],
         '每个词的音频/视觉源行数量由CSR indptr相邻差计算，所有1932条均保留，包括零来源和mask词。可用词覆盖ECDF条件于mask=1。Stage 3词级NPZ自带CSR，不能与仍缺失的Formal样本级map_*.npz混为一谈；来源数量多不等于质量高。')

    segments=pd.read_csv(S3/'vision_segment_stage3.csv')
    segments=segments.merge(status[['sample_id','sample_index']],on='sample_id',validate='many_to_one')
    order=segments.groupby('sample_id').sample_index.first().sort_values().index.tolist()
    segment_states=['VERIFIED_ACTIVE','VERIFIED_CONTINUITY','AMBIGUOUS','MISSING','UNRESOLVED']
    palette=dict(zip(segment_states,STATE_COLORS))
    fig,ax=plt.subplots(figsize=(12,7),layout='constrained')
    for row,sid in enumerate(order):
        g=segments[segments.sample_id.eq(sid)]; duration=g.end.max()
        for state,z in g.groupby('segment_status'):
            ax.broken_barh(list(zip(z.start/duration*100,(z.end-z.start)/duration*100)),(row-.4,.8),facecolors=palette[state])
    ax.set(xlim=(0,100),ylim=(-.7,len(order)-.3),yticks=range(len(order)),yticklabels=[f'S{int(status.set_index("sample_id").loc[sid,"sample_index"]):03}' for sid in order],xlabel='Relative clip time (%)',title='Observed visual-state timelines / 41 affected clips')
    ax.tick_params(axis='y',labelsize=7);ax.invert_yaxis()
    ax.legend(handles=[Patch(color=palette[s],label=l) for s,l in zip(segment_states,names)],loc='upper center',bbox_to_anchor=(.5,1.0),ncol=5,fontsize=9)
    save(fig,'Q1_22_visual_timelines','受影响样本的视觉状态时间分布',segments,[S3/'vision_segment_stage3.csv',S3/'final_resolution_status_stage3.csv'],
         '全部41条受影响样本，包括37条身份风险与4条no-face。颜色表示真实时间段状态；横轴按各clip时长归一化用于比较结构，原始start/end秒数保留在CSV中。传播未跨越歧义或自然缺失，不把mask补成平滑连续数据。')


if __name__ == '__main__':
    old=json.loads((base.OUT/'figure_manifest.json').read_text(encoding='utf8'))
    build()
    ids={e['id'] for e in base.ENTRIES}
    base.ENTRIES[:]=sorted([e for e in old if e['id'] not in ids]+base.ENTRIES,key=lambda e:e['id'])
    base.index()
