"""Rebuild paper figures from saved observations; no model training or invented data.

Run from any directory with Python, numpy, pandas, matplotlib, seaborn and Pillow.
Every figure has PNG/PDF/SVG, an exact plotting table and a provenance entry.
"""
from pathlib import Path
import json
import re
import html
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'doc/paper_figures'
for sub in ('figures', 'data', 'previews'):
    (OUT / sub).mkdir(parents=True, exist_ok=True)
COLORS = ['#277DA8', '#F08A62', '#45A990', '#9874B8', '#D6AA3B', '#D15D78']
MOD = dict(zip(['T', 'A', 'V', 'J'], COLORS))
sns.set_theme(style='ticks', context='paper', font='DejaVu Sans', palette=COLORS)
np.random.seed(20260926)
plt.rcParams.update({'font.size': 10, 'axes.titlesize': 12, 'axes.labelsize': 10,
                     'legend.fontsize': 9, 'axes.spines.top': False,
                     'axes.spines.right': False, 'svg.fonttype': 'none',
                     'pdf.fonttype': 42, 'savefig.facecolor': 'white'})
ENTRIES = []

def readj(p):
    return json.loads((ROOT / p).read_text(encoding='utf-8'))

def readc(p):
    return pd.read_csv(ROOT / p)

def canvas(n=1, size=None, **kwargs):
    return plt.subplots(1, n, figsize=size or (5.2*n, 3.9), layout='constrained', **kwargs)

def save(fig, name, title, data, sources, caption):
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)
    data.to_csv(OUT / 'data' / f'{name}.csv', index=False, encoding='utf-8-sig')
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(OUT / 'figures' / f'{name}.{ext}', dpi=220, bbox_inches='tight', pad_inches=.08)
    plt.close(fig)
    ENTRIES.append(dict(id=name, title=title, rows=len(data), sources=sources, caption=caption))
    print(name, flush=True)

def nice(ax, xlabel=None, ylabel=None):
    ax.grid(axis='y', color='#DFE5EB', alpha=.65, linewidth=.6)
    ax.set_axisbelow(True)
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)

def zoom_positive(ax, data):
    """Explicitly label a point/line plot with a positive, nonzero lower bound."""
    lower=float(data.ci_low.min()); upper=float(data.ci_high.max())
    pad=max((upper-lower)*.15,abs(upper)*.03)
    ax.set_ylim(lower-pad,upper+pad)
    ax.text(.02,.03,'Zoomed y-axis',transform=ax.transAxes,fontsize=8,color='#6A7787')

def q1_figures():
    src='Q1/q1_features/runs/q1_full_20260924/reports/quality.csv'
    q=readc(src)
    fig,ax=canvas(2)
    sns.histplot(q.duration, bins=18, ax=ax[0], color=COLORS[0], edgecolor='white')
    nice(ax[0], 'Decoded duration (s)', 'Samples')
    ax[0].axvline(q.duration.median(), color=COLORS[1], ls='--', label=f'Median {q.duration.median():.2f} s')
    ax[0].legend(); ax[0].set_title('100 original clips')
    sns.scatterplot(data=q,x='spoken_words',y='duration',hue='paired_use',palette={0:COLORS[1],1:COLORS[0]},ax=ax[1],s=38,alpha=.8)
    nice(ax[1], 'Reference words', 'Decoded duration (s)'); ax[1].set_title('Length and pairing eligibility')
    save(fig,'Q1_01_duration','原始样本长度与可配对状态',q, [src], '100条实际解码时长与词数。paired_use为当前流程可配对标志，不是人工正确率；解码时长与题面标称时长口径不同。')
    fig,ax=canvas(2)
    cols=['aligned_word_fraction','vision_eligible_fraction']
    long=q.melt(id_vars='sample_id',value_vars=cols,var_name='measure',value_name='fraction')
    long['measure']=long.measure.map(dict(zip(cols,['Accepted words','Eligible vision'])))
    sns.violinplot(data=long,x='measure',y='fraction',hue='measure',palette=COLORS[:2],cut=0,inner='quart',legend=False,ax=ax[0])
    sns.stripplot(data=long,x='measure',y='fraction',color='#354657',size=2,alpha=.45,jitter=.17,ax=ax[0])
    ax[0].set(xlabel='',ylabel='Fraction',ylim=(-.03,1.04),title='Coverage across all clips')
    sns.scatterplot(data=q,x='diagnostic_wer',y='aligned_word_fraction',hue='paired_use',palette={0:COLORS[1],1:COLORS[0]},ax=ax[1],s=32)
    nice(ax[1],'Diagnostic WER','Accepted-word fraction'); ax[1].set_title('Alignment acceptance is not correctness')
    save(fig,'Q1_02_coverage','文本定位与视觉有效覆盖率',q,[src], '小提琴由100条样本的实际比例估计，cut=0，保留极端值。CTC接受率不等于人工对齐精度；WER来自诊断转写，不是情感分类误差。')
    fig,ax=canvas(size=(7,3.5))
    order=q.sort_values('terminal_difference_s')
    ax.plot(range(1,101),order.first_pts_difference_s,color=COLORS[2],lw=2,label='First PTS')
    ax.plot(range(1,101),order.terminal_difference_s,color=COLORS[3],lw=2,label='Terminal')
    ax.fill_between(range(1,101),order.first_pts_difference_s,order.terminal_difference_s,color=COLORS[3],alpha=.12)
    nice(ax,'Clip rank by terminal difference','Audio − video clock (s)'); ax.legend(); ax.set_xlim(1,100)
    save(fig,'Q1_03_clock','声画时钟起点与终点差异',q[['sample_id','first_pts_difference_s','terminal_difference_s']],[src], '声画媒体时钟诊断，不代表说话人身份正确或单词对齐精度。全体样本保留，零线方便区分提前与滞后。')
    fig,ax=canvas(2)
    sns.scatterplot(data=q,x='vision_episode_count',y='vision_eligible_fraction',hue='paired_use',palette={0:COLORS[1],1:COLORS[0]},ax=ax[0],s=36,alpha=.8)
    nice(ax[0],'Face episodes','Eligible vision fraction')
    rank=q.sort_values('aligned_word_fraction').reset_index(drop=True)
    ax[1].plot(np.arange(1,101),rank.aligned_word_fraction,color=COLORS[0],label='Accepted words')
    ax[1].plot(np.arange(1,101),rank.vision_eligible_fraction,color=COLORS[2],alpha=.85,label='Eligible vision')
    nice(ax[1],'Clip rank by word acceptance','Fraction'); ax[1].legend()
    save(fig,'Q1_04_quality_rank','逐样本质量差异与视觉片段数',q,[src], '右图按文本接受率升序排列，视觉值使用同一排列而非独立排序。揭示低覆盖样本，不能声称所有样本均达到高质量。')
    p='Q1/q1_features/runs/q1_full_20260924/reports/auto_resolution/problem_partition.csv'
    part=readc(p); c=part.group.value_counts().rename_axis('group').reset_index(name='n')
    fig,ax=canvas(size=(6.5,3.5)); ax.barh(c.group.str.replace('_',' '),c.n,color=COLORS[:len(c)])
    for i,v in enumerate(c.n): ax.text(v+.3,i,str(v),va='center')
    ax.set(xlabel='Samples',title=f'Automatic follow-up subset (n={len(part)})'); ax.set_xlim(0,c.n.max()*1.15)
    save(fig,'Q1_05_problem_partition','自动补查样本的问题分区',c,[p], '仅统计problem_partition中进入补查的样本，分母不是全部100条；身份与声文问题可以并存。该图是困难样本构成，不是消融提升。')
    p='Q1/q1_features/runs/q1_full_20260924/reports/auto_resolution/arcface_episode_similarity.csv'; arc=readc(p)
    fig,ax=canvas(2)
    sns.histplot(data=arc,x='cosine_similarity',bins=22,ax=ax[0],color=COLORS[3],edgecolor='white')
    nice(ax[0],'Episode cosine similarity','Episode pairs')
    sns.scatterplot(data=arc,x='episode_a_dispersion',y='cosine_similarity',ax=ax[1],color=COLORS[3],alpha=.5,s=23)
    nice(ax[1],'Episode A embedding dispersion','Cosine similarity')
    save(fig,'Q1_06_identity_similarity','人脸片段身份相似度与嵌入离散程度',arc,[p],'所有已保存片段对的ArcFace余弦值；片段对共享样本且不独立。余弦阈值属于自动身份聚类判断，不是人工身份准确率；该图诊断质量，不证明说话人已经确认。')
    p='Q1/q1_features/runs/q1_full_20260924/reports/auto_resolution/av_sync_scores.csv'; sync=readc(p)
    fig,ax=canvas(2)
    sns.scatterplot(data=sync,x='null_p95',y='real_sync_score',hue='visual_signal_basis',palette=COLORS[:sync.visual_signal_basis.nunique()],ax=ax[0],alpha=.65,s=26)
    lo=min(sync.null_p95.min(),sync.real_sync_score.min())-.02; hi=max(sync.null_p95.max(),sync.real_sync_score.max())+.02
    ax[0].plot([lo,hi],[lo,hi],'--',c='#627387',lw=1); nice(ax[0],'Null 95th percentile','Observed sync score'); ax[0].legend(fontsize=6)
    sns.histplot(data=sync,x='empirical_percentile',bins=np.linspace(0,1,11),ax=ax[1],color=COLORS[2],edgecolor='white'); nice(ax[1],'Rank within diagnostic null sets','Identity clusters')
    save(fig,'Q1_07_sync_null','视听同步得分与错位随机参照',sync,[p],'嘴部运动与音频能量的诊断相关统计，对照来自移位与跨样本组合。零假设排序不是经过多重比较校正的显著性，也不是说话人准确率；低于null p95的真实分数如实保留。')
    p='Q1/q1_features/runs/q1_full_20260924/reports/auto_resolution/arcface_episode_quality.csv'; quality=readc(p)
    fig,ax=canvas(2)
    sns.scatterplot(data=quality,x='selected_frame_count',y='valid_embedding_count',hue='face_id',palette='viridis',ax=ax[0],s=35,alpha=.65)
    nice(ax[0],'Selected frames','Valid face embeddings'); ax[0].legend(title='Face ID',fontsize=7)
    sns.histplot(data=quality,x='dispersion',bins=20,ax=ax[1],color=COLORS[3],edgecolor='white'); nice(ax[1],'Within-episode dispersion','Face episodes')
    save(fig,'Q1_08_identity_quality','身份嵌入的成功数量与稳定性',quality,[p],'统计已检测到的人脸episode，分母不是100条原始样本。成功嵌入数量与片段内一致性分开呈现，未把没有人脸的样本当成人脸聚类正确。')

def q2_figures():
    src='Q2/results/experiment_ledger.json'; ledger=readj(src)
    hist=[]
    for t in ledger['trials']:
        trial=t['checkpoint'].split('/')[1] if t['checkpoint'].startswith('experiments/') else t['checkpoint'].replace('/best.pt','')
        for row in t.get('history',[]): hist.append(dict(trial=trial,seed=t['seed'],checkpoint=t['checkpoint'],**row))
    history=pd.DataFrame(hist)
    for col in history.columns.difference(['trial','checkpoint']): history[col]=pd.to_numeric(history[col],errors='coerce')
    history.to_csv(OUT/'data/Q2_all_epoch_history.csv',index=False,encoding='utf-8-sig')
    final=history[history.trial=='attn_bert8_all_v1']
    fig,ax=canvas(2)
    for col,c,lab in zip(['total','task_clean','task_corrupt'],COLORS,['Total','Clean','Corrupt']): ax[0].plot(final.epoch,final[col],marker='o',ms=3,color=c,label=lab)
    ax[0].axvspan(.7,2.5,color='#DDE7EF',alpha=.55); nice(ax[0],'Epoch','Mean batch loss'); ax[0].legend(); ax[0].set_title('Saved epoch aggregates')
    for col,c,lab in zip(['clean_macro_f1','missing_macro_f1'],COLORS,['Clean','72-scenario mean']):
        f=final.dropna(subset=[col]); ax[1].plot(f.epoch,f[col],marker='o',color=c,label=lab)
    ax[1].axvline(10,color='#657487',ls='--',label='Selected epoch 10'); nice(ax[1],'Epoch','Validation Macro-F1'); ax[1].legend(); ax[1].set_title('Validation only at 5, 10, 15, 20')
    save(fig,'Q2_01_training','最终模型训练损失与验证曲线',final,[src], '左图是逐轮内各batch等权平均，不是逐step。第3轮总损失增加是加入受损视图损失；右图只连接实际记录的四个验证点。训练checkpoint与FP16导出包数值口径不同。')
    fig,ax=canvas(2)
    for i,t in enumerate(['attn_bert8_all_v1','attn_bert8_all_seed1112_v1','attn_bert8_all_seed1113_v1']):
        h=history[history.trial==t]
        ax[0].plot(h.epoch,h.task_clean,color=COLORS[i],label=str(int(h.seed.iloc[0])) if len(h) else t)
        h=h.dropna(subset=['missing_macro_f1']); ax[1].plot(h.epoch,h.missing_macro_f1,'o-',color=COLORS[i])
    nice(ax[0],'Epoch','Clean loss'); ax[0].legend(title='Seed'); nice(ax[1],'Epoch','Validation missing Macro-F1')
    save(fig,'Q2_02_seed_training','固定结构三种子训练波动',history[history.trial.str.startswith('attn_bert8_all')],[src], '三个种子均为BERT8全层微调；部署种子预先固定1111。图呈现训练随机性，不能用单seed优势代表多seed稳定优势。')
    fig,ax=canvas(2)
    ax[0].plot(final.epoch,final.lr*1e5,'o-',color=COLORS[0],label='Fusion / projection'); ax[0].plot(final.epoch,final.lr*1e4,'o-',color=COLORS[3],label='BERT')
    nice(ax[0],'Epoch',r'Learning rate ($10^{-5}$)'); ax[0].legend()
    h=history[history.trial.isin(['attn_40_v4','attn_bert8_all_v1','bert8_ablate_no_corruption_v1'])]
    for i,(t,g) in enumerate(h.groupby('trial')):
        g=g.dropna(subset=['missing_macro_f1']); ax[1].plot(g.epoch,g.missing_macro_f1,'o-',color=COLORS[i],label={'attn_40_v4':'BERT4 / 40 epochs','attn_bert8_all_v1':'Final BERT8','bert8_ablate_no_corruption_v1':'No corruption'}[t])
    nice(ax[1],'Epoch','Validation missing Macro-F1'); ax[1].legend()
    save(fig,'Q2_03_schedule','学习率与延长训练的验证表现',h,[src], 'BERT学习率为下游学习率0.1倍，左图由保存lr与最终配置计算；40轮BERT4与BERT8结构和配置不同，只能视为探索，不能把差值归于训练轮数。左图数据见Q2_01_training.csv的lr列。')
    evals=pd.DataFrame(ledger['evaluations'])
    evals.to_csv(OUT/'data/Q2_all_evaluations.csv',index=False,encoding='utf-8-sig')
    rows=[]
    names={'attn_bert8_all_v1':'Final BERT8','bert8_ablate_no_cls_v1':'No CLS','bert8_ablate_no_corruption_v1':'No corruption','bert8_ablate_mean_pool_v1':'Mean pool'}
    for key,label in names.items():
        t=next(t for t in ledger['trials'] if '/'+key+'/' in t['checkpoint'])
        for split in ['valid','test']:
            if split=='valid': val=t['valid']
            else:
                choices=[e for e in ledger['evaluations'] if e.get('checkpoint')==t['checkpoint'] and e.get('split')=='test' and not e.get('class_bias')]
                val=choices[-1] if choices else {}
            if val: rows.append(dict(model=label,split=split,**{k:val[k] for k in ['clean_macro_f1','missing_macro_f1','missing_mae']}))
    ab=pd.DataFrame(rows)
    fig,ax=canvas(2)
    for a,split in zip(ax,['valid','test']):
        d=ab[ab.split==split].set_index('model').reindex(names.values())
        for i,(lab,r) in enumerate(d.iterrows()):
            a.plot([r.missing_macro_f1,r.clean_macro_f1],[i,i],c=COLORS[i],lw=2)
            a.scatter([r.missing_macro_f1,r.clean_macro_f1],[i,i],c=[COLORS[i]]*2,marker='o',s=45)
        a.set(yticks=range(4),yticklabels=list(names.values()),xlabel='Macro-F1',title=split.title()+' / missing → clean')
        a.invert_yaxis(); a.grid(axis='x',alpha=.2)
    save(fig,'Q2_04_ablations','最终结构的三个直接消融',ab,[src], '短线左端为72场景均值，右端为clean（按实际数值绘制）。同seed的去CLS、去人工缺失训练、均值池化实验；test已参与探索。这里使用ledger训练checkpoint评估，不与导出包最后一位小数混用。')
    ev=evals[(evals.split=='test') & evals.class_bias.apply(lambda v:v is None)].copy()
    ev['family']=ev.checkpoint.str.contains('bert8').map({True:'BERT8',False:'Other exploration'})
    fig,ax=canvas()
    sns.scatterplot(data=ev,x='clean_macro_f1',y='missing_macro_f1',hue='family',palette=[COLORS[0],COLORS[3]],ax=ax,s=43,alpha=.75)
    fin=ev[ev.checkpoint.str.contains('/attn_bert8_all_v1/')]
    ax.scatter(fin.clean_macro_f1,fin.missing_macro_f1,marker='*',s=200,color=COLORS[1],edgecolor='white',label='Deployed seed')
    nice(ax,'Clean Macro-F1','Missing Macro-F1'); ax.legend(fontsize=8)
    save(fig,'Q2_05_exploration','保存的无偏置探索结果全景',ev,[src], '展示保存的test无类别偏置评价记录，点可能属于同一架构的不同配置/种子，非独立试验。两坐标使用明确的局部范围以显示接近分数，不作为独立泛化证据。')
    metrics=[]
    for split in ['valid','test']:
        m=readc(f'Q2/results/final/{split}/metrics_per_scenario.csv'); m=m[m.scope=='all_samples'].copy(); m['split']=split; metrics.append(m)
    metrics=pd.concat(metrics,ignore_index=True)
    mask=metrics.scenario.str.match(r'^(T|A|V|TA|TV|AV)_(front|middle|rear)_0\.[2468]$')
    grid=metrics[mask].copy(); grid[['modality','position','rho']]=grid.scenario.str.split('_',expand=True); grid.rho=grid.rho.astype(float)
    sources=['Q2/results/final/'+s+'/metrics_per_scenario.csv' for s in ['valid','test']]
    fig,ax=canvas(2,size=(12,4.2))
    for a,split in zip(ax,['valid','test']):
        d=grid[grid.split==split].copy(); d['condition']=d.position.map({'front':'F','middle':'M','rear':'R'})+' '+(d.rho*100).astype(int).astype(str)
        p=d.pivot(index='modality',columns='condition',values='macro_f1').reindex(['T','A','V','TA','TV','AV'])
        p=p[[f'{pos} {r}' for pos in ['F','M','R'] for r in [20,40,60,80]]]
        sns.heatmap(p,ax=a,cmap='YlGnBu',vmin=grid.macro_f1.min(),vmax=grid.macro_f1.max(),annot=True,fmt='.2f',annot_kws={'fontsize':7},linewidths=.4,cbar=split=='test',cbar_kws={'label':'Macro-F1'})
        a.set(title=split.title()+' / 72 conditions',xlabel='Position and nominal span (%)',ylabel='Removed modality')
        a.tick_params(axis='x',rotation=45); a.tick_params(axis='y',rotation=0)
    save(fig,'Q2_06_grid','验证与测试72场景完整热图',grid,sources,'F/M/R为前/中/后，20–80为名义区间比例，两个面板共享色标。每格是一个场景的全样本Macro-F1；自然缺失仍保留。全部场景均展示。')
    fig,ax=canvas(2)
    agg=grid.groupby(['split','modality','rho'],as_index=False)[['macro_f1','mae']].mean()
    for a,split in zip(ax,['valid','test']):
        sns.lineplot(data=agg[agg.split==split],x='rho',y='macro_f1',hue='modality',hue_order=['T','A','V','TA','TV','AV'],marker='o',palette=COLORS,ax=a)
        nice(a,'Nominal span','Macro-F1'); a.set_title(split.title()); a.set_xticks([.2,.4,.6,.8]); a.legend(ncol=2,title='Removed')
    save(fig,'Q2_07_robustness','按模态分解的缺失强度曲线',agg,sources,'各点是前中后三个位置等权平均。连线仅引导阅读，未拟合未评估缺失率；无独立重复误差带。文本相关缺失退化较大。')
    fig,ax=canvas(2)
    agg=grid.groupby(['split','position','rho'],as_index=False)[['macro_f1','mae']].mean()
    for a,col in zip(ax,['macro_f1','mae']):
        sns.lineplot(data=agg,x='rho',y=col,hue='position',style='split',markers=True,dashes=True,ax=a,palette=COLORS[:3]); nice(a,'Nominal span','Macro-F1' if col=='macro_f1' else 'MAE'); a.set_xticks([.2,.4,.6,.8])
    save(fig,'Q2_08_positions','缺失位置对分类和回归的影响',agg,sources,'每条曲线均跨六种模态条件平均，valid和test分开编码。位置差异可能同时受到实际删除率和共同受损样本构成影响，应结合配对图。')
    deletion=readc('Q2/results/final/test/deletion_rates.csv'); deletion['nominal']=deletion.scenario.str.rsplit('_',n=1).str[-1].astype(float); deletion['modality']=deletion.scenario.str.split('_').str[0]
    fig,ax=canvas()
    for i,(m,g) in enumerate(deletion.groupby('modality',sort=False)):
        ax.errorbar(g.nominal+(i-2.5)*.007,g.actual_rate_median,yerr=[g.actual_rate_median-g.actual_rate_q25,g.actual_rate_q75-g.actual_rate_median],fmt='o',ms=3,c=COLORS[i],alpha=.75,label=m,capsize=2)
    ax.plot([.1,.9],[.1,.9],ls='--',c='#67788A',lw=1); nice(ax,'Nominal span (display offsets ±0.018)','Actual deletion fraction'); ax.legend(ncol=3)
    save(fig,'Q2_09_actual_deletion','名义缺失跨度与真实删除率',deletion,['Q2/results/final/test/deletion_rates.csv'],'点为场景内样本实际删除率中位数，线为四分位范围而非置信区间。各模态横向偏移仅避免重叠，原始名义比例保留CSV；不把缺失区间长度当作实际内容删除比例。')
    paired=readc('Q2/results/final/test/paired_conditions.csv')
    fig,ax=canvas(2)
    for a,f in zip(ax,['modality','position']):
        d=paired[paired.factor==f].copy(); d['delta']=d.macro_f1-d.clean_macro_f1
        sns.boxplot(data=d,x='condition',y='delta',hue='condition',palette=COLORS[:d.condition.nunique()],legend=False,ax=a,fliersize=2,width=.55)
        a.axhline(0,color='#596778',ls='--',lw=1); nice(a,'Matched '+f,'Macro-F1 − matched clean')
    save(fig,'Q2_10_paired_conditions','共同受损样本上的配对条件比较',paired,['Q2/results/final/test/paired_conditions.csv'],'每个点/箱体汇集条件组的指标差值，非逐样本F1。比较在同组共同受损样本上完成；不同组仍可能具有不同分母，不能当成独立重复。')
    stress=metrics[(~mask)&(metrics.scenario!='clean')]
    fig,ax=canvas(2,size=(11,4))
    for a,split in zip(ax,['valid','test']):
        d=stress[stress.split==split].copy(); names_=d.scenario.str.replace('stress_','',regex=False).str.replace('_',' ',regex=False)
        a.scatter(d.macro_f1,np.arange(len(d)),c=COLORS[0],s=40); a.set(yticks=np.arange(len(d)),yticklabels=names_,xlabel='Macro-F1',title=split.title()+' / stress conditions'); a.grid(axis='x',alpha=.25)
    save(fig,'Q2_11_stress','整模态与多模态压力场景',stress,sources,'压力测试与主72场景分开，不纳入主均值；点图使用明确局部横坐标，不以截断柱形夸大差异。')
    per=[]
    for split in ['valid','test']:
        rows=readj(f'Q2/results/final/{split}/metrics.json')
        for row in rows:
            if row['scope']=='all_samples' and row['scenario']=='clean':
                for cls,v in row['metrics']['per_class'].items(): per.append(dict(split=split,cls=cls,**v))
    per=pd.DataFrame(per)
    fig,ax=canvas(2)
    for a,split in zip(ax,['valid','test']):
        p=per[per.split==split].set_index('cls').reindex(['Negative','Neutral','Positive'])
        sns.heatmap(p[['precision','recall','f1']],annot=True,fmt='.3f',cmap='YlGnBu',vmin=0,vmax=1,ax=a,cbar=split=='test',linewidths=2)
        a.set(title=split.title()+' / class performance',ylabel='',xlabel=''); a.tick_params(axis='y',rotation=0)
    save(fig,'Q2_12_class_metrics','验证与测试类别瓶颈',per,['Q2/results/final/valid/metrics.json','Q2/results/final/test/metrics.json'],'中性类召回和F1明显低于正负类。表格记录每类support，所有色值固定0–1，避免通过各格独立缩放夸大。')
    p='Q2/results/final/figures/test_clean_predictions.csv'; pred=readc(p); pred['residual']=pred.pred_score-pred.true_score; pred['abs_error']=abs(pred.residual); pred['class']=pred.true_class.map({0:'Neg',1:'Neu',2:'Pos'})
    fig,ax=canvas(2)
    sns.violinplot(data=pred,x='class',y='residual',hue='class',order=['Neg','Neu','Pos'],palette=dict(zip(['Neg','Neu','Pos'],COLORS)),legend=False,cut=0,inner='quart',ax=ax[0]); ax[0].axhline(0,c='#566779',ls='--'); nice(ax[0],'True class','Prediction − truth')
    for cl,g in pred.groupby('class',sort=False):
        x=np.sort(g.abs_error); ax[1].plot(x,np.arange(1,len(x)+1)/len(x),color=dict(zip(['Neg','Neu','Pos'],COLORS))[cl],label=f'{cl} (n={len(x)})')
    nice(ax[1],'Absolute error','Cumulative fraction'); ax[1].legend(); ax[1].set_ylim(0,1.02)
    save(fig,'Q2_13_error_distribution','回归残差小提琴与绝对误差分布',pred,[p],'727条test真实逐样本预测；不裁去离群误差。各真值类残差方向反映回归向中部收缩。累计曲线由原始误差排序计算，无平滑拟合。')
    fig,ax=canvas(2)
    cm=pd.crosstab(pred.true_class,pred.pred_class).reindex(index=range(3),columns=range(3),fill_value=0)
    rate=cm.div(cm.sum(axis=1),axis=0); labels=np.array([[f'{cm.iloc[i,j]}\n{rate.iloc[i,j]:.1%}' for j in range(3)] for i in range(3)])
    sns.heatmap(rate,annot=labels,fmt='',cmap='Blues',vmin=0,vmax=1,ax=ax[0],xticklabels=['Neg','Neu','Pos'],yticklabels=['Neg','Neu','Pos'],cbar=False,square=True)
    ax[0].set(xlabel='Predicted class',ylabel='True class',title='Test clean / counts and row fractions')
    hb=ax[1].hexbin(pred.true_score,pred.pred_score,gridsize=23,mincnt=1,cmap='YlGnBu',linewidths=.1); fig.colorbar(hb,ax=ax[1],label='Samples')
    lo=min(pred.true_score.min(),pred.pred_score.min())-.1; hi=max(pred.true_score.max(),pred.pred_score.max())+.1
    ax[1].plot([lo,hi],[lo,hi],'--',c=COLORS[1],lw=1); ax[1].set(xlabel='True intensity',ylabel='Predicted intensity',xlim=(lo,hi),ylim=(lo,hi),title='Density without coordinate jitter')
    save(fig,'Q2_14_prediction_diagnostics','测试混淆矩阵与回归密度',pred,[p],'回归图使用真实坐标聚合为六边形密度，不对真值抖动。虚线为理想预测。test已参与探索；验证集逐类诊断见Q2_12。')
    p='Q2/results/final/q2_special_state.csv'; state=readc(p)
    fig,ax=canvas(size=(10,3.3)); cols=['observed_text_positions','observed_audio_positions','observed_visual_positions']
    frac=state[cols].div(state.structural_positions,axis=0).T
    sns.heatmap(frac,ax=ax,cmap='YlGnBu',vmin=0,vmax=1,xticklabels=[f'{i:02}' for i in range(1,31)],yticklabels=['T','A','V'],linewidths=.4,cbar_kws={'label':'Observed / structural'})
    ax.set(xlabel='Attachment 3 sample',ylabel='',title='Availability in all 30 unlabeled samples'); ax.tick_params(axis='x',rotation=0); ax.tick_params(axis='y',rotation=0)
    save(fig,'Q2_15_special_availability','附件3全部样本的当前可观测状态',state,[p],'可观测位置占结构位置比例；自然零行不推断为某种具体提取失败。补偿计数均为0，最终模型没有补偿模块。专项无真值，不报告准确率。')
    p='doc/paper_figures/data/Q2_valid_reloaded_predictions.csv'
    if (ROOT/p).exists():
        valid=readc(p); valid['residual']=valid.pred_score-valid.true_score; valid['abs_error']=abs(valid.residual); valid['class']=valid.true_class.map({0:'Neg',1:'Neu',2:'Pos'}); valid['correct']=valid.true_class==valid.pred_class
        fig,ax=canvas(2)
        cm=pd.crosstab(valid.true_class,valid.pred_class).reindex(index=range(3),columns=range(3),fill_value=0); rate=cm.div(cm.sum(axis=1),axis=0)
        labels=np.array([[f'{cm.iloc[i,j]}\n{rate.iloc[i,j]:.1%}' for j in range(3)] for i in range(3)])
        sns.heatmap(rate,annot=labels,fmt='',cmap='Blues',vmin=0,vmax=1,ax=ax[0],xticklabels=['Neg','Neu','Pos'],yticklabels=['Neg','Neu','Pos'],cbar=False,square=True)
        ax[0].set(xlabel='Predicted class',ylabel='True class',title='Validation clean / n = 728')
        hb=ax[1].hexbin(valid.true_score,valid.pred_score,gridsize=23,mincnt=1,cmap='YlGnBu',linewidths=.1); fig.colorbar(hb,ax=ax[1],label='Samples')
        ax[1].plot([-3,3],[-3,3],'--',c=COLORS[1],lw=1); ax[1].set(xlabel='True intensity',ylabel='Predicted intensity',title='Reloaded delivery model',xlim=(-3.1,3.1),ylim=(-3.1,3.1))
        save(fig,'Q2_16_valid_diagnostics','验证集真实重载的混淆与回归密度',valid,[p,'doc/paper_figures/data/Q2_valid_reloaded_diagnostics.json'],'本地离线交付包真实推理，Accuracy/Macro-F1与Q3保存结果完全一致，MAE差约3×10^-9；验证集已用于选模，不是独立test。中性类184条中65条判对，88条误判正类。')
        fig,ax=canvas(2)
        sns.violinplot(data=valid,x='class',y='residual',hue='class',order=['Neg','Neu','Pos'],palette=dict(zip(['Neg','Neu','Pos'],COLORS)),legend=False,cut=0,inner='quart',ax=ax[0]); ax[0].axhline(0,c='#677889',ls='--'); nice(ax[0],'True class','Prediction − truth')
        for cl,g in valid.groupby('class'):
            x=np.sort(g.abs_error); ax[1].plot(x,np.arange(1,len(x)+1)/len(x),color=dict(zip(['Neg','Neu','Pos'],COLORS))[cl],label=f'{cl} (n={len(g)})')
        nice(ax[1],'Absolute error','Cumulative fraction'); ax[1].legend()
        save(fig,'Q2_17_valid_residuals','验证集分情感类别残差与误差分布',valid,[p],'未删除离群样本，小提琴cut=0。负向类偏高估、正向类偏低估可作为向中性回缩的描述，不能仅凭图推断训练因果机制。')
        valid['confidence']=valid[['p_neg','p_neu','p_pos']].max(axis=1)
        valid['confidence_bin']=pd.cut(valid.confidence,bins=np.linspace(1/3,1,8),include_lowest=True)
        rel=valid.groupby('confidence_bin',observed=True).agg(n=('correct','size'),accuracy=('correct','mean'),mean_confidence=('confidence','mean')).reset_index(); rel['confidence_bin']=rel.confidence_bin.astype(str)
        fig,ax=canvas(2)
        for correct,color in [(True,COLORS[0]),(False,COLORS[1])]:
            values=valid.loc[valid.correct==correct,'confidence']; ax[0].hist(values,bins=np.linspace(1/3,1,14),histtype='stepfilled',alpha=.5,color=color,label=f'{"Correct" if correct else "Wrong"} (n={len(values)})')
        nice(ax[0],'Maximum class probability','Samples'); ax[0].legend()
        ax[1].plot([.3,1],[.3,1],'--',c='#708093',lw=1); ax[1].scatter(rel.mean_confidence,rel.accuracy,s=rel.n*1.2,color=COLORS[0],alpha=.7)
        for _,r in rel.iterrows(): ax[1].annotate(str(r.n),(r.mean_confidence,r.accuracy),xytext=(0,9),textcoords='offset points',ha='center',fontsize=8)
        nice(ax[1],'Mean confidence','Observed accuracy'); ax[1].set(title='Fixed bins / labels show sample counts',xlim=(.3,1.02),ylim=(.2,1.02))
        rel.to_csv(OUT/'data/Q2_valid_confidence_bins.csv',index=False,encoding='utf-8-sig')
        save(fig,'Q2_18_valid_confidence','验证集置信度与错误关系',valid,[p],'右图固定7个等宽置信度区间，圆面积与样本量成比例，标注样本数；斜线是概率与正确率相等参照。仅做事后诊断，不进行校准拟合或阈值重新选择。分箱精确数据另见Q2_valid_confidence_bins.csv。')
        valid['length_bin']=pd.cut(valid.text_observed,bins=[0,10,20,30,50],labels=['1–10','11–20','21–30','31–50'])
        length=valid.groupby('length_bin',observed=True).agg(n=('correct','size'),accuracy=('correct','mean'),mae=('abs_error','mean')).reset_index(); length['length_bin']=length.length_bin.astype(str)
        fig,ax=canvas(2)
        for a,col,lab in zip(ax,['accuracy','mae'],['Accuracy','MAE']):
            a.scatter(length.length_bin,length[col],s=55,c=COLORS[0]); nice(a,'Observed text positions',lab)
            for i,r in length.iterrows(): a.annotate(f'n={r.n}',(i,r[col]),xytext=(0,10),textcoords='offset points',ha='center',fontsize=8)
            a.margins(y=.25)
        save(fig,'Q2_19_valid_groups','验证集按有效文本长度分层',length,[p],'固定长度分层的描述性准确率与MAE，各组标签组成可能不同，不能声称长度造成性能变化。点图和明确局部纵轴减少空白；每组分母标出。')

def q3_figures():
    base='Q3/runs/valid_v2/summaries/'
    faith=readj(base+'faithfulness.json'); f=pd.DataFrame([dict(selection=k,**v) for k,v in faith.items()])
    f[['kind','modality','budget','grain']]=f.selection.str.split('.',expand=True).iloc[:,:4]; f['pct']=f.budget.str[1:].astype(int)
    point=f[(f.grain.isin(['point','joint']))&(~f.selection.str.contains('cost'))]
    fig,ax=canvas(2)
    for a,mods in zip(ax,[['T','J'],['A','V']]):
        for m in mods:
            g=point[point.modality==m].sort_values('pct'); a.errorbar(g.pct,g.mean_G,yerr=[g.mean_G-g.ci_low,g.ci_high-g.mean_G],fmt='o-',capsize=4,color=MOD[m],label=m)
        a.set_xticks([10,20,30]); nice(a,'Nominal budget (%)','Mean paired gain G'); a.legend(); a.set_title('Text / joint' if mods[0]=='T' else 'Audio / vision (separate scale)'); zoom_positive(a,point[point.modality.isin(mods)])
    save(fig,'Q3_01_faithfulness','三预算忠实性增益与视频簇区间',point,[base+'faithfulness.json'],'G为关键集合D减严格匹配随机集合平均D。误差线为video_id簇bootstrap 95%区间；T/J与A/V分别缩放并显式标注，纵轴为带刻度的局部范围，不从0开始，不能据线条高度比较跨面板效应。')
    pair=readc(base+'paired_samples.csv'); pair['gain']=pair.target-pair.random
    ids=['key.T.p20.point','key.A.p20.point','key.V.p20.point','key.J.p20.joint']
    pp=pair[(pair.operation=='delete')&pair.selection_id.isin(ids)].dropna(subset=['target','random']).copy(); pp['modality']=pp.selection_id.str.split('.').str[1]
    fig,ax=canvas(2)
    for a,mods in zip(ax,[['T','J'],['A','V']]):
        p=pp[pp.modality.isin(mods)]
        sns.violinplot(data=p,x='modality',y='gain',hue='modality',order=mods,palette=MOD,inner='quart',cut=0,legend=False,ax=a)
        a.axhline(0,c='#596B7B',ls='--',lw=1); nice(a,'Selected modality','Paired gain G'); a.set_title('All comparable samples / separate scales')
    save(fig,'Q3_02_gain_violin','20%预算逐样本忠实性增益分布',pp,[base+'paired_samples.csv'],'所有可比较样本进入小提琴，cut=0不外推密度；负增益反例全部保留。不同模态可比较分母不同，原始CSV含sample_id/video_id/n_random。')
    fig,ax=canvas(2)
    for a,mods in zip(ax,[['T','J'],['A','V']]):
        for m in mods:
            g=np.sort(pp.loc[pp.modality==m,'gain']); a.plot(g,np.arange(1,len(g)+1)/len(g),c=MOD[m],label=f'{m}: n={len(g)}')
        a.axvline(0,c='#657185',ls='--',lw=1); nice(a,'Paired gain G','Cumulative fraction'); a.legend(); a.set_ylim(0,1.02)
    save(fig,'Q3_03_gain_ecdf','忠实性增益累计分布与负增益比例',pp,[base+'paired_samples.csv'],'经验分布无需核带宽，显示负增益与长尾，补充小提琴的平滑显示。样本相关性不允许把24k条重复记录当成独立样本。')
    fig,ax=canvas(2)
    coverage=point.copy(); coverage['no_compare']=coverage.n_planned-coverage.n_eligible
    for a,mods in zip(ax,[['T','J'],['A','V']]):
        d=coverage[coverage.modality.isin(mods)].sort_values(['modality','pct']); labels=d.modality+' '+d.pct.astype(str)
        a.bar(labels,d.n_eligible,color=[MOD[m] for m in d.modality],label='Comparable')
        a.bar(labels,d.no_compare,bottom=d.n_eligible,color='#CCD3DC',label='No matched comparison')
        a.set_ylim(0,790); nice(a,'Modality / budget (%)','Samples'); a.legend(fontsize=8)
    save(fig,'Q3_04_control_coverage','严格随机对照的可比较覆盖率',coverage,[base+'faithfulness.json'],'灰色包含无可用模态/无匹配对照，具体拆分见CSV的not_applicable与without_controls。每条主路径计划728条；未补齐无合法随机集合的样本。')
    fig,ax=canvas(size=(7,3.8))
    for m,g in pp.groupby('modality'):
        counts=g.n_random.value_counts().sort_index(); ax.step(counts.index,counts.cumsum()/counts.sum(),where='post',color=MOD[m],label=m)
    ax.axvline(20,c='#69798A',ls='--',lw=1); nice(ax,'Available matched random sets','Cumulative sample fraction'); ax.legend(); ax.set_xlim(0,51)
    save(fig,'Q3_05_random_counts','逐样本严格随机集合数量',pp,[base+'paired_samples.csv'],'每样本最多50组，少于20组的样本仍用于描述均值，但不足以支持细尾概率判断；集合数不是独立样本量。')
    keep=readj(base+'keep.json'); kk=pd.DataFrame([dict(selection=k,**v) for k,v in keep.items() if k.split(' / ')[0] in ids]); kk['modality']=kk.selection.str.split('.').str[1]; kk['operation']=kk.selection.str.split(' / ').str[-1]
    fig,ax=canvas(2)
    for a,op in zip(ax,['conditional_keep','global_keep']):
        d=kk[kk.operation==op].set_index('modality').reindex(['T','A','V','J']).dropna(subset=['mean_G'])
        a.errorbar(d.mean_G,range(len(d)),xerr=[d.mean_G-d.ci_low,d.ci_high-d.mean_G],fmt='o',c=COLORS[0],capsize=4)
        a.axvline(0,c='#667488',ls='--'); a.set(yticks=range(len(d)),yticklabels=d.index,xlabel='Target S − random S',title=op.replace('_',' ').title()); a.grid(axis='x',alpha=.2)
    save(fig,'Q3_06_keep','保留实验的配对增益与反例',kk,[base+'keep.json'],'S=1−D。global保留只留下关键集合；conditional只移除当前模态域内其余内容。关键删除集合不必是最佳保留集合，负增益如实展示。')
    gran=readj(base+'granularity.json'); gr=pd.DataFrame([dict(selection=k,**v) for k,v in gran.items()])
    fig,ax=canvas(2)
    for a,idx in zip(ax,[[1,2],[0,3]]):
        d=gr.iloc[idx]; a.errorbar(d.mean_G,range(len(d)),xerr=[d.mean_G-d.ci_low,d.ci_high-d.mean_G],fmt='o',capsize=4,c=COLORS[3])
        a.axvline(0,c='#667488',ls='--'); a.set(yticks=range(len(d)),yticklabels=[s.replace('key.','').replace('.p20.',' / ') for s in d.selection],xlabel='Group D − point D',title='Cost mismatches disclosed in caption'); a.grid(axis='x',alpha=.2)
    save(fig,'Q3_07_granularity','整词和分块选择的粒度敏感性',gr,[base+'granularity.json'],'历史汇总中block3与point存在成本不等配对：T/A各22条、V26条；因此不能将该图block3差值解释为纯同成本粒度因果效应。word成本差异0，区间跨0；整词可读性不等于忠实性显著提升。')
    mod=readj(base+'modality.json'); rows=[]
    for row in mod['rows']:
        for i,m in enumerate(['T','A','V']): rows.append(dict(sample_id=row['sample_id'],modality=m,D=row['D'][i],phi_class=row['phi_class'][i],phi_score_raw=row['phi_score_raw'][i],observed=row['n_observed'][i]))
    md=pd.DataFrame(rows)
    fig,ax=canvas(3,size=(12.5,3.8))
    for a,col,lab in zip(ax,['D','phi_class','phi_score_raw'],['Whole-modality effect D','Class Shapley','Score Shapley (raw)']):
        sns.boxplot(data=md,x='modality',y=col,hue='modality',palette=MOD,legend=False,showfliers=False,ax=a,width=.5)
        # ECDF figure below retains tails; boxplot is explicitly labelled.
        a.axhline(0,c='#6C7988',ls='--',lw=1); nice(a,'Modality',lab)
    save(fig,'Q3_08_modality_shapley','全验证集整模态影响与双输出Shapley',md,[base+'modality.json'],'箱体Q1–Q3，中位线与1.5IQR须；图中不显示离群点以保持主体可读，全部值在CSV且Q3_09显示全分布。带符号Shapley区分抑制与支持，不能和绝对影响D混称。')
    fig,ax=canvas(2)
    for a,col in zip(ax,['D','phi_class']):
        for m,g in md.groupby('modality'):
            x=np.sort(g[col].dropna()); a.plot(x,np.arange(1,len(x)+1)/len(x),color=MOD[m],label=m)
        nice(a,'Whole-modality D' if col=='D' else 'Class Shapley','Cumulative fraction'); a.legend()
    save(fig,'Q3_09_modality_ecdf','整模态影响全范围分布',md,[base+'modality.json'],'所有有限观测均进入经验累计分布，包含零观测模态按现有汇总定义的零贡献；可与Q3_08主体箱体互补，不隐藏长尾。')
    sm=readj('Q3/runs/special_v2/summaries/modality.json'); data=[]
    for row in sm['rows']:
        for i,m in enumerate(['T','A','V']): data.append(dict(sample_id=row['sample_id'],modality=m,D=row['D'][i],phi_class=row['phi_class'][i],phi_score_raw=row['phi_score_raw'][i],observed=row['n_observed'][i]))
    sp=pd.DataFrame(data); fig,ax=canvas(3,size=(13,4))
    for a,col,cmap in zip(ax,['D','phi_class','phi_score_raw'],['YlGnBu','vlag','vlag']):
        p=sp.pivot(index='sample_id',columns='modality',values=col).reindex(columns=['T','A','V'])
        sns.heatmap(p,ax=a,cmap=cmap,center=0 if col!='D' else None,annot=True,fmt='.2f',annot_kws={'fontsize':6.8},cbar_kws={'label':col},linewidths=.3)
        a.set(xlabel='',ylabel='Sample' if col=='D' else '',title=col); a.tick_params(axis='y',rotation=0)
    save(fig,'Q3_10_special_modality','附件4全部20条模态作用与双输出贡献',sp,['Q3/runs/special_v2/summaries/modality.json'],'各面板各自的明确色标：D非负，Shapley有符号。13号视觉无观测；14号分类与强度首位模态不一致，不能用一张平均模态图掩盖。')
    direction=readj(base+'direction.json'); failures=pd.DataFrame(direction['failures']); fig,ax=canvas(size=(8,3.8))
    labels=failures.sample_id.str.replace('$_$',' / ',regex=False)
    ax.barh(labels,failures.delta_class,color=[COLORS[0] if x<0 else COLORS[1] for x in failures.delta_class]); ax.axvline(0,c='#657488',lw=1); ax.set(xlabel='Original-class probability drop',title=f"All {direction['failed']} direction reversals / {direction['evaluated']} retested")
    save(fig,'Q3_11_direction_failures','方向重测的全部验证集失败样本',failures,[base+'direction.json'],'8个suppress候选联合删除后概率反而降低，1个support候选反而升高。方向只相对于固定原预测类别；局部单点符号不能保证联合集合方向。')
    low=readj(base+'low_comparison.json'); ll=pd.DataFrame([dict(selection=k,**v) for k,v in low.items()]); ll['modality']=ll.selection.str.split('.').str[1]; ll['pct']=ll.selection.str.split('.').str[2].str[1:].astype(int)
    fig,ax=canvas(2)
    for a,mods in zip(ax,[['T'],['A','V']]):
        for m in mods:
            g=ll[ll.modality==m].sort_values('pct'); a.errorbar(g.pct,g.mean_G,yerr=[g.mean_G-g.ci_low,g.ci_high-g.mean_G],fmt='o-',capsize=4,c=MOD[m],label=m)
        nice(a,'Nominal budget (%)','Key D − low-impact D'); a.set_xticks([10,20,30]); a.legend(); zoom_positive(a,ll[ll.modality.isin(mods)])
    save(fig,'Q3_12_low_impact','关键集合与低影响集合对照',ll,[base+'low_comparison.json'],'同样本关键集合与低影响集合的差值，区间为video_id簇bootstrap。文本与AV使用标明数值的不同局部纵轴；该对照补充严格随机对照，不取代随机集合几何匹配。')
    st=readj(base+'stratified.json'); ss=pd.DataFrame([dict(selection=k,**v) for k,v in st.items()]); ss['path']=ss.selection.str.split(' / ').str[0]; ss['stratum']=ss.selection.str.split(' / ').str[1]
    fig,ax=canvas(2)
    paths=[['key.T.p20.point','key.J.p20.joint'],['key.A.p20.point','key.V.p20.point']]
    for a,ps in zip(ax,paths):
        for i,p in enumerate(ps):
            g=ss[ss.path==p].set_index('stratum').reindex(['original_correct','original_wrong','true_neutral']); x=np.arange(3)+(i-.5)*.13
            m=p.split('.')[1]; a.errorbar(x,g.mean_G,yerr=[g.mean_G-g.ci_low,g.ci_high-g.mean_G],fmt='o',capsize=4,c=MOD[m],label=m)
        a.set_xticks(range(3),['Correct','Wrong','Neutral']); nice(a,'Original prediction / true class','Mean paired gain G'); a.legend(); zoom_positive(a,ss[ss.path.isin(ps)])
    save(fig,'Q3_13_stratification','正确错误与中性类分层忠实性',ss,[base+'stratified.json'],'分层存在重叠，中性类可同时属于正确或错误组；并非三个互斥分组。错误预测也有正忠实性增益，说明解释可以忠实反映错误决策，不代表预测正确。')
    pert=readj(base+'perturbation_metrics.json'); rr=[]
    for p,v in pert.items():
        for name,key in [('Clean','clean_same_subset'),('Key','key')]: rr.append(dict(path=p,condition=name,n=v['n'],f1=v[key]['macro_f1'],mae=v[key]['mae']))
        rr.append(dict(path=p,condition='Random',n=v['n'],f1=v['random']['macro_f1']['mean'],mae=v['random']['mae']['mean']))
    rr=pd.DataFrame(rr); rr['label']=rr.path.map(lambda s:s.split('.')[1]+(' word' if s.endswith('word') else ''))
    fig,ax=canvas(2)
    for a,col,lab in zip(ax,['f1','mae'],['Macro-F1','MAE']):
        for i,(cond,g) in enumerate(rr.groupby('condition',sort=False)): a.scatter(np.arange(len(g))+(i-1)*.12,g[col],c=COLORS[i],label=cond,s=38)
        a.set_xticks(np.arange(rr.label.nunique()),rr.label.unique())
        nice(a,'Intervention path',lab); a.legend()
    save(fig,'Q3_14_task_after_deletion','关键删除与随机删除后的任务表现',rr,[base+'perturbation_metrics.json'],'每条路径使用自己可比较样本集合上的clean、key和random；跨路径分母不同，不能直接解释路径间性能差。随机结果先逐重复计算任务指标再取均值，未用平均概率代替。')
    subset_rows=[]; budget_rows=[]
    for directory in sorted((ROOT/'Q3/runs/special_v2/samples').iterdir()):
        if not directory.is_dir(): continue
        ori=json.loads((directory/'original.json').read_text('utf-8')); sid=json.loads((directory/'mapping.json').read_text('utf-8'))['sample_id']
        sh=json.loads((directory/'shapley.json').read_text('utf-8'))
        for s in sh['subsets']:
            subset_rows.append(dict(sample_id=sid,subset=''.join(m for i,m in enumerate('TAV') if s['bitmask']&(1<<i)) or 'Empty',prob=s['probs'][ori['c_star']],score=s['score']))
        for s in (directory/'selections.jsonl').read_text('utf-8').splitlines():
            r=json.loads(s)
            if r['selection_id'] in ids: budget_rows.append(dict(sample_id=sid,selection=r['selection_id'],target=r['target_budget'],actual=r['actual_cost'],segments=len(r['intervals'])))
    sub=pd.DataFrame(subset_rows); fig,ax=canvas(2,size=(12,4.5))
    order=['Empty','T','A','TA','V','TV','AV','TAV']
    for a,col in zip(ax,['prob','score']):
        p=sub.pivot(index='sample_id',columns='subset',values=col).reindex(columns=order)
        sns.heatmap(p,cmap='YlGnBu' if col=='prob' else 'vlag',center=0 if col=='score' else None,ax=a,cbar_kws={'label':'Original-class probability' if col=='prob' else 'Intensity'},linewidths=.2)
        a.set(title='Eight exact modality subsets',xlabel='Kept modalities',ylabel='Sample'); a.tick_params(axis='y',rotation=0)
    save(fig,'Q3_15_subset_predictions','20条样本八个模态保留子集的实际输出',sub,['Q3/runs/special_v2/samples/*/shapley.json','Q3/runs/special_v2/samples/*/original.json'],'Shapley实际使用的八子集全枚举输出，概率始终跟踪原预测类别；不是各子集最大类别概率。Empty为训练先验回退；不同样本原预测类别不同。')
    budgets=pd.DataFrame(budget_rows); budgets['shortfall']=budgets.target-budgets.actual; budgets['modality']=budgets.selection.str.split('.').str[1]
    fig,ax=canvas(2)
    b=budgets.pivot(index='sample_id',columns='modality',values='shortfall').reindex(columns=['T','A','V','J'])
    sns.heatmap(b,cmap='YlOrBr',annot=True,fmt='g',ax=ax[0],cbar_kws={'label':'Target − actual cost'},linewidths=.3); ax[0].set(xlabel='Path',ylabel='Sample',title='Integer-budget feasibility'); ax[0].tick_params(axis='y',rotation=0)
    b=budgets.groupby(['modality','segments']).size().unstack(fill_value=0).reindex(['T','A','V','J']); b.plot.bar(stacked=True,ax=ax[1],color=COLORS[:len(b.columns)],rot=0); nice(ax[1],'Path','Samples'); ax[1].legend(title='Segments'); ax[1].set_title('At most three intervals')
    save(fig,'Q3_16_budget_geometry','实际预算可达性与连续片段数',budgets,['Q3/runs/special_v2/samples/*/selections.jsonl'],'实际成本取不超过目标预算的最大合法值，不强行补齐。零观测模态可能无适用选择；对照匹配实际几何与成本。仅为20条专项的算法可行性诊断。')
    # Twenty local data plots, one per special sample, with observed slots only.
    for directory in sorted((ROOT/'Q3/runs/special_v2/samples').iterdir()):
        if not directory.is_dir(): continue
        mapping=json.loads((directory/'mapping.json').read_text('utf-8')); local=json.loads((directory/'local.json').read_text('utf-8'))
        selections=[json.loads(s) for s in (directory/'selections.jsonl').read_text('utf-8').splitlines()]
        sid=mapping['sample_id']; records=[]; token={r['official_t']:r['token'] for r in mapping['token_rows']}
        for t,vals in enumerate(local['D']):
            for j,m in enumerate(['T','A','V']):
                if vals[j] is not None:
                    key=next((r for r in selections if r['selection_id']==f'key.{m}.p20.point'),{})
                    records.append(dict(sample_id=sid,position=t,token=token.get(t,''),modality=m,D=vals[j],delta_class=local['delta_class'][t][j],selected=t*3+j in key.get('delete_j',[])))
        d=pd.DataFrame(records); mods=[m for m in ['T','A','V'] if m in set(d.modality)]
        fig,axs=plt.subplots(len(mods),1,figsize=(11,2.15*len(mods)),layout='constrained')
        for a,m in zip(np.atleast_1d(axs),mods):
            g=d[d.modality==m]
            if g.empty:
                a.text(.5,.5,'No observed visual content',transform=a.transAxes,ha='center'); a.set_axis_off(); continue
            x=np.arange(len(g)); a.bar(x,g.D,color=MOD[m],width=.78,alpha=.8)
            sel=g.selected.to_numpy(); a.scatter(x[sel],g.D.to_numpy()[sel],marker='v',c='#212F3E',s=25,zorder=3,label='20% selected')
            if m=='T':
                a.set_xticks(x,[f'{int(t)}:{str(w)}' for t,w in zip(g.position,g.token)],rotation=55,ha='right',fontsize=7)
            else:
                stride=max(1,int(np.ceil(len(g)/15))); a.set_xticks(x[::stride],g.position.iloc[::stride].astype(int),fontsize=8)
            nice(a,'Official position / token' if m=='T' else 'Official position',m+' / local D')
            med=float(g.D.median())
            if med>0 and g.D.max()>8*med:
                a.set_yscale('symlog',linthresh=2*med)
                a.set_ylabel(m+' / D (symlog)')
            a.set_xlim(-.7,len(g)-.3); a.legend(loc='upper right',fontsize=7)
        suffix=' · V unobserved' if 'V' not in mods else ''
        fig.suptitle(f'Sample {sid} · single-position effects · separate modality scales'+suffix,fontsize=12)
        source=str(directory.relative_to(ROOT)).replace('\\','/')
        save(fig,f'Q3_case_{sid}',f'附件4样本{sid}的三模态逐位置重要性',d,[source+'/local.json',source+'/mapping.json',source+'/selections.jsonl'],'仅显示已观测位置，三角标记20% point集合；三模态独立纵轴，峰值超过中位数8倍时使用标明的symlog轴（线性阈值=2×中位数），保留峰值与小效应，禁止跨面板用柱高比较贡献。无观测V省略并在标题标出。所有位置为官方索引，不是秒数；单点D之和不等于联合D。')

def index():
    (OUT/'figure_manifest.json').write_text(json.dumps(ENTRIES,ensure_ascii=False,indent=2)+'\n','utf-8')
    lines=['# 论文数据图与可复算材料','',f'本目录包含 **{len(ENTRIES)}组数据图**，每组同时提供220 dpi PNG、矢量PDF和可编辑SVG，以及对应CSV。全部来自保存的实际结果，不生成流程图或方法主图。','',
           '打开 [图册](index.html) 可按问题筛选和查看大图；论文排版优先使用PDF，修改标签使用SVG。图内使用简短英文，以下提供中文图意与可用结论。','',
           '## 使用与证据边界','',
           '- Q2 test曾影响后续结构探索，图中test结果只能作为已观察测试集上的探索结果，不能写成独立留出确认。',
           '- 三种子训练曲线保留波动；消融不声称统计显著。valid为开发集，附件3/4无标签。',
           '- 历史训练只保存epoch平均损失和间隔验证，没有逐batch梯度/损失；全量逐轮明细见 [Q2_all_epoch_history.csv](data/Q2_all_epoch_history.csv)。未插值或伪造逐step记录。',
           '- 对接近评分使用点图及有刻度的局部坐标；比例热图固定0–1，效应跨量级时分面且标明不同尺度。零线、负增益与反例均保留。',
           '- Q3 block3历史汇总存在实际成本不一致（T/A各22，V26），不作为严格同成本优势证据；word无此成本问题。',
           '- Q1接受率与可用率是流程指标，非人工准确率；Q3忠实性不等于情感预测正确或人类因果解释。',
           '', '## 建议论文选图','',
           '正文优先：Q1_02、Q1_03；Q2_01、Q2_04、Q2_06、Q2_07、Q2_12、Q2_14；Q3_01、Q3_02、Q3_06、Q3_10，以及01/14/18号局部解释。其余图用于补充实验和错误归因；不要将几十张图全部挤入正文。','',
           '## 图表清单','', '| 图号 | 内容 | 数据行数 | 图与数据 |','|---|---|---:|---|']
    for e in ENTRIES: lines.append(f"| {e['id']} | {e['title']} | {e['rows']} | [PNG](figures/{e['id']}.png) · [PDF](figures/{e['id']}.pdf) · [SVG](figures/{e['id']}.svg) · [CSV](data/{e['id']}.csv) |")
    for e in ENTRIES:
        links=[]
        for s in e['sources']:
            target=s.split('*')[0] if '*' in s else s
            links.append(f'[{s}](../../{target})')
        lines.extend(['',f"## {e['id']} {e['title']}",'',e['caption'],'','来源：'+ '；'.join(links)+'。'])
    lines.extend(['','## 复算','', '`python scripts/build_paper_figures.py`','', '依赖：Python、numpy、pandas、matplotlib、seaborn、Pillow。源码相对自身定位仓库，无需网络和服务器；不会重新训练或更改任何实验值。'])
    (OUT/'README.md').write_text('\n'.join(lines)+'\n','utf-8')
    cards=''.join(f'<article data-q="{e["id"][:2]}"><h2>{e["id"]} · {html.escape(e["title"])}</h2><a href="figures/{e["id"]}.png"><img loading="lazy" src="figures/{e["id"]}.png"></a><p>{html.escape(e["caption"])}</p><nav><a href="figures/{e["id"]}.pdf">PDF</a><a href="figures/{e["id"]}.svg">SVG</a><a href="data/{e["id"]}.csv">CSV</a></nav></article>' for e in ENTRIES)
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>论文数据图册</title><style>body{font:15px/1.6 system-ui;background:#eef2f6;color:#26374a;margin:0}header{padding:28px 5vw;background:#19334b;color:white}main{max-width:1250px;margin:24px auto;padding:0 18px}article{background:white;padding:20px;margin:24px 0;border-radius:12px;box-shadow:0 3px 18px #1d344512}h2{font-size:19px}img{width:100%;height:auto}p{color:#536578}nav a,button{display:inline-block;margin-right:14px;padding:6px 14px;border-radius:6px;background:#e5eef5;color:#1c597e;border:0;cursor:pointer;text-decoration:none}.filters{position:sticky;top:0;background:#eef2f6ef;padding:12px;z-index:3}</style><header><h1>Q1 · Q2 · Q3 论文数据图册</h1><p style="color:#dce7f0">实际实验数据 · 可追溯CSV · PNG / PDF / SVG · 不含流程图</p></header><main><div class="filters"><button onclick="filter('all')">全部</button><button onclick="filter('Q1')">Q1 特征质量</button><button onclick="filter('Q2')">Q2 鲁棒预测</button><button onclick="filter('Q3')">Q3 可解释性</button></div>'''+cards+'''</main><script>function filter(q){document.querySelectorAll('article').forEach(a=>a.hidden=q!=='all'&&a.dataset.q!==q)}</script></html>'''
    (OUT/'index.html').write_text(page,'utf-8')
    from PIL import Image,ImageOps,ImageDraw
    for start in range(0,len(ENTRIES),12):
        entries=ENTRIES[start:start+12]; board=Image.new('RGB',(1500,330*((len(entries)+2)//3)), '#e8eef3'); draw=ImageDraw.Draw(board)
        for k,e in enumerate(entries):
            im=Image.open(OUT/'figures'/f'{e["id"]}.png').convert('RGB'); im.thumbnail((480,285))
            x=(k%3)*500+(500-im.width)//2; y=(k//3)*330+30
            board.paste(im,(x,y)); draw.text(((k%3)*500+15,(k//3)*330+8),e['id'],fill='#142a40')
        board.save(OUT/'previews'/f'contact_{start//12+1}.jpg',quality=90)

if __name__=='__main__':
    q1_figures(); q2_figures(); q3_figures(); index()
    print(f'Completed {len(ENTRIES)} figure groups.',flush=True)
