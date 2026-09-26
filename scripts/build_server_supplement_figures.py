"""Server-detail supplement. Rebuild from checked-in CSV; --extract accepts a private gzip export."""
from pathlib import Path
import argparse
import gzip
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import build_paper_figures as base

DATA = base.OUT / 'data'


def extract(path):
    raw = json.load(gzip.open(path, 'rt', encoding='utf-8'))
    pred = pd.read_csv(DATA / 'Q2_valid_reloaded_predictions.csv')
    quality = pd.read_csv(base.ROOT / 'Q1/q1_features/runs/q1_full_20260924/reports/quality.csv').set_index('sample_id')
    local, grains, words, diagnostics = [], [], [], []
    score_errors = []
    for row in raw['q3']:
        reference = pred.iloc[row['index']]
        original = row['original']
        assert original['c_star'] == reference.pred_class
        assert original['n_observed'] == [reference.text_observed, reference.audio_observed, reference.vision_observed]
        score_errors.append(abs(original['score'] - reference.pred_score))
        sid = reference.sample_id
        for pos, values in enumerate(row['local']['D']):
            for mod, value in enumerate(values):
                if value is not None:
                    local.append(dict(sample_id=sid, video_id=sid.split('$_$')[0], position=pos, modality='TAV'[mod], D=value,
                                      delta_class=row['local']['delta_class'][pos][mod], delta_score=row['local']['delta_score_raw'][pos][mod]))
        sels = {s['selection_id']: s for s in row['selections']}
        exps = {(e['selection_id'], e['operation'], e['control_role']): e for e in row['experiments']}
        for key, s in sels.items():
            if s['experiment_kind'] != 'key' or s['granularity'] not in ('word', 'block3') or s['status'] != 'complete':
                continue
            point_id = s.get('cost_matched_point_id', f"key.{s['path']}.p20.point")
            a, b = exps.get((key, 'delete', 'key')), exps.get((point_id, 'delete', 'key'))
            if a is not None and b is not None:
                grains.append(dict(sample_id=sid, video_id=sid.split('$_$')[0], selection=key,
                                   group_cost=s['actual_cost'], point_cost=sels[point_id]['actual_cost'], group_D=a['D'], point_D=b['D']))
    for row in raw['q1']:
        sid = row['media']['sample_id']
        d = row['diagnostic']
        assert abs(d['diagnostic_wer'] - quality.loc[sid, 'diagnostic_wer']) < 1e-6
        diagnostics.append(dict(sample_id=sid, substitutions=d['substitutions'], deletions=d['deletions'], insertions=d['insertions'],
                                reference_words=len(d['normalized_reference'].split()), wer=d['diagnostic_wer']))
        for w in row['words']:
            interval = w.get('accepted_interval')
            words.append(dict(sample_id=sid, word_id=w['word_id'], score=w.get('ctc_log_score'), accepted=interval is not None,
                              duration=interval[1]-interval[0] if interval else np.nan))
    assert len(raw['q3']) == 728 and len(raw['q1']) == 100 and max(score_errors) < 2e-6
    for name, rows in [('server_Q3_local',local),('server_Q3_granularity',grains),('server_Q1_words',words),('server_Q1_diagnostics',diagnostics)]:
        pd.DataFrame(rows).to_csv(DATA / (name+'.csv'),index=False)
    (DATA/'server_detail_verification.json').write_text(json.dumps(dict(q3_samples=728,q1_samples=100,
        max_original_score_difference=max(score_errors),prediction_classes_match=True,observation_counts_match=True,
        q1_diagnostic_wer_matches=True,word_records=len(words),accepted_words=sum(w['accepted'] for w in words),
        source_root=raw['server_root'],date='2026-09-26'),indent=2)+'\n',encoding='utf8')


def build(m=base):
    source = 'doc/paper_figures/data/'
    loc = pd.read_csv(DATA/'server_Q3_local.csv')
    rows=[]
    for (sid,mod),g in loc.groupby(['sample_id','modality']):
        values=np.sort(g.D.to_numpy())[::-1]; total=values.sum(); n=len(values)
        rows.append(dict(sample_id=sid,modality=mod,n_positions=n,
                         top20_share=values[:int(np.ceil(.2*n))].sum()/total if total else np.nan,
                         positive_fraction=(g.delta_class>6.5563e-6).mean(),negative_fraction=(g.delta_class < -6.5563e-6).mean(),
                         median_D=g.D.median()))
    conc=pd.DataFrame(rows)
    fig,ax=m.canvas(2)
    sns.violinplot(data=conc,x='modality',y='top20_share',order=list('TAV'),hue='modality',palette=m.MOD,cut=0,inner='quart',legend=False,ax=ax[0])
    ax[0].set(xlabel='',ylabel='Top 20% share of local D',ylim=(0,1.02),title='Local influence concentration')
    sns.ecdfplot(data=conc,x='top20_share',hue='modality',palette=m.MOD,ax=ax[1])
    ax[1].set(xlabel='Top 20% share',ylabel='Cumulative fraction',title='One value per clip / modality',xlim=(0,1))
    m.save(fig,'Q3_17_local_concentration','验证集逐位置影响集中度',conc,[source+'server_Q3_local.csv'],
           '728条开发样本，T/A/V有观测样本728/728/713。每样本按单位置D排序，前ceil(20%n)占全部单位置D之和的比例；不是联合删除忠实性或人类情绪贡献。')
    fig,ax=m.canvas(2)
    sns.violinplot(data=conc,x='modality',y='negative_fraction',order=list('TAV'),hue='modality',palette=m.MOD,cut=0,inner='quart',legend=False,ax=ax[0])
    ax[0].set(xlabel='',ylabel='Fraction with delta p < -epsilon',ylim=(0,1.02),title='Opposing single-position effects')
    for mod in 'TAV':
        g=loc[loc.modality==mod]; sns.ecdfplot(x=g.delta_class,ax=ax[1],color=m.MOD[mod],label=f'{mod} (n={len(g)})')
    ax[1].set_xscale('symlog',linthresh=1e-4); ax[1].axvline(0,color='#777777',lw=.8)
    ax[1].set(xlabel='Original-class delta p (symlog)',ylabel='Cumulative fraction',title='All observed positions');ax[1].legend(fontsize=8)
    m.save(fig,'Q3_18_signed_local','验证集正负单位置影响',conc,[source+'server_Q3_local.csv'],
           'delta p=p原类别(x)-p原类别(删除后)。左面板以既有数值容差epsilon=6.5563e-6区分负向影响；右面板保留全部原始值，位置嵌套于样本，仅描述分布；横轴明确使用symlog，完整保留负值。')
    grain=pd.read_csv(DATA/'server_Q3_granularity.csv'); grain['gain']=grain.group_D-grain.point_D
    grain['matched']=grain.group_cost==grain.point_cost
    summary=[]
    for key,g in grain.groupby('selection'):
        for subset,z in [('All historical',g),('Equal cost',g[g.matched])]:
            rng=np.random.default_rng(20260926+sum(map(ord,key)))
            clusters=z.groupby('video_id').gain.agg(['sum','count']).to_numpy()
            draws=rng.integers(0,len(clusters),(1000,len(clusters))); sampled=clusters[draws].sum(axis=1)
            lo,hi=np.quantile(sampled[:,0]/sampled[:,1],[.025,.975])
            summary.append(dict(selection=key,subset=subset,n=len(z),clusters=len(clusters),mean=z.gain.mean(),ci_low=lo,ci_high=hi,
                                excluded=len(g)-len(z)))
    summary=pd.DataFrame(summary); fig,axes=plt.subplots(1,4,figsize=(12,3.6),layout='constrained')
    for ax,(key,g) in zip(axes,summary.groupby('selection')):
        for i,(_,r) in enumerate(g.iterrows()):
            ax.errorbar(i,r['mean'],yerr=[[r['mean']-r.ci_low],[r.ci_high-r['mean']]],fmt='o',capsize=5,color=m.COLORS[i],ms=7)
            ax.annotate(f'n={r.n}',(i,r['mean']),xytext=(7,7),textcoords='offset points',fontsize=8)
        ax.axhline(0,color='#777',ls='--',lw=1);ax.set_xlim(-.4,1.7);ax.set_xticks([0,1],['All','Equal cost'])
        ax.set_title(key.replace('key.','').replace('.p20.',' / '));ax.set_ylabel('Group D − point D')
        ax.ticklabel_format(axis='y',style='sci',scilimits=(-2,2));m.nice(ax)
    m.save(fig,'Q3_19_equal_cost_granularity','排除成本不一致后的粒度敏感性',summary,[source+'server_Q3_granularity.csv'],
           '新增只读重汇总，不重训/重选/更改原报告。Equal cost只保留group_cost=point_cost的原配对记录；视频簇bootstrap1000次、样本加权95%区间。面板采用不同纵轴尺度，保留零线；子集过滤仍不能证明粒度因果优越。')
    words=pd.read_csv(DATA/'server_Q1_words.csv'); fig,ax=m.canvas(2)
    valid=words.dropna(subset=['score']).copy(); valid['state']=np.where(valid.accepted,'Accepted interval','No accepted interval')
    sns.ecdfplot(data=valid,x='score',hue='state',ax=ax[0],palette=m.COLORS[:2])
    ax[0].set_xscale('symlog',linthresh=.01)
    ax[0].set_xlim(valid.score.min()*1.03, 0)
    ax[0].axvline(-5,color='#444',ls='--',lw=1);ax[0].set(xlabel='CTC log score (symlog)',ylabel='Cumulative fraction',title='Recorded word scores')
    sns.histplot(data=words[words.accepted],x='duration',bins=35,ax=ax[1],color=m.COLORS[2])
    ax[1].set(xlabel='Accepted interval duration (s)',ylabel='Words',title='1793 accepted intervals')
    m.save(fig,'Q1_09_word_alignment','逐词CTC分数与接受区间时长',words,[source+'server_Q1_words.csv'],
           '服务器1932条原始词记录（含未形成有效spoken词的条目），1793条有accepted_interval；因此不将1932替代生产spoken分母1926。左图为各状态内ECDF，排除6条无分数记录，symlog横轴完整保留尾部。分数不是校准概率，区间存在不等于人工确认；不采用陈旧pending状态字段判定接受。')
    d=pd.read_csv(DATA/'server_Q1_diagnostics.csv'); long=d.melt(id_vars=['sample_id','reference_words','wer'],value_vars=['substitutions','deletions','insertions'],var_name='error',value_name='count')
    long['rate']=long['count']/long.reference_words
    fig,ax=m.canvas(2)
    sns.boxplot(data=long,x='error',y='rate',hue='error',palette=m.COLORS[:3],legend=False,ax=ax[0],fliersize=2)
    ax[0].set_xticks([0,1,2],['Sub.','Del.','Ins.']); ax[0].set(xlabel='',ylabel='Errors / reference words',title='Diagnostic transcription errors')
    ranked=d.sort_values('wer'); bottom=np.zeros(len(d))
    for col,color in zip(['substitutions','deletions','insertions'],m.COLORS):
        values=ranked[col]/ranked.reference_words;ax[1].bar(np.arange(100),values,bottom=bottom,width=1,color=color,label=col[:3].capitalize());bottom+=values
    ax[1].set(xlabel='Clip rank by WER',ylabel='Diagnostic WER',title='All 100 clips',xlim=(-1,100));ax[1].legend()
    m.save(fig,'Q1_10_transcription_errors','转写误差类型分解',long,[source+'server_Q1_diagnostics.csv'],
           '独立诊断转写的替换/删除/插入数按规范化参考词数归一化，三者相加等于WER。WER可超过1；它不是CTC时间对齐准确率，更不是情感分类误差。')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--extract',type=Path);args=parser.parse_args()
    if args.extract: extract(args.extract)
    old=json.loads((base.OUT/'figure_manifest.json').read_text(encoding='utf8'))
    build(); ids={e['id'] for e in base.ENTRIES}
    base.ENTRIES[:]=sorted([e for e in old if e['id'] not in ids]+base.ENTRIES,key=lambda e:e['id'])
    base.index()
