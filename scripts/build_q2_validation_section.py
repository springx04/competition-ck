"""Recompute Q2(4) diagnostic tables and plots from saved validation predictions."""
import json
import numpy as np
import pandas as pd
import seaborn as sns
import build_paper_figures as base

SOURCE = 'doc/paper_figures/data/Q2_valid_reloaded_predictions.csv'
SCENARIOS = 'Q2/results/final/valid/metrics_per_scenario.csv'
NAMES = ['Negative', 'Neutral', 'Positive']


def build(m=base):
    d = pd.read_csv(m.ROOT / SOURCE)
    d['correct'] = d.true_class == d.pred_class
    d['residual'] = d.pred_score - d.true_score
    d['abs_error'] = abs(d.residual)
    d['confidence'] = d[['p_neg','p_neu','p_pos']].max(axis=1)
    assert len(d)==728 and d.sample_id.nunique()==728
    assert np.array_equal(d[['p_neg','p_neu','p_pos']].to_numpy().argmax(axis=1),d.pred_class)
    cm = pd.crosstab(d.true_class,d.pred_class).reindex(index=range(3),columns=range(3),fill_value=0)
    assert cm.to_numpy().tolist()==[[126,31,49],[31,65,88],[37,44,257]]
    flows=[]
    for i in range(3):
        for j in range(3):
            if i==j: continue
            g=d[(d.true_class==i)&(d.pred_class==j)]
            flows.append(dict(true_class=NAMES[i],pred_class=NAMES[j],n=len(g),
                              fraction_of_errors=len(g)/(~d.correct).sum(),
                              fraction_of_true_class=len(g)/(d.true_class==i).sum()))
    flows=pd.DataFrame(flows).sort_values('n',ascending=False)
    fig,ax=m.canvas(2,size=(11,3.8))
    ax[0].barh(range(6),flows.n,color=[m.COLORS[NAMES.index(x)] for x in flows.true_class])
    ax[0].set(yticks=range(6),yticklabels=[f'{a[:3]} → {b[:3]}' for a,b in zip(flows.true_class,flows.pred_class)],
              xlabel='Misclassified clips',title='All 280 errors / validation',xlim=(0,105))
    ax[0].invert_yaxis()
    for i,r in enumerate(flows.itertuples()):ax[0].text(r.n+1,i,f'{r.n} ({r.fraction_of_errors:.1%})',va='center',fontsize=8)
    sns.ecdfplot(data=d,x='confidence',hue='correct',palette={True:m.COLORS[0],False:m.COLORS[1]},ax=ax[1])
    ax[1].axvline(.8,color='#707C87',ls='--',lw=1)
    ax[1].set(xlabel='Maximum class probability',ylabel='Cumulative fraction',title='Correct vs wrong predictions',xlim=(1/3,1))
    legend=ax[1].get_legend();legend.set_title('Classification')
    for t in legend.texts:t.set_text({'False':'Wrong (n=280)','True':'Correct (n=448)'}.get(t.get_text(),t.get_text()))
    m.save(fig,'Q2_20_valid_error_flows','验证集全部误分类流向与置信度',d,[SOURCE],
           '左图百分比分母为280条全部误分类，六个非对角流向完整保留，柱色按真实类别区分。右图各正确性组内ECDF；0.8仅为事后诊断参考线，不改变预测或重新选阈值。confidence≥0.8的212条中42条错误。')
    flows.to_csv(m.OUT/'data/Q2_valid_error_flows.csv',index=False)

    # Fixed bins with exact neutral separated; these boundaries are not fitted to outcomes.
    d['score_group']=np.select([d.true_score<=-1,(d.true_score>-1)&(d.true_score<0),d.true_score==0,
                               (d.true_score>0)&(d.true_score<1),d.true_score>=1],
                              ['[-3,-1]','(-1,0)','0','(0,1)','[1,3]'],default='outside')
    order=['[-3,-1]','(-1,0)','0','(0,1)','[1,3]']
    bins=d.groupby('score_group').agg(n=('correct','size'),accuracy=('correct','mean'),mae=('abs_error','mean'),
                                    bias=('residual','mean'),mean_true=('true_score','mean'),mean_pred=('pred_score','mean')).reindex(order).reset_index()
    fig,ax=m.canvas(2,size=(11,4))
    sns.violinplot(data=d,x='score_group',y='residual',order=order,hue='score_group',palette=m.COLORS[:5],
                   cut=0,inner='quart',legend=False,ax=ax[0])
    ax[0].axhline(0,color='#707C87',ls='--',lw=1)
    ax[0].set(xlabel='True intensity bin',ylabel='Prediction − truth',title='Signed errors / full observed range')
    x=np.arange(5)
    ax[1].plot(x,bins.mean_true,'o-',color=m.COLORS[0],label='Mean truth')
    ax[1].plot(x,bins.mean_pred,'s-',color=m.COLORS[1],label='Mean prediction')
    ax[1].axhline(0,color='#707C87',ls='--',lw=1)
    ax[1].set(xticks=x,xticklabels=[f'{r.score_group}\nn={r.n}' for r in bins.itertuples()],
              xlabel='True intensity bin',ylabel='Mean intensity',title='Prediction shrinkage by fixed bins');ax[1].legend()
    m.save(fig,'Q2_21_valid_intensity_bias','验证集情感强度分层与有符号回归偏差',d,[SOURCE],
           '固定分组[-3,-1]、(-1,0)、{0}、(0,1)、[1,3]；不移除大误差。残差为预测减真值。组均值比较可检验幅度收缩现象；不据此断言损失函数、讽刺或模态冲突是已验证原因。每组样本数、MAE、bias见Q2_valid_intensity_bins.csv。')
    bins.to_csv(m.OUT/'data/Q2_valid_intensity_bins.csv',index=False)

    allsc=pd.read_csv(m.ROOT/SCENARIOS)
    grid=allsc[(allsc.scope=='all_samples')&allsc.scenario.str.match(r'^(T|A|V|TA|TV|AV)_(front|middle|rear)_0\.[2468]$')].copy()
    assert len(grid)==72
    grid[['modality','position','rho']]=grid.scenario.str.split('_',expand=True);grid['rho']=grid.rho.astype(float)
    fig,ax=m.canvas(2,size=(12,4.2))
    pivot=grid.groupby(['modality','rho']).macro_f1.mean().unstack().reindex(['T','A','V','TA','TV','AV'])
    sns.heatmap(pivot,annot=True,fmt='.3f',cmap='YlGnBu',vmin=0,vmax=1,linewidths=.7,
                cbar_kws={'label':'Macro-F1'},ax=ax[0])
    ax[0].set(xlabel='Nominal missing span',ylabel='Masked modalities',title='Validation / mean over three positions');ax[0].tick_params(axis='y',rotation=0)
    clean=allsc[(allsc.scope=='all_samples')&(allsc.scenario=='clean')].iloc[0]
    for j,pos in enumerate(['front','middle','rear']):
        z=grid[grid.position==pos].groupby('rho').macro_f1.mean()
        ax[1].plot(z.index,z.values,'o-',color=m.COLORS[j],label=pos.capitalize())
    ax[1].axhline(clean.macro_f1,color='#647483',ls='--',label='Clean')
    ax[1].set(xticks=[.2,.4,.6,.8],xlabel='Nominal missing span',ylabel='Macro-F1',
              title='Position trend / mean over six modalities',ylim=(grid.groupby(['position','rho']).macro_f1.mean().min()-.012,clean.macro_f1+.018))
    ax[1].legend(ncol=2,fontsize=8);ax[1].text(.03,.03,'Zoomed y-axis',transform=ax[1].transAxes,fontsize=8,color='#647483')
    m.save(fig,'Q2_22_valid_missing_profile','验证集局部缺失的模态、位置与跨度规律',grid,[SCENARIOS],
           '仅使用valid的72个all_samples场景；每场景728条、相同样本复用，不是52416个独立样本。左图每格平均3个位置、色标固定0–1；右图每点平均6种模态组合，纵轴局部放大且标注。跨度是名义ρ，不等于样本实际删除比例。')

    perclass=d.groupby('true_class').agg(n=('correct','size'),accuracy=('correct','mean'),mae=('abs_error','mean'),
                    bias=('residual','mean'),mean_true=('true_score','mean'),mean_pred=('pred_score','mean'))
    perclass.to_csv(m.OUT/'data/Q2_valid_class_error_summary.csv')
    errors=d[~d.correct].copy(); errors['confidence_rank']=errors.confidence.rank(method='first',ascending=False).astype(int)
    errors.sort_values('confidence_rank').to_csv(m.OUT/'data/Q2_valid_all_errors.csv',index=False)
    high=d[d.confidence>=.8]
    result=dict(n=len(d),correct=int(d.correct.sum()),incorrect=int((~d.correct).sum()),
                confusion=cm.to_numpy().tolist(),mae=float(d.abs_error.mean()),
                pearson=float(d.true_score.corr(d.pred_score)),rmse=float(np.sqrt((d.residual**2).mean())),
                mean_confidence=float(d.confidence.mean()),high_confidence_n=len(high),high_confidence_errors=int((~high.correct).sum()),
                video_clusters=int(d.sample_id.str.split('$_$',regex=False).str[0].nunique()),
                stage='post-hoc validation diagnosis, no new training or selection')
    (m.OUT/'data/Q2_valid_section_summary.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')


if __name__=='__main__':
    old=json.loads((base.OUT/'figure_manifest.json').read_text(encoding='utf8'))
    build(); ids={e['id'] for e in base.ENTRIES}
    base.ENTRIES[:]=sorted([e for e in old if e['id'] not in ids]+base.ENTRIES,key=lambda e:e['id'])
    base.index()
