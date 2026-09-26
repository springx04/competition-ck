"""Paired summaries with source-video bootstrap and explicit eligibility counts."""
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, precision_recall_fscore_support
from .data import load_samples
from .results import write_json
from .statistics import cluster_bootstrap, summarize_faithfulness
from .workflow import read_rows, run_dir


def paired_summary(rows, total, repeats, seed):
    eligible=[r for r in rows if r.get('target') is not None and r.get('random') is not None]
    result=summarize_faithfulness(eligible)
    result.update(n_total=total,n_planned=len(rows),n_complete=sum(r.get('target') is not None for r in rows),
                  n_eligible=len(eligible),n_not_applicable=sum(r.get('target') is None for r in rows),
                  n_without_controls=sum(r.get('target') is not None and r.get('random') is None for r in rows))
    if not eligible: return result
    values=np.asarray([r['target']-r['random'] for r in eligible]); clusters=np.asarray([r['video_id'] for r in eligible])
    count=len(np.unique(clusters))
    ci=np.quantile(cluster_bootstrap(values,clusters,repeats,seed),[.025,.975]).tolist() if count>=2 else [None,None]
    result.update(ci_low=ci[0],ci_high=ci[1],bootstrap_unit='video_id cluster; sample-weighted mean',bootstrap_clusters=count,
                  random_count_min=min(r.get('n_random',0) for r in eligible),random_count_max=max(r.get('n_random',0) for r in eligible),
                  n_random_below_20=sum(r.get('n_random',0)<20 for r in eligible),
                  mean_target=float(np.mean([r['target'] for r in eligible])),mean_random=float(np.mean([r['random'] for r in eligible])))
    return result


def task_metrics(labels, predictions, truth_scores, scores):
    labels=np.asarray(labels); predictions=np.asarray(predictions);truth_scores=np.asarray(truth_scores);scores=np.asarray(scores)
    p,r,f,support=precision_recall_fscore_support(labels,predictions,labels=[0,1,2],zero_division=0)
    correlation=float(np.corrcoef(truth_scores,scores)[0,1]) if len(scores)>1 and np.std(scores)>0 and np.std(truth_scores)>0 else None
    return dict(n=len(labels),accuracy=float(accuracy_score(labels,predictions)),
                macro_f1=float(f1_score(labels,predictions,labels=[0,1,2],average='macro',zero_division=0)),
                weighted_f1=float(f1_score(labels,predictions,labels=[0,1,2],average='weighted',zero_division=0)),
                mae=float(mean_absolute_error(truth_scores,scores)),pearson=correlation,
                per_class=[dict(class_id=i,precision=float(p[i]),recall=float(r[i]),f1=float(f[i]),support=int(support[i])) for i in range(3)])


def summarize(config, split):
    run=run_dir(config,split);samples=load_samples(config,split);destination=run/'summaries'
    pairs=defaultdict(list);direction=Counter();failures=[];lows=defaultdict(list);grains=defaultdict(list);strata=defaultdict(list);tasks=defaultdict(list)
    originals=[];modal=[];flat=[]
    for sample in samples:
        directory=run/'samples'/f'{sample.sample_index:04d}'
        if not (directory/'iteration_status.json').exists(): continue
        def read(name):return json.loads((directory/name).read_text(encoding='utf-8'))
        original=read('original.json');originals.append((sample,original));groups=defaultdict(list)
        for e in read_rows(directory/'experiments.jsonl'):groups[(e['selection_id'],e['operation'],e['control_role'])].append(e)
        sels={s['selection_id']:s for s in read_rows(directory/'selections.jsonl')}
        modality=read('modality.json');shapley=read('shapley.json')
        modal.append(dict(sample_id=sample.sample_id,video_id=sample.video_id,D=modality['D'],n_observed=original['n_observed'],phi_class=shapley['phi_class'],phi_score_raw=shapley['phi_score_raw']))
        planned=[(s['selection_id'],op) for s in sels.values() if s['experiment_kind']=='key' for op in (['delete','global_keep','conditional_keep'] if s['budget_pct']==20 else ['delete'])]
        planned.append(('key.U.p20.union','global_keep'))
        for sid,op in planned:
            key=groups[(sid,op,'key')];randoms=groups[(sid,op,'random')];metric='D' if op=='delete' else 'S'
            row=dict(sample_id=sample.sample_id,video_id=sample.video_id,target=key[0][metric] if key else None,
                     random=float(np.mean([r[metric] for r in randoms])) if randoms else None,n_random=len(randoms))
            pairs[(sid,op)].append(row)
            flat.append(dict(split=split,selection_id=sid,operation=op,**row))
            if split=='valid' and op=='delete' and sid in [f'key.{m}.p20.point' for m in 'TAV']+['key.J.p20.joint','key.T.p20.word']:
                for label,eligible in [('true_neutral',sample.label==1),('original_correct',sample.label==original['c_star']),('original_wrong',sample.label!=original['c_star'])]:
                    if eligible:strata[(sid,label)].append(row)
                if key and randoms:
                    tasks[sid].append((sample,original,key[0],randoms))
        for s in sels.values():
            sid=s['selection_id'];role=s['experiment_kind']
            if role in ('support','suppress'):
                direction['planned']+=1;found=groups[(sid,'delete',role)]
                if not found:direction['not_applicable']+=1;continue
                direction['evaluated']+=1;direction['retained' if found[0]['direction_retained'] else 'failed']+=1
                if not found[0]['direction_retained']:failures.append(dict(sample_id=sample.sample_id,selection_id=sid,delta_class=found[0]['delta_class'],delta_score_raw=found[0]['delta_score_raw']))
            if role=='key' and s['granularity']=='point':
                key=groups[(sid,'delete','key')];low=groups[(sid.replace('key.','low.',1),'delete','low')]
                if key and low and key[0]['cost']==low[0]['cost']:
                    lows[sid].append(dict(sample_id=sample.sample_id,video_id=sample.video_id,target=key[0]['D'],random=low[0]['D']))
            if role=='key' and s['granularity'] in ('word','block3') and s['status']=='complete':
                point_id=s.get('cost_matched_point_id',f'key.{s["path"]}.p20.point')
                key=groups[(sid,'delete','key')];point=groups[(point_id,'delete','key')]
                if key and point:
                    left=set(s['delete_j']);right=set(sels[point_id]['delete_j'])
                    grains[sid].append(dict(sample_id=sample.sample_id,video_id=sample.video_id,target=key[0]['D'],random=point[0]['D'],actual_cost=s['actual_cost'],point_cost=sels[point_id]['actual_cost'],jaccard=len(left&right)/len(left|right)))
    repeats=int(config['method']['bootstrap_repeats']);seed=int(config['runtime']['seed'])
    def summarize_dict(source):
        return {' / '.join(k) if isinstance(k,tuple) else k:paired_summary(v,len(samples),repeats,seed+sum(map(ord,str(k)))) for k,v in sorted(source.items())}
    combined=summarize_dict(pairs)
    deletion={key.split(' / ')[0]:val for key,val in combined.items() if key.endswith(' / delete')}
    keep={key:val for key,val in combined.items() if not key.endswith(' / delete')}
    grain_summary=summarize_dict(grains)
    for key,rows in grains.items():
        grain_summary[key].update(mean_jaccard=float(np.mean([r['jaccard'] for r in rows])),cost_mismatches=sum(r['actual_cost']!=r['point_cost'] for r in rows),contrast='group D minus same-cost point D; not a causal interaction')
    modality_summary=dict(n=len(modal),mean_D=np.mean([r['D'] for r in modal],axis=0).tolist(),
                          mean_phi_class=np.mean([r['phi_class'] for r in modal],axis=0).tolist(),mean_phi_score_raw=np.mean([r['phi_score_raw'] for r in modal],axis=0).tolist(),
                          observed_samples=np.sum(np.asarray([r['n_observed'] for r in modal])>0,axis=0).tolist(),rows=modal)
    outputs=dict(faithfulness=deletion,keep=keep,low_comparison=summarize_dict(lows),granularity=grain_summary,
                 stratified=summarize_dict(strata),direction=dict(**direction,failures=failures),modality=modality_summary,
                 completeness=dict(expected=len(samples),complete=len(originals),video_clusters=len({s.video_id for s,o in originals})))
    if split=='valid':
        outputs['prediction_metrics']=task_metrics([s.label for s,o in originals],[o['c_star'] for s,o in originals],[s.score_label for s,o in originals],[o['score'] for s,o in originals])
        perturb={}
        for key,rows in tasks.items():
            labels=[s.label for s,o,k,r in rows];truth=[s.score_label for s,o,k,r in rows]
            clean=task_metrics(labels,[o['c_star'] for s,o,k,r in rows],truth,[o['score'] for s,o,k,r in rows])
            selected=task_metrics(labels,[k['predicted_class'] for s,o,k,r in rows],truth,[k['score'] for s,o,k,r in rows])
            random_metrics=[]
            for trial in range(50):
                # Stored trials are independently shuffled within each sample.
                random_metrics.append(task_metrics(labels,[r[trial%len(r)]['predicted_class'] for s,o,k,r in rows],truth,[r[trial%len(r)]['score'] for s,o,k,r in rows]))
            random_summary={name:dict(mean=float(np.mean([r[name] for r in random_metrics])),min=float(min(r[name] for r in random_metrics)),max=float(max(r[name] for r in random_metrics))) for name in ('accuracy','macro_f1','weighted_f1','mae')}
            perturb[key]=dict(n=len(rows),clean_same_subset=clean,key=selected,random=random_summary,random_repeats=50,
                             interpretation='Each trial computes dataset metrics before averaging; samples with <50 controls cycle their shuffled set. Ranges are descriptive, not independent confidence intervals.')
        outputs['perturbation_metrics']=perturb
    for name,value in outputs.items():write_json(destination/f'{name}.json',value)
    with (destination/'paired_samples.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    print(f'summarize {split}: {len(originals)}/{len(samples)} samples, {len(deletion)} deletion groups, {len(keep)} keep groups')
    return outputs


def plots(outputs, destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    figs=destination/'figures';figs.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(1,4,figsize=(12,3.6))
    for axis,path in zip(axes,'TAVJ'):
        rows=[outputs['faithfulness'][f'key.{path}.p{budget}.'+('joint' if path=='J' else 'point')] for budget in (10,20,30)]
        y=np.asarray([r['mean_G'] for r in rows]);lo=np.asarray([r['ci_low'] for r in rows]);hi=np.asarray([r['ci_high'] for r in rows])
        axis.errorbar([10,20,30],y,yerr=[y-lo,hi-y],marker='o',capsize=4,color='#166591')
        axis.axhline(0,color='#aaa',lw=1);axis.set_title(path+' | n='+','.join(str(r['n_eligible']) for r in rows));axis.set_xticks([10,20,30]);axis.set_xlabel('Budget (%)');axis.set_ylabel('Mean paired G')
    fig.suptitle('Deletion faithfulness | 95% source-video cluster bootstrap | axes differ')
    fig.tight_layout();fig.savefig(figs/'faithfulness_budgets.png',dpi=180);fig.savefig(figs/'faithfulness_budgets.svg');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(10,3.4))
    for axis,key,title in zip(axes,['mean_D','mean_phi_class','mean_phi_score_raw'],['Whole-modality deletion D','Signed Shapley: class','Signed Shapley: score']):
        axis.bar(list('TAV'),outputs['modality'][key],color=['#166591','#399a85','#bd8040']);axis.axhline(0,color='#888',lw=.8);axis.set_title(title);axis.set_ylabel('Mean across 728 samples')
    fig.tight_layout();fig.savefig(figs/'modality_shapley.png',dpi=180);fig.savefig(figs/'modality_shapley.svg');plt.close(fig)
    chosen=[(k,v) for k,v in outputs['keep'].items() if '.p20.point / ' in k or '.p20.union / ' in k or '.p20.joint / ' in k]
    fig,axis=plt.subplots(figsize=(10,5.5))
    for i,(key,value) in enumerate(chosen):
        mean=value['mean_G'];axis.errorbar(mean,i,xerr=[[mean-value['ci_low']],[value['ci_high']-mean]],fmt='o',capsize=3,color='#166591')
    axis.set_yticks(range(len(chosen)),[f'{k} (n={v["n_eligible"]})' for k,v in chosen]);axis.axvline(0,color='#aaa');axis.set_xlabel('Key S - matched-random mean S; cluster 95% interval');fig.tight_layout();fig.savefig(figs/'keep_comparison.png',dpi=180);fig.savefig(figs/'keep_comparison.svg');plt.close(fig)


def report(config, split):
    outputs=summarize(config,split)
    if split=='special':
        from .report import build_special_rows,write_special_outputs
        write_special_outputs(build_special_rows(config),Path(config['project']['root'])/'outputs_v2')
        return
    root=Path(config['project']['root']);destination=root/'reports/iteration_v2';destination.mkdir(parents=True,exist_ok=True)
    plots(outputs,destination)
    metrics=outputs['prediction_metrics'];direction=outputs['direction']
    lines=['# Q3 迭代2实验报告','',
           '全量 valid 为开发评价；原预测器、遮蔽规则、20%主预算和最多3段均保持原方法。附件4无标签，不能报告专项准确率。旧运行目录保留，新结果位于 valid_v2 / special_v2。','',
           f'原样预测：n={metrics["n"]}，Accuracy={metrics["accuracy"]:.6f}，Macro-F1={metrics["macro_f1"]:.6f}，Weighted-F1={metrics["weighted_f1"]:.6f}，MAE={metrics["mae"]:.6f}，Pearson={metrics["pearson"]:.6f}。',
           f'有效来源视频簇：{outputs["completeness"]["video_clusters"]}。全部标签仅进入评价，未进入归因或选段。','',
           '## 严格匹配后的忠实性','',
           'G = D(关键集合) − 匹配随机集合平均 D。每个随机集合保持官方片段长度多重集合、完整单元和三路成本，排除目标。置信区间以 video_id 为整簇抽样，簇内保留所有片段后计算样本均值。少于20个随机集合仍可计算描述均值，但不能支持精细尾部判断。','',
           '| 实验 | 完整/计划 | 可比较 | 无对照 | mean G | median G | G>0 | 95% CI |',
           '|---|---:|---:|---:|---:|---:|---:|---|']
    for key,value in outputs['faithfulness'].items():
        if not value['n_eligible']:continue
        ci=f'[{value["ci_low"]:.6f}, {value["ci_high"]:.6f}]' if value['ci_low'] is not None else 'N/A'
        lines.append(f'| {key} | {value["n_complete"]}/{value["n_planned"]} | {value["n_eligible"]} | {value["n_without_controls"]} | {value["mean_G"]:.6f} | {value["median_G"]:.6f} | {value["positive_fraction"]:.3f} | {ci} |')
    lines+=['','![三预算忠实性](figures/faithfulness_budgets.png)','',
            '文本与联合路径有稳定的正平均增益；A/V 的绝对增益明显较小。95%区间不跨0不等于人类情绪解释准确，也不保证每条样本有效。小样本同成本子组的区间应与有效样本数一起阅读。','',
            '## 保留、低影响与粒度','',
            '保留定义 S=1−D。global_keep 删除全输入集合外内容；conditional_keep 仅删除对应模态域的其余内容。以下均为同样本配对，不把不同成本直接比较。','',
            '![仅保留证据](figures/keep_comparison.png)','',
            '| 比较 | n | 配对平均差 | 95% CI |','|---|---:|---:|---|']
    for kind,values in [('key D − low D',outputs['low_comparison']),('group D − 同成本point D',outputs['granularity'])]:
        for key,value in values.items():
            if not value['n_eligible']:continue
            lines.append(f'| {kind}：{key} | {value["n_eligible"]} | {value["mean_G"]:.6f} | [{value["ci_low"]:.6f}, {value["ci_high"]:.6f}] |')
    lines+=['','整词和3位置分块以组为完整单元先做真实联合遮蔽；预算取不超过目标的最大可达成本。同成本 point 对照随实际成本重选。block3 为固定不重叠的官方索引块，只是粒度敏感性，不能叫音视频复制组或真实时间片。分组 Jaccard、成本一致性与各样本数见 granularity.json。','',
            '## 方向与双输出','',f'valid 方向候选 {direction["planned"]} 个，重测 {direction["evaluated"]} 个，保持 {direction["retained"]} 个，失败 {direction["failed"]} 个，不适用 {direction["not_applicable"]} 个。全部失败记录保留在 direction.json。',
            '专项09、18号的文本 suppress 候选方向反转；14号分类首位V、强度首位T；13号视觉无观测。卡片保留这些反例。','',
            '![整模态与Shapley](figures/modality_shapley.png)','',
            '## 中性类与错误预测分层','',
            '| 路径 / 分层 | 可比较 n | mean G | G>0 |','|---|---:|---:|---:|']
    for key,value in outputs['stratified'].items():
        if value['n_eligible']:lines.append(f'| {key} | {value["n_eligible"]} | {value["mean_G"]:.6f} | {value["positive_fraction"]:.3f} |')
    lines+=['','## 干预后的任务表现','',
            '每个随机重复先在同一可比较样本集合上计算任务指标，再对指标取平均；没有把平均概率当作一次随机模型输出。随机集合不足50时循环其随机排列，重复范围仅是描述变化，不是独立置信区间。','',
            '| 路径 | n | clean F1 | key F1 | random mean F1 | clean MAE | key MAE | random mean MAE |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for key,value in outputs['perturbation_metrics'].items():
        c=value['clean_same_subset'];k=value['key'];r=value['random']
        lines.append(f'| {key} | {value["n"]} | {c["macro_f1"]:.5f} | {k["macro_f1"]:.5f} | {r["macro_f1"]["mean"]:.5f} | {c["mae"]:.5f} | {k["mae"]:.5f} | {r["mae"]["mean"]:.5f} |')
    lines+=['','## 素材核验与结论边界','',
            '20条专项均可回到实际遮蔽的原文字符；CTC共有479个词时候选，其中14个被算法拒绝。边界与媒体时钟可机械检查，但尚无逐词听审确认。官方A/V特征行生成来源仍unresolved，不能按位置比例伪造时间。逐词播放页及真实PTS参考帧位于 outputs_v2/assets；参考帧只是素材上下文。','',
            '修正后的文本/联合解释具有可用的模型忠实性证据，但A/V效果较弱，功能词或标点仍可能成为高影响集合。计算结果完整不等于素材证据链完整，也不等于模型预测已达到很高准确率。无需用改变主要模态、删除失败案例或重训Q2来美化这些结论。','',
            '可复算材料：runs/valid_v2/summaries 下的 JSON 与 paired_samples.csv；全部逐视图/逐实验明细保留在服务器。验证报告与预测重算记录独立保存。']
    content='\n'.join(lines)+'\n'
    (destination/'Q3_实验报告.md').write_text(content,encoding='utf-8')
    (root/'reports/Q3_实验报告_v2_valid.md').write_text(content.replace('(figures/','(iteration_v2/figures/'),encoding='utf-8')
