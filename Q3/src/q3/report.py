"""Readable case cards, with independent prediction, perturbation and material status."""
from pathlib import Path
from collections import defaultdict
import csv
import html
import json
import numpy as np
from .results import write_json
from .workflow import read_rows, run_dir
from .data import load_samples

FIELDS = ['sample_id','source_id','status','predicted_class_id','predicted_class_name','sentiment_score',
          'p_negative','p_neutral','p_positive','n_T','n_A','n_V','D_T','D_A','D_V','D_sum',
          'w_T','w_A','w_V','primary_modalities','head_top_disagreement','evidence_T','evidence_A','evidence_V',
          'selected_text','mapping_status','ctc_human_verified','ctc_words','case_json','case_html']
STYLE = 'body{font:16px/1.6 system-ui,sans-serif;color:#203047;background:#f5f7fa;margin:0}main{max-width:1120px;margin:36px auto;padding:0 24px}section{background:white;padding:24px;margin:20px 0;border-radius:12px;border:1px solid #dae2eb}h1{font-size:30px}h2{font-size:21px;color:#174b75}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:9px;text-align:left;border-bottom:1px solid #dae2eb}th{background:#edf3f8}mark{background:#ffd579;padding:0 2px}.note{border-left:4px solid #dd9d20;padding:12px;background:#fff6df}.bad{color:#ac2730;font-weight:600}.muted{color:#687689}.scroll{overflow-x:auto}svg{width:100%;height:auto}a{color:#155b9a}.num{font-variant-numeric:tabular-nums}details{margin:18px 0}summary{cursor:pointer;font-weight:600}'

def _json(value): return json.dumps(value,ensure_ascii=False,separators=(',',':'))
def esc(value): return html.escape(str(value))
def fmt(value): return 'N/A' if value is None else f'{float(value):.6f}'

def top_modalities(values, observed, epsilon):
    valid=[i for i,n in enumerate(observed) if n>0]
    if not valid or max(abs(values[i]) for i in valid)<=epsilon: return []
    maximum=max(abs(values[i]) for i in valid)
    return ['TAV'[i] for i in valid if abs(values[i])>=maximum-epsilon]

def _case(config, sample):
    index=sample.sample_index; sid=sample.sample_id
    d=run_dir(config,'special')/'samples'/f'{index:04d}'
    required=['original.json','modality.json','shapley.json','local.json','mapping.json','selections.jsonl','experiments.jsonl','iteration_status.json']
    missing=[f for f in required if not (d/f).exists()]
    if missing:
        return dict(sample_id=sid,source_id=sid,status='partial',missing_files=missing,raw_text=sample.raw_text or '',mapping_status='unresolved')
    def read(name): return json.loads((d/name).read_text(encoding='utf-8'))
    original=read('original.json'); modality=read('modality.json'); shapley=read('shapley.json'); mapping=read('mapping.json')
    selections=read_rows(d/'selections.jsonl'); experiments=read_rows(d/'experiments.jsonl')
    numeric=json.loads((Path(config['project']['root'])/'reports/numeric_tolerance_v2.json').read_text(encoding='utf-8'))
    vals=np.asarray(modality['D'],float); total=float(vals.sum()); weights=(vals/total).tolist() if total>numeric['eps_D'] else [None]*3
    subsets=shapley['subsets']; cls=int(original['c_star'])
    p=np.asarray([x['probs'] for x in subsets]); scores=np.asarray([x['score'] for x in subsets])
    dc=[float(p[7,cls]-p[7^(1<<m),cls]) for m in range(3)]
    ds=[float(scores[7]-scores[7^(1<<m)]) for m in range(3)]
    ct=top_modalities(dc,original['n_observed'],numeric['eps_prob']); st=top_modalities(ds,original['n_observed'],numeric['eps_score'])
    disagreement=None if ct==st else dict(classification=ct[0] if len(ct)==1 else ct,intensity=st[0] if len(st)==1 else st)
    evidence={m:next((s['intervals'] for s in selections if s['selection_id']==f'key.{m}.p20.point'),[]) for m in 'TAV'}
    row=dict(sample_id=sid,source_id=sid,status=read('iteration_status.json')['status'],predicted_class_id=cls,
             predicted_class_name=config['method']['class_names'][cls],sentiment_score=original['score'],
             p_negative=original['probs'][0],p_neutral=original['probs'][1],p_positive=original['probs'][2],
             D_sum=total,primary_modalities=top_modalities(vals,original['n_observed'],numeric['eps_D']),
             head_top_disagreement=disagreement,classification_top=ct,intensity_top=st,
             delta_class_by_modality=dc,delta_score_by_modality=ds,raw_text=sample.raw_text,
             mapping_status=mapping['coverage']['status'],ctc_human_verified=mapping['coverage']['ctc_word_verified'],
             ctc_words=mapping['coverage']['ctc_word_denominator'],
             selected_text=mapping['evidence'].get('key.T.p20.point',{}).get('selected_text',''),
             original=original,modality=modality,shapley=shapley,mapping=mapping,local=read('local.json'),
             selections=selections,experiments=experiments,numeric_tolerance=numeric)
    for i,m in enumerate('TAV'):
        row.update({f'n_{m}':original['n_observed'][i],f'D_{m}':float(vals[i]),f'w_{m}':weights[i],f'evidence_{m}':evidence[m]})
    return row

def build_special_rows(config): return [_case(config,s) for s in load_samples(config,'special')]

def highlighted_text(row):
    evidence=row['mapping']['evidence'].get('key.T.p20.point',{})
    spans=sorted(evidence.get('char_spans',[])); text=row.get('raw_text') or ''
    cursor=0; parts=[]
    for start,end in spans:
        start=max(cursor,start)
        if end<=start: continue
        parts += [esc(text[cursor:start]),'<mark>'+esc(text[start:end])+'</mark>'];cursor=end
    parts.append(esc(text[cursor:]))
    return ''.join(parts)

def local_heatmap(row):
    values=row['local']['D']; width=1000; cells=[]
    for m,name in enumerate('TAV'):
        maximum=max((float(v[m]) for v in values if v[m] is not None),default=0.)
        y=32+m*55; cells.append(f'<text x="5" y="{y+18}" font-size="16">{name}</text>')
        for t in range(50):
            value=values[t][m]; ratio=0 if value is None or maximum==0 else float(value)/maximum
            color='#dbe1e8' if value is None else f'rgb({int(247-210*ratio)},{int(251-139*ratio)},{int(255-93*ratio)})'
            x=35+t*17
            cells.append(f'<rect x="{x}" y="{y}" width="16" height="25" fill="{color}"><title>{name} t={t}: D={fmt(value)}</title></rect>')
        cells.append(f'<text x="894" y="{y+18}" font-size="12">max {maximum:.5f}</text>')
    for t in range(0,50,5): cells.append(f'<text x="{35+t*17}" y="210" font-size="12">{t}</text>')
    return '<svg role="img" aria-label="三模态50个官方位置的局部影响" viewBox="0 0 1000 225">'+''.join(cells)+'</svg>'

def grouped_trials(row):
    result=defaultdict(list)
    for experiment in row['experiments']:
        result[(experiment['selection_id'],experiment['operation'],experiment['control_role'])].append(experiment)
    return result

def _render(row):
    sid=esc(row['sample_id'])
    start=f'<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Q3 样本 {sid}</title><style>{STYLE}</style><main><p><a href="../index.html">全部20条案例</a></p><h1>Q3 · 样本 {sid}</h1>'
    if row['status']!='complete':
        return start+f'<section><h2>计算尚未完整</h2><p>{esc(row.get("missing_files",row["status"]))}</p></section></main></html>'
    lines=[start,f'<section><h2>原样预测与整模态影响</h2><p><b>{esc(row["predicted_class_name"])}</b> · 强度 <b>{row["sentiment_score"]:.5f}</b> · 概率 [负/中/正] = [{row["p_negative"]:.4f}, {row["p_neutral"]:.4f}, {row["p_positive"]:.4f}]</p>',
           f'<p>综合影响首位：{esc(", ".join(row["primary_modalities"]) or "数值上未区分")}；分类头：{esc(row["classification_top"])}；强度头：{esc(row["intensity_top"])}；总 D = {row["D_sum"]:.6f}。</p>',
           '<p class="muted">模态比例是整路删除的相对输出敏感性；不等于情感占比、可靠性或因果贡献。</p><table><tr><th>模态</th><th>观测行数</th><th>D</th><th>相对权重</th><th>Δp 原类别</th><th>Δs 原单位</th><th>Shapley 分类 / 强度</th></tr>']
    for i,m in enumerate('TAV'):
        weight=row['w_'+m]; w='未区分' if weight is None else f'{weight:.1%}'
        phi=f'{row["shapley"]["phi_class"][i]:.5f} / {row["shapley"]["phi_score_raw"][i]:.5f}'
        lines.append(f'<tr><td>{m}</td><td>{row["n_"+m]}</td><td>{row["D_"+m]:.6f}</td><td>{w}</td><td>{row["delta_class_by_modality"][i]:+.6f}</td><td>{row["delta_score_by_modality"][i]:+.6f}</td><td>{phi}</td></tr>')
    lines+=['</table></section><section><h2>实际遮蔽的文本与局部影响</h2>',f'<p>{highlighted_text(row)}</p>',
            '<p class="muted">高亮仅表示 20% 主文本集合实际遮蔽的字符，其他原文仅作上下文。热图逐行归一化，右侧给出各模态绝对最大 D；灰色表示无观测，不是零影响。横轴为官方位置，不能当作秒数。</p>',local_heatmap(row),
            '<table><tr><th>主证据</th><th>官方半开区间 [lo, hi)</th><th>实际成本 / 预算</th><th>素材证据</th></tr>']
    sels={s['selection_id']:s for s in row['selections']}; trials=grouped_trials(row)
    for m in 'TAV':
        s=sels[f'key.{m}.p20.point']; material='原文字符已匹配；词时待听审' if m=='T' else '官方特征行来源 unresolved'
        if s['status']!='complete': material='无观测/不适用'
        lines.append(f'<tr><td>{m}</td><td>{esc(s["intervals"])}</td><td>{s["actual_cost"]} / {s["target_budget"]}</td><td>{material}</td></tr>')
    lines+=['</table></section><section><h2>严格对照：删除后影响</h2><p>G = 关键集合 D − 匹配随机集合平均 D。正值表示模型对选中集合更敏感；不保证符合人类情绪直觉。随机集合同时匹配模态成本、片段长度和完整单元。</p>',
            '<div class="scroll"><table><tr><th>选择路径</th><th>关键 D</th><th>低影响 D</th><th>随机均值</th><th>G</th><th>随机 n</th></tr>']
    for s in row['selections']:
        sid0=s['selection_id']
        if s['experiment_kind']!='key': continue
        target=trials[(sid0,'delete','key')]; randoms=trials[(sid0,'delete','random')]
        low=trials[(sid0.replace('key.','low.',1),'delete','low')]
        if not target:
            lines.append(f'<tr><td>{esc(sid0)}</td><td colspan="5">N/A：{esc(s.get("reason"))}</td></tr>');continue
        d=target[0]['D']; mean=float(np.mean([r['D'] for r in randoms])) if randoms else None
        lines.append(f'<tr><td>{esc(sid0)}</td><td>{fmt(d)}</td><td>{fmt(low[0]["D"] if low else None)}</td><td>{fmt(mean)}</td><td>{fmt(d-mean if mean is not None else None)}</td><td>{len(randoms)}</td></tr>')
    lines+=['</table></div></section><section><h2>方向候选：整段重测后的判定</h2><p>support 应使 Δp &gt; 容差；suppress 应使 Δp &lt; −容差。失败候选保留，不重新挑选以消除反例。</p><table><tr><th>候选</th><th>Δp 原类别</th><th>Δs 原单位</th><th>D</th><th>方向保持</th></tr>']
    for s in row['selections']:
        if s['experiment_kind'] not in ('support','suppress'): continue
        candidates=trials[(s['selection_id'],'delete',s['experiment_kind'])]
        if not candidates:
            lines.append(f'<tr><td>{esc(s["selection_id"])}</td><td colspan="4">N/A：{esc(s.get("reason"))}</td></tr>');continue
        e=candidates[0]; css='' if e['direction_retained'] else ' class="bad"'
        label='保持' if e['direction_retained'] else '失败：不可当作该方向的证据'
        lines.append(f'<tr><td>{esc(s["selection_id"])}</td><td>{e["delta_class"]:+.6f}</td><td>{e["delta_score_raw"]:+.6f}</td><td>{e["D"]:.6f}</td><td{css}>{label}</td></tr>')
    lines+=['</table></section><section><h2>仅保留证据</h2><p>S = 1 − D。global_keep 删除全输入中集合外的内容；conditional_keep 只删除当前路径域内的其他内容。此处比较相同样本、相同干预定义的关键与随机保留。</p><details><summary>展开保留对照数值</summary><table><tr><th>选择路径 / 保留方式</th><th>关键 S</th><th>随机 S 均值</th><th>差值</th><th>n</th></tr>']
    for (sid0,operation,role),target in sorted(trials.items()):
        if role!='key' or operation not in ('global_keep','conditional_keep') or not target: continue
        randoms=trials[(sid0,operation,'random')];mean=float(np.mean([r['S'] for r in randoms])) if randoms else None;s=target[0]['S']
        lines.append(f'<tr><td>{esc(sid0)} / {operation}</td><td>{fmt(s)}</td><td>{fmt(mean)}</td><td>{fmt(s-mean if mean is not None else None)}</td><td>{len(randoms)}</td></tr>')
    lines+=['</table></details></section><section><h2>素材核验状态</h2>',
            f'<p class="note">词时候选 {row["ctc_words"]} 个，人工确认 {row["ctc_human_verified"]} 个。官方 A/V 行来源仍 unresolved。算法接受、可播放和边界正确均不等于人工听审通过。</p>',
            f'<p><a href="../assets/{sid}_review.html">逐词候选播放与复核页面</a> · <a href="{sid}.json">完整案例 JSON</a></p>',
            '<p>参考帧来自真实视频解码 PTS，仅作素材上下文；不能据此声称对应模型使用的视觉行。13 号即使原视频可见人物，模型的官方视觉输入仍无观测。</p></section></main></html>']
    return ''.join(lines)

def write_special_outputs(rows, output_dir):
    output=Path(output_dir); cases=output/'cases';cases.mkdir(parents=True,exist_ok=True)
    with (output/'Q3_attachment4_predictions.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=FIELDS);writer.writeheader()
        for row in rows:
            sid=row['sample_id'];write_json(cases/f'{sid}.json',row);(cases/f'{sid}.html').write_text(_render(row),encoding='utf-8')
            vals={k:row.get(k) for k in FIELDS}
            for k in ('primary_modalities','head_top_disagreement','evidence_T','evidence_A','evidence_V'):vals[k]=_json(vals[k])
            vals.update(case_json=f'cases/{sid}.json',case_html=f'cases/{sid}.html');writer.writerow(vals)
    cards=''.join(f'<tr><td><a href="cases/{esc(r["sample_id"])}.html">{esc(r["sample_id"])}</a></td><td>{esc(r.get("predicted_class_name","未完成"))}</td><td>{fmt(r.get("sentiment_score"))}</td><td>{esc(r.get("primary_modalities",[]))}</td><td>{esc(r.get("selected_text",""))}</td><td>{esc(r["mapping_status"])}</td></tr>' for r in rows)
    (output/'index.html').write_text(f'<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Q3 全部20条专项解释</title><style>{STYLE}</style><main><h1>Q3 · 20 条专项解释</h1><section><p>附件4无标签；预测与解释不是正确性证明。所有候选、失败与素材缺口均保留。</p><table><tr><th>样本</th><th>预测</th><th>强度</th><th>主要模态</th><th>实际遮蔽文本</th><th>素材状态</th></tr>{cards}</table></section></main></html>',encoding='utf-8')
