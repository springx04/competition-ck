"""Independent validation of saved interventions, geometry and final deliverables."""
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import numpy as np

from .data import load_samples
from .grouping import Unit, singleton_units
from .results import write_json
from .workflow import read_rows, run_dir


def canonical_intervals(units, atoms):
    """Rebuild spans from complete units; reject partial or outside atoms."""
    atoms = set(atoms)
    domain = {a for unit in units for a in unit.atoms}
    if not atoms <= domain:
        raise ValueError('outside observed unit domain')
    intervals, opened = [], False
    for unit in units:
        picked = atoms.intersection(unit.atoms)
        if picked and picked != set(unit.atoms):
            raise ValueError('partial indivisible unit')
        if picked:
            if opened:
                intervals[-1][1] = unit.hi
            else:
                intervals.append([unit.lo, unit.hi])
            opened = True
        else:
            opened = False
    return intervals


def sample_checks(directory, eps_prob):
    issues, counts = [], Counter()
    def check(condition, detail):
        if not condition:
            issues.append(detail)
    required = ['original.json','local.json','shapley.json','modality.json',
                'selections.jsonl','controls.jsonl','control_spaces.jsonl',
                'experiments.jsonl','groups.jsonl','iteration_status.json']
    missing = [name for name in required if not (directory/name).exists()]
    if missing:
        return ['missing: '+','.join(missing)], counts
    original = json.loads((directory/'original.json').read_text(encoding='utf-8'))
    check(json.loads((directory/'iteration_status.json').read_text(encoding='utf-8')).get('status')=='complete','iteration not complete')
    observed = np.asarray(original['U0'], bool)
    domain = {3*t+m for t,m in zip(*np.where(observed))}
    selections = {s['selection_id']:s for s in read_rows(directory/'selections.jsonl')}
    unit_paths = {(m,'point'): singleton_units(observed,m) for m in 'TAV'}
    unit_paths[('J','joint')] = [Unit(f'J.{t}',tuple(3*t+m for m in range(3) if observed[t,m]),t,t+1,int(observed[t].sum())) for t in range(50) if observed[t].any()]
    for group in read_rows(directory/'groups.jsonl'):
        unit_paths.setdefault((group['path'],group['granularity']),[]).append(Unit(group['unit_id'],tuple(group['delete_j']),group['lo'],group['hi'],group['cost']))
    random_sets = defaultdict(dict)
    for s in selections.values():
        atoms = s['delete_j']; sid = s['selection_id']
        check(len(set(atoms))==len(atoms) and set(atoms)<=domain, f'{sid}: invalid target atoms')
        check(len(atoms)==s['actual_cost']<=s['target_budget'],f'{sid}: target cost')
        if s['status']!='complete':
            counts['not_applicable_selections']+=1
            continue
        counts['complete_selections']+=1
        try:
            spans = canonical_intervals(unit_paths[(s['path'],s['granularity'])], atoms)
            check(spans==s['intervals'] and len(spans)<=3, f'{sid}: target geometry')
        except ValueError as exc:
            issues.append(f'{sid}: {exc}')
    for c in read_rows(directory/'controls.jsonl'):
        sid=c['selection_id']; s=selections[sid]; atoms=c['delete_j']
        counts['controls']+=1
        if s['granularity']=='word': counts['word_controls']+=1
        try:
            spans=canonical_intervals(unit_paths[(s['path'],s['granularity'])],atoms)
            check(sorted(b-a for a,b in spans)==sorted(b-a for a,b in s['intervals']),f'{sid}: control geometry')
        except ValueError as exc:
            issues.append(f'{sid}: control {exc}')
        costs=[sum(a%3==m for a in atoms) for m in range(3)]
        check(costs==s['counts_by_modality'] and len(set(atoms))==len(atoms),f'{sid}: control modality cost')
        check(tuple(atoms)!=tuple(s['delete_j']),f'{sid}: target included as random')
        random_sets[sid][c['control_index']]=tuple(atoms)
    for sid, trials in random_sets.items():
        check(len(set(trials.values()))==len(trials),f'{sid}: duplicate random mask')
    for space in read_rows(directory/'control_spaces.jsonl'):
        sid=space['selection_id']
        check(len(random_sets[sid])==space['n_sampled']==min(50,space['feasible_count']),f'{sid}: control count')
    experiments=read_rows(directory/'experiments.jsonl')
    counts['experiments']=len(experiments)
    seen=set()
    for e in experiments:
        sid=e['selection_id']; role=e['control_role']; op=e['operation']; idx=e['control_index']
        key=(sid,role,op,idx)
        check(key not in seen,f'{sid}: duplicate experiment'); seen.add(key)
        if sid=='key.U.p20.union':
            parts=[s for s in selections.values() if s['selection_id'] in [f'key.{m}.p20.point' for m in 'TAV'] and s['status']=='complete']
            picked=set().union(*(set(s['delete_j']) if role=='key' else set(random_sets[s['selection_id']][idx]) for s in parts))
            expected=domain-picked
        else:
            s=selections[sid]
            picked=set(random_sets[sid][idx] if role=='random' else s['delete_j'])
            local_domain=domain if s['path']=='J' else {a for a in domain if a%3=='TAV'.index(s['path'])}
            expected=picked if op=='delete' else (domain if op=='global_keep' else local_domain)-picked
        check(set(e['delete_j'])==expected,f'{sid}/{op}: intervention mask')
        p=np.asarray(e['probs'],float); score=e['score']; cls=original['c_star']
        dc=original['probs'][cls]-p[cls]; ds=original['score']-score
        check(np.isfinite(p).all() and np.isfinite(score) and abs(p.sum()-1)<1e-5,f'{sid}: prediction numeric')
        check(int(p.argmax())==e['predicted_class'],f'{sid}: predicted class')
        check(abs(e['D']-(abs(dc)/2+abs(ds)/12))<1e-10 and abs(e['delta_class']-dc)<1e-10,f'{sid}: saved difference')
        if op!='delete': check(abs(e['S']-(1-e['D']))<1e-10,f'{sid}: keep score')
        if role in ('support','suppress'):
            retained=dc*(1 if role=='support' else -1)>eps_prob
            check(e['direction_retained']==retained,f'{sid}: direction verdict')
            counts['direction_candidates']+=1
            counts['direction_failed']+=not retained
    for s in selections.values():
        if s['status']!='complete': continue
        sid=s['selection_id']; role=s['experiment_kind']
        operations=['delete']+(['global_keep','conditional_keep'] if role=='key' and s['budget_pct']==20 else [])
        for op in operations:
            check((sid,role,op,None) in seen,f'{sid}/{op}: missing target experiment')
            for idx in random_sets[sid]: check((sid,'random',op,idx) in seen,f'{sid}/{op}: missing random experiment')
    local=json.loads((directory/'local.json').read_text(encoding='utf-8'))
    for name in ('D','delta_class','delta_score_raw'):
        values=local[name]
        check(len(values)==50 and all(len(r)==3 for r in values),f'local/{name}: shape')
        check(all((values[t][m] is not None)==bool(observed[t,m]) for t in range(50) for m in range(3)),f'local/{name}: incomplete scan')
    check('joint' in local and all(len(local['joint'].get(name,[]))==50 and all((local['joint'][name][t] is not None)==bool(observed[t].any()) for t in range(50)) for name in ('D','delta_class','delta_score_raw')),'joint local scan incomplete')
    shapley=json.loads((directory/'shapley.json').read_text(encoding='utf-8'))
    check(len(shapley['subsets'])==8 and max(map(abs,shapley['residual']))<1e-10,'Shapley completeness/efficiency')
    check(observed.sum(0).tolist()==original['n_observed'],'observed count')
    return issues,counts


def validate(config,stage):
    root=Path(config['project']['root']); result={'stage':stage,'implementation_version':2,'issues':[]}
    numeric=json.loads((root/'reports/numeric_tolerance_v2.json').read_text(encoding='utf-8')) if stage=='final' else {}
    for split,expected in [('valid',728),('special',20)]:
        samples=load_samples(config,split); counts=Counter()
        if len(samples)!=expected or any(s.error for s in samples): result['issues'].append(f'{split}: invalid input count or input failure')
        for sample in samples if stage=='final' else []:
            directory=run_dir(config,split)/'samples'/f'{sample.sample_index:04d}'
            issues,nums=sample_checks(directory,numeric['eps_prob']);counts.update(nums)
            if not issues:counts['complete_samples']+=1
            result['issues'].extend(f'{split}/{sample.sample_id}: {issue}' for issue in issues)
            if split=='special' and not (directory/'mapping.json').exists(): result['issues'].append(f'special/{sample.sample_id}: missing mapping')
        result[split]={'expected':expected,'loaded':len(samples),**counts}
    if stage=='final':
        output=root/'outputs_v2'; path=output/'Q3_attachment4_predictions.csv'
        if not path.exists():result['issues'].append('missing final CSV')
        else:
            with path.open(encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
            if [r['sample_id'] for r in rows]!=[f'{i:02d}' for i in range(1,21)] or any(r['status']!='complete' for r in rows):result['issues'].append('CSV incomplete or incorrect ID order')
        for i in range(1,21):
            for ext in ('html','json'):
                if not (output/'cases'/f'{i:02d}.{ext}').exists():result['issues'].append(f'missing case {i:02d}.{ext}')
        if not (run_dir(config,'valid')/'summaries/faithfulness.json').exists():result['issues'].append('missing valid summary')
    result['status']='complete' if not result['issues'] else 'failed'
    result['material_status']='partial_until_word_review_and_AV_provenance'
    write_json(root/f'reports/validation_v2_{stage}.json',result)
    print(json.dumps(result,ensure_ascii=False))
    return int(bool(result['issues']))
