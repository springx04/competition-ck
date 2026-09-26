"""Q3 iteration 2: matched interventions with complete, reusable view records."""
from collections import Counter, defaultdict
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from transformers import BertTokenizerFast

from .attribution import view_result, shapley
from .controls import MatchedControlSpace
from .data import load_samples
from .grouping import Unit, singleton_units, block3_units, text_word_units
from .interventions import apply_delete, mask_from_atoms
from .predictor import Predictor, Prediction
from .results import write_json
from .selection import budget_count, select_units
from .statistics import cluster_bootstrap


def run_dir(config, split):
    return Path(config['output'][f'{split}_run'])


def read_rows(path):
    path = Path(path)
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line] if path.exists() else []


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n' for row in rows), encoding='utf-8')


def tokenizer_for(config):
    return BertTokenizerFast.from_pretrained(Path(config['project']['bundle_dir'])/'assets/text_encoder', local_files_only=True)


def seed_run(config, split, samples):
    run = run_dir(config, split)
    source = Path(config['project']['root']) / 'runs' / f'{split}_main'
    if run.resolve() == source.resolve():
        raise ValueError('Iteration 2 requires a new run directory; preserve the original run.')
    for sample in samples:
        dest = run/'samples'/f'{sample.sample_index:04d}'
        dest.mkdir(parents=True, exist_ok=True)
        for name in ('original.json', 'views.jsonl'):
            old = source/'samples'/dest.name/name
            if old.exists() and not (dest/name).exists():
                shutil.copy2(old, dest/name)
    if split == 'special' and (source/'alignment').exists() and not (run/'alignment').exists():
        shutil.copytree(source/'alignment', run/'alignment')
    write_rows(run/'manifest.jsonl', [{'sample_index': s.sample_index, 'sample_id': s.sample_id,
               'video_id': s.video_id, 'raw_text': s.raw_text, 'input_status': 'failed' if s.error else 'ready'} for s in samples])
    write_json(run/'run.json', {'implementation_version': 2, 'split': split, 'expected_samples': len(samples),
               'source_run': str(source), 'evaluation_role': 'development' if split == 'valid' else 'unlabeled_cases'})


class ViewEvaluator:
    def __init__(self, sample, predictor, directory, batch_size):
        self.sample, self.predictor, self.directory = sample, predictor, directory
        self.batch_size = batch_size
        path = directory/'original.json'
        if path.exists():
            row = json.loads(path.read_text())
            self.original = Prediction(np.asarray([row['logits']]), np.asarray([row['probs']]), np.asarray([row['score']]),
                                       np.asarray([row['U0']], bool), np.asarray([row['J0']], bool))
        else:
            self.original = predictor.predict({k: v.unsqueeze(0) for k, v in sample.raw.items()})
        self.observed = self.original.U[0]
        self.atoms = tuple(int(3*t+m) for t, m in zip(*np.where(self.observed)))
        self.c_star = int(self.original.probs[0].argmax())
        self.cache = {}
        for row in read_rows(directory/'views.jsonl'):
            key = tuple(row['delete_j'])
            pred = Prediction(np.asarray([row['logits']]), np.asarray([row['probs']]), np.asarray([row['score']]),
                              self.original.U, self.original.J)
            self.cache[key] = self._row(key, pred)
        self.cache[()] = self._row((), self.original)

    def _row(self, key, prediction):
        view = view_result(key, self.original, prediction, self.c_star)
        row = asdict(view)
        for name in ('logits', 'probs'):
            row[name] = row[name].tolist()
        row['delete_j'] = list(key)
        return row

    def evaluate(self, atom_sets):
        missing = sorted({tuple(sorted(atoms)) for atoms in atom_sets} - self.cache.keys())
        offset = 0
        while offset < len(missing):
            keys = missing[offset:offset+self.batch_size]
            changed = [apply_delete(self.sample.raw, mask_from_atoms(key), self.observed) for key in keys]
            batch = {k: torch.stack([item[k] for item in changed]) for k in self.sample.raw}
            try:
                pred = self.predictor.predict(batch)
            except torch.cuda.OutOfMemoryError:
                if self.batch_size <= 16:
                    raise
                self.batch_size = max(16, self.batch_size//2)
                del batch, changed
                torch.cuda.empty_cache()
                continue
            for i, key in enumerate(keys):
                one = Prediction(pred.logits[i:i+1], pred.probs[i:i+1], pred.score[i:i+1], pred.U[i:i+1], pred.J[i:i+1])
                self.cache[key] = self._row(key, one)
            offset += len(keys)

    def save(self):
        o = self.original
        write_json(self.directory/'original.json', {'logits': o.logits[0].tolist(), 'probs': o.probs[0].tolist(),
                   'score': float(o.score[0]), 'c_star': self.c_star, 'U0': self.observed.tolist(),
                   'J0': o.J[0].tolist(), 'n_observed': self.observed.sum(0).tolist(), 'prediction_status': 'complete'})
        write_rows(self.directory/'views.jsonl', [self.cache[k] for k in sorted(self.cache)])


def build_selections(evaluator, sample, tokenizer, config):
    u = evaluator.observed
    paths = {}
    for m, name in enumerate('TAV'):
        paths[(name, 'point')] = singleton_units(u, name)
        paths[(name, 'block3')] = block3_units(u, name)
    encoded = tokenizer(sample.raw_text or '', truncation=True, max_length=50, padding='max_length', return_offsets_mapping=True)
    text_ok = all(encoded[k] == sample.raw[v].tolist() for k, v in [('input_ids','input_ids'),
                        ('attention_mask','stored_attention'), ('token_type_ids','token_type_ids')])
    paths[('T','word')] = text_word_units(sample.raw_text, encoded['offset_mapping'], u) if text_ok else []
    paths[('J','joint')] = [Unit(f'J.{t}', tuple(3*t+m for m in range(3) if u[t,m]), t, t+1,
                                   int(u[t].sum()), 0., 'J') for t in range(50) if u[t].any()]
    evaluator.evaluate([unit.atoms for units in paths.values() for unit in units] +
                       [tuple(a for a in evaluator.atoms if a%3 not in {m for m in range(3) if subset&(1<<m)}) for subset in range(8)])
    for key, units in paths.items():
        paths[key] = [Unit(unit.unit_id, unit.atoms, unit.lo, unit.hi, unit.cost,
                           evaluator.cache[unit.atoms]['D'], unit.modality) for unit in units]
    selections, spaces = [], {}
    def add(path, granularity, pct, role='key', fixed_cost=None):
        units = paths[(path, granularity)]
        budget = fixed_cost if fixed_cost is not None else budget_count(sum(unit.cost for unit in units), pct)
        direction = role in ('support', 'suppress')
        ranked = units
        if direction:
            sign = 1 if role == 'support' else -1
            ranked = [Unit(x.unit_id, x.atoms, x.lo, x.hi, x.cost,
                      max(sign*evaluator.cache[x.atoms]['delta_class'], 0.), x.modality) for x in units]
        selection = select_units(ranked, budget, int(config['method']['max_intervals']),
                                 'min' if role == 'low' else 'max', granularity == 'point' and not direction,
                                 fill_budget=not direction)
        sid = f'{role}.{path}.p{pct}.{granularity}' + (f'.cost{fixed_cost}' if fixed_cost is not None else '')
        row = {'selection_id': sid, 'experiment_kind': role, 'path': path, 'granularity': granularity,
               'budget_pct': pct, 'target_budget': budget, 'actual_cost': selection.actual_cost,
               'counts_by_modality': selection.counts_by_modality, 'intervals': selection.intervals,
               'delete_j': list(selection.atoms), 'proxy_value': selection.proxy_value,
               'status': selection.status, 'reason': selection.reason}
        if direction and selection.proxy_value <= config['numeric']['eps_prob']:
            row.update(status='not_applicable', reason='no_directional_proxy')
        if granularity == 'word' and not text_ok:
            row.update(status='not_applicable', reason='tokenizer_mismatch')
        selections.append(row)
        spaces[sid] = units
        return row
    for name in 'TAV':
        for pct in (10,20,30):
            add(name,'point',pct)
            add(name,'point',pct,'low')
        for role in ('support','suppress'):
            add(name,'point',20,role)
        for granularity in (['block3','word'] if name == 'T' else ['block3']):
            row = add(name,granularity,20)
            if row['actual_cost'] and row['actual_cost'] != budget_count(int(u[:, 'TAV'.index(name)].sum()),20):
                sid = f"key.{name}.p20.point.cost{row['actual_cost']}"
                if sid not in spaces:
                    add(name,'point',20,fixed_cost=row['actual_cost'])
                row['cost_matched_point_id'] = sid
            else:
                row['cost_matched_point_id'] = f'key.{name}.p20.point'
    for pct in (10,20,30):
        add('J','joint',pct)
    groups = [{'path': name, 'granularity': grain, 'unit_id': x.unit_id, 'delete_j': list(x.atoms),
               'lo':x.lo,'hi':x.hi,'cost':x.cost,'D':x.proxy} for (name,grain),units in paths.items()
              if grain in ('word','block3') for x in units]
    return selections, spaces, groups


def build_controls(selections, spaces, config, split, index):
    controls, diagnostics = [], []
    for sel in selections:
        if sel['experiment_kind'] == 'low' or sel['status'] != 'complete':
            continue
        space = MatchedControlSpace(spaces[sel['selection_id']], sel['delete_j'], sel['intervals'])
        if space.target_rank is None:
            raise ValueError(f"Target not in its own matched space: {sel['selection_id']}")
        entropy = [int(config['runtime']['seed']), 101, 1 if split == 'valid' else 2, index,
                   *sel['selection_id'].encode('ascii')]
        seed = int(np.random.SeedSequence(entropy).generate_state(1, dtype=np.uint64)[0])
        samples = space.sample(int(config['method']['controls_per_set']), seed)
        diagnostics.append({'selection_id':sel['selection_id'], 'seed':seed, 'sampling':'Floyd without replacement; independently shuffled order', 'feasible_count':space.available,
                            'n_sampled':len(samples), 'status':'complete' if samples else 'no_alternative'})
        for i, atoms in enumerate(samples):
            controls.append({'selection_id':sel['selection_id'], 'control_index':i, 'delete_j':list(atoms),
                             'counts_by_modality':sel['counts_by_modality'], 'interval_lengths':sorted(b-a for a,b in sel['intervals'])})
    return controls, diagnostics


def evaluate_experiments(evaluator, selections, controls, config):
    all_atoms = set(evaluator.atoms)
    rows, jobs = [], []
    selected = {s['selection_id']:s for s in selections if s['status']=='complete'}
    def queue(selection, keep_atoms, role, control_index=None):
        sid, path = selection['selection_id'], selection['path']
        atoms = set(keep_atoms)
        operations = {'delete':atoms}
        if selection['budget_pct']==20 and selection['experiment_kind']=='key':
            domain = all_atoms if path=='J' else {a for a in all_atoms if a%3=='TAV'.index(path)}
            operations.update(global_keep=all_atoms-atoms, conditional_keep=domain-atoms)
        for operation, deleted in operations.items():
            jobs.append(({'selection_id':sid, 'operation':operation, 'control_role':role,
                          'control_index':control_index,'cost':len(atoms), 'status':'complete'},tuple(sorted(deleted))))
    for selection in selected.values():
        queue(selection,selection['delete_j'],selection['experiment_kind'])
    for control in controls:
        queue(selected[control['selection_id']],control['delete_j'],'random',control['control_index'])
    # Per-modality 20% union: the random union uses independently matched controls.
    union = set()
    per_modality = []
    for name in 'TAV':
        sid = f'key.{name}.p20.point'
        if sid in selected:
            union.update(selected[sid]['delete_j'])
            per_modality.append([set(c['delete_j']) for c in controls if c['selection_id']==sid])
    for role, kept, index in [('key',union,None)] + [('random',set().union(*(group[i%len(group)] for group in per_modality)),i)
            for i in range(min((len(g) for g in per_modality), default=0)) if all(per_modality)]:
        jobs.append(({'selection_id':'key.U.p20.union','operation':'global_keep','control_role':role,
                      'control_index':index,'cost':len(kept),'status':'complete'},tuple(sorted(all_atoms-kept))))
    evaluator.evaluate([key for _,key in jobs])
    for metadata, key in jobs:
        row = dict(evaluator.cache[key], **metadata)
        row['experiment_id'] = f"{row['selection_id']}.{row['operation']}.{row['control_role']}.{row['control_index']}"
        if row['operation']!='delete':
            row['S']=1-row['D']
        if row['control_role'] in ('support','suppress'):
            signed = row['delta_class'] * (1 if row['control_role']=='support' else -1)
            row['direction_retained'] = bool(signed > config['numeric']['eps_prob'])
        rows.append(row)
    return rows


def process(config, split, resume=False, limit=None):
    samples = load_samples(config,split)
    seed_run(config,split,samples)
    tokenizer = tokenizer_for(config)
    predictor = Predictor(config['project']['bundle_dir'],config['runtime']['device'])
    tolerance = Path(config['project']['root'])/'reports/numeric_tolerance_v2.json'
    config['numeric'] = json.loads(tolerance.read_text()) if tolerance.exists() else {'eps_prob':2e-6,'eps_score':5e-6,'eps_D':1.5e-6}
    total = len(samples if limit is None else samples[:limit])
    for i,sample in enumerate(samples if limit is None else samples[:limit]):
        directory = run_dir(config,split)/'samples'/f'{sample.sample_index:04d}'
        status = directory/'iteration_status.json'
        if resume and status.exists() and json.loads(status.read_text()).get('status')=='complete':
            continue
        if sample.error:
            raise ValueError(f'{sample.sample_id}: {sample.error}')
        started = time.monotonic()
        evaluator = ViewEvaluator(sample,predictor,directory,int(config['runtime']['view_batch_size']))
        selections,spaces,groups = build_selections(evaluator,sample,tokenizer,config)
        controls,diagnostics = build_controls(selections,spaces,config,split,sample.sample_index)
        experiments = evaluate_experiments(evaluator,selections,controls,config)
        evaluator.save()
        write_rows(directory/'selections.jsonl',selections)
        write_rows(directory/'controls.jsonl',controls)
        write_rows(directory/'control_spaces.jsonl',diagnostics)
        write_rows(directory/'groups.jsonl',groups)
        write_rows(directory/'experiments.jsonl',experiments)
        original = json.loads((directory/'original.json').read_text())
        subsets = [evaluator.cache[tuple(a for a in evaluator.atoms if not (bit&(1<<(a%3))))] for bit in range(8)]
        phi,residual = shapley(np.asarray([s['probs'] for s in subsets]),np.asarray([s['score'] for s in subsets]),evaluator.c_star)
        write_json(directory/'shapley.json',{'phi_class':phi[:,0].tolist(),'phi_score_raw':(phi[:,1]*6).tolist(),
                  'residual':residual.tolist(),'subsets':[dict(s,bitmask=i) for i,s in enumerate(subsets)]})
        d=[evaluator.cache[tuple(a for a in evaluator.atoms if a%3==m)]['D'] for m in range(3)]
        write_json(directory/'modality.json',{'D':d,'n_observed':original['n_observed']})
        local = {key:[[evaluator.cache[(3*t+m,)][key] if evaluator.observed[t,m] else None for m in range(3)]
                      for t in range(50)] for key in ('D','delta_class','delta_score_raw')}
        local['observed']=evaluator.observed.tolist()
        local['joint']={key:[evaluator.cache[tuple(3*t+m for m in range(3) if evaluator.observed[t,m])][key] if evaluator.observed[t].any() else None for t in range(50)] for key in ('D','delta_class','delta_score_raw')}
        write_json(directory/'local.json',local)
        if split=='special':
            from .mapping import build_mapping
            mapping=build_mapping(config,sample,original,selections,tokenizer)
            write_json(directory/'mapping.json',mapping)
        write_json(status,{'status':'complete','implementation_version':2,'sample_id':sample.sample_id,
                   'unique_views':len(evaluator.cache),'experiments':len(experiments),'elapsed_s':time.monotonic()-started,
                   'replication_groups_status':'unresolved_no_official_provenance; block3_sensitivity_used'})
        print(f'{split}: {i+1}/{total} {sample.sample_id} views={len(evaluator.cache)} seconds={time.monotonic()-started:.1f}',flush=True)
