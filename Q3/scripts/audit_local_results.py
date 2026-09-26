"""Read-only audit of saved Q3 cases; writes only the requested audit JSON.

Run from the repository root with PYTHONPATH=Q3/src. This does not approve
word timings, edit existing results, or run the full validation experiment.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np
import torch
from transformers import BertTokenizerFast

from q3.config import load_config
from q3.data import load_samples
from q3.grouping import Unit, text_word_units
from q3.interventions import apply_delete
from q3.predictor import Predictor
from q3.selection import select_units


def audit(config_path, reproduce):
    config = load_config(config_path)
    root = Path(config['project']['root'])
    samples = load_samples(config, 'special')
    tokenizer = BertTokenizerFast.from_pretrained(
        Path(config['project']['bundle_dir']) / 'assets/text_encoder', local_files_only=True)
    cases = [json.loads((root / 'outputs' / 'cases' / f'{i:02d}.json').read_text(encoding='utf-8'))
             for i in range(1, 21)]
    result = {'scope': 'local special cases; no auditory review; no full valid rerun',
              'case_count': len(cases), 'cases': [], 'direction_reversals': [],
              'head_disagreements': [], 'random_geometry_mismatches': [],
              'word_random_partial_words': [], 'summary_mismatches': []}
    totals = Counter()
    gains = defaultdict(list)
    residuals = []
    for sample, case in zip(samples, cases):
        sid = sample.sample_id
        original = json.loads((root / 'runs/special_main/samples' / f'{sample.sample_index:04d}' / 'original.json').read_text())
        observed = np.asarray(original['U0'], bool)
        encoded = tokenizer(sample.raw_text, add_special_tokens=True, truncation=True,
                            max_length=50, padding='max_length', return_offsets_mapping=True)
        matches = all(list(encoded[k]) == sample.raw[v].tolist() for k, v in
                      [('input_ids', 'input_ids'), ('attention_mask', 'stored_attention'), ('token_type_ids', 'token_type_ids')])
        totals['tokenizer_triplet_matches'] += matches
        groups = [set(u.atoms) for u in text_word_units(sample.raw_text, encoded['offset_mapping'], observed)]
        selections = {s['selection_id']: s for s in case['selections']}
        experiments = defaultdict(list)
        for row in case['experiments']:
            experiments[row['selection_id']].append(row)
        for key, rows in experiments.items():
            selection = selections[key]
            randoms = [r for r in rows if r['control_role'] == 'random']
            targets = [r for r in rows if r['operation'] == 'delete' and r['control_role'] != 'random']
            if key.startswith('key.') and randoms and targets:
                gains[key].append(targets[0]['D'] - np.mean([r['D'] for r in randoms]))
            if targets and key.startswith(('support.', 'suppress.')):
                totals['direction_candidates'] += 1
                delta = targets[0]['delta_class']
                if (key.startswith('support.') and delta <= 0) or (key.startswith('suppress.') and delta >= 0):
                    result['direction_reversals'].append({'sample_id': sid, 'selection_id': key, 'delta_class': delta})
            for row in randoms:
                totals['random_sets'] += 1
                positions = sorted({a // 3 for a in row['delete_j']})
                runs = []
                for t in positions:
                    if runs and t == runs[-1][-1] + 1:
                        runs[-1].append(t)
                    else:
                        runs.append([t])
                actual = sorted(map(len, runs))
                expected = sorted(b-a for a, b in selection['intervals'])
                if actual != expected:
                    result['random_geometry_mismatches'].append([sid, key, row['control_index'], actual, expected])
                if key == 'key.T.p20.word':
                    totals['word_random_sets'] += 1
                    atoms = set(row['delete_j'])
                    if any(atoms & group and not group <= atoms for group in groups):
                        result['word_random_partial_words'].append([sid, row['control_index']])
        subsets = case['shapley']['subsets']
        p, s = np.asarray([r['probs'] for r in subsets]), np.asarray([r['score'] for r in subsets])
        cls = case['predicted_class_id']
        dc = np.asarray([p[7, cls] - p[7 ^ (1 << m), cls] for m in range(3)])
        ds = np.asarray([s[7] - s[7 ^ (1 << m)] for m in range(3)])
        if abs(dc).argmax() != abs(ds).argmax():
            result['head_disagreements'].append({'sample_id': sid, 'class_top': 'TAV'[abs(dc).argmax()],
                                                 'score_top': 'TAV'[abs(ds).argmax()],
                                                 'delta_class': dc.tolist(), 'delta_score_raw': ds.tolist()})
        residuals.extend(case['shapley']['residual'])
        selected = selections['key.T.p20.point']
        evidence = []
        for lo, hi in selected['intervals']:
            evidence.append({'official_interval_half_open': [lo, hi],
                             'tokens': [tokenizer.convert_ids_to_tokens(int(sample.raw['input_ids'][t])) for t in range(lo, hi)],
                             'raw_character_span': [encoded['offset_mapping'][lo][0], encoded['offset_mapping'][hi-1][1]],
                             'text': sample.raw_text[encoded['offset_mapping'][lo][0]:encoded['offset_mapping'][hi-1][1]]})
        result['cases'].append({'sample_id': sid, 'tokenizer_triplet_matches': matches,
                                'primary_modalities': json.loads(case['primary_modalities']),
                                'raw_text': sample.raw_text, 'text_main_evidence': evidence,
                                'ctc_review_status': dict(Counter(w['review_status'] for w in case['ctc_words'])),
                                'ctc_algorithm_status': dict(Counter(w['algorithm_status'] for w in case['ctc_words']))})
    summary = json.loads((root / 'runs/special_main/summaries/faithfulness.json').read_text())
    for key, values in gains.items():
        if key not in summary or abs(float(np.mean(values)) - summary[key]['mean_G']) > 1e-10:
            result['summary_mismatches'].append(key)
    result['counts'] = dict(totals)
    result['max_shapley_residual'] = max(map(abs, residuals))
    result['mirrors_equal'] = {name: (root / name).read_bytes() == (root.parent / name).read_bytes()
                              for name in ['reports/Q3_实验报告.md', 'reports/Q3_素材定位缺口.md',
                                           'outputs/Q3_attachment4_predictions.csv', 'delivery/q3_submission.zip',
                                           'runs/valid_main/summaries/faithfulness.json']}
    toy = [Unit('a', (0, 3), 0, 2, 2, 10, 'T'), Unit('b', (9, 12, 15), 3, 6, 3, 1, 'T')]
    selection = select_units(toy, 3, 3, 'max', False)
    result['max_reachable_cost_counterexample'] = {'budget': 3, 'reachable_cost': 3, 'selected_cost': selection.actual_cost}
    if reproduce:
        torch.set_num_threads(4)
        predictor = Predictor(config['project']['bundle_dir'], 'cpu')
        raw = {k: torch.stack([sample.raw[k] for sample in samples]) for k in samples[0].raw}
        prediction = predictor.predict(raw)
        saved_probs = np.asarray([[c[f'p_{name}'] for name in ['negative', 'neutral', 'positive']] for c in cases])
        saved_scores = np.asarray([c['sentiment_score'] for c in cases])
        result['prediction_reproduction'] = {'device': 'cpu', 'batch_size': 20,
            'class_agreement': int((saved_probs.argmax(1) == prediction.probs.argmax(1)).sum()),
            'max_probability_difference': float(abs(saved_probs - prediction.probs).max()),
            'max_score_difference': float(abs(saved_scores - prediction.score).max())}
        for issue in result['direction_reversals']:
            i = int(issue['sample_id']) - 1
            selection = next(s for s in cases[i]['selections'] if s['selection_id'] == issue['selection_id'])
            mask = np.zeros((50, 3), bool)
            for atom in selection['delete_j']:
                mask[atom // 3, atom % 3] = True
            altered = apply_delete(samples[i].raw, mask, prediction.U[i])
            perturbed = predictor.predict({k: v.unsqueeze(0) for k, v in altered.items()})
            cls = cases[i]['predicted_class_id']
            issue['reproduced_delta_class'] = float(prediction.probs[i, cls] - perturbed.probs[0, cls])
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='Q3/configs/q3.yaml')
    parser.add_argument('--output', default='Q3/reports/review_20260926/local_audit.json')
    parser.add_argument('--reproduce', action='store_true')
    args = parser.parse_args()
    result = audit(args.config, args.reproduce)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, separators=(',', ':')) + '\n', encoding='utf-8')
    print(json.dumps({'counts': result['counts'], 'direction_reversals': result['direction_reversals'],
                      'geometry_mismatches': len(result['random_geometry_mismatches']),
                      'word_partial_controls': len(result['word_random_partial_words']),
                      'prediction_reproduction': result.get('prediction_reproduction')}, ensure_ascii=False, indent=2))
