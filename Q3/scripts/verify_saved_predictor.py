"""Recompute saved interventions on the fixed exported predictor."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from q3.config import load_config
from q3.data import load_samples
from q3.interventions import apply_delete, mask_from_atoms
from q3.predictor import Predictor
from q3.results import write_json
from q3.workflow import read_rows, run_dir


def verify(config):
    root=Path(config['project']['root'])
    numeric=json.loads((root/'reports/numeric_tolerance_v2.json').read_text(encoding='utf-8'))
    predictor=Predictor(config['project']['bundle_dir'],config['runtime']['device'])
    records=[]; failures=[]
    for split,indices in [('valid',[0,364,727]),('special',list(range(20)))]:
        samples=load_samples(config,split)
        for index in indices:
            sample=samples[index]; directory=run_dir(config,split)/'samples'/f'{index:04d}'
            original=json.loads((directory/'original.json').read_text(encoding='utf-8'))
            observed=np.asarray(original['U0'],bool)
            rows=read_rows(directory/'experiments.jsonl'); picked=[dict(original,delete_j=[],predicted_class=original['c_star'])]
            for sid in ('key.T.p20.point','key.A.p20.point','key.V.p20.point','key.J.p20.joint','key.T.p20.word','key.A.p20.block3','suppress.T.p20.point'):
                for operation,role in [('delete','key'),('delete','random'),('global_keep','key'),('conditional_keep','random')]:
                    match=next((r for r in rows if r['selection_id']==sid and r['operation']==operation and r['control_role']==role),None)
                    if match:picked.append(match)
            altered=[apply_delete(sample.raw,mask_from_atoms(r['delete_j']),observed) for r in picked]
            pred=predictor.predict({k:torch.stack([a[k] for a in altered]) for k in sample.raw})
            pd=float(abs(pred.probs-np.asarray([r['probs'] for r in picked])).max())
            sd=float(abs(pred.score-np.asarray([r['score'] for r in picked])).max())
            same=bool(np.all(pred.probs.argmax(1)==[r['predicted_class'] for r in picked]))
            shuffled={k:v.clone() for k,v in sample.raw.items()}
            rng=np.random.default_rng(20260926+index)
            for key in ('audio','vision'):shuffled[key]=shuffled[key][rng.permutation(50)]
            permutation=predictor.predict({k:v.unsqueeze(0) for k,v in shuffled.items()})
            pp=float(abs(permutation.probs[0]-pred.probs[0]).max());ps=float(abs(permutation.score[0]-pred.score[0]))
            okay=pd<=numeric['eps_prob'] and sd<=numeric['eps_score'] and same and pp<=numeric['eps_prob'] and ps<=numeric['eps_score']
            records.append(dict(split=split,sample_id=sample.sample_id,interventions=len(picked),max_probability_difference=pd,max_score_difference=sd,class_consistent=same,av_permutation_probability_difference=pp,av_permutation_score_difference=ps,passed=okay))
            if not okay:failures.append(f'{split}/{sample.sample_id}')
    result=dict(samples=len(records),interventions=sum(r['interventions'] for r in records),records=records,failures=failures,status='pass' if not failures else 'fail',numeric_tolerance=numeric)
    write_json(root/'reports/iteration_v2/predictor_reproduction.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('records','numeric_tolerance')}))
    if failures:raise RuntimeError('Saved predictor reproduction failed')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);verify(load_config(p.parse_args().config))
