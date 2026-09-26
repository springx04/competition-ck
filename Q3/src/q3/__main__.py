import argparse
import json
from pathlib import Path
import numpy as np

from .config import load_config
from .data import load_samples
from .predictor import Predictor
from .results import write_json
from .workflow import process, seed_run, run_dir, read_rows


def check_predictor(config):
    predictor=Predictor(config['project']['bundle_dir'],config['runtime']['device'])
    samples=load_samples(config,'valid')[:8]+load_samples(config,'special')
    single_batch=dict(logits=0.,probability=0.,score=0.);repeat=dict(single_batch);classes=True
    batch_size=int(config['runtime']['view_batch_size'])
    for sample in samples:
        one=predictor.predict({k:v.unsqueeze(0) for k,v in sample.raw.items()})
        raw_batch={k:v.unsqueeze(0).expand(batch_size,*v.shape).clone() for k,v in sample.raw.items()}
        first,second=predictor.predict(raw_batch),predictor.predict(raw_batch)
        for label,attr in [('logits','logits'),('probability','probs'),('score','score')]:
            a,b,c=getattr(one,attr),getattr(first,attr),getattr(second,attr)
            single_batch[label]=max(single_batch[label],float(abs(a-b).max()))
            repeat[label]=max(repeat[label],float(abs(b-c).max()))
        classes &= bool(np.all(first.probs.argmax(1)==one.probs.argmax(1)[0]) and np.all(first.probs.argmax(1)==second.probs.argmax(1)))
    prob=max(single_batch['probability'],repeat['probability']);score=max(single_batch['score'],repeat['score'])
    result={'probe_count':len(samples),'batch_size':batch_size,'single_batch_difference':single_batch,
            'repeat_difference':repeat,'class_consistent':classes,'eps_prob':max(1e-7,10*prob),'eps_score':max(1e-6,10*score)}
    result['eps_D']=.5*result['eps_prob']+result['eps_score']/12
    write_json(Path(config['project']['root'])/'reports/numeric_tolerance_v2.json',result)
    if len(samples)!=28 or not classes or prob>1e-4 or score>1e-3:
        raise RuntimeError('Predictor numerical check failed; see numeric_tolerance_v2.json')
    print(json.dumps(result))


def main():
    parser=argparse.ArgumentParser(prog='q3');parser.add_argument('--config',required=True);parser.add_argument('--device')
    sub=parser.add_subparsers(dest='command',required=True)
    for command in ('run','prepare','attribute','select','controls','evaluate','summarize','map','report'):
        item=sub.add_parser(command);item.add_argument('--split',choices=['valid','special'],required=True)
        item.add_argument('--resume',action='store_true');item.add_argument('--limit',type=int)
    sub.add_parser('check-predictor');sub.add_parser('export')
    item=sub.add_parser('validate');item.add_argument('--stage',choices=['inputs','final'],required=True)
    args=parser.parse_args();config=load_config(args.config)
    if args.device:config['runtime']['device']=args.device
    if args.command=='check-predictor':check_predictor(config)
    elif args.command=='validate':
        from .validation import validate
        return validate(config,args.stage)
    elif args.command=='prepare':seed_run(config,args.split,load_samples(config,args.split))
    elif args.command in ('run','attribute','select','controls','evaluate'):process(config,args.split,args.resume,args.limit)
    elif args.command=='summarize':
        from .summary import summarize
        summarize(config,args.split)
    elif args.command=='map':
        if args.split != 'special':
            parser.error('map requires --split special')
        from .mapping import build_mapping
        from .workflow import tokenizer_for
        tokenizer = tokenizer_for(config)
        for sample in load_samples(config, 'special'):
            directory = run_dir(config, 'special')/'samples'/f'{sample.sample_index:04d}'
            original = json.loads((directory/'original.json').read_text(encoding='utf-8'))
            mapping = build_mapping(config, sample, original, read_rows(directory/'selections.jsonl'), tokenizer)
            write_json(directory/'mapping.json', mapping)
        print('map special: 20 records refreshed without prediction changes')
    elif args.command=='report':
        from .summary import report
        report(config,args.split)
    elif args.command=='export':
        from .export import export_delivery
        print(export_delivery(config))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
