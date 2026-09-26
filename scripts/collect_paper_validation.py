"""Reload the existing delivery model for validation diagnostics, without training."""
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'Q2/delivery/q2_final/src'))
# Some saved modules use PEP 604 annotations without the future import.
# Postpone annotations only on Python 3.9; model source and tensors are unchanged.
if sys.version_info < (3,10):
    import __future__
    import importlib.abc
    import importlib.machinery
    class AnnotationLoader(importlib.machinery.SourceFileLoader):
        def source_to_code(self,data,path,*,_optimize=-1):
            return compile(data,path,'exec',flags=__future__.annotations.compiler_flag,dont_inherit=True,optimize=_optimize)
        def get_code(self,fullname):
            return self.source_to_code(self.get_data(self.path),self.path)
    class AnnotationFinder(importlib.abc.MetaPathFinder):
        def find_spec(self,fullname,path=None,target=None):
            if fullname=='q2' or fullname.startswith('q2.'):
                spec=importlib.machinery.PathFinder.find_spec(fullname,path)
                if spec and spec.origin and spec.origin.endswith('.py'):
                    spec.loader=AnnotationLoader(fullname,spec.origin)
                return spec
    sys.meta_path.insert(0,AnnotationFinder())
from q2.data import load_pickle, unpack_record
from q2.ensemble import load_ensemble
from q2.metrics import compute_metrics

def main():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    device='cuda:0' if torch.cuda.is_available() else 'cpu'
    record=unpack_record(load_pickle(ROOT/'E题数据/附件2-数据集特征文件/aligned_50.pkl')['valid'],'attachment2')
    bundle=load_ensemble(ROOT/'Q2/delivery/q2_final',device)
    logits=[]; scores=[]
    for start in range(0,len(record['input_ids']),64):
        batch={key:torch.from_numpy(np.ascontiguousarray(record[key][start:start+64])) for key in ('input_ids','stored_attention','token_type_ids','audio','vision')}
        out=bundle.predict_raw(batch,return_details=False)
        logits.append(out.logits.detach().cpu().numpy()); scores.append(out.score.detach().cpu().numpy())
    logits=np.concatenate(logits); scores=np.concatenate(scores)
    p=torch.tensor(logits).softmax(-1).numpy()
    frame=pd.DataFrame(dict(sample_id=record['id'],true_class=record['class_id'],pred_class=logits.argmax(1),true_score=record['score'],pred_score=scores,p_neg=p[:,0],p_neu=p[:,1],p_pos=p[:,2]))
    frame['audio_observed']=np.any(record['audio']!=0,axis=-1).sum(1)
    frame['vision_observed']=np.any(record['vision']!=0,axis=-1).sum(1)
    frame['text_observed']=((record['stored_attention']>0)&~np.isin(record['input_ids'],[0,101,102,103])).sum(1)
    outdir=ROOT/'doc/paper_figures/data'; outdir.mkdir(parents=True,exist_ok=True)
    frame.to_csv(outdir/'Q2_valid_reloaded_predictions.csv',index=False,encoding='utf-8-sig')
    metrics=compute_metrics(record['class_id'],logits,record['score'],scores)
    original=json.loads((ROOT/'Q3/runs/valid_v2/summaries/prediction_metrics.json').read_text('utf-8'))
    differences={k:float(metrics[k])-original[k] for k in ['accuracy','macro_f1','mae','pearson']}
    metadata=dict(source='Q2/delivery/q2_final',split='valid',n=len(frame),device=device,torch=torch.__version__,python=sys.version,postponed_annotations=sys.version_info<(3,10),metrics=metrics,difference_from_saved_q3_valid=differences,note='Independent local forward pass; original experimental reports remain authoritative. No retraining or checkpoint selection. Python 3.9 postpones q2 type annotations in memory only; computation source is unchanged.')
    (outdir/'Q2_valid_reloaded_diagnostics.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n','utf-8')
    print(json.dumps(differences,indent=2),flush=True)

if __name__=='__main__': main()
