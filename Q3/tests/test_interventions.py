import numpy as np
from q3.interventions import apply_delete
def test_only_selected_fields_change():
    raw={'input_ids':np.arange(50), 'stored_attention':np.ones(50,dtype=np.int64), 'token_type_ids':np.zeros(50,dtype=np.int64), 'audio':np.ones((50,74)), 'vision':np.ones((50,35))}
    u=np.ones((50,3),bool);m=np.zeros((50,3),bool);m[2,0]=True;m[4,1]=True
    out=apply_delete(raw,m,u);assert out['input_ids'][2]==103;assert np.all(out['audio'][4]==0);assert out['input_ids'][4]==raw['input_ids'][4];assert np.all(out['vision']==raw['vision'])
