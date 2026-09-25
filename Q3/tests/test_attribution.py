import numpy as np
from q3.attribution import shapley
def test_shapley_efficiency():
    probs=np.array([[.2,.5,.3]]*8,float); scores=np.arange(8,dtype=float);phi,res=shapley(probs,scores,1);assert np.allclose(res,0);assert np.allclose(phi.sum(0),[0,7/6])
