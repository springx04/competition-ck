import sys, torch
sys.path.insert(0,'Q2/audit_20260924/optimized/src')
from q2.state import infer_state as opt
def ref(raw):
 ids,stored,audio,vision=raw['input_ids'],raw['stored_attention'],raw['audio'],raw['vision']; active=stored==1; boundary=(ids==101)|(ids==102); ts=active&(ids!=0)&~boundary; U=torch.stack((ts&(ids!=103),(audio!=0).any(-1),(vision!=0).any(-1)),-1); J=ts|U[...,1]|U[...,2]; B,L=ids.shape; pos=torch.arange(L); first=torch.where(J,pos[None],L).min(1).values; last=torch.where(J,pos[None],-1).max(1).values; env=(last-first+1).clamp(min=1); holes=J[...,None]&~U; f=torch.zeros(B,L,3,dtype=torch.long); b=torch.zeros_like(f); c=torch.zeros(B,3,dtype=torch.long)
 for t in range(L): c=torch.where(holes[:,t],c+1,0); f[:,t]=c
 c.zero_()
 for t in range(L-1,-1,-1): c=torch.where(holes[:,t],c+1,0); b[:,t]=c
 gap=torch.where(holes,(f+b-1).float()/env[:,None,None],0); return U,J,gap
for seed in range(20):
 torch.manual_seed(seed); B,L=3,50; raw={'input_ids':torch.randint(0,110,(B,L)), 'stored_attention':torch.randint(0,2,(B,L)), 'audio':torch.randn(B,L,4), 'vision':torch.randn(B,L,3)}; raw['audio'][torch.rand(B,L,4)<.7]=0; raw['vision'][torch.rand(B,L,3)<.7]=0
 a=opt(raw); u,j,g=ref(raw); assert torch.equal(a.U,u); assert torch.equal(a.J,j); assert torch.equal(a.q[...,1],g), (seed,(a.q[...,1]-g).abs().max())
print('vectorized state exact: 20 random cases')
