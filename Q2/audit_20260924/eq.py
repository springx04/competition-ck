import sys,torch
sys.path.insert(0,'Q2/audit_20260924/optimized/src')
from q2.model.network import Student,ModelInput
from q2.state import infer_state
raw={'input_ids':torch.tensor([[101,100,101,102]+[0]*46]),'stored_attention':torch.tensor([[1,1,1,1]+[0]*46]),'token_type_ids':torch.zeros(1,50,dtype=torch.long),'audio':torch.zeros(1,50,74),'vision':torch.zeros(1,50,35)}; st=infer_state(raw); inp=ModelInput(torch.randn(1,50,256),torch.zeros(1,50,74),torch.zeros(1,50,35),st.U,st.J,st.q); m=Student('full',[.3,.3,.4],0).eval(); torch.manual_seed(2)
with torch.no_grad(): a=m(inp,return_details=True); torch.manual_seed(2); b=m(inp,return_details=True,return_attention=False)
print('same_outputs',torch.equal(a.logits,b.logits),torch.equal(a.score,b.score), 'attention_shapes',a.attention.shape,b.attention)
