import sys,time,torch
sys.path.insert(0,'C:/Users/30738/Desktop/sm/中文题目/E题/.audit/Q2/src'); from q2.state import infer_state
B,L=32,50; torch.manual_seed(1); raw={'input_ids':torch.randint(0,30522,(B,L)), 'stored_attention':torch.ones(B,L,dtype=torch.long), 'audio':torch.randn(B,L,74), 'vision':torch.randn(B,L,35)}
for _ in range(20): infer_state(raw)
t=time.perf_counter()
for _ in range(200): infer_state(raw)
print('baseline_ms', (time.perf_counter()-t)*1000/200)
