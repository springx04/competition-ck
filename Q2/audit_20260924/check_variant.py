import torch
from q2.model.network import Student
m=Student('late_balanced',[.285,.223,.492],0.0)
print(m.variant,m.late,m.use_msd,sum(p.numel() for p in m.parameters()))
