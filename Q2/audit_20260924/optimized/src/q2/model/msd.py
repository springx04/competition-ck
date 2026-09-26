import torch
from torch import nn

from ..vendor.ebmc_msd import MSDModule


class Decomposition(nn.Module):
    def __init__(self, hidden_dim=128, use_msd=True, include_aux_heads=True):
        super().__init__()
        self.use_msd = use_msd
        self.modules_by_modality = nn.ModuleList([MSDModule(hidden_dim) for _ in range(3)]) if use_msd else None
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim, eps=1e-5) for _ in range(3)])
        self.aux_class = nn.ModuleList([nn.Linear(hidden_dim, 3) for _ in range(3)]) if use_msd and include_aux_heads else None
        self.aux_score = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(3)]) if use_msd and include_aux_heads else None

    def forward(self, H, U):
        c_list, s_list, f_list = [], [], []
        for m in range(3):
            h = H[:, :, m]
            mask = U[:, :, m, None]
            c, s = self.modules_by_modality[m](h) if self.use_msd else (h / 2, h / 2)
            c, s = c * mask, s * mask
            f = self.norms[m](c + s) * mask
            c_list.append(c)
            s_list.append(s)
            f_list.append(f)
        return torch.stack(c_list, dim=2), torch.stack(s_list, dim=2), torch.stack(f_list, dim=2)
