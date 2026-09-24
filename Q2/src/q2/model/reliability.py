import torch
from torch import nn
from torch.nn import functional as F


class ErrorEstimator(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4 * dim + 5, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, F_hat, C, F_original, U, q, B_comp, modality_embedding):
        batch, length, _, dim = F_hat.shape
        estimates = []
        for target in range(3):
            self_count = U[:, :, target].sum(dim=1, keepdim=True).clamp(min=1)
            self_context = (F_original[:, :, target] * U[:, :, target, None]).sum(dim=1) / self_count
            others = [m for m in range(3) if m != target]
            other_mask = U[:, :, others]
            cross_count = other_mask.sum(dim=(1, 2)).clamp(min=1)
            cross_context = (C[:, :, others] * other_mask[..., None]).sum(dim=(1, 2)) / cross_count[:, None]
            features = torch.cat((F_hat[:, :, target], self_context[:, None].expand(-1, length, -1),
                cross_context[:, None].expand(-1, length, -1),
                modality_embedding[target][None, None].expand(batch, length, -1), q[:, :, target]), dim=-1)
            estimate = F.softplus(self.net(features.detach()).squeeze(-1))
            estimates.append(estimate * B_comp[:, :, target])
        return torch.stack(estimates, dim=2)
