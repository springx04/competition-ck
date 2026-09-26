import torch
from torch import nn
from torch.nn import functional as F


class ContentFusion(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.content_score = nn.Sequential(nn.Linear(dim * 2 + 3 + 5, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, F_original, F_hat, U, B_comp, e_hat, q, modality_embedding, use_reliability):
        source = U.long() + 2 * B_comp.long()
        available = source > 0
        value = torch.where(U[..., None], F_original, torch.where(B_comp[..., None], F_hat, torch.zeros_like(F_hat)))
        batch, length, _, dim = value.shape
        embedding = modality_embedding[None, None].expand(batch, length, -1, -1)
        features = torch.cat((value, embedding, F.one_hot(source, num_classes=3).float(), q), dim=-1)
        scores = self.content_score(features).squeeze(-1)
        if use_reliability and e_hat is not None:
            scores = scores - torch.log1p(e_hat.detach()) * B_comp
        valid_position = available.any(dim=-1)
        alpha = torch.zeros_like(scores)
        if valid_position.any():
            candidate = scores[valid_position].masked_fill(~available[valid_position], float("-inf"))
            alpha[valid_position] = torch.softmax(candidate, dim=-1)
        fused = (alpha[..., None] * value).sum(dim=2)
        denom = valid_position.sum(dim=1).clamp(min=1)
        pooled = fused.sum(dim=1) / denom[:, None]
        return pooled, source, alpha, value
