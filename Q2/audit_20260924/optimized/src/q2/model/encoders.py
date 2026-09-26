import math
import torch
from torch import nn


def sinusoidal_positions(length: int, dim: int, device=None) -> torch.Tensor:
    positions = torch.arange(length, device=device).float()[:, None]
    k = torch.arange(0, dim, 2, device=device).float()
    angles = positions / torch.pow(10000.0, k / dim)
    result = torch.zeros(length, dim, device=device)
    result[:, 0::2] = torch.sin(angles)
    result[:, 1::2] = torch.cos(angles)
    return result


def masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int = 1):
    counts = mask.sum(dim=dim, keepdim=True)
    pooled = (x * mask.unsqueeze(-1)).sum(dim=dim) / counts.clamp(min=1)
    return pooled, counts.squeeze(dim) > 0


class ModalityEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.projection = nn.Linear(input_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim, eps=1e-5)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, dim_feedforward=256,
                                       dropout=.1, activation="gelu", batch_first=True,
                                       norm_first=True, layer_norm_eps=1e-5)
            for _ in range(2)
        ])
        self.output_norm = nn.LayerNorm(hidden_dim, eps=1e-5)

    def forward(self, x, mask, position, modality_embedding):
        out = (self.input_norm(self.projection(x)) + position[None] + modality_embedding) * mask[..., None]
        active = mask.any(dim=1)
        if not active.any():
            return torch.zeros_like(out)
        active_out = out[active]
        active_mask = mask[active]
        for layer in self.layers:
            active_out = layer(active_out, src_key_padding_mask=~active_mask)
            active_out = active_out * active_mask[..., None]
        out = torch.zeros_like(out)
        out[active] = self.output_norm(active_out) * active_mask[..., None]
        return out
