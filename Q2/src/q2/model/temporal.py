"""Compact audio/vision context from the current observed sequence only."""
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class ObservedGRU(nn.Module):
    def __init__(self, dimension=128):
        super().__init__()
        self.gru = nn.GRU(dimension, dimension // 2, batch_first=True, bidirectional=True)
        self.norm = nn.LayerNorm(dimension)

    def forward(self, features, observed):
        counts = observed.sum(dim=1)
        samples = torch.where(counts > 0)[0]
        if samples.numel() == 0:
            return features * 0
        length = features.shape[1]
        # Stable chronological compaction: observed rows precede masked rows.
        positions = torch.arange(length, device=features.device)[None]
        order = (positions + (~observed[samples]).long() * length).argsort(dim=1)
        index = order[..., None].expand(-1, -1, features.shape[-1])
        compact = features[samples].gather(1, index)
        packed = pack_padded_sequence(compact, counts[samples].cpu(),
                                      batch_first=True, enforce_sorted=False)
        contextual, _ = self.gru(packed)
        contextual, _ = pad_packed_sequence(contextual, batch_first=True, total_length=length)
        aligned = torch.zeros_like(compact).scatter(1, index, contextual)
        context = torch.zeros_like(features).index_copy(0, samples, aligned)
        return self.norm(features + context) * observed[..., None]
