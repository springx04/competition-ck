import torch
from torch import nn
from torch.nn import functional as F

from .encoders import masked_mean


class SharedCompensator(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.query_projection = nn.Linear(dim * 3, dim)
        self.self_source = nn.Linear(dim * 2, dim)
        self.other_source = nn.Linear(dim, dim)
        self.attention = nn.MultiheadAttention(dim, 4, dropout=.1, batch_first=True)
        self.norm1 = nn.LayerNorm(dim, eps=1e-5)
        self.ffn = nn.Sequential(nn.Linear(dim, 256), nn.GELU(), nn.Linear(256, dim))
        self.norm2 = nn.LayerNorm(dim, eps=1e-5)
        self.dropout = nn.Dropout(.1)
        self.outputs = nn.ModuleList([nn.Linear(dim, dim) for _ in range(3)])

    def forward(self, C, S, F_original, U, J, position, modality_embedding,
                return_attention=False, target_order=(0, 1, 2)):
        batch, length, _, dim = C.shape
        hats, masks, attentions = [None] * 3, [None] * 3, [None] * 3
        source_valid = U.permute(0, 2, 1).reshape(batch, 3 * length)
        has_source = source_valid.any(dim=1)
        # Every target reads this immutable, pre-compensation C/S/F snapshot.
        for target in target_order:
            self_context, _ = masked_mean(F_original[:, :, target], U[:, :, target])
            query = self.query_projection(torch.cat((position[None].expand(batch, -1, -1),
                modality_embedding[target][None, None].expand(batch, length, -1),
                self_context[:, None].expand(batch, length, -1)), dim=-1))
            memories = []
            for source in range(3):
                value = (self.self_source(torch.cat((C[:, :, source], S[:, :, source]), dim=-1))
                         if source == target else self.other_source(C[:, :, source]))
                value = value + position[None] + modality_embedding[source]
                memories.append(value * U[:, :, source, None])
            memory = torch.cat(memories, dim=1)
            query_mask = J & ~U[:, :, target] & has_source[:, None]
            result = torch.zeros_like(query)
            weights = None
            if has_source.any():
                index = torch.where(has_source)[0]
                attended, weights = self.attention(query[index], memory[index], memory[index],
                    key_padding_mask=~source_valid[index], need_weights=return_attention,
                    average_attn_weights=True)
                residual = self.norm1(query[index] + self.dropout(attended))
                result[index] = self.norm2(residual + self.dropout(self.ffn(residual)))
            hats[target] = self.outputs[target](result) * query_mask[..., None]
            masks[target] = query_mask
            if return_attention:
                full = torch.zeros(batch, length, length * 3, device=C.device, dtype=C.dtype)
                if weights is not None:
                    full[has_source] = weights
                attentions[target] = full * query_mask[..., None]
        return (torch.stack(hats, dim=2), torch.stack(masks, dim=2),
                torch.stack(attentions, dim=2) if return_attention else None)
