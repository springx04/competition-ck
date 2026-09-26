"""Observation state inferred solely from the current raw model input."""
from dataclasses import dataclass

import torch


PAD, UNK, CLS, SEP, MASK = 0, 100, 101, 102, 103


@dataclass
class ObservationState:
    U: torch.Tensor  # B,T,3: currently observed content
    J: torch.Tensor  # B,T: current structural or content position
    q: torch.Tensor  # B,T,3,5
    text_slot: torch.Tensor
    bert_attention: torch.Tensor


def infer_state(raw_batch: dict) -> ObservationState:
    ids = raw_batch["input_ids"]
    stored = raw_batch["stored_attention"]
    audio = raw_batch["audio"]
    vision = raw_batch["vision"]
    if ids.ndim != 2 or audio.shape[:2] != ids.shape or vision.shape[:2] != ids.shape:
        raise ValueError("RawBatch must have a common B,T axis")
    active = stored == 1
    boundary = (ids == CLS) | (ids == SEP)
    text_slot = active & (ids != PAD) & ~boundary
    u_text = text_slot & (ids != MASK)
    u_audio = (audio != 0).any(dim=-1)
    u_vision = (vision != 0).any(dim=-1)
    U = torch.stack((u_text, u_audio, u_vision), dim=-1)
    J = text_slot | u_audio | u_vision
    bert_attention = active & (ids != PAD) & (ids != MASK)

    batch, length = ids.shape
    positions = torch.arange(length, device=ids.device)
    n_j = J.sum(dim=1).clamp(min=1)
    first = torch.where(J, positions[None], length).min(dim=1).values
    last = torch.where(J, positions[None], -1).max(dim=1).values
    envelope = (last - first + 1).clamp(min=1)
    holes = J[..., None] & ~U
    forward = torch.zeros(batch, length, 3, dtype=torch.long, device=ids.device)
    backward = torch.zeros_like(forward)
    # Run lengths ending at each position.  The cumulative-count form is
    # exactly equivalent to the old Python loop, but keeps the operation on
    # one tensor kernel and avoids 2*seq_len interpreter iterations per call.
    def _run_lengths(mask: torch.Tensor) -> torch.Tensor:
        cumulative = mask.to(torch.long).cumsum(dim=1)
        reset_values = torch.where(mask, torch.zeros_like(cumulative), cumulative)
        previous = torch.cummax(reset_values, dim=1).values
        return (cumulative - previous) * mask

    forward = _run_lengths(holes)
    backward = torch.flip(_run_lengths(torch.flip(holes, dims=(1,))), dims=(1,))
    gap = torch.where(holes, (forward + backward - 1).float() / envelope[:, None, None], 0)
    distance = (positions[:, None] - positions[None, :]).abs()
    nearest = torch.where(U[:, None], distance[None, :, :, None], length).min(dim=2).values.float() / length
    nearest = torch.where(U.any(dim=1)[:, None], nearest, 1.0)
    observed_counts = U.sum(dim=1).float()
    q = torch.stack((
        (observed_counts / n_j[:, None])[:, None].expand(batch, length, 3),
        gap,
        (positions.float() / max(length - 1, 1))[None, :, None].expand(batch, length, 3),
        nearest,
        ((observed_counts.sum(dim=1, keepdim=True) - observed_counts) / (2 * n_j[:, None]))[:, None].expand(batch, length, 3),
    ), dim=-1)
    q = q * J[:, :, None, None]
    return ObservationState(U=U, J=J, q=q, text_slot=text_slot, bert_attention=bert_attention)
