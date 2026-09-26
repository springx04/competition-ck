"""Single student used for training, evaluation, deployment and Q3 inspection."""
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from ..state import ObservationState, infer_state
from .encoders import ModalityEncoder, masked_mean, sinusoidal_positions
from .msd import Decomposition
from .compensation import SharedCompensator
from .reliability import ErrorEstimator
from .fusion import ContentFusion


VARIANTS = ("full", "late_clean", "late_aug", "no_msd", "no_comp", "no_reliability",
            "no_cons", "no_span", "no_teacher", "uniform_spans")


@dataclass
class ModelInput:
    text_features: torch.Tensor
    audio_norm: torch.Tensor
    vision_norm: torch.Tensor
    U: torch.Tensor
    J: torch.Tensor
    q: torch.Tensor


@dataclass
class ForwardOutput:
    logits: torch.Tensor
    score: torch.Tensor
    U: torch.Tensor
    J: torch.Tensor
    source: torch.Tensor
    B_comp: torch.Tensor
    fusion_type: str
    H: Optional[torch.Tensor] = None
    C: Optional[torch.Tensor] = None
    S: Optional[torch.Tensor] = None
    F: Optional[torch.Tensor] = None
    F_hat: Optional[torch.Tensor] = None
    e_hat: Optional[torch.Tensor] = None
    r: Optional[torch.Tensor] = None
    alpha: Optional[torch.Tensor] = None
    attention: Optional[torch.Tensor] = None
    q: Optional[torch.Tensor] = None
    source_positions: Optional[torch.Tensor] = None
    time_axis: str = "official_aligned_positions"


def encode_view(raw_batch: dict, frozen_text_encoder, normalizer, text_cache=None,
                text_recompute=None) -> ModelInput:
    from ..text import encode_text
    state = infer_state(raw_batch)
    if text_cache is None:
        text_features = encode_text(raw_batch, frozen_text_encoder, state)
    else:
        text_features = text_cache.clone()
        if text_recompute is not None and bool(text_recompute.any()):
            indices = torch.where(text_recompute)[0]
            subset = {key: value[indices] for key, value in raw_batch.items() if isinstance(value, torch.Tensor)}
            substate = infer_state(subset)
            text_features[indices] = encode_text(subset, frozen_text_encoder, substate)
    text_features = text_features * state.U[:, :, 0, None]
    return ModelInput(text_features=text_features, audio_norm=normalizer.transform(raw_batch, "audio"),
                      vision_norm=normalizer.transform(raw_batch, "vision"),
                      U=state.U, J=state.J, q=state.q)


class Student(nn.Module):
    def __init__(self, variant: str, class_prior, score_prior: float, include_aux_heads=True):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant: {variant}")
        self.variant = variant
        self.late = variant.startswith("late_")
        self.use_msd = not self.late and variant != "no_msd"
        self.use_comp = not self.late and variant != "no_comp"
        self.use_reliability = self.use_comp and variant not in ("no_reliability", "no_teacher")
        self.register_buffer("class_prior", torch.as_tensor(class_prior, dtype=torch.float32))
        self.register_buffer("score_prior", torch.as_tensor(score_prior, dtype=torch.float32))
        self.modalities = nn.Embedding(3, 128)
        self.encoders = nn.ModuleList([ModalityEncoder(dim) for dim in (256, 74, 35)])
        if self.late:
            self.late_projection = nn.Sequential(nn.Linear(387, 128), nn.GELU(), nn.Dropout(.1))
            self.decomposition = None
            self.compensator = None
            self.estimator = None
            self.fusion = None
        else:
            self.decomposition = Decomposition(use_msd=self.use_msd, include_aux_heads=include_aux_heads)
            self.compensator = SharedCompensator() if self.use_comp else None
            self.estimator = ErrorEstimator() if self.use_reliability else None
            self.fusion = ContentFusion()
        self.classifier = nn.Linear(128, 3)
        self.regressor = nn.Linear(128, 1)

    def forward(self, model_input: ModelInput, return_details=False, return_attention=None) -> ForwardOutput:
        # Training needs the intermediate tensors for losses but never the
        # 3 x 50 x 150 attention map.  Keep the historical default for
        # callers that request details, while allowing training/evaluation
        # code to opt out of materializing this large diagnostic tensor.
        if return_attention is None:
            return_attention = return_details
        U, J = model_input.U, model_input.J
        batch, length, _ = U.shape
        position = sinusoidal_positions(length, 128, U.device)
        embeds = self.modalities.weight
        inputs = (model_input.text_features, model_input.audio_norm, model_input.vision_norm)
        H = torch.stack([self.encoders[m](inputs[m], U[:, :, m], position, embeds[m])
                         for m in range(3)], dim=2)
        if self.late:
            pools, flags = zip(*(masked_mean(H[:, :, m], U[:, :, m]) for m in range(3)))
            pooled = self.late_projection(torch.cat((*pools, torch.stack(flags, dim=-1).float()), dim=-1))
            source = U.long()
            B_comp = torch.zeros_like(U)
            C = S = F = F_hat = e_hat = alpha = attention = None
            r = None
            fusion_type = "late_concat"
        else:
            C, S, F = self.decomposition(H, U)
            if self.use_comp:
                F_hat, B_comp, attention = self.compensator(C, S, F, U, J, position, embeds, return_attention)
                e_hat = self.estimator(F_hat, C, F, U, model_input.q, B_comp, embeds) if self.estimator else None
            else:
                F_hat, B_comp, e_hat, attention = torch.zeros_like(F), torch.zeros_like(U), None, None
            pooled, source, alpha, _ = self.fusion(F, F_hat, U, B_comp, e_hat, model_input.q,
                                                    embeds, self.use_reliability)
            r = torch.where(U, torch.ones_like(alpha), torch.where(B_comp,
                1 / (1 + e_hat.detach()) if e_hat is not None else torch.ones_like(alpha),
                torch.zeros_like(alpha)))
            fusion_type = "position_content"
        logits = self.classifier(pooled)
        score = 3 * torch.tanh(self.regressor(pooled).squeeze(-1))
        no_content = ~J.any(dim=1)
        if no_content.any():
            logits = torch.where(no_content[:, None], self.class_prior.clamp_min(1e-12).log()[None], logits)
            score = torch.where(no_content, self.score_prior.expand_as(score), score)
        if not return_details:
            H = C = S = F = F_hat = e_hat = r = alpha = attention = None
        source_positions = (torch.stack((torch.arange(3, device=U.device).repeat_interleave(length),
                           torch.arange(length, device=U.device).repeat(3)), dim=-1)
                           if return_details else None)
        return ForwardOutput(logits=logits, score=score, U=U, J=J, source=source, B_comp=B_comp,
                             fusion_type=fusion_type, H=H, C=C, S=S, F=F, F_hat=F_hat,
                             e_hat=e_hat, r=r, alpha=alpha, attention=attention,
                             q=model_input.q if return_details else None,
                             source_positions=source_positions)
