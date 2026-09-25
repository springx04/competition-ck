"""Shared optional logit correction for evaluation and deployment."""
import numpy as np
import torch


def validate_class_bias(value):
    if value is None:
        return None
    bias = np.asarray(value, dtype=np.float32)
    if bias.shape != (3,) or not np.isfinite(bias).all():
        raise ValueError("class_bias must be a finite length-3 vector")
    return bias.tolist()


def apply_class_bias(logits, observed, class_bias):
    """Keep the train-prior fallback intact for samples with no observations."""
    if class_bias is None:
        return logits
    bias = torch.as_tensor(class_bias, dtype=logits.dtype, device=logits.device)
    has_content = observed.any(dim=(1, 2))
    return logits + has_content[:, None].to(logits.dtype) * bias
