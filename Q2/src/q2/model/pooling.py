"""Aggregation weights derived only from currently observed feature rows."""
import torch


def consecutive_run_weights(features, observed):
    """Give each consecutive identical observed run total weight one.

    A missing row always breaks a run. This does not alter the observation
    mask or official sequence positions, and never reads the clean view.
    """
    starts = observed.clone()
    same = (features[:, 1:] == features[:, :-1]).all(dim=-1)
    starts[:, 1:] &= ~(same & observed[:, :-1])
    groups = starts.long().cumsum(dim=1)
    counts = features.new_zeros(features.shape[0], features.shape[1] + 1)
    counts.scatter_add_(1, groups, observed.to(features.dtype))
    return observed.to(features.dtype) / counts.gather(1, groups).clamp_min(1)
