"""Train-only sampling helpers for repeated video clips.

The sampler operates on the video IDs already stored in the attachment 2
training metadata.  It does not inspect labels, features, or any other split.
"""

from __future__ import annotations

from collections import Counter
import math
from typing import Iterable

import numpy as np


def video_sample_weights(video_ids: Iterable[object], power: float = 0.0) -> np.ndarray:
    """Return one replacement-sampling weight per training clip.

    If a video contributes ``n`` clips, each of its clips receives
    ``n ** (-power)``.  ``power=0`` therefore reproduces ordinary uniform
    sampling, while ``power=1`` gives every video the same total weight.
    Values between zero and one reduce the influence of videos with many
    clips without discarding any examples.

    The caller is responsible for passing only training video IDs.  Keeping
    this function independent of a dataset object makes it harder to
    accidentally derive weights from validation or test metadata.
    """

    if isinstance(power, bool) or not isinstance(power, (int, float)):
        raise TypeError("power must be a non-negative finite number")
    power = float(power)
    if not math.isfinite(power) or power < 0.0:
        raise ValueError("power must be a non-negative finite number")

    ids = list(video_ids)
    if not ids:
        return np.empty(0, dtype=np.float64)
    try:
        counts = Counter(ids)
    except TypeError as exc:
        raise TypeError("video_ids must contain hashable IDs") from exc
    return np.asarray(
        [counts[video_id] ** (-power) for video_id in ids],
        dtype=np.float64,
    )
