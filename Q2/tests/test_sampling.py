from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from q2.sampling import video_sample_weights  # noqa: E402


def test_video_weights_reduce_long_video_bias_without_dropping_clips() -> None:
    video_ids = ["short", "long", "long", "long", "long"]

    weights = video_sample_weights(video_ids, power=0.5)

    np.testing.assert_allclose(weights, [1.0, 0.5, 0.5, 0.5, 0.5])
    assert weights.shape == (len(video_ids),)
    assert weights.dtype == np.float64
    # Total group weights are 1 for the one-clip video and 2 for the
    # four-clip video (sqrt(n)); every original clip remains represented.
    assert weights[0] == pytest.approx(1.0)
    assert weights[1:].sum() == pytest.approx(2.0)


def test_power_zero_is_uniform_sampling() -> None:
    weights = video_sample_weights(["a", "a", "b", "c", "c", "c"])

    np.testing.assert_array_equal(weights, np.ones(6, dtype=np.float64))


def test_video_weights_validate_power_and_empty_input() -> None:
    assert video_sample_weights([]).shape == (0,)
    with pytest.raises(ValueError):
        video_sample_weights(["a"], power=-0.1)
    with pytest.raises(ValueError):
        video_sample_weights(["a"], power=float("nan"))
    with pytest.raises(TypeError):
        video_sample_weights(["a"], power=True)
