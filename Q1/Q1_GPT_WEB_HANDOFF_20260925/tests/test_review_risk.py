from __future__ import annotations

import pytest

from q1_features.review_risk import percentile_ranks


def test_percentile_ranks_average_ties() -> None:
    assert percentile_ranks([1.0, 2.0, 2.0, 4.0]) == [0.0, 0.5, 0.5, 1.0]


def test_percentile_ranks_singleton_is_centered() -> None:
    assert percentile_ranks([3.0]) == [0.5]


def test_percentile_ranks_reject_non_finite_values() -> None:
    with pytest.raises(ValueError, match="finite"):
        percentile_ranks([1.0, float("nan")])

