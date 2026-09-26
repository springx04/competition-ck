from __future__ import annotations

import math

from q1_features.review_evaluation import average_precision, confusion


def test_confusion_metrics() -> None:
    result = confusion([1, 1, 0, 0], [1, 0, 1, 0])
    assert result == {
        "tp": 1,
        "fp": 1,
        "tn": 1,
        "fn": 1,
        "precision": 0.5,
        "recall": 0.5,
        "f1": 0.5,
    }


def test_average_precision_perfect_ranking() -> None:
    assert average_precision([1, 0, 1, 0], [0.9, 0.2, 0.8, 0.1]) == 1.0


def test_average_precision_without_positives_is_nan() -> None:
    assert math.isnan(average_precision([0, 0], [0.9, 0.1]))

