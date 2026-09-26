from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from q2.metrics import (  # noqa: E402
    classification_metrics,
    compute_metrics,
    degradation_metrics,
    regression_metrics,
    symbol_inconsistency_metrics,
)


def test_classification_uses_three_fixed_classes_and_marks_missing_class() -> None:
    y_true = np.array([0, 1, 1, 0])
    logits = np.array(
        [
            [3.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 4.0, 0.0],
        ]
    )

    metrics = classification_metrics(y_true, logits)

    assert metrics["accuracy"] == pytest.approx(0.75)
    assert metrics["missing_classes"] == [2]
    assert metrics["missing_class_names"] == ["Positive"]
    assert metrics["per_class"]["Positive"] == {
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "support": 0,
    }
    assert metrics["per_class"]["Negative"]["support"] == 2
    assert metrics["per_class"]["Neutral"]["support"] == 2


def test_regression_reports_pearson_and_explicit_undefined_reasons() -> None:
    metrics = regression_metrics([1.0, 2.0, 3.0], [1.5, 2.0, 2.5])
    assert metrics["mae"] == pytest.approx(1.0 / 3.0)
    assert metrics["pearson"] == pytest.approx(1.0)
    assert metrics["pearson_reason"] is None

    assert regression_metrics([1.0], [1.0])["pearson_reason"] == "insufficient_samples"
    assert regression_metrics([1.0, 2.0], [1.0, 1.0])["pearson_reason"] == "constant_prediction"
    assert regression_metrics([1.0, 1.0], [1.0, 2.0])["pearson_reason"] == "constant_target"
    assert regression_metrics([1.0, 1.0], [2.0, 2.0])["pearson_reason"] == (
        "constant_prediction_and_target"
    )

    with pytest.raises(ValueError, match="non-finite"):
        regression_metrics([1.0, 2.0], [1.0, np.nan])


def test_symbol_inconsistency_uses_strict_boundaries_and_reports_neutral_scores() -> None:
    y_true = [0, 0, 2, 2, 1, 1]
    scores = [-0.4, 0.0, 0.0, 0.2, 0.0, 0.5]

    metrics = symbol_inconsistency_metrics(y_true, scores)

    # Negative zero, positive zero, and non-zero Neutral are all inconsistent.
    assert metrics["symbol_inconsistency_count"] == 3
    assert metrics["symbol_inconsistency_rate"] == pytest.approx(0.5)
    assert metrics["symbol_inconsistency_by_class"]["Negative"]["inconsistent"] == 1
    assert metrics["symbol_inconsistency_by_class"]["Positive"]["inconsistent"] == 1
    assert metrics["symbol_inconsistency_by_class"]["Neutral"]["inconsistent"] == 1
    assert metrics["neutral_abs_score_mean"] == pytest.approx(0.25)
    assert metrics["neutral_abs_score_quantiles"] == {
        "q50": pytest.approx(0.25),
        "q90": pytest.approx(0.45),
        "q95": pytest.approx(0.475),
    }


def test_compute_metrics_includes_predicted_neutral_audit_statistics() -> None:
    result = compute_metrics(
        [0, 1, 2],
        [[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]],
        [-1.0, 0.4, 1.0],
        [-0.5, 0.2, 0.5],
    )
    assert result["class_order"] == ["Negative", "Neutral", "Positive"]
    assert result["pearson"] == pytest.approx(1.0)
    assert result["predicted_neutral_score"]["n"] == 1
    assert result["predicted_neutral_abs_score_mean"] == pytest.approx(0.2)


def test_degradation_follows_clean_corrupt_sign_convention_and_keeps_undefined_pcc_empty() -> None:
    clean = {"accuracy": 0.80, "macro_f1": 0.75, "mae": 0.20, "pearson": 0.90}
    corrupt = {"accuracy": 0.60, "macro_f1": 0.50, "mae": 0.35, "pearson": 0.40}
    delta = degradation_metrics(clean, corrupt)
    assert delta == {
        "delta_accuracy": pytest.approx(0.20),
        "delta_macro_f1": pytest.approx(0.25),
        "delta_mae": pytest.approx(0.15),
        "delta_pearson": pytest.approx(0.50),
        "delta_pearson_reason": None,
    }

    undefined = degradation_metrics(
        {"accuracy": 0.8, "macro_f1": 0.7, "mae": 0.2, "pearson": None},
        {"accuracy": 0.7, "macro_f1": 0.6, "mae": 0.3, "pearson": 0.2},
    )
    assert undefined["delta_pearson"] is None
    assert undefined["delta_pearson_reason"] == "clean_pearson_undefined"
