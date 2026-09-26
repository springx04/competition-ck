"""Metrics used by the Q2 validation and test reports.

The reporting contract fixes the class order to ``0=Negative``,
``1=Neutral`` and ``2=Positive``.  Regression metrics always consume the
original continuous score; the classification prediction is never used to
rewrite it.

All public functions return plain Python dictionaries so their results can be
written to JSON or converted to a CSV row without carrying numpy scalar
objects.  Numpy arrays and torch-like tensors (objects implementing
``detach().cpu().numpy()``) are accepted as inputs.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    precision_recall_fscore_support,
)


CLASS_IDS = (0, 1, 2)
CLASS_NAMES = ("Negative", "Neutral", "Positive")
DEFAULT_NEUTRAL_QUANTILES = (0.50, 0.90, 0.95)


def _to_numpy(value: Any) -> np.ndarray:
    """Convert numpy/scalar/torch-like values to a numpy array."""

    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _vector(value: Any, name: str, *, dtype: Any = float) -> np.ndarray:
    array = _to_numpy(value)
    if array.ndim == 0:
        array = array.reshape(1)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {array.shape}")
    return array.astype(dtype, copy=False)


def _labels(value: Any, name: str = "labels") -> np.ndarray:
    labels = _vector(value, name, dtype=float)
    if not np.isfinite(labels).all():
        raise ValueError(f"{name} contains non-finite values")
    if not np.equal(labels, np.floor(labels)).all():
        raise ValueError(f"{name} must contain integer class ids 0, 1, or 2")
    labels = labels.astype(np.int64, copy=False)
    if not np.isin(labels, CLASS_IDS).all():
        raise ValueError(f"{name} must contain only class ids 0, 1, or 2")
    return labels


def _check_same_length(first: np.ndarray, second: np.ndarray, first_name: str, second_name: str) -> None:
    if len(first) != len(second):
        raise ValueError(
            f"{first_name} and {second_name} must have the same length "
            f"({len(first)} != {len(second)})"
        )


def _finite_vector(value: Any, name: str) -> np.ndarray:
    vector = _vector(value, name, dtype=float)
    if not np.isfinite(vector).all():
        raise ValueError(f"{name} contains non-finite values")
    return vector


def _quantile_values(values: np.ndarray, quantiles: Sequence[float]) -> dict[str, float | None]:
    """Return quantiles with stable JSON-friendly names (q50, q90, ...)."""

    checked: list[float] = []
    for quantile in quantiles:
        q = float(quantile)
        if not 0.0 <= q <= 1.0:
            raise ValueError(f"quantile must be in [0, 1], got {q}")
        checked.append(q)
    def quantile_name(q: float) -> str:
        percentage = 100.0 * q
        if percentage.is_integer():
            return f"q{int(percentage)}"
        return f"q{q:g}".replace(".", "p")

    if values.size == 0:
        return {quantile_name(q): None for q in checked}
    return {
        quantile_name(q): float(np.quantile(values, q))
        for q in checked
    }


def _score_stats(
    scores: np.ndarray,
    mask: np.ndarray,
    quantiles: Sequence[float],
) -> dict[str, Any]:
    values = np.abs(scores[mask])
    return {
        "n": int(values.size),
        "abs_mean": float(values.mean()) if values.size else None,
        "abs_quantiles": _quantile_values(values, quantiles),
    }


def classification_metrics(y_true: Any, logits: Any) -> dict[str, Any]:
    """Compute the fixed three-class metrics from class logits.

    Parameters
    ----------
    y_true:
        Integer class labels in the fixed order ``0/1/2``.
    logits:
        An ``(N, 3)`` array.  Predictions are always ``argmax(logits, axis=1)``.

    Returns
    -------
    dict
        Accuracy, macro/weighted F1, per-class precision/recall/F1/support,
        and the classes absent from this group.
    """

    labels = _labels(y_true)
    logit_array = _to_numpy(logits)
    if logit_array.ndim != 2 or logit_array.shape[1] != len(CLASS_IDS):
        raise ValueError(
            f"logits must have shape (N, 3), got shape {logit_array.shape}"
        )
    if not np.isfinite(logit_array).all():
        raise ValueError("logits contains non-finite values")
    _check_same_length(labels, logit_array, "y_true", "logits")
    if len(labels) == 0:
        raise ValueError("cannot compute classification metrics for an empty group")

    predicted = np.argmax(logit_array, axis=1).astype(np.int64, copy=False)
    fixed_labels = list(CLASS_IDS)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels,
        predicted,
        labels=fixed_labels,
        zero_division=0,
    )
    per_class = {
        class_name: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index, class_name in enumerate(CLASS_NAMES)
    }
    missing_ids = [class_id for class_id, count in zip(CLASS_IDS, support) if count == 0]
    missing_names = [CLASS_NAMES[class_id] for class_id in missing_ids]

    return {
        "n": int(len(labels)),
        "class_order": list(CLASS_NAMES),
        "accuracy": float(accuracy_score(labels, predicted)),
        "macro_f1": float(
            f1_score(labels, predicted, labels=fixed_labels, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(labels, predicted, labels=fixed_labels, average="weighted", zero_division=0)
        ),
        "per_class": per_class,
        "missing_classes": missing_ids,
        "missing_class_names": missing_names,
    }


def regression_metrics(y_true: Any, predicted_score: Any) -> dict[str, Any]:
    """Compute continuous MAE and Pearson correlation.

    A constant vector or a group with fewer than two samples makes Pearson
    undefined.  In that case ``pearson`` is ``None`` and ``pearson_reason``
    records the exact reason.  Non-finite predictions are rejected instead of
    being silently replaced with zero.
    """

    target = _finite_vector(y_true, "y_true_score")
    prediction = _finite_vector(predicted_score, "predicted_score")
    _check_same_length(target, prediction, "y_true_score", "predicted_score")
    if len(target) == 0:
        raise ValueError("cannot compute regression metrics for an empty group")

    prediction_constant = bool(np.all(prediction == prediction[0]))
    target_constant = bool(np.all(target == target[0]))
    if len(target) < 2:
        pearson = None
        pearson_reason = "insufficient_samples"
    elif prediction_constant and target_constant:
        pearson = None
        pearson_reason = "constant_prediction_and_target"
    elif prediction_constant:
        pearson = None
        pearson_reason = "constant_prediction"
    elif target_constant:
        pearson = None
        pearson_reason = "constant_target"
    else:
        # The variance checks above avoid numpy/scipy warnings and undefined
        # NaNs for the cases that the reporting contract calls out explicitly.
        centered_prediction = prediction - prediction.mean()
        centered_target = target - target.mean()
        pearson = float(
            np.dot(centered_prediction, centered_target)
            / np.sqrt(
                np.dot(centered_prediction, centered_prediction)
                * np.dot(centered_target, centered_target)
            )
        )
        pearson_reason = None

    return {
        "n": int(len(target)),
        "mae": float(mean_absolute_error(target, prediction)),
        "pearson": pearson,
        "pearson_reason": pearson_reason,
    }


def symbol_inconsistency_metrics(
    y_true: Any,
    predicted_score: Any,
    predicted_labels: Any | None = None,
    *,
    neutral_quantiles: Sequence[float] = DEFAULT_NEUTRAL_QUANTILES,
) -> dict[str, Any]:
    """Measure disagreement between class labels and the continuous score.

    The strict reporting rules are:

    * Negative (0) with ``score >= 0``;
    * Positive (2) with ``score <= 0``;
    * Neutral (1) with ``score != 0``.

    ``neutral_score`` describes true Neutral samples, which is the population
    used by the inconsistency rule.  When ``predicted_labels`` is supplied,
    ``predicted_neutral_score`` is also reported for auditing the independent
    classification head.
    """

    labels = _labels(y_true)
    scores = _finite_vector(predicted_score, "predicted_score")
    _check_same_length(labels, scores, "y_true", "predicted_score")
    if len(labels) == 0:
        raise ValueError("cannot compute symbol metrics for an empty group")

    negative_inconsistent = (labels == 0) & (scores >= 0)
    positive_inconsistent = (labels == 2) & (scores <= 0)
    neutral_inconsistent = (labels == 1) & (scores != 0)
    inconsistent = negative_inconsistent | positive_inconsistent | neutral_inconsistent

    by_class = {
        "Negative": {
            "n": int(np.sum(labels == 0)),
            "inconsistent": int(np.sum(negative_inconsistent)),
        },
        "Neutral": {
            "n": int(np.sum(labels == 1)),
            "inconsistent": int(np.sum(neutral_inconsistent)),
        },
        "Positive": {
            "n": int(np.sum(labels == 2)),
            "inconsistent": int(np.sum(positive_inconsistent)),
        },
    }
    for values in by_class.values():
        values["rate"] = (
            float(values["inconsistent"] / values["n"]) if values["n"] else None
        )

    result: dict[str, Any] = {
        "n": int(len(labels)),
        "symbol_inconsistency_count": int(np.sum(inconsistent)),
        "symbol_inconsistency_rate": float(np.mean(inconsistent)),
        "symbol_inconsistency_by_class": by_class,
        "neutral_score": _score_stats(scores, labels == 1, neutral_quantiles),
    }
    # Flat names make CSV export convenient while the nested object preserves
    # the full meaning of each statistic.
    result["neutral_abs_score_mean"] = result["neutral_score"]["abs_mean"]
    result["neutral_abs_score_quantiles"] = result["neutral_score"]["abs_quantiles"]

    if predicted_labels is not None:
        predicted = _labels(predicted_labels, "predicted_labels")
        _check_same_length(labels, predicted, "y_true", "predicted_labels")
        predicted_neutral = _score_stats(scores, predicted == 1, neutral_quantiles)
        result["predicted_neutral_score"] = predicted_neutral
        result["predicted_neutral_abs_score_mean"] = predicted_neutral["abs_mean"]
        result["predicted_neutral_abs_score_quantiles"] = predicted_neutral["abs_quantiles"]
    return result


def compute_metrics(
    y_true_class: Any,
    logits: Any,
    y_true_score: Any,
    predicted_score: Any,
    *,
    neutral_quantiles: Sequence[float] = DEFAULT_NEUTRAL_QUANTILES,
) -> dict[str, Any]:
    """Compute all Q2 metrics for one identical sample batch."""

    classification = classification_metrics(y_true_class, logits)
    regression = regression_metrics(y_true_score, predicted_score)
    logits_array = _to_numpy(logits)
    predicted_labels = np.argmax(logits_array, axis=1).astype(np.int64, copy=False)
    symbol = symbol_inconsistency_metrics(
        y_true_class,
        predicted_score,
        predicted_labels,
        neutral_quantiles=neutral_quantiles,
    )
    # ``n`` is shared by all three sections.  Keeping the result flat makes a
    # metric row straightforward to serialize while preserving per-class and
    # neutral audit details in their nested fields.
    return {**classification, **regression, **symbol}


def _metric_value(metrics: Mapping[str, Any], *names: str) -> float | None:
    for name in names:
        if name in metrics:
            value = metrics[name]
            return None if value is None else float(value)
    return None


def _difference(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return float(left - right)


def degradation_metrics(
    clean: Mapping[str, Any],
    corrupt: Mapping[str, Any],
) -> dict[str, Any]:
    """Compute clean-vs-corrupt degradation on the same sample batch.

    Positive values mean the clean result is better for accuracy/F1/Pearson or
    that corruption increased MAE, matching the Q2 reporting convention:
    ``clean - corrupt`` for accuracy/F1/Pearson and ``corrupt - clean`` for
    MAE.  Undefined Pearson values remain ``None`` and are never replaced by
    zero.
    """

    clean_accuracy = _metric_value(clean, "accuracy")
    corrupt_accuracy = _metric_value(corrupt, "accuracy")
    clean_macro_f1 = _metric_value(clean, "macro_f1")
    corrupt_macro_f1 = _metric_value(corrupt, "macro_f1")
    clean_mae = _metric_value(clean, "mae")
    corrupt_mae = _metric_value(corrupt, "mae")
    clean_pearson = _metric_value(clean, "pearson", "pcc")
    corrupt_pearson = _metric_value(corrupt, "pearson", "pcc")

    pearson_reason = None
    if clean_pearson is None or corrupt_pearson is None:
        if clean_pearson is None and corrupt_pearson is None:
            pearson_reason = "clean_and_corrupt_pearson_undefined"
        elif clean_pearson is None:
            pearson_reason = "clean_pearson_undefined"
        else:
            pearson_reason = "corrupt_pearson_undefined"

    return {
        "delta_accuracy": _difference(clean_accuracy, corrupt_accuracy),
        "delta_macro_f1": _difference(clean_macro_f1, corrupt_macro_f1),
        "delta_mae": _difference(corrupt_mae, clean_mae),
        "delta_pearson": _difference(clean_pearson, corrupt_pearson),
        "delta_pearson_reason": pearson_reason,
    }


# Descriptive aliases used by callers that prefer the report terminology.
compute_classification_metrics = classification_metrics
compute_regression_metrics = regression_metrics
compute_symbol_metrics = symbol_inconsistency_metrics
compute_degradation = degradation_metrics


__all__ = [
    "CLASS_IDS",
    "CLASS_NAMES",
    "DEFAULT_NEUTRAL_QUANTILES",
    "classification_metrics",
    "compute_classification_metrics",
    "compute_degradation",
    "compute_metrics",
    "compute_regression_metrics",
    "compute_symbol_metrics",
    "degradation_metrics",
    "regression_metrics",
    "symbol_inconsistency_metrics",
]
