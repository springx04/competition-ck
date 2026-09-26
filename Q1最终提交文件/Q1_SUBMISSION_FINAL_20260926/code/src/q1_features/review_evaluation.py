from __future__ import annotations

import csv
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Sequence

from .storage import write_json


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _truth(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


def confusion(labels: Sequence[int], predictions: Sequence[int]) -> dict[str, Any]:
    if len(labels) != len(predictions) or not labels:
        raise ValueError("confusion requires non-empty equally sized arrays")
    tp = sum(y == 1 and p == 1 for y, p in zip(labels, predictions))
    fp = sum(y == 0 and p == 1 for y, p in zip(labels, predictions))
    tn = sum(y == 0 and p == 0 for y, p in zip(labels, predictions))
    fn = sum(y == 1 and p == 0 for y, p in zip(labels, predictions))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1,
    }


def average_precision(labels: Sequence[int], scores: Sequence[float]) -> float:
    if len(labels) != len(scores) or not labels:
        raise ValueError("average precision requires non-empty equally sized arrays")
    positive_count = sum(labels)
    if positive_count == 0:
        return float("nan")
    order = sorted(range(len(labels)), key=lambda index: (-scores[index], index))
    seen_positive = 0
    precision_sum = 0.0
    for rank, index in enumerate(order, start=1):
        if labels[index]:
            seen_positive += 1
            precision_sum += seen_positive / rank
    return precision_sum / positive_count


def _percentile(values: Sequence[float], q: float) -> float | None:
    finite = sorted(value for value in values if math.isfinite(value))
    if not finite:
        return None
    position = (len(finite) - 1) * q
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return finite[lower]
    return finite[lower] * (upper - position) + finite[upper] * (position - lower)


def _grouped_bootstrap(
    rows: Sequence[dict[str, Any]],
    metric: Callable[[Sequence[dict[str, Any]]], float],
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["video_id"]].append(row)
    group_ids = sorted(groups)
    rng = random.Random(seed)
    values = []
    for _ in range(iterations):
        sampled = [rng.choice(group_ids) for _ in group_ids]
        replicate = [row for group_id in sampled for row in groups[group_id]]
        values.append(metric(replicate))
    return {
        "iterations": iterations,
        "seed": seed,
        "group_count": len(group_ids),
        "valid_replicates": sum(math.isfinite(value) for value in values),
        "ci95": [_percentile(values, 0.025), _percentile(values, 0.975)],
    }


def evaluate_audio_review(
    final_decisions_csv: str | Path,
    risk_csv: str | Path,
    quality_csv: str | Path,
    output_json: str | Path,
    *,
    iterations: int = 10_000,
    seed: int = 2026,
) -> dict[str, Any]:
    decisions = _read_csv(final_decisions_csv)
    risks = {row["sample_id"]: row for row in _read_csv(risk_csv)}
    quality = {row["sample_id"]: row for row in _read_csv(quality_csv)}
    if len(decisions) != 100 or len({row["sample_id"] for row in decisions}) != 100:
        raise ValueError("final decisions must contain exactly 100 unique samples")
    if set(row["sample_id"] for row in decisions) != set(risks) or set(risks) != set(quality):
        raise ValueError("final decisions, risk ranking, and quality table must contain identical sample IDs")

    evaluated: list[dict[str, Any]] = []
    unresolved_ids: list[str] = []
    unsafe_released_ids: list[str] = []
    for decision in decisions:
        sample_id = decision["sample_id"]
        transcript = decision["final_transcript_status"]
        visual = decision["final_visual_status"]
        paired = _truth(quality[sample_id]["paired_use"])
        if paired and (transcript != "match" or visual != "active_face_identified"):
            unsafe_released_ids.append(sample_id)
        if transcript == "unverifiable" or visual == "unverifiable":
            unresolved_ids.append(sample_id)
        if transcript not in {"match", "mismatch"}:
            continue
        risk = risks[sample_id]
        evaluated.append({
            "sample_id": sample_id,
            "video_id": decision["video_id"],
            "label": int(transcript == "mismatch"),
            "baseline": int(_truth(risk["existing_ctc_alert"])),
            "risk_score": float(risk["risk_score"]),
        })
    if not evaluated:
        raise ValueError("no adjudicated match/mismatch transcript decisions available")
    labels = [row["label"] for row in evaluated]
    baseline = [row["baseline"] for row in evaluated]
    scores = [row["risk_score"] for row in evaluated]
    baseline_metrics = confusion(labels, baseline)
    auprc = average_precision(labels, scores)
    false_negative_ids = [
        row["sample_id"] for row in evaluated
        if row["label"] == 1 and row["baseline"] == 0
    ]
    result = {
        "evaluated_match_or_mismatch": len(evaluated),
        "gold_mismatch_count": sum(labels),
        "excluded_unverifiable_or_other": 100 - len(evaluated),
        "baseline_ctc": baseline_metrics,
        "ctc_plus_optional_whisperx_ranking": {"auprc": auprc},
        "original_no_alert_false_negative_count": len(false_negative_ids),
        "original_no_alert_false_negative_ids": false_negative_ids,
        "unresolved_ids": sorted(unresolved_ids),
        "false_release": len(unsafe_released_ids),
        "false_release_ids": sorted(unsafe_released_ids),
        "grouped_bootstrap": {
            "baseline_f1": _grouped_bootstrap(
                evaluated,
                lambda sample: confusion(
                    [row["label"] for row in sample],
                    [row["baseline"] for row in sample],
                )["f1"],
                iterations=iterations,
                seed=seed,
            ),
            "risk_auprc": _grouped_bootstrap(
                evaluated,
                lambda sample: average_precision(
                    [row["label"] for row in sample],
                    [row["risk_score"] for row in sample],
                ),
                iterations=iterations,
                seed=seed,
            ),
        },
        "warning": "Auxiliary scores are evaluation-only and do not create human confirmations.",
    }
    write_json(output_json, result)
    return result

