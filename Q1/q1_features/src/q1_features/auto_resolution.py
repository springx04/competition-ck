from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .storage import read_json, read_jsonl, write_csv, write_json


TA_ISSUES = {"suspected_text_audio_mismatch", "low_aligned_word_fraction"}
PARTITION_FIELDS = [
    "sample_id",
    "ta_issue",
    "identity_issue",
    "no_face",
    "current_paired_use",
    "vision_episode_count",
    "group",
]
EXPECTED_PARTITION = {
    "identity_only": 28,
    "text_audio_only_single_episode": 10,
    "identity_and_text_audio": 9,
    "vision_no_face_and_text_audio": 4,
}


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def truth(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def normalize_words(text: str) -> list[str]:
    return re.findall(r"[A-Z0-9]+(?:'[A-Z0-9]+)?", str(text).upper())


def normalize_text(text: str) -> str:
    return " ".join(normalize_words(text))


def edit_distance(left: Sequence[Any], right: Sequence[Any]) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for i, item_left in enumerate(left, start=1):
        current = [i]
        for j, item_right in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (item_left != item_right),
                )
            )
        previous = current
    return previous[-1]


def error_rate(reference: Sequence[Any], hypothesis: Sequence[Any]) -> float:
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return edit_distance(reference, hypothesis) / len(reference)


def word_overlap(reference: Sequence[str], hypothesis: Sequence[str]) -> dict[str, float]:
    ref = Counter(reference)
    hyp = Counter(hypothesis)
    overlap = sum((ref & hyp).values())
    precision = overlap / len(hypothesis) if hypothesis else 0.0
    recall = overlap / len(reference) if reference else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    union = sum((ref | hyp).values())
    return {
        "word_overlap_count": overlap,
        "word_overlap_precision": precision,
        "word_overlap_recall": recall,
        "word_overlap_f1": f1,
        "word_overlap_jaccard": overlap / union if union else 1.0,
    }


def text_similarity(reference: str, hypothesis: str) -> float:
    ref = normalize_words(reference)
    hyp = normalize_words(hypothesis)
    denominator = max(len(ref), len(hyp), 1)
    return max(0.0, 1.0 - edit_distance(ref, hyp) / denominator)


def build_problem_partition(
    quality_csv: str | Path,
    alignment_issues_csv: str | Path,
    *,
    expected: Mapping[str, int] | None = EXPECTED_PARTITION,
) -> list[dict[str, Any]]:
    quality = {row["sample_id"]: row for row in read_csv(quality_csv)}
    issue_types: dict[str, set[str]] = defaultdict(set)
    for row in read_csv(alignment_issues_csv):
        issue_types[row["sample_id"]].add(row["issue_type"])

    rows: list[dict[str, Any]] = []
    for sample_id, quality_row in sorted(quality.items()):
        if truth(quality_row.get("paired_use", "")):
            continue
        types = issue_types.get(sample_id, set())
        ta_issue = bool(types & TA_ISSUES)
        identity_issue = "identity_unknown" in types
        no_face = "vision_no_face" in types
        episode_count = int(float(quality_row.get("vision_episode_count", 0) or 0))
        if no_face and ta_issue and not identity_issue:
            group = "vision_no_face_and_text_audio"
        elif identity_issue and ta_issue and not no_face:
            group = "identity_and_text_audio"
        elif identity_issue and not ta_issue and not no_face:
            group = "identity_only"
        elif ta_issue and not identity_issue and not no_face and episode_count == 1:
            group = "text_audio_only_single_episode"
        else:
            group = "unexpected"
        rows.append(
            {
                "sample_id": sample_id,
                "ta_issue": int(ta_issue),
                "identity_issue": int(identity_issue),
                "no_face": int(no_face),
                "current_paired_use": int(truth(quality_row.get("paired_use", ""))),
                "vision_episode_count": episode_count,
                "group": group,
            }
        )

    counts = Counter(row["group"] for row in rows)
    if len(rows) != 51:
        raise ValueError(f"expected 51 paired_use=false samples, found {len(rows)}")
    if expected is not None and dict(counts) != dict(expected):
        raise ValueError(f"problem partition mismatch: expected={dict(expected)}, actual={dict(counts)}")
    return rows


def write_problem_partition(
    quality_csv: str | Path,
    alignment_issues_csv: str | Path,
    output_csv: str | Path,
) -> dict[str, Any]:
    rows = build_problem_partition(quality_csv, alignment_issues_csv)
    write_csv(output_csv, rows, PARTITION_FIELDS)
    return {"rows": len(rows), "groups": dict(Counter(row["group"] for row in rows))}


def _matching_word_pairs(
    official_words: Sequence[Mapping[str, Any]],
    whisper_words: Sequence[Mapping[str, Any]],
) -> list[tuple[Mapping[str, Any], Mapping[str, Any]]]:
    from difflib import SequenceMatcher

    left = [normalize_text(str(row.get("raw_word", ""))) for row in official_words]
    right = [normalize_text(str(row.get("word", ""))) for row in whisper_words]
    pairs: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    for block in SequenceMatcher(a=left, b=right, autojunk=False).get_matching_blocks():
        for offset in range(block.size):
            a = official_words[block.a + offset]
            b = whisper_words[block.b + offset]
            if a.get("accepted_interval") and b.get("start") is not None and b.get("end") is not None:
                pairs.append((a, b))
    return pairs


def summarize_whisperx_sample(
    sample_id: str,
    official_text: str,
    official_words: Sequence[Mapping[str, Any]],
    whisper_result: Mapping[str, Any],
    diagnostic_wer: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    transcript = str(whisper_result.get("transcript", ""))
    normalized_reference = normalize_text(official_text)
    normalized_hypothesis = normalize_text(transcript)
    reference_tokens = normalize_words(official_text)
    hypothesis_tokens = normalize_words(transcript)
    whisper_words = list(whisper_result.get("words", []))
    spoken = [row for row in official_words if str(row.get("ctc_text", "")).strip()]
    accepted = [row for row in spoken if row.get("accepted_interval")]
    pairs = _matching_word_pairs(spoken, whisper_words)
    comparison: list[dict[str, Any]] = []
    start_differences: list[float] = []
    end_differences: list[float] = []
    midpoint_differences: list[float] = []
    for official, whisper in pairs:
        ctc_start, ctc_end = map(float, official["accepted_interval"])
        whisper_start, whisper_end = float(whisper["start"]), float(whisper["end"])
        start_difference = abs(ctc_start - whisper_start)
        end_difference = abs(ctc_end - whisper_end)
        midpoint_difference = abs((ctc_start + ctc_end - whisper_start - whisper_end) / 2.0)
        start_differences.append(start_difference)
        end_differences.append(end_difference)
        midpoint_differences.append(midpoint_difference)
        comparison.append(
            {
                "sample_id": sample_id,
                "word_id": official.get("word_id"),
                "raw_word": official.get("raw_word", ""),
                "ctc_start": ctc_start,
                "ctc_end": ctc_end,
                "whisperx_start": whisper_start,
                "whisperx_end": whisper_end,
                "start_abs_difference_s": start_difference,
                "end_abs_difference_s": end_difference,
                "midpoint_abs_difference_s": midpoint_difference,
                "whisperx_score": whisper.get("score", ""),
            }
        )
    overlap = word_overlap(reference_tokens, hypothesis_tokens)
    result = {
        "sample_id": sample_id,
        "status": whisper_result.get("status", "ok"),
        "normalized_official_text": normalized_reference,
        "normalized_whisperx_transcript": normalized_hypothesis,
        "whisperx_word_timestamps_json": json.dumps(whisper_words, ensure_ascii=False, separators=(",", ":")),
        "whisperx_wer": error_rate(reference_tokens, hypothesis_tokens),
        "whisperx_cer": error_rate(list(normalized_reference.replace(" ", "")), list(normalized_hypothesis.replace(" ", ""))),
        **overlap,
        "ctc_diagnostic_wer": diagnostic_wer,
        "ctc_accepted_word_fraction": len(accepted) / len(spoken) if spoken else 0.0,
        "common_timed_word_count": len(pairs),
        "common_timed_word_coverage": len(pairs) / len(spoken) if spoken else 0.0,
        "median_start_difference_s": float(np.median(start_differences)) if start_differences else "",
        "p90_start_difference_s": float(np.quantile(start_differences, 0.9)) if start_differences else "",
        "median_end_difference_s": float(np.median(end_differences)) if end_differences else "",
        "p90_end_difference_s": float(np.quantile(end_differences, 0.9)) if end_differences else "",
        "median_midpoint_difference_s": float(np.median(midpoint_differences)) if midpoint_differences else "",
        "p90_midpoint_difference_s": float(np.quantile(midpoint_differences, 0.9)) if midpoint_differences else "",
        "coverage": len([row for row in whisper_words if row.get("start") is not None and row.get("end") is not None]) / max(1, len(hypothesis_tokens)),
    }
    return result, comparison


def build_text_permutation_test(
    manifest_rows: Sequence[Mapping[str, Any]],
    whisper_rows: Sequence[Mapping[str, Any]],
    *,
    negative_count: int = 20,
) -> list[dict[str, Any]]:
    manifest = {str(row["sample_id"]): row for row in manifest_rows}
    whisper = {str(row["sample_id"]): row for row in whisper_rows}
    if set(manifest) != set(whisper):
        raise ValueError("manifest and WhisperX results must contain identical sample IDs")
    rows: list[dict[str, Any]] = []
    for sample_id in sorted(manifest):
        item = manifest[sample_id]
        duration = float(item["duration"])
        word_count = len(normalize_words(str(item["raw_text"])))
        transcript = whisper[sample_id]["normalized_whisperx_transcript"]
        candidates = []
        for other_id, other in manifest.items():
            if other_id == sample_id or str(other["video_id"]) == str(item["video_id"]):
                continue
            other_duration = float(other["duration"])
            other_words = len(normalize_words(str(other["raw_text"])))
            distance = abs(duration - other_duration) / max(duration, other_duration, 1e-9)
            distance += abs(word_count - other_words) / max(word_count, other_words, 1)
            candidates.append((distance, other_id))
        candidates.sort(key=lambda value: (value[0], value[1]))
        chosen = candidates[:negative_count]
        if len(chosen) < negative_count:
            raise ValueError(f"{sample_id}: fewer than {negative_count} different-video controls")
        true_similarity = text_similarity(str(item["raw_text"]), transcript)
        negative_similarities = [
            text_similarity(str(manifest[other_id]["raw_text"]), transcript)
            for _, other_id in chosen
        ]
        true_rank = 1 + sum(value > true_similarity for value in negative_similarities)
        ties = sum(value == true_similarity for value in negative_similarities)
        p_like = (1 + sum(value >= true_similarity for value in negative_similarities)) / (len(negative_similarities) + 1)
        percentile = (sum(value < true_similarity for value in negative_similarities) + 0.5 * ties) / len(negative_similarities)
        rows.append(
            {
                "sample_id": sample_id,
                "video_id": item["video_id"],
                "negative_count": len(negative_similarities),
                "true_similarity": true_similarity,
                "negative_similarity_median": float(np.median(negative_similarities)),
                "negative_similarity_p95": float(np.quantile(negative_similarities, 0.95)),
                "true_rank": true_rank,
                "empirical_percentile": percentile,
                "p_like_score": p_like,
                "negative_sample_ids_json": json.dumps([other_id for _, other_id in chosen], separators=(",", ":")),
            }
        )
    return rows


def classify_text_audio_evidence(
    whisper_row: Mapping[str, Any],
    permutation_row: Mapping[str, Any],
    *,
    semantic_rank_max: int = 1,
    semantic_percentile_min: float = 0.95,
    overlap_f1_min: float = 0.60,
    ctc_wer_max: float = 0.65,
    ctc_fraction_min: float = 0.80,
    common_coverage_min: float = 0.60,
    median_midpoint_max: float = 0.35,
    p90_midpoint_max: float = 0.75,
) -> tuple[str, str, dict[str, bool]]:
    semantic_support = (
        int(float(permutation_row["true_rank"])) <= semantic_rank_max
        and float(permutation_row["empirical_percentile"]) >= semantic_percentile_min
        and float(whisper_row["word_overlap_f1"]) >= overlap_f1_min
    )
    ctc_support = (
        float(whisper_row["ctc_diagnostic_wer"]) <= ctc_wer_max
        and float(whisper_row["ctc_accepted_word_fraction"]) >= ctc_fraction_min
    )
    median_value = whisper_row.get("median_midpoint_difference_s", "")
    p90_value = whisper_row.get("p90_midpoint_difference_s", "")
    boundary_support = (
        float(whisper_row["common_timed_word_coverage"]) >= common_coverage_min
        and median_value not in {"", None}
        and p90_value not in {"", None}
        and float(median_value) <= median_midpoint_max
        and float(p90_value) <= p90_midpoint_max
    )
    flags = {
        "semantic_support": semantic_support,
        "ctc_support": ctc_support,
        "boundary_support": boundary_support,
    }
    if semantic_support and ctc_support and boundary_support:
        return "TA_STRONG", "WhisperX semantics, negative controls, CTC coverage, and word timing agree", flags
    if semantic_support:
        return "TA_SEMANTIC_ONLY", "WhisperX and negative controls support the official text, but CTC coverage or timing is partial", flags
    if ctc_support:
        return "TA_CONFLICT", "CTC supports the official text but WhisperX/negative-control semantic evidence does not", flags
    return "TA_CONFLICT", "CTC and WhisperX/negative-control evidence do not establish the official pairing", flags


def safe_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None
