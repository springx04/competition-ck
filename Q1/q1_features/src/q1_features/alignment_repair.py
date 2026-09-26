from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .auto_resolution import edit_distance, normalize_text, normalize_words, word_overlap


SEMANTIC_STATUSES = {"MATCH_STRONG", "MATCH_WEAK", "CONFLICT", "UNRESOLVED"}
TEMPORAL_STATUSES = {"VERIFIED", "PARTIAL", "FAILED", "NOT_APPLICABLE"}
VISUAL_STATUSES = {"verified", "ambiguous", "no_active", "missing"}



def official_alignment_segment(official_text: str, duration: float) -> dict[str, Any]:
    """Build the only valid WhisperX forced-alignment input for Stage 2."""
    if duration <= 0:
        raise ValueError("audio duration must be positive")
    return {"start": 0.0, "end": float(duration), "text": str(official_text)}


def usable_word_interval(
    semantic_status: str,
    forced_word: Mapping[str, Any] | None,
) -> tuple[float, float] | None:
    """Never manufacture word/audio time when semantic pairing is unsupported."""
    if semantic_status not in {"MATCH_STRONG", "MATCH_WEAK"} or not forced_word:
        return None
    start, end = forced_word.get("start"), forced_word.get("end")
    if start is None or end is None or float(end) <= float(start):
        return None
    return float(start), float(end)


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def text_pair_metrics(reference: str, hypothesis: str) -> dict[str, float]:
    ref_tokens = normalize_words(reference)
    hyp_tokens = normalize_words(hypothesis)
    ref_chars = list(normalize_text(reference).replace(" ", ""))
    hyp_chars = list(normalize_text(hypothesis).replace(" ", ""))
    overlap = word_overlap(ref_tokens, hyp_tokens)
    wer = edit_distance(ref_tokens, hyp_tokens) / max(1, len(ref_tokens))
    cer = edit_distance(ref_chars, hyp_chars) / max(1, len(ref_chars))
    character_similarity = max(0.0, 1.0 - edit_distance(ref_chars, hyp_chars) / max(1, len(ref_chars), len(hyp_chars)))
    return {
        "wer": float(wer),
        "cer": float(cer),
        "token_precision": float(overlap["word_overlap_precision"]),
        "token_recall": float(overlap["word_overlap_recall"]),
        "token_f1": float(overlap["word_overlap_f1"]),
        "character_similarity": float(character_similarity),
        "score": float((overlap["word_overlap_f1"] + character_similarity) / 2.0),
    }


def classify_semantic_pairing(
    *,
    chain_ok: bool,
    mapping_suspected: bool,
    official_whisper: Mapping[str, float],
    official_ctc: Mapping[str, float],
    ctc_whisper: Mapping[str, float],
    true_pair_rank: int,
    top1_minus_true_margin: float,
    strong_f1: float = 0.60,
    strong_character_similarity: float = 0.60,
    weak_f1: float = 0.45,
    weak_rank_max: int = 5,
    conflict_official_f1_max: float = 0.35,
    conflict_asr_agreement_f1_min: float = 0.60,
    conflict_global_rank_min: int = 10,
    conflict_margin_min: float = 0.15,
) -> tuple[str, str]:
    """Classify semantic pairing without using CTC diagnostic WER as a veto."""
    if not chain_ok:
        return "UNRESOLVED", "data_chain_not_verified"
    if mapping_suspected:
        return "UNRESOLVED", "within_video_clip_mapping_suspected"
    whisper_strong = (
        float(official_whisper["token_f1"]) >= strong_f1
        and float(official_whisper["character_similarity"]) >= strong_character_similarity
        and int(true_pair_rank) == 1
    )
    if whisper_strong:
        return "MATCH_STRONG", "WhisperX_and_global_retrieval_support_official_text"
    both_asr_reject_official = (
        float(official_whisper["token_f1"]) <= conflict_official_f1_max
        and float(official_ctc["token_f1"]) <= conflict_official_f1_max
        and float(ctc_whisper["token_f1"]) >= conflict_asr_agreement_f1_min
    )
    global_conflict = int(true_pair_rank) >= conflict_global_rank_min and float(top1_minus_true_margin) >= conflict_margin_min
    if both_asr_reject_official or global_conflict:
        reasons = []
        if both_asr_reject_official:
            reasons.append("CTC_and_WhisperX_agree_but_both_conflict_with_official")
        if global_conflict:
            reasons.append("global_retrieval_prefers_other_audio")
        return "CONFLICT", ";".join(reasons)
    if float(official_whisper["token_f1"]) >= weak_f1 and int(true_pair_rank) <= weak_rank_max:
        return "MATCH_WEAK", "WhisperX_partially_supports_official_text"
    if float(official_ctc["token_f1"]) >= strong_f1 and float(ctc_whisper["token_f1"]) >= weak_f1:
        return "MATCH_WEAK", "CTC_and_cross_ASR_evidence_support_official_text"
    return "UNRESOLVED", "independent_semantic_evidence_is_inconclusive"


def classify_temporal_alignment(
    semantic_status: str,
    *,
    forced_alignment_success: bool,
    forced_alignment_coverage: float,
    common_coverage: float | None,
    midpoint_median_s: float | None,
    midpoint_p90_s: float | None,
    verified_common_coverage_min: float = 0.80,
    verified_midpoint_median_s_max: float = 0.30,
    verified_midpoint_p90_s_max: float = 0.60,
    partial_alignment_coverage_min: float = 0.10,
) -> tuple[str, str]:
    if semantic_status not in {"MATCH_STRONG", "MATCH_WEAK"}:
        return "NOT_APPLICABLE", "semantic_pairing_not_supported"
    if not forced_alignment_success or forced_alignment_coverage < partial_alignment_coverage_min:
        return "FAILED", "official_text_forced_alignment_failed_or_has_negligible_coverage"
    if (
        common_coverage is not None
        and midpoint_median_s is not None
        and midpoint_p90_s is not None
        and common_coverage >= verified_common_coverage_min
        and midpoint_median_s <= verified_midpoint_median_s_max
        and midpoint_p90_s <= verified_midpoint_p90_s_max
    ):
        return "VERIFIED", "CTC_and_official_forced_alignment_meet_frozen_time_thresholds"
    return "PARTIAL", "official_text_has_reliable_partial_times_but_time_agreement_is_below_VERIFIED"


def classify_conflict_cause(
    *,
    chain_ok: bool,
    mapping_suspected: bool,
    official_whisper_f1: float,
    official_ctc_f1: float,
    ctc_whisper_f1: float,
) -> str:
    if not chain_ok:
        return "data_chain_error"
    if mapping_suspected:
        return "clip_mapping_suspected"
    if official_whisper_f1 <= 0.35 and official_ctc_f1 <= 0.35 and ctc_whisper_f1 >= 0.60:
        return "two_ASRs_agree_but_both_conflict_with_official"
    if ctc_whisper_f1 < 0.40:
        return "ASR_unstable"
    return "unresolved_algorithmic_or_content_conflict"


def best_assignment(score_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from scipy.optimize import linear_sum_assignment

    matrix = np.asarray(score_matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("within-video score matrix must be square")
    rows, cols = linear_sum_assignment(-matrix)
    return rows.astype(np.int64), cols.astype(np.int64)


def mapping_candidate_rows(
    sample_ids: Sequence[str],
    scores: np.ndarray,
    *,
    off_diagonal_margin: float = 0.10,
) -> list[dict[str, Any]]:
    matrix = np.asarray(scores, dtype=float)
    if matrix.shape != (len(sample_ids), len(sample_ids)):
        raise ValueError("sample IDs and score matrix shape differ")
    assignment_rows, assignment_cols = best_assignment(matrix)
    assigned = {int(row): int(col) for row, col in zip(assignment_rows, assignment_cols)}
    output = []
    for i, sample_id in enumerate(sample_ids):
        best_col = int(np.argmax(matrix[i]))
        diagonal = float(matrix[i, i])
        best = float(matrix[i, best_col])
        hungarian_col = assigned[i]
        suspected = (
            best_col != i
            and hungarian_col != i
            and best - diagonal >= off_diagonal_margin
        )
        output.append({
            "sample_id": sample_id,
            "diagonal_score": diagonal,
            "row_best_sample_id": sample_ids[best_col],
            "row_best_score": best,
            "row_best_minus_diagonal": best - diagonal,
            "hungarian_sample_id": sample_ids[hungarian_col],
            "hungarian_score": float(matrix[i, hungarian_col]),
            "mapping_suspected": int(suspected),
        })
    return output


def av_sync_evidence(percentile: float | None, *, support_min: float = 0.80, conflict_max: float = 0.20) -> str:
    if percentile is None or not math.isfinite(float(percentile)):
        return "neutral"
    if float(percentile) >= support_min:
        return "support"
    if float(percentile) <= conflict_max:
        return "conflict"
    return "neutral"


def decide_active_speaker_window(
    visible_clusters: Sequence[int],
    cluster_logits: Mapping[int, float | None],
    *,
    threshold: float = 0.0,
) -> tuple[str, int | None]:
    visible = sorted(set(int(value) for value in visible_clusters))
    if not visible:
        return "missing", None
    active = [
        cluster_id for cluster_id in visible
        if cluster_logits.get(cluster_id) is not None and float(cluster_logits[cluster_id]) >= threshold
    ]
    if len(active) == 1:
        return "verified", active[0]
    if len(active) > 1:
        return "ambiguous", None
    return "no_active", None


def build_talknet_identity_timeline(
    *,
    sample_id: str,
    duration: float,
    episode_to_cluster: Mapping[str, int],
    visible_rows: Sequence[Mapping[str, Any]],
    talknet_rows: Sequence[Mapping[str, Any]],
    av_percentiles: Mapping[int, float],
    window_s: float = 0.50,
    stride_s: float = 0.25,
    logit_threshold: float = 0.0,
    av_support_min: float = 0.80,
    av_conflict_max: float = 0.20,
) -> list[dict[str, Any]]:
    if duration < 0 or window_s <= 0 or stride_s <= 0:
        raise ValueError("invalid timeline parameters")
    starts = np.arange(0.0, max(duration, 1e-12), stride_s)
    output: list[dict[str, Any]] = []
    for start in starts:
        end = min(duration, float(start + window_s))
        visible = set()
        for row in visible_rows:
            if row.get("face_id") is None or row.get("episode_id") is None:
                continue
            episode_key = f"{int(row['face_id'])}:{int(row['episode_id'])}"
            cluster = episode_to_cluster.get(episode_key)
            if cluster is None:
                continue
            if float(row["end"]) > start and float(row["start"]) < end:
                visible.add(int(cluster))
        logits: dict[int, float | None] = {}
        for cluster in sorted(visible):
            values = []
            for row in talknet_rows:
                if row.get("face_id") is None or row.get("episode_id") is None:
                    continue
                episode_key = f"{int(row['face_id'])}:{int(row['episode_id'])}"
                if episode_to_cluster.get(episode_key) != cluster:
                    continue
                if float(row["end"]) > start and float(row["start"]) < end:
                    values.append(float(row["class1_logit"]))
            logits[cluster] = float(np.median(values)) if values else None
        status, active_cluster = decide_active_speaker_window(sorted(visible), logits, threshold=logit_threshold)
        av_evidence = av_sync_evidence(
            av_percentiles.get(active_cluster) if active_cluster is not None else None,
            support_min=av_support_min,
            conflict_max=av_conflict_max,
        )
        output.append({
            "sample_id": sample_id,
            "start": float(start),
            "end": float(end),
            "visible_clusters": json.dumps(sorted(visible), separators=(",", ":")),
            "cluster_logits": json.dumps({str(k): v for k, v in logits.items()}, separators=(",", ":")),
            "active_cluster": "" if active_cluster is None else int(active_cluster),
            "window_status": status,
            "av_sync_evidence": av_evidence,
            "quality_flag": "talknet_av_sync_conflict" if status == "verified" and av_evidence == "conflict" else "",
        })
    return output


def merge_visual_timeline(
    timeline: Sequence[Mapping[str, Any]],
    *,
    duration: float,
    stride_s: float = 0.25,
) -> list[dict[str, Any]]:
    """Convert overlapping inference windows to non-overlapping local mask segments."""
    cells = []
    for row in timeline:
        start = float(row["start"])
        end = min(duration, start + stride_s)
        if end <= start:
            continue
        cells.append({
            "start": start,
            "end": end,
            "status": str(row["window_status"]),
            "active_cluster": row.get("active_cluster", ""),
            "av_sync_evidence": row.get("av_sync_evidence", "neutral"),
            "quality_flag": row.get("quality_flag", ""),
        })
    merged: list[dict[str, Any]] = []
    for cell in cells:
        key = (cell["status"], str(cell["active_cluster"]), cell["av_sync_evidence"], cell["quality_flag"])
        if merged:
            last = merged[-1]
            last_key = (last["status"], str(last["active_cluster"]), last["av_sync_evidence"], last["quality_flag"])
            if key == last_key and abs(float(last["end"]) - float(cell["start"])) <= 1e-9:
                last["end"] = cell["end"]
                continue
        merged.append(dict(cell))
    return merged


def interval_overlap(start: float, end: float, intervals: Iterable[tuple[float, float]]) -> float:
    pieces = sorted(
        (max(float(start), float(left)), min(float(end), float(right)))
        for left, right in intervals
        if min(float(end), float(right)) > max(float(start), float(left))
    )
    if not pieces or end <= start:
        return 0.0
    total = 0.0
    left, right = pieces[0]
    for next_left, next_right in pieces[1:]:
        if next_left <= right:
            right = max(right, next_right)
        else:
            total += right - left
            left, right = next_left, next_right
    return total + right - left


def compare_word_intervals(
    ctc_words: Sequence[Mapping[str, Any]],
    forced_words: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    forced_by_id = {int(row["word_id"]): row for row in forced_words if row.get("start") is not None and row.get("end") is not None}
    spoken = [row for row in ctc_words if str(row.get("ctc_text", "")).strip()]
    comparisons = []
    for row in spoken:
        word_id = int(row["word_id"])
        accepted = row.get("accepted_interval")
        forced = forced_by_id.get(word_id)
        if not accepted or forced is None:
            continue
        ctc_start, ctc_end = map(float, accepted)
        forced_start, forced_end = float(forced["start"]), float(forced["end"])
        comparisons.append({
            "word_id": word_id,
            "raw_word": row.get("raw_word", ""),
            "ctc_start": ctc_start,
            "ctc_end": ctc_end,
            "forced_start": forced_start,
            "forced_end": forced_end,
            "start_error": abs(ctc_start - forced_start),
            "end_error": abs(ctc_end - forced_end),
            "midpoint_error": abs((ctc_start + ctc_end - forced_start - forced_end) / 2.0),
        })
    mids = [row["midpoint_error"] for row in comparisons]
    starts = [row["start_error"] for row in comparisons]
    ends = [row["end_error"] for row in comparisons]
    ctc_coverage = sum(bool(row.get("accepted_interval")) for row in spoken) / max(1, len(spoken))
    forced_coverage = len(forced_by_id) / max(1, len(spoken))
    summary = {
        "common_word_count": len(comparisons),
        "common_word_fraction": len(comparisons) / max(1, len(spoken)),
        "start_median": float(np.median(starts)) if starts else None,
        "end_median": float(np.median(ends)) if ends else None,
        "midpoint_median": float(np.median(mids)) if mids else None,
        "midpoint_p90": float(np.quantile(mids, 0.90)) if mids else None,
        "max_error": float(max(starts + ends)) if comparisons else None,
        "CTC_coverage": float(ctc_coverage),
        "new_alignment_coverage": float(forced_coverage),
    }
    return summary, comparisons


def validate_statuses(semantic_status: str, temporal_status: str) -> None:
    if semantic_status not in SEMANTIC_STATUSES:
        raise ValueError(f"invalid semantic status: {semantic_status}")
    if temporal_status not in TEMPORAL_STATUSES:
        raise ValueError(f"invalid temporal status: {temporal_status}")
    if semantic_status in {"CONFLICT", "UNRESOLVED"} and temporal_status != "NOT_APPLICABLE":
        raise ValueError("unsupported semantic pairing must have NOT_APPLICABLE temporal status")
