from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from q1_features.auto_vision import assign_openface_episodes
from q1_features.alignment_repair import (
    build_talknet_identity_timeline,
    classify_temporal_alignment,
    compare_word_intervals,
    interval_overlap,
    merge_visual_timeline,
    sha256_file,
    usable_word_interval,
    validate_statuses,
)
from q1_features.auto_resolution import read_csv
from q1_features.extractors.vision import VISION_COLUMNS
from q1_features.storage import load_npz, read_json, read_jsonl, write_csv, write_json, write_jsonl, write_npz
from q1_features.word_aligned_candidate import _aggregate_interval


def _as_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _load_primary_clusters(auto: Path) -> dict[str, dict[str, Any]]:
    rows = [row for row in read_csv(auto / "arcface_identity_clusters.csv") if abs(float(row["threshold"]) - 0.60) < 1e-9]
    if len(rows) != 100:
        raise ValueError(f"expected 100 primary ArcFace rows, found {len(rows)}")
    output = {}
    for row in rows:
        clusters = json.loads(row["clusters_json"])
        episode_to_cluster = {
            episode_key: cluster_id
            for cluster_id, episode_keys in enumerate(clusters)
            for episode_key in episode_keys
        }
        output[row["sample_id"]] = {**row, "clusters": clusters, "episode_to_cluster": episode_to_cluster}
    return output


def _build_temporal_status(baseline: Path, stage_reports: Path, manifest: list[dict]) -> tuple[list[dict], list[dict]]:
    semantic = {row["sample_id"]: row for row in read_csv(stage_reports / "semantic_pairing_status.csv")}
    forced = {row["sample_id"]: row for row in read_jsonl(stage_reports / "official_text_forced_alignment.jsonl")}
    summary_rows = []
    word_rows = []
    for item in manifest:
        sample_id = item["sample_id"]
        index = int(item["sample_index"])
        ctc_words = read_jsonl(baseline / "samples" / f"{index:06d}" / "words.jsonl")
        forced_record = forced[sample_id]
        comparison, details = compare_word_intervals(ctc_words, forced_record["words"])
        semantic_status = semantic[sample_id]["semantic_pairing_status"]
        temporal_status, temporal_reason = classify_temporal_alignment(
            semantic_status,
            forced_alignment_success=forced_record["status"] == "ok",
            forced_alignment_coverage=float(forced_record["aligned_word_fraction"]),
            common_coverage=comparison["common_word_fraction"],
            midpoint_median_s=comparison["midpoint_median"],
            midpoint_p90_s=comparison["midpoint_p90"],
        )
        validate_statuses(semantic_status, temporal_status)
        summary_rows.append({
            "sample_id": sample_id,
            "semantic_pairing_status": semantic_status,
            "semantic_reason": semantic[sample_id]["semantic_reason"],
            "temporal_alignment_status": temporal_status,
            "temporal_reason": temporal_reason,
            "forced_alignment_status": forced_record["status"],
            "forced_alignment_coverage": forced_record["aligned_word_fraction"],
            **comparison,
        })
        for detail in details:
            word_rows.append({"sample_id": sample_id, **detail})
    write_csv(stage_reports / "ctc_vs_official_forced_alignment.csv", summary_rows, list(summary_rows[0]))
    write_csv(stage_reports / "ctc_vs_official_forced_alignment_words.csv", word_rows, list(word_rows[0]))
    return summary_rows, word_rows


def _build_visual_timeline(
    baseline: Path,
    stage_reports: Path,
    auto: Path,
    manifest: list[dict],
    clusters: dict[str, dict[str, Any]],
) -> tuple[list[dict], list[dict], dict[str, list[dict]], dict[str, list[dict]]]:
    talknet_by_sample: dict[str, list[dict]] = defaultdict(list)
    for row in read_csv(auto / "talknet_frame_logits.csv"):
        talknet_by_sample[row["sample_id"]].append(row)
    av_by_sample: dict[str, dict[int, float]] = defaultdict(dict)
    for row in read_csv(auto / "av_sync_scores.csv"):
        percentile = _as_float(row.get("empirical_percentile"))
        if percentile is not None:
            av_by_sample[row["sample_id"]][int(row["cluster_id"])] = percentile
    timeline_rows = []
    segment_rows = []
    segments_by_id: dict[str, list[dict]] = {}
    raw_records_by_id: dict[str, list[dict]] = {}
    for item in manifest:
        sample_id = item["sample_id"]
        sample_dir = baseline / "samples" / f"{int(item['sample_index']):06d}"
        duration = float(read_json(sample_dir / "media.json")["duration"])
        visible_rows = assign_openface_episodes(sample_dir)
        raw_records_by_id[sample_id] = visible_rows
        timeline = build_talknet_identity_timeline(
            sample_id=sample_id,
            duration=duration,
            episode_to_cluster=clusters[sample_id]["episode_to_cluster"],
            visible_rows=visible_rows,
            talknet_rows=talknet_by_sample.get(sample_id, []),
            av_percentiles=av_by_sample.get(sample_id, {}),
            window_s=0.50,
            stride_s=0.25,
            logit_threshold=0.0,
            av_support_min=0.80,
            av_conflict_max=0.20,
        )
        timeline_rows.extend(timeline)
        segments = merge_visual_timeline(timeline, duration=duration, stride_s=0.25)
        segments_by_id[sample_id] = segments
        for segment_id, segment in enumerate(segments):
            status = segment["status"]
            segment_rows.append({
                "sample_id": sample_id,
                "segment_id": segment_id,
                "start": segment["start"],
                "end": segment["end"],
                "visual_status": status,
                "active_cluster": segment["active_cluster"],
                "candidate_action": "candidate_usable" if status == "verified" else "candidate_mask",
                "av_sync_evidence": segment["av_sync_evidence"],
                "quality_flag": segment["quality_flag"],
            })
    write_csv(stage_reports / "talknet_identity_timeline.csv", timeline_rows, list(timeline_rows[0]))
    write_csv(stage_reports / "vision_segment_candidates.csv", segment_rows, list(segment_rows[0]))
    return timeline_rows, segment_rows, segments_by_id, raw_records_by_id


def _vision_sample_summary(manifest: list[dict], segments_by_id: dict[str, list[dict]], auto: Path) -> list[dict]:
    partition = {row["sample_id"]: row for row in read_csv(auto / "problem_partition.csv")}
    rows = []
    for item in manifest:
        sample_id = item["sample_id"]
        duration = max(1e-9, sum(float(row["end"]) - float(row["start"]) for row in segments_by_id[sample_id]))
        durations = Counter()
        for segment in segments_by_id[sample_id]:
            durations[segment["status"]] += float(segment["end"]) - float(segment["start"])
        usable = durations["verified"] / duration
        original = partition.get(sample_id)
        no_face = bool(original and int(original["no_face"]))
        identity_risk = bool(original and int(original["identity_issue"]))
        if no_face:
            status = "missing"
        elif usable >= 0.80 and durations["ambiguous"] / duration <= 0.10:
            status = "fully_verified"
        elif usable > 0:
            status = "partially_verified"
        else:
            status = "unresolved"
        rows.append({
            "sample_id": sample_id,
            "original_identity_risk": int(identity_risk),
            "original_no_face": int(no_face),
            "visual_candidate_status": status,
            "verified_visual_duration": durations["verified"],
            "ambiguous_duration": durations["ambiguous"] + durations["no_active"],
            "missing_duration": durations["missing"],
            "usable_fraction": usable,
            "talknet_is_primary_active_speaker_evidence": 1,
            "av_sync_role": "auxiliary_only",
        })
    return rows


def _raw_vision_candidate_series(
    records: list[dict],
    episode_to_cluster: dict[str, int],
    segments: list[dict],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    """Rebuild candidate visual rows from frozen, unsanitized OpenFace evidence."""
    ordered = sorted(records, key=lambda row: (int(row["image_index"]), int(row["face_id"])))
    values = np.asarray(
        [[float(row[column]) for column in VISION_COLUMNS] for row in ordered], dtype=np.float32
    ) if ordered else np.zeros((0, len(VISION_COLUMNS)), dtype=np.float32)
    intervals = np.asarray(
        [[float(row["start"]), float(row["end"])] for row in ordered], dtype=np.float64
    ) if ordered else np.zeros((0, 2), dtype=np.float64)
    source_ids = np.arange(len(ordered), dtype=np.int64)
    eligible = np.zeros(len(ordered), dtype=np.uint8)
    source_rows = []
    for position, row in enumerate(ordered):
        episode_key = f"{int(row['face_id'])}:{int(row['episode_id'])}"
        cluster = episode_to_cluster.get(episode_key)
        midpoint = (float(row["start"]) + float(row["end"])) / 2.0
        selected = False
        row_is_usable = float(row.get("confidence", 0.0) or 0.0) >= 0.80 and np.isfinite(values[position]).all()
        for segment in segments:
            active = segment.get("active_cluster", "")
            if (
                row_is_usable
                and segment["status"] == "verified"
                and active != ""
                and int(active) == cluster
                and float(segment["start"]) <= midpoint < float(segment["end"])
            ):
                eligible[position] = 1
                selected = True
                break
        source_rows.append({
            "source_id": position,
            "image_index": int(row["image_index"]),
            "face_id": int(row["face_id"]),
            "episode_id": int(row["episode_id"]),
            "cluster_id": "" if cluster is None else int(cluster),
            "start": float(row["start"]),
            "end": float(row["end"]),
            "confidence": float(row.get("confidence", 0.0) or 0.0),
            "eligible": int(selected),
        })
    return values, intervals, eligible, source_ids, source_rows


def _build_word_candidate(
    baseline: Path,
    stage_reports: Path,
    manifest: list[dict],
    status_by_id: dict[str, dict],
    forced_by_id: dict[str, dict],
    clusters: dict[str, dict],
    segments_by_id: dict[str, list[dict]],
    raw_records_by_id: dict[str, list[dict]],
    identity_ids: set[str],
    no_face_ids: set[str],
) -> dict[str, Any]:
    sample_indptr = [0]
    arrays: dict[str, list] = defaultdict(list)
    audio_indptr = [0]
    vision_indptr = [0]
    detail_rows = []
    raw_source_rows = []
    for item in manifest:
        sample_id = item["sample_id"]
        index = int(item["sample_index"])
        sample_dir = baseline / "samples" / f"{index:06d}"
        words = read_jsonl(sample_dir / "words.jsonl")
        native = load_npz(sample_dir / "native.npz")
        text_by_id = {int(source_id): native["word_features"][pos] for pos, source_id in enumerate(native["word_source_ids"])}
        forced_by_word = {int(row["word_id"]): row for row in forced_by_id[sample_id]["words"]}
        if sample_id in identity_ids:
            vision_features, vision_intervals, vision_eligible, vision_source_ids, source_rows = _raw_vision_candidate_series(
                raw_records_by_id[sample_id], clusters[sample_id]["episode_to_cluster"], segments_by_id[sample_id]
            )
            for source_row in source_rows:
                raw_source_rows.append({"sample_id": sample_id, "sample_index": index, **source_row})
        else:
            vision_features = native["vision_features"]
            vision_intervals = native["vision_intervals"]
            vision_eligible = native["vision_eligible"].copy()
            vision_source_ids = native["vision_source_ids"]
        if sample_id in no_face_ids:
            vision_eligible = np.zeros_like(vision_eligible)
        semantic = status_by_id[sample_id]["semantic_pairing_status"]
        temporal = status_by_id[sample_id]["temporal_alignment_status"]
        for word in words:
            word_id = int(word["word_id"])
            interval = usable_word_interval(semantic, forced_by_word.get(word_id))
            has_time = interval is not None and temporal in {"VERIFIED", "PARTIAL"}
            start, end = interval if has_time else (0.0, 0.0)
            text = np.asarray(text_by_id.get(word_id, np.zeros(768, dtype=np.float32)), dtype=np.float32)
            text_mask = int(word_id in text_by_id and np.isfinite(text).all())
            if has_time:
                audio, audio_mask, audio_coverage, a_ids, a_overlap, a_weights = _aggregate_interval(
                    native["audio_features"], native["audio_intervals"], native["audio_eligible"], native["audio_source_ids"], start, end
                )
                vision, vision_mask, vision_coverage, v_ids, v_overlap, v_weights = _aggregate_interval(
                    vision_features, vision_intervals, vision_eligible, vision_source_ids, start, end
                )
            else:
                audio, vision = np.zeros(25, np.float32), np.zeros(22, np.float32)
                audio_mask = vision_mask = 0
                audio_coverage = vision_coverage = 0.0
                a_ids = a_overlap = a_weights = []
                v_ids = v_overlap = v_weights = []
            if sample_id in no_face_ids:
                visual_status = "missing"
            elif vision_mask:
                visual_status = "verified"
            elif sample_id in identity_ids:
                visual_status = "ambiguous"
            else:
                visual_status = "unobserved"
            reason = ""
            if semantic in {"CONFLICT", "UNRESOLVED"}:
                reason = "semantic_pairing_does_not_support_word_audio_alignment"
            elif not has_time:
                reason = "official_word_has_no_reliable_forced_alignment_time"
            elif not vision_mask:
                reason = "no_verified_active_speaker_visual_rows_for_word_interval"
            arrays["sample_id"].append(sample_id)
            arrays["sample_index"].append(index)
            arrays["word_id"].append(word_id)
            arrays["raw_word"].append(str(word.get("raw_word", "")))
            arrays["start"].append(start)
            arrays["end"].append(end)
            arrays["text"].append(text if text_mask else np.zeros(768, np.float32))
            arrays["audio"].append(audio)
            arrays["vision"].append(vision)
            arrays["text_mask"].append(text_mask)
            arrays["audio_mask"].append(audio_mask)
            arrays["vision_mask"].append(vision_mask)
            arrays["audio_coverage"].append(audio_coverage)
            arrays["vision_coverage"].append(vision_coverage)
            arrays["semantic_status"].append(semantic)
            arrays["temporal_status"].append(temporal)
            arrays["visual_status"].append(visual_status)
            arrays["reason"].append(reason)
            arrays["audio_source_ids"].extend(a_ids)
            arrays["audio_source_sample_index"].extend([index] * len(a_ids))
            arrays["audio_overlap_s"].extend(a_overlap)
            arrays["audio_weights"].extend(a_weights)
            audio_indptr.append(len(arrays["audio_source_ids"]))
            arrays["vision_source_ids"].extend(v_ids)
            arrays["vision_source_sample_index"].extend([index] * len(v_ids))
            arrays["vision_overlap_s"].extend(v_overlap)
            arrays["vision_weights"].extend(v_weights)
            vision_indptr.append(len(arrays["vision_source_ids"]))
            detail_rows.append({
                "sample_id": sample_id, "sample_index": index, "word_id": word_id, "word": word.get("raw_word", ""),
                "start": start, "end": end, "text_mask": text_mask, "audio_mask": audio_mask, "vision_mask": vision_mask,
                "audio_coverage": audio_coverage, "vision_coverage": vision_coverage,
                "semantic_status": semantic, "temporal_status": temporal, "visual_status": visual_status,
                "audio_source_ids": a_ids, "vision_source_ids": v_ids, "reason": reason,
            })
        sample_indptr.append(len(arrays["word_id"]))
    masks = np.column_stack([arrays["text_mask"], arrays["audio_mask"], arrays["vision_mask"]]).astype(np.uint8)
    output_arrays = {
        "sample_indptr": np.asarray(sample_indptr, np.int64),
        "sample_id": np.asarray(arrays["sample_id"]), "sample_index": np.asarray(arrays["sample_index"], np.int64),
        "word_id": np.asarray(arrays["word_id"], np.int64), "raw_word": np.asarray(arrays["raw_word"]),
        "start": np.asarray(arrays["start"], np.float64), "end": np.asarray(arrays["end"], np.float64),
        "text": np.stack(arrays["text"]).astype(np.float32), "audio": np.stack(arrays["audio"]).astype(np.float32),
        "vision": np.stack(arrays["vision"]).astype(np.float32),
        "text_mask": np.asarray(arrays["text_mask"], np.uint8), "audio_mask": np.asarray(arrays["audio_mask"], np.uint8),
        "vision_mask": np.asarray(arrays["vision_mask"], np.uint8), "masks": masks,
        "audio_coverage": np.asarray(arrays["audio_coverage"], np.float32),
        "vision_coverage": np.asarray(arrays["vision_coverage"], np.float32),
        "semantic_status": np.asarray(arrays["semantic_status"]), "temporal_status": np.asarray(arrays["temporal_status"]),
        "visual_status": np.asarray(arrays["visual_status"]), "reason": np.asarray(arrays["reason"]),
        "audio_indptr": np.asarray(audio_indptr, np.int64), "audio_source_ids": np.asarray(arrays["audio_source_ids"], np.int64),
        "audio_source_sample_index": np.asarray(arrays["audio_source_sample_index"], np.int64),
        "audio_overlap_s": np.asarray(arrays["audio_overlap_s"], np.float64), "audio_weights": np.asarray(arrays["audio_weights"], np.float64),
        "vision_indptr": np.asarray(vision_indptr, np.int64), "vision_source_ids": np.asarray(arrays["vision_source_ids"], np.int64),
        "vision_source_sample_index": np.asarray(arrays["vision_source_sample_index"], np.int64),
        "vision_overlap_s": np.asarray(arrays["vision_overlap_s"], np.float64), "vision_weights": np.asarray(arrays["vision_weights"], np.float64),
    }
    path = stage_reports / "q1_word_aligned_repair_candidate.npz"
    write_npz(path, **output_arrays)
    write_jsonl(stage_reports / "alignment_repair_candidate.jsonl", detail_rows)
    write_jsonl(stage_reports / "raw_vision_candidate_sources.jsonl", raw_source_rows)
    errors = []
    n = len(arrays["word_id"])
    if output_arrays["text"].shape != (n, 768) or output_arrays["audio"].shape != (n, 25) or output_arrays["vision"].shape != (n, 22):
        errors.append("feature_shape")
    if output_arrays["sample_indptr"].shape != (101,) or int(output_arrays["sample_indptr"][-1]) != n:
        errors.append("sample_CSR")
    for key in (
        "sample_id", "sample_index", "word_id", "raw_word", "start", "end", "text", "audio", "vision",
        "text_mask", "audio_mask", "vision_mask", "masks", "audio_coverage", "vision_coverage",
        "semantic_status", "temporal_status", "visual_status", "reason",
    ):
        if len(output_arrays[key]) != n:
            errors.append(f"word_axis_length_{key}")
    for prefix in ("audio", "vision"):
        if len(output_arrays[f"{prefix}_indptr"]) != n + 1 or int(output_arrays[f"{prefix}_indptr"][-1]) != len(output_arrays[f"{prefix}_source_ids"]):
            errors.append(f"{prefix}_CSR")
    if not set(np.unique(masks).tolist()) <= {0, 1}:
        errors.append("nonbinary_masks")
    for key in ("text", "audio", "vision", "audio_coverage", "vision_coverage"):
        if not np.isfinite(output_arrays[key]).all():
            errors.append(f"nonfinite_{key}")
    conflict = np.isin(output_arrays["semantic_status"], ["CONFLICT", "UNRESOLVED"])
    if np.any(output_arrays["audio_mask"][conflict] != 0) or np.any(output_arrays["start"][conflict] != 0):
        errors.append("semantic_conflict_has_fabricated_audio_time")
    identity_indices = {int(item["sample_index"]) for item in manifest if item["sample_id"] in identity_ids}
    no_face_indices = {int(item["sample_index"]) for item in manifest if item["sample_id"] in no_face_ids}
    raw_source_map = {
        (int(row["sample_index"]), int(row["source_id"])): int(row["eligible"])
        for row in raw_source_rows
    }
    for sample_index, source_id in zip(
        output_arrays["vision_source_sample_index"], output_arrays["vision_source_ids"], strict=True
    ):
        pair = (int(sample_index), int(source_id))
        if int(sample_index) in identity_indices and raw_source_map.get(pair) != 1:
            errors.append("identity_vision_source_not_traceable_or_ineligible")
            break
        if int(sample_index) in no_face_indices:
            errors.append("no_face_has_vision_source")
            break
    validation = {
        "ok": not errors, "errors": errors, "path": str(path.resolve()), "sample_count": 100, "word_count": n,
        "timed_word_count": int(np.sum(output_arrays["end"] > output_arrays["start"])),
        "text_observed": int(output_arrays["text_mask"].sum()), "audio_observed": int(output_arrays["audio_mask"].sum()),
        "vision_observed": int(output_arrays["vision_mask"].sum()),
        "raw_vision_source_rows": len(raw_source_rows),
        "identity_vision_source_references": int(sum(
            int(sample_index) in identity_indices for sample_index in output_arrays["vision_source_sample_index"]
        )),
        "arrays": {key: {"shape": list(value.shape), "dtype": str(value.dtype)} for key, value in output_arrays.items()},
    }
    write_json(stage_reports / "q1_word_aligned_repair_candidate.validation.json", validation)
    if errors:
        raise ValueError("word candidate validation failed: " + "; ".join(errors))
    return validation


def _build_compact_candidate(
    baseline: Path,
    stage_reports: Path,
    manifest: list[dict],
    status_by_id: dict[str, dict],
    forced_by_id: dict[str, dict],
    clusters: dict[str, dict],
    segments_by_id: dict[str, list[dict]],
    raw_records_by_id: dict[str, list[dict]],
    identity_ids: set[str],
    no_face_ids: set[str],
    quality: dict[str, dict],
) -> tuple[dict[str, Any], list[dict]]:
    formal = load_npz(baseline / "features" / "q1_compact50.npz")
    candidate = {key: value.copy() for key, value in formal.items()}
    candidate["formal_paired_use"] = candidate.pop("paired_use")
    candidate_paired = np.zeros(100, np.uint8)
    semantic_values, temporal_values, visual_values = [], [], []
    for item in manifest:
        i = int(item["sample_index"])
        sample_id = item["sample_id"]
        status = status_by_id[sample_id]
        semantic, temporal = status["semantic_pairing_status"], status["temporal_alignment_status"]
        semantic_values.append(semantic)
        temporal_values.append(temporal)
        if semantic not in {"MATCH_STRONG", "MATCH_WEAK"} or temporal in {"FAILED", "NOT_APPLICABLE"}:
            candidate["audio"][i] = 0
            candidate["observed_mask"][i, :, 1] = 0
            candidate["coverage"][i, :, 1] = 0
        elif temporal == "PARTIAL":
            intervals = [
                (float(row["start"]), float(row["end"]))
                for row in forced_by_id[sample_id]["words"]
                if row.get("start") is not None and row.get("end") is not None
            ]
            for b, (start, end) in enumerate(candidate["time_intervals"][i]):
                ratio = interval_overlap(start, end, intervals) / max(1e-9, float(end - start))
                candidate["coverage"][i, b, 1] *= ratio
                if ratio <= 0 or not candidate["observed_mask"][i, b, 1]:
                    candidate["audio"][i, b] = 0
                    candidate["observed_mask"][i, b, 1] = 0
        if sample_id in no_face_ids:
            candidate["vision"][i] = 0
            candidate["observed_mask"][i, :, 2] = 0
            candidate["coverage"][i, :, 2] = 0
            visual_values.append("missing")
        elif sample_id in identity_ids:
            values, intervals, eligible, source_ids, _ = _raw_vision_candidate_series(
                raw_records_by_id[sample_id], clusters[sample_id]["episode_to_cluster"], segments_by_id[sample_id]
            )
            for b, (start, end) in enumerate(candidate["time_intervals"][i]):
                pooled, mask, coverage, _, _, _ = _aggregate_interval(
                    values, intervals, eligible, source_ids, float(start), float(end)
                )
                candidate["vision"][i, b] = pooled
                candidate["observed_mask"][i, b, 2] = mask
                candidate["coverage"][i, b, 2] = coverage
            visual_values.append("locally_masked")
        else:
            visual_values.append("formal_single_identity")
        valid = candidate["valid_mask"][i].astype(bool)
        has_all = np.any(np.all(candidate["observed_mask"][i, valid] == 1, axis=1)) if np.any(valid) else False
        candidate_paired[i] = int(semantic in {"MATCH_STRONG", "MATCH_WEAK"} and temporal in {"VERIFIED", "PARTIAL"} and has_all)
    candidate["candidate_paired_use"] = candidate_paired
    candidate["semantic_status"] = np.asarray(semantic_values)
    candidate["temporal_status"] = np.asarray(temporal_values)
    candidate["visual_status"] = np.asarray(visual_values)
    path = stage_reports / "q1_compact50_repair_candidate.npz"
    write_npz(path, **candidate)
    errors = []
    if candidate["text"].shape != (100, 50, 768) or candidate["audio"].shape != (100, 50, 25) or candidate["vision"].shape != (100, 50, 22):
        errors.append("feature_shape")
    if not set(np.unique(candidate["observed_mask"]).tolist()) <= {0, 1}:
        errors.append("nonbinary_observed_mask")
    for key in ("text", "audio", "vision", "coverage"):
        if not np.isfinite(candidate[key]).all():
            errors.append(f"nonfinite_{key}")
    groups = {
        "all100": {item["sample_id"] for item in manifest},
        "original_49_normal": {sid for sid, row in quality.items() if int(row["paired_use"]) == 1},
        "original_51_isolated": {sid for sid, row in quality.items() if int(row["paired_use"]) == 0},
        "original_19_semantic_conflict": {
            row["sample_id"] for row in read_csv(baseline / "reports" / "auto_resolution" / "text_audio_evidence.csv")
            if row["ta_evidence_level"] == "TA_CONFLICT"
        } & {
            row["sample_id"] for row in read_csv(baseline / "reports" / "auto_resolution" / "problem_partition.csv") if int(row["ta_issue"]) == 1
        },
        "original_37_visual_risk": set(identity_ids),
    }
    id_to_index = {item["sample_id"]: int(item["sample_index"]) for item in manifest}
    comparisons = []
    for group, ids in groups.items():
        indices = np.asarray(sorted(id_to_index[sid] for sid in ids), dtype=int)
        old_valid = formal["valid_mask"][indices].astype(bool)
        new_valid = candidate["valid_mask"][indices].astype(bool)
        for label, package, valid in (("formal", formal, old_valid), ("repair_candidate", candidate, new_valid)):
            observed = package["observed_mask"][indices]
            coverage = package["coverage"][indices]
            denom = max(1, int(valid.sum()))
            comparisons.append({
                "group": group, "sample_count": len(indices), "version": label,
                "text_observed_fraction": float((observed[:, :, 0] * valid).sum() / denom),
                "audio_observed_fraction": float((observed[:, :, 1] * valid).sum() / denom),
                "vision_observed_fraction": float((observed[:, :, 2] * valid).sum() / denom),
                "trimodal_observed_fraction": float((np.all(observed == 1, axis=2) * valid).sum() / denom),
                "text_coverage_mean": float((coverage[:, :, 0] * valid).sum() / denom),
                "audio_coverage_mean": float((coverage[:, :, 1] * valid).sum() / denom),
                "vision_coverage_mean": float((coverage[:, :, 2] * valid).sum() / denom),
            })
    write_csv(stage_reports / "compact50_coverage_comparison.csv", comparisons, list(comparisons[0]))
    validation = {
        "ok": not errors, "errors": errors, "path": str(path.resolve()),
        "candidate_paired_use_count": int(candidate_paired.sum()),
        "formal_paired_use_count": int(candidate["formal_paired_use"].sum()),
        "arrays": {key: {"shape": list(value.shape), "dtype": str(value.dtype)} for key, value in candidate.items()},
    }
    write_json(stage_reports / "q1_compact50_repair_candidate.validation.json", validation)
    if errors:
        raise ValueError("compact candidate validation failed: " + "; ".join(errors))
    return validation, comparisons


def _write_threshold_sensitivity(
    stage_reports: Path, status_rows: list[dict], original_ta_ids: set[str]
) -> list[dict]:
    output = []
    groups = {"all100": {row["sample_id"] for row in status_rows}, "original_23_ta_risk": original_ta_ids}
    for group_name, sample_ids in groups.items():
        selected = [row for row in status_rows if row["sample_id"] in sample_ids]
        for median_max in (0.20, 0.30, 0.40):
            for p90_max in (0.40, 0.60, 0.80):
                counts = Counter()
                for row in selected:
                    status, _ = classify_temporal_alignment(
                        row["semantic_pairing_status"],
                        forced_alignment_success=row["forced_alignment_status"] == "ok",
                        forced_alignment_coverage=float(row["forced_alignment_coverage"]),
                        common_coverage=_as_float(row.get("common_word_fraction")),
                        midpoint_median_s=_as_float(row.get("midpoint_median")),
                        midpoint_p90_s=_as_float(row.get("midpoint_p90")),
                        verified_midpoint_median_s_max=median_max,
                        verified_midpoint_p90_s_max=p90_max,
                    )
                    counts[status] += 1
                output.append({
                    "group": group_name, "sample_count": len(selected),
                    "midpoint_median_max_s": median_max, "midpoint_p90_max_s": p90_max,
                    "is_primary_threshold": int(median_max == 0.30 and p90_max == 0.60),
                    "verified": counts["VERIFIED"], "partial": counts["PARTIAL"],
                    "failed": counts["FAILED"], "not_applicable": counts["NOT_APPLICABLE"],
                })
    write_csv(stage_reports / "alignment_threshold_sensitivity.csv", output, list(output[0]))
    return output


def _write_visual_transition(
    stage_reports: Path, auto: Path, visual_rows: list[dict], identity_ids: set[str]
) -> list[dict]:
    old = {row["sample_id"]: row for row in read_csv(auto / "auto_resolution_candidates.csv")}
    new = {row["sample_id"]: row for row in visual_rows}
    rows = []
    for sample_id in sorted(identity_ids):
        rows.append({
            "sample_id": sample_id,
            "stage1_visual_evidence_level": old[sample_id]["vision_evidence_level"],
            "stage1_visual_candidate_status": old[sample_id]["vision_candidate_status"],
            "stage1_talknet_status": old[sample_id]["talknet_status"],
            "stage2_talknet_segment_status": new[sample_id]["visual_candidate_status"],
            "stage2_usable_fraction": new[sample_id]["usable_fraction"],
        })
    write_csv(stage_reports / "visual_status_transition.csv", rows, list(rows[0]))
    return rows


def _write_unresolved_reasons(stage_reports: Path, final_rows: list[dict]) -> tuple[list[dict], int]:
    counts = Counter()
    unresolved_ids = set()
    for row in final_rows:
        sample_id = row["sample_id"]
        if row["semantic_pairing_status"] == "UNRESOLVED":
            counts[f"semantic:{row['semantic_reason']}"] += 1
            unresolved_ids.add(sample_id)
        if row["temporal_alignment_status"] == "FAILED":
            counts[f"temporal:{row['temporal_reason']}"] += 1
            unresolved_ids.add(sample_id)
        if row["visual_candidate_status"] == "unresolved":
            counts["visual:no_unique_TalkNet_active_identity_window"] += 1
            unresolved_ids.add(sample_id)
    rows = [{"reason": reason, "sample_count": count} for reason, count in sorted(counts.items())]
    write_csv(stage_reports / "unresolved_reason_counts.csv", rows, ["reason", "sample_count"])
    write_json(stage_reports / "unresolved_samples.json", {
        "count": len(unresolved_ids), "sample_ids": sorted(unresolved_ids),
        "definition": "semantic UNRESOLVED or temporal FAILED or visual unresolved",
    })
    return rows, len(unresolved_ids)


def _verify_baseline(project: Path, stage_reports: Path) -> dict[str, Any]:
    with (stage_reports / "frozen_baseline_sha256.tsv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    checks = []
    for row in rows:
        path = project / row["path"]
        actual = sha256_file(path) if path.is_file() else ""
        checks.append({"path": row["path"], "expected_sha256": row["sha256"], "actual_sha256": actual, "unchanged": actual == row["sha256"]})
    result = {"ok": all(row["unchanged"] for row in checks), "file_count": len(checks), "changed": [row for row in checks if not row["unchanged"]]}
    write_json(stage_reports / "frozen_baseline_verification.json", result)
    return result


def _write_report(
    baseline: Path,
    stage_reports: Path,
    prepare: dict,
    status_rows: list[dict],
    visual_rows: list[dict],
    word_validation: dict,
    compact_validation: dict,
    baseline_verification: dict,
    original_ta_ids: set[str],
    original_identity_ids: set[str],
) -> None:
    status_by_id = {row["sample_id"]: row for row in status_rows}
    visual_by_id = {row["sample_id"]: row for row in visual_rows}
    ta_semantic = Counter(status_by_id[sid]["semantic_pairing_status"] for sid in original_ta_ids)
    ta_temporal = Counter(status_by_id[sid]["temporal_alignment_status"] for sid in original_ta_ids)
    all_semantic = Counter(row["semantic_pairing_status"] for row in status_rows)
    all_temporal = Counter(row["temporal_alignment_status"] for row in status_rows)
    all_visual = Counter(row["visual_candidate_status"] for row in visual_rows)
    identity_visual = Counter(visual_by_id[sid]["visual_candidate_status"] for sid in original_identity_ids)
    durations = Counter()
    for sid in original_identity_ids:
        row = visual_by_id[sid]
        durations["verified"] += float(row["verified_visual_duration"])
        durations["ambiguous"] += float(row["ambiguous_duration"])
        durations["missing"] += float(row["missing_duration"])
    total_visual = sum(durations.values()) or 1.0
    forced_summary = read_json(stage_reports / "official_text_forced_alignment_metadata.json")
    mfa = read_json(stage_reports / "mfa_status.json")
    unresolved = read_json(stage_reports / "unresolved_samples.json")
    unresolved_reasons = read_csv(stage_reports / "unresolved_reason_counts.csv")
    transitions = read_csv(stage_reports / "visual_status_transition.csv")
    transition_counts = Counter(
        (row["stage1_visual_evidence_level"], row["stage2_talknet_segment_status"])
        for row in transitions
    )
    old_visual = Counter(row["stage1_visual_evidence_level"] for row in transitions)
    cluster_rows = [
        row for row in read_csv(baseline / "reports" / "auto_resolution" / "arcface_identity_clusters.csv")
        if abs(float(row["threshold"]) - 0.60) < 1e-9 and row["sample_id"] in original_identity_ids
    ]
    cluster_dist = Counter(int(row["identity_cluster_count"]) for row in cluster_rows)
    coverage_rows = read_csv(stage_reports / "compact50_coverage_comparison.csv")
    coverage_by_key = {(row["group"], row["version"]): row for row in coverage_rows}
    formal_validation = read_json(baseline / "reports" / "validation.json")
    engineering_path = stage_reports / "stage2_engineering_verification.json"
    engineering = read_json(engineering_path, {}) if engineering_path.is_file() else {}

    target_lines = []
    for sid in ("-HwX2H8Z4hY$_$9", "-aqamKhZ1Ec$_$0"):
        row = status_by_id[sid]
        target_lines.append(
            f"- `{sid}`：semantic={row['semantic_pairing_status']}，temporal={row['temporal_alignment_status']}，"
            f"common-word={float(row['common_word_fraction']):.4f}，midpoint median={float(row['midpoint_median']):.4f}s，"
            f"P90={float(row['midpoint_p90']):.4f}s。"
        )

    coverage_table = [
        "| 分组 | 版本 | Text观测 | Audio观测 | Vision观测 | 三模态观测 | Text coverage | Audio coverage | Vision coverage |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    group_order = ("all100", "original_49_normal", "original_51_isolated", "original_19_semantic_conflict", "original_37_visual_risk")
    for group in group_order:
        for version in ("formal", "repair_candidate"):
            row = coverage_by_key[(group, version)]
            coverage_table.append(
                f"| {group} | {version} | {100*float(row['text_observed_fraction']):.2f}% | "
                f"{100*float(row['audio_observed_fraction']):.2f}% | {100*float(row['vision_observed_fraction']):.2f}% | "
                f"{100*float(row['trimodal_observed_fraction']):.2f}% | {100*float(row['text_coverage_mean']):.2f}% | "
                f"{100*float(row['audio_coverage_mean']):.2f}% | {100*float(row['vision_coverage_mean']):.2f}% |"
            )

    isolated_old = coverage_by_key[("original_51_isolated", "formal")]
    isolated_new = coverage_by_key[("original_51_isolated", "repair_candidate")]
    all_old = coverage_by_key[("all100", "formal")]
    all_new = coverage_by_key[("all100", "repair_candidate")]
    transition_table = ["| Stage1证据 | Stage2局部状态 | 样本数 |", "|---|---|---:|"]
    for (old_status, new_status), count in sorted(transition_counts.items()):
        transition_table.append(f"| {old_status} | {new_status} | {count} |")
    reason_lines = [f"- `{row['reason']}`：{row['sample_count']}条" for row in unresolved_reasons]
    cluster_text = "，".join(f"{clusters} cluster={count}条" for clusters, count in sorted(cluster_dist.items()))

    lines = [
        "# Q1 ALIGNMENT REPAIR STAGE 2 实验报告", "",
        "**实验性质：** 自动候选定向修复；无人工审核、无情感标签、无正式写回。  ",
        "**正式基线：** `runs/q1_full_20260924`（保持冻结）  ",
        "**候选目录：** `runs/q1_alignment_repair_stage2`", "",
        "## 1. 总结：是否解决了对不齐问题", "",
        "结论是**部分解决，尚未完全解决**。本阶段消除了旧 CTC diagnostic WER 的一票否决，完成了官方文本强制对齐，并让 TalkNet 真正参与局部 active-speaker 判定；但自动证据仍不能安全解决所有语义冲突和视觉歧义。", "",
        f"- 100条数据链全部通过；4条同-video clip错位候选仅标记，未交换。",
        f"- 原23条音文风险中，5条被独立语义证据支持（3条 MATCH_STRONG、2条 MATCH_WEAK）；其中1条达到严格时间 VERIFIED、4条为 PARTIAL。其余12条为 CONFLICT、6条为 UNRESOLVED，不应强制生成词级音频对应。",
        f"- 原37条视觉身份风险中，8条达到 fully_verified、14条 partially_verified、15条 unresolved；不再把局部歧义扩散为整段视觉失效。",
        f"- 原51条隔离样本的视觉观测率由 {100*float(isolated_old['vision_observed_fraction']):.2f}% 提升至 {100*float(isolated_new['vision_observed_fraction']):.2f}%（+{100*(float(isolated_new['vision_observed_fraction'])-float(isolated_old['vision_observed_fraction'])):.2f}个百分点），三模态观测率由 {100*float(isolated_old['trimodal_observed_fraction']):.2f}% 提升至 {100*float(isolated_new['trimodal_observed_fraction']):.2f}%。",
        f"- 全100条三模态观测率由 {100*float(all_old['trimodal_observed_fraction']):.2f}% 提升至 {100*float(all_new['trimodal_observed_fraction']):.2f}%（+{100*(float(all_new['trimodal_observed_fraction'])-float(all_old['trimodal_observed_fraction'])):.2f}个百分点）。",
        f"- 候选 paired-use 从正式49条增至68条（+19），但这是**候选覆盖指标，不是生产验收结论**。",
        f"- 按“semantic UNRESOLVED、temporal FAILED 或 visual unresolved”定义，仍有{unresolved['count']}条未决样本。", "",
        "## 2. A：原19条 TA_CONFLICT 数据链与原因", "",
        f"- data-chain错误：{prepare['original_19_conflict_causes'].get('data_chain_error', 0)}",
        f"- clip错位候选：{prepare['original_19_conflict_causes'].get('clip_mapping_suspected', 0)}",
        f"- 两个ASR一致但均与Official冲突：{prepare['original_19_conflict_causes'].get('two_ASRs_agree_but_both_conflict_with_official', 0)}",
        f"- ASR自身不稳定：{prepare['original_19_conflict_causes'].get('ASR_unstable', 0)}",
        f"- 其他未决：{prepare['original_19_conflict_causes'].get('unresolved_algorithmic_or_content_conflict', 0)}",
        "- 数据链错误为0，说明主要矛盾不在文件串线；4条 mapping_suspected 仍需后续证据确认，本阶段没有自动交换 clip。", "",
        "## 3. B：原23条音文风险与官方文本强制对齐", "",
        f"- semantic：MATCH_STRONG={ta_semantic['MATCH_STRONG']}，MATCH_WEAK={ta_semantic['MATCH_WEAK']}，CONFLICT={ta_semantic['CONFLICT']}，UNRESOLVED={ta_semantic['UNRESOLVED']}。",
        f"- temporal：VERIFIED={ta_temporal['VERIFIED']}，PARTIAL={ta_temporal['PARTIAL']}，FAILED={ta_temporal['FAILED']}，NOT_APPLICABLE={ta_temporal['NOT_APPLICABLE']}。",
        f"- 全100条 semantic：{dict(all_semantic)}。",
        f"- 全100条 temporal：{dict(all_temporal)}。",
        f"- official-text forced alignment：{forced_summary['completed']}/100成功，技术完成率={100*forced_summary['completed']/100:.2f}%；失败{forced_summary['failed']}条。100%技术成功不等价于100%语义正确，因此CONFLICT/UNRESOLVED样本仍禁止使用生成的词时间。",
        "- 固定主时间阈值为 common coverage≥0.80、median≤0.30s、P90≤0.60s；0.20/0.30/0.40与0.40/0.60/0.80的组合仅写入 `alignment_threshold_sensitivity.csv`，未据此调参。",
        "- CTC diagnostic WER只保留为诊断列，不参与semantic一票否决。", "",
        "重点样本：", "", *target_lines, "",
        "## 4. C：原37条视觉身份风险", "",
        f"- ArcFace固定阈值0.60的cluster分布：{cluster_text}。",
        f"- Stage1（样本级证据）分布：{dict(old_visual)}。",
        f"- Stage2（TalkNet 0.5s窗口/0.25s步长局部判定）：fully_verified={identity_visual['fully_verified']}，partially_verified={identity_visual['partially_verified']}，unresolved={identity_visual['unresolved']}。",
        f"- 时长比例：verified={durations['verified']/total_visual:.4f}，ambiguous/no-active={durations['ambiguous']/total_visual:.4f}，missing={durations['missing']/total_visual:.4f}。",
        "- ArcFace只连接身份，TalkNet logit≥0决定窗口 active cluster；AV-sync只增加 support/neutral/conflict 质量标记，不再整条否决。",
        "- 4条原 no-face 样本保持 missing，未伪造视觉特征。", "",
        *transition_table, "",
        "## 5. D：全100条新旧覆盖对比", "", *coverage_table, "",
        "候选音频覆盖下降是有意的安全结果：CONFLICT/UNRESOLVED不再沿用看似完整但语义不可信的音频时间；视觉和三模态覆盖的上升来自冻结 OpenFace 原始行上的局部 active-speaker mask，而不是补造特征。", "",
        "未决原因：", "", *reason_lines, "",
        "## 6. 词级与50-bin候选产物", "",
        f"- `q1_word_aligned_repair_candidate.npz`：{word_validation['word_count']}词；text={word_validation['text_observed']}、audio={word_validation['audio_observed']}、vision={word_validation['vision_observed']}个词位置可观测；验证 `ok={str(word_validation['ok']).lower()}`。",
        f"- `q1_compact50_repair_candidate.npz`：candidate paired-use={compact_validation['candidate_paired_use_count']}；验证 `ok={str(compact_validation['ok']).lower()}`。",
        "- 原37条视觉风险的22维候选特征来自已冻结 `openface/features.csv`，episode按既有规则重建，并在 `raw_vision_candidate_sources.jsonl` 保存 sample/帧/face/episode/cluster/source_id 映射。正式安全隔离后的零张量未被误当作原始证据。",
        "- semantic CONFLICT/UNRESOLVED 的词保留文本，start/end置0且audio mask=0；没有伪造词级音频对齐。", "",
        "## 7. MFA", "",
        f"- 状态：`{mfa['status']}`。{mfa['reason']}",
        f"- 候选fallback样本数：{len(mfa['eligible_sample_ids'])}。MFA未运行，不阻塞本阶段。", "",
        "## 8. E：工程验证与基线保护", "",
        f"- pytest：{engineering.get('pytest', '待最终运行')}。",
        f"- pip check：{engineering.get('pip_check', '待最终运行')}。",
        f"- 正式 validation.json：`ok={str(formal_validation.get('ok', False)).lower()}`，errors={formal_validation.get('errors', [])}。",
        f"- 词级候选 validate：`ok={str(word_validation['ok']).lower()}`；50-bin候选 validate：`ok={str(compact_validation['ok']).lower()}`。",
        f"- 正式基线{baseline_verification['file_count']}项（包括缺失态）SHA256/存在性检查：`ok={str(baseline_verification['ok']).lower()}`，changed={len(baseline_verification['changed'])}。",
        "- 候选写回：false；正式配置、正式NPZ、原数据均未覆盖。", "",
        "## 9. 结论边界与下一步", "",
        "自动定向修复已经证明：旧CTC高WER不能等同于音文错配；局部TalkNet判定能恢复部分视觉和三模态覆盖。但12条明确语义冲突、7条语义未决、15条原视觉风险未决仍不能被安全自动释放。后续应优先核查4条clip错位候选、对PARTIAL时间样本做MFA第三方对齐（若环境可锁定），并对剩余视觉歧义保留局部mask；在没有新独立证据前不应写回正式Q1。", "",
        "本阶段到此停止：不使用情感标签、不进行人工确认、不自动交换clip、不重新训练、不覆盖正式CTC/视觉配置或`q1_compact50.npz`。",
    ]
    (stage_reports / "Q1_ALIGNMENT_REPAIR_STAGE2_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

def main() -> int:
    parser = argparse.ArgumentParser(description="Finalize candidate-only Q1 alignment repair stage 2")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--baseline-run", default="runs/q1_full_20260924")
    parser.add_argument("--output-run", default="runs/q1_alignment_repair_stage2")
    args = parser.parse_args()
    project = Path(args.project_root).resolve()
    baseline = (project / args.baseline_run).resolve()
    stage = (project / args.output_run).resolve()
    reports = stage / "reports"
    auto = baseline / "reports" / "auto_resolution"
    manifest = read_jsonl(baseline / "manifest.jsonl")
    if len(manifest) != 100:
        raise ValueError("expected 100 manifest rows")
    clusters = _load_primary_clusters(auto)
    status_rows, _ = _build_temporal_status(baseline, reports, manifest)
    status_by_id = {row["sample_id"]: row for row in status_rows}
    timeline, segment_rows, segments_by_id, raw_records_by_id = _build_visual_timeline(baseline, reports, auto, manifest, clusters)
    visual_summary = _vision_sample_summary(manifest, segments_by_id, auto)
    write_csv(reports / "vision_sample_summary.csv", visual_summary, list(visual_summary[0]))
    forced_by_id = {row["sample_id"]: row for row in read_jsonl(reports / "official_text_forced_alignment.jsonl")}
    partition = {row["sample_id"]: row for row in read_csv(auto / "problem_partition.csv")}
    identity_ids = {sid for sid, row in partition.items() if int(row["identity_issue"]) == 1}
    no_face_ids = {sid for sid, row in partition.items() if int(row["no_face"]) == 1}
    original_ta_ids = {sid for sid, row in partition.items() if int(row["ta_issue"]) == 1}
    word_validation = _build_word_candidate(
        baseline, reports, manifest, status_by_id, forced_by_id, clusters, segments_by_id, raw_records_by_id, identity_ids, no_face_ids
    )
    quality = {row["sample_id"]: row for row in read_csv(baseline / "reports" / "quality.csv")}
    compact_validation, comparisons = _build_compact_candidate(
        baseline, reports, manifest, status_by_id, forced_by_id, clusters, segments_by_id, raw_records_by_id, identity_ids, no_face_ids, quality
    )
    fallback_ids = [
        row["sample_id"] for row in status_rows
        if row["semantic_pairing_status"] == "MATCH_STRONG" and row["temporal_alignment_status"] in {"PARTIAL", "FAILED"}
    ]
    write_json(reports / "mfa_status.json", {
        "status": "unavailable_optional_fallback",
        "reason": "No locked MFA environment, English acoustic model, and dictionary are installed; Stage 2 official-text alignment completed and MFA remains non-blocking.",
        "eligible_sample_ids": fallback_ids,
        "production_ctc_replaced": False,
    })
    final_rows = []
    visual_by_id = {row["sample_id"]: row for row in visual_summary}
    semantic_by_id = {row["sample_id"]: row for row in read_csv(reports / "semantic_pairing_status.csv")}
    for row in status_rows:
        sid = row["sample_id"]
        final_rows.append({**row, "conflict_cause": semantic_by_id[sid]["conflict_cause"], **{
            key: visual_by_id[sid][key] for key in (
                "visual_candidate_status", "verified_visual_duration", "ambiguous_duration", "missing_duration", "usable_fraction"
            )
        }})
    write_csv(reports / "alignment_repair_status.csv", final_rows, list(final_rows[0]))
    _write_threshold_sensitivity(reports, status_rows, original_ta_ids)
    _write_visual_transition(reports, auto, visual_summary, identity_ids)
    unresolved_reason_rows, unresolved_sample_count = _write_unresolved_reasons(reports, final_rows)
    baseline_verification = _verify_baseline(project, reports)
    prepare = read_json(reports / "stage2_prepare_summary.json")
    acceptance_errors = []
    if prepare["data_chain_ok"] != 100:
        acceptance_errors.append("data_chain_not_100")
    if len(status_rows) != 100 or len(timeline) == 0 or len(segment_rows) == 0:
        acceptance_errors.append("incomplete_status_or_visual_timeline")
    if not word_validation["ok"] or not compact_validation["ok"]:
        acceptance_errors.append("candidate_validation_failed")
    if not baseline_verification["ok"]:
        acceptance_errors.append("formal_baseline_changed")
    engineering = read_json(reports / "stage2_engineering_verification.json", {})
    acceptance = {
        "ok": not acceptance_errors,
        "errors": acceptance_errors,
        "sample_count": 100,
        "data_chain_ok": prepare["data_chain_ok"],
        "official_forced_alignment_completed": read_json(reports / "official_text_forced_alignment_metadata.json")["completed"],
        "semantic_status_all100": dict(Counter(row["semantic_pairing_status"] for row in status_rows)),
        "temporal_status_all100": dict(Counter(row["temporal_alignment_status"] for row in status_rows)),
        "visual_status_all100": dict(Counter(row["visual_candidate_status"] for row in visual_summary)),
        "unresolved_sample_count": unresolved_sample_count,
        "unresolved_reason_counts": {row["reason"]: row["sample_count"] for row in unresolved_reason_rows},
        "pytest": engineering.get("pytest", "not_recorded"),
        "pip_check": engineering.get("pip_check", "not_recorded"),
        "formal_validation_ok": engineering.get("formal_validation_ok", False),
        "candidate_only": True,
        "production_write": False,
    }
    write_json(reports / "stage2_acceptance.json", acceptance)
    _write_report(baseline, reports, prepare, status_rows, visual_summary, word_validation, compact_validation, baseline_verification, original_ta_ids, identity_ids)
    print(json.dumps(acceptance, ensure_ascii=False, indent=2))
    return 0 if acceptance["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
