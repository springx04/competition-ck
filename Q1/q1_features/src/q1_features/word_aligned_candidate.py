from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .storage import load_npz, read_jsonl, write_json, write_npz


def _aggregate_interval(
    values: np.ndarray,
    intervals: np.ndarray,
    eligible: np.ndarray,
    source_ids: np.ndarray,
    start: float,
    end: float,
) -> tuple[np.ndarray, int, float, list[int], list[float], list[float]]:
    dimension = int(values.shape[1])
    overlaps = np.maximum(0.0, np.minimum(intervals[:, 1], end) - np.maximum(intervals[:, 0], start))
    rows = np.flatnonzero((overlaps > 0) & eligible.astype(bool))
    if not len(rows) or end <= start:
        return np.zeros(dimension, dtype=np.float32), 0, 0.0, [], [], []
    selected_overlap = overlaps[rows]
    weights = selected_overlap / selected_overlap.sum()
    pooled = np.sum(values[rows] * weights[:, None], axis=0).astype(np.float32)
    pieces = sorted((max(start, float(intervals[row, 0])), min(end, float(intervals[row, 1]))) for row in rows)
    union = 0.0
    left, right = pieces[0]
    for next_left, next_right in pieces[1:]:
        if next_left <= right:
            right = max(right, next_right)
        else:
            union += right - left
            left, right = next_left, next_right
    union += right - left
    coverage = min(1.0, union / (end - start))
    return (
        pooled,
        1,
        coverage,
        source_ids[rows].astype(int).tolist(),
        selected_overlap.astype(float).tolist(),
        weights.astype(float).tolist(),
    )


def build_word_aligned_candidate(run_dir: Path, output_npz: Path, validation_json: Path) -> dict[str, Any]:
    if output_npz.resolve() == (run_dir / "features" / "q1_compact50.npz").resolve():
        raise ValueError("word-aligned candidate must not overwrite q1_compact50.npz")
    manifest = read_jsonl(run_dir / "manifest.jsonl")
    if len(manifest) != 100:
        raise ValueError(f"expected 100 samples, found {len(manifest)}")

    sample_indptr = [0]
    sample_ids = []
    sample_indices = []
    word_ids = []
    raw_words = []
    starts = []
    ends = []
    text_values = []
    audio_values = []
    vision_values = []
    masks = []
    coverage = []
    alignment_status = []
    alignment_reason = []
    audio_indptr = [0]
    audio_source_ids = []
    audio_source_sample_index = []
    audio_overlap_s = []
    audio_weights = []
    vision_indptr = [0]
    vision_source_ids = []
    vision_source_sample_index = []
    vision_overlap_s = []
    vision_weights = []

    for item in manifest:
        sample_index = int(item["sample_index"])
        sample_id = str(item["sample_id"])
        sample_dir = run_dir / "samples" / f"{sample_index:06d}"
        words = read_jsonl(sample_dir / "words.jsonl")
        native = load_npz(sample_dir / "native.npz")
        text_by_id = {
            int(source_id): native["word_features"][row]
            for row, source_id in enumerate(native["word_source_ids"])
        }
        for word in words:
            word_id = int(word["word_id"])
            interval = word.get("accepted_interval")
            has_time = bool(interval and float(interval[1]) > float(interval[0]))
            start, end = (float(interval[0]), float(interval[1])) if has_time else (0.0, 0.0)
            text = np.asarray(text_by_id.get(word_id, np.zeros(768, dtype=np.float32)), dtype=np.float32)
            text_mask = int(word_id in text_by_id and np.isfinite(text).all())
            if has_time:
                audio, audio_mask, audio_coverage, a_ids, a_overlap, a_weights = _aggregate_interval(
                    native["audio_features"], native["audio_intervals"], native["audio_eligible"],
                    native["audio_source_ids"], start, end,
                )
                vision, vision_mask, vision_coverage, v_ids, v_overlap, v_weights = _aggregate_interval(
                    native["vision_features"], native["vision_intervals"], native["vision_eligible"],
                    native["vision_source_ids"], start, end,
                )
            else:
                audio = np.zeros(25, dtype=np.float32)
                vision = np.zeros(22, dtype=np.float32)
                audio_mask = vision_mask = 0
                audio_coverage = vision_coverage = 0.0
                a_ids = a_overlap = a_weights = []
                v_ids = v_overlap = v_weights = []

            sample_ids.append(sample_id)
            sample_indices.append(sample_index)
            word_ids.append(word_id)
            raw_words.append(str(word.get("raw_word", "")))
            starts.append(start)
            ends.append(end)
            text_values.append(text if text_mask else np.zeros(768, dtype=np.float32))
            audio_values.append(audio)
            vision_values.append(vision)
            masks.append([text_mask, audio_mask, vision_mask, int(has_time)])
            coverage.append([float(text_mask), audio_coverage, vision_coverage])
            alignment_status.append("accepted" if has_time else str(word.get("alignment_status", "unlocated")))
            reasons = list(word.get("alignment_reasons", []))
            if not has_time and not reasons:
                reasons = ["no_accepted_ctc_interval"]
            alignment_reason.append(";".join(map(str, reasons)))

            audio_source_ids.extend(a_ids)
            audio_source_sample_index.extend([sample_index] * len(a_ids))
            audio_overlap_s.extend(a_overlap)
            audio_weights.extend(a_weights)
            audio_indptr.append(len(audio_source_ids))
            vision_source_ids.extend(v_ids)
            vision_source_sample_index.extend([sample_index] * len(v_ids))
            vision_overlap_s.extend(v_overlap)
            vision_weights.extend(v_weights)
            vision_indptr.append(len(vision_source_ids))
        sample_indptr.append(len(word_ids))

    arrays = {
        "sample_indptr": np.asarray(sample_indptr, dtype=np.int64),
        "sample_id": np.asarray(sample_ids),
        "sample_index": np.asarray(sample_indices, dtype=np.int64),
        "word_id": np.asarray(word_ids, dtype=np.int64),
        "raw_word": np.asarray(raw_words),
        "start": np.asarray(starts, dtype=np.float64),
        "end": np.asarray(ends, dtype=np.float64),
        "text": np.stack(text_values).astype(np.float32),
        "audio": np.stack(audio_values).astype(np.float32),
        "vision": np.stack(vision_values).astype(np.float32),
        "masks": np.asarray(masks, dtype=np.uint8),
        "coverage": np.asarray(coverage, dtype=np.float32),
        "alignment_status": np.asarray(alignment_status),
        "alignment_reason": np.asarray(alignment_reason),
        "audio_indptr": np.asarray(audio_indptr, dtype=np.int64),
        "audio_source_ids": np.asarray(audio_source_ids, dtype=np.int64),
        "audio_source_sample_index": np.asarray(audio_source_sample_index, dtype=np.int64),
        "audio_overlap_s": np.asarray(audio_overlap_s, dtype=np.float64),
        "audio_weights": np.asarray(audio_weights, dtype=np.float64),
        "vision_indptr": np.asarray(vision_indptr, dtype=np.int64),
        "vision_source_ids": np.asarray(vision_source_ids, dtype=np.int64),
        "vision_source_sample_index": np.asarray(vision_source_sample_index, dtype=np.int64),
        "vision_overlap_s": np.asarray(vision_overlap_s, dtype=np.float64),
        "vision_weights": np.asarray(vision_weights, dtype=np.float64),
    }
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    write_npz(output_npz, **arrays)
    result = validate_word_aligned_candidate(output_npz, sample_count=100)
    write_json(validation_json, result)
    return result


def validate_word_aligned_candidate(path: Path, *, sample_count: int) -> dict[str, Any]:
    arrays = load_npz(path)
    errors = []
    required = {
        "sample_indptr", "sample_id", "sample_index", "word_id", "raw_word", "start", "end",
        "text", "audio", "vision", "masks", "coverage", "alignment_status", "alignment_reason",
        "audio_indptr", "audio_source_ids", "audio_source_sample_index", "audio_overlap_s", "audio_weights",
        "vision_indptr", "vision_source_ids", "vision_source_sample_index", "vision_overlap_s", "vision_weights",
    }
    missing = sorted(required - set(arrays))
    if missing:
        errors.append(f"missing arrays: {missing}")
    total = len(arrays.get("word_id", []))
    if arrays.get("sample_indptr", np.empty(0)).shape != (sample_count + 1,):
        errors.append("sample_indptr shape")
    elif int(arrays["sample_indptr"][-1]) != total or np.any(np.diff(arrays["sample_indptr"]) < 0):
        errors.append("sample_indptr structure")
    expected_shapes = {"text": (total, 768), "audio": (total, 25), "vision": (total, 22), "masks": (total, 4), "coverage": (total, 3)}
    for key, shape in expected_shapes.items():
        if arrays.get(key, np.empty(0)).shape != shape:
            errors.append(f"{key} shape")
    for key in ("text", "audio", "vision", "coverage", "start", "end", "audio_overlap_s", "audio_weights", "vision_overlap_s", "vision_weights"):
        if key in arrays and not np.isfinite(arrays[key]).all():
            errors.append(f"{key} nonfinite")
    if "masks" in arrays and not set(np.unique(arrays["masks"]).tolist()) <= {0, 1}:
        errors.append("masks nonbinary")
    if "coverage" in arrays and (np.any(arrays["coverage"] < 0) or np.any(arrays["coverage"] > 1 + 1e-6)):
        errors.append("coverage range")
    for prefix in ("audio", "vision"):
        indptr = arrays.get(f"{prefix}_indptr", np.empty(0))
        ids = arrays.get(f"{prefix}_source_ids", np.empty(0))
        if indptr.shape != (total + 1,) or (len(indptr) and (int(indptr[-1]) != len(ids) or np.any(np.diff(indptr) < 0))):
            errors.append(f"{prefix} CSR structure")
        for suffix in ("source_sample_index", "overlap_s", "weights"):
            if len(arrays.get(f"{prefix}_{suffix}", [])) != len(ids):
                errors.append(f"{prefix}_{suffix} length")
    result = {
        "ok": not errors,
        "errors": errors,
        "path": str(path.resolve()),
        "sample_count": sample_count,
        "word_count": total,
        "aligned_word_count": int(arrays.get("masks", np.zeros((0, 4)))[:, 3].sum()) if total else 0,
        "text_observed": int(arrays.get("masks", np.zeros((0, 4)))[:, 0].sum()) if total else 0,
        "audio_observed": int(arrays.get("masks", np.zeros((0, 4)))[:, 1].sum()) if total else 0,
        "vision_observed": int(arrays.get("masks", np.zeros((0, 4)))[:, 2].sum()) if total else 0,
        "arrays": {key: {"shape": list(value.shape), "dtype": str(value.dtype)} for key, value in arrays.items()},
    }
    if errors:
        raise ValueError("word-aligned candidate validation failed: " + "; ".join(errors))
    return result
