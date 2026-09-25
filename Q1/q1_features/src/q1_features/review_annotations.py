from __future__ import annotations

import csv
import hashlib
import math
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .storage import read_json, read_jsonl, write_csv, write_json
from .review_experiment import TRANSCRIPT_STATUSES, VISUAL_STATUSES, export_proposed_configs


VISUAL_SEGMENT_LABELS = {
    "active_speaking",
    "visible_not_speaking",
    "no_visible_target",
    "multiple_ambiguous",
    "unverifiable",
}

VISUAL_SEGMENT_FIELDS = [
    "sample_id",
    "segment_id",
    "start",
    "end",
    "visual_label",
    "face_id",
    "episode_id",
    "confidence",
    "evidence_note",
]

ANCHOR_FIELDS = [
    "sample_id",
    "word_id",
    "raw_word",
    "gold_start",
    "gold_end",
    "confidence",
    "evidence_note",
]

ANCHOR_SELECTION_FIELDS = [
    "sample_id",
    "video_id",
    "selection_group",
    "risk_quintile",
    "risk_score",
    "existing_ctc_alert",
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_float(value: Any, field: str, sample_id: str) -> float:
    try:
        result = float(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{sample_id}: {field} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{sample_id}: {field} must be finite")
    return result


def _sample_dirs(run_dir: Path) -> dict[str, Path]:
    return {
        str(row["sample_id"]): run_dir / "samples" / f"{int(row['sample_index']):06d}"
        for row in read_jsonl(run_dir / "manifest.jsonl")
    }


def _sample_video_ids(run_dir: Path) -> dict[str, str]:
    return {
        str(row["sample_id"]): str(row.get("video_id", ""))
        for row in read_jsonl(run_dir / "manifest.jsonl")
    }


def temporal_iou(
    left_start: float,
    left_end: float,
    right_start: float,
    right_end: float,
) -> float:
    intersection = max(0.0, min(left_end, right_end) - max(left_start, right_start))
    union = max(left_end, right_end) - min(left_start, right_start)
    return intersection / union if union > 0 else 0.0


def validate_visual_segments(
    rows: Sequence[Mapping[str, Any]],
    run_dir: Path,
    *,
    expected_sample_ids: Iterable[str] | None = None,
    require_sample_coverage: bool = True,
) -> list[dict[str, str]]:
    sample_dirs = _sample_dirs(run_dir)
    expected = set(expected_sample_ids) if expected_sample_ids is not None else set(sample_dirs)
    normalized: list[dict[str, str]] = []
    seen_keys: set[tuple[str, str]] = set()
    intervals_by_identity: dict[tuple[str, str, str], list[tuple[float, float, str]]] = defaultdict(list)

    durations: dict[str, float] = {}
    valid_faces: dict[str, set[tuple[str, str]]] = {}

    for raw in rows:
        row = {field: str(raw.get(field, "")).strip() for field in VISUAL_SEGMENT_FIELDS}
        sample_id = row["sample_id"]
        segment_id = row["segment_id"]
        if sample_id not in expected:
            raise ValueError(f"unknown sample_id in visual segments: {sample_id}")
        if not segment_id:
            raise ValueError(f"{sample_id}: segment_id is required")
        key = (sample_id, segment_id)
        if key in seen_keys:
            raise ValueError(f"duplicate visual segment: {sample_id}/{segment_id}")
        seen_keys.add(key)

        label = row["visual_label"]
        if label not in VISUAL_SEGMENT_LABELS:
            raise ValueError(f"{sample_id}/{segment_id}: invalid visual_label {label!r}")
        confidence = row["confidence"]
        if confidence not in {"1", "2", "3"}:
            raise ValueError(f"{sample_id}/{segment_id}: confidence must be 1, 2, or 3")
        if not row["evidence_note"]:
            raise ValueError(f"{sample_id}/{segment_id}: evidence_note is required")

        start = _parse_float(row["start"], "start", sample_id)
        end = _parse_float(row["end"], "end", sample_id)
        if sample_id not in durations:
            media = read_json(sample_dirs[sample_id] / "media.json")
            durations[sample_id] = float(media["duration"])
        if start < 0 or end <= start or end > durations[sample_id] + 1e-6:
            raise ValueError(
                f"{sample_id}/{segment_id}: interval [{start}, {end}] is outside duration "
                f"{durations[sample_id]}"
            )

        face_id = row["face_id"]
        episode_id = row["episode_id"]
        requires_face = label in {"active_speaking", "visible_not_speaking"}
        if requires_face and (not face_id or not episode_id):
            raise ValueError(f"{sample_id}/{segment_id}: face_id and episode_id are required")
        if bool(face_id) != bool(episode_id):
            raise ValueError(f"{sample_id}/{segment_id}: face_id and episode_id must be supplied together")
        if face_id:
            if sample_id not in valid_faces:
                valid_faces[sample_id] = {
                    (str(item.get("face_id", "")), str(item.get("episode_id", "")))
                    for item in read_jsonl(sample_dirs[sample_id] / "vision_rows.jsonl")
                }
            if (face_id, episode_id) not in valid_faces[sample_id]:
                raise ValueError(
                    f"{sample_id}/{segment_id}: unknown OpenFace pair "
                    f"face_id={face_id}, episode_id={episode_id}"
                )
            identity_key = (sample_id, face_id, episode_id)
            for old_start, old_end, old_segment_id in intervals_by_identity[identity_key]:
                if min(end, old_end) - max(start, old_start) > 1e-9:
                    raise ValueError(
                        f"{sample_id}: overlapping segments for face_id={face_id}, "
                        f"episode_id={episode_id}: {old_segment_id} and {segment_id}"
                    )
            intervals_by_identity[identity_key].append((start, end, segment_id))

        normalized.append(row)

    if require_sample_coverage:
        covered = {row["sample_id"] for row in normalized}
        missing = sorted(expected - covered)
        if missing:
            raise ValueError(f"visual segment form does not cover samples: {missing}")
    return normalized


def validate_anchor_rows(
    rows: Sequence[Mapping[str, Any]],
    run_dir: Path,
    *,
    expected_keys: Iterable[tuple[str, str]] | None = None,
) -> list[dict[str, str]]:
    sample_dirs = _sample_dirs(run_dir)
    normalized: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    counts: dict[str, int] = defaultdict(int)
    word_maps: dict[str, dict[str, dict[str, Any]]] = {}
    durations: dict[str, float] = {}

    for raw in rows:
        row = {field: str(raw.get(field, "")).strip() for field in ANCHOR_FIELDS}
        sample_id = row["sample_id"]
        word_id = row["word_id"]
        if sample_id not in sample_dirs:
            raise ValueError(f"unknown sample_id in anchor form: {sample_id}")
        key = (sample_id, word_id)
        if key in seen:
            raise ValueError(f"duplicate anchor word: {sample_id}/{word_id}")
        seen.add(key)
        counts[sample_id] += 1
        if counts[sample_id] > 5:
            raise ValueError(f"{sample_id}: at most five anchor words are allowed")

        if sample_id not in word_maps:
            word_maps[sample_id] = {
                str(item.get("word_id", "")): item
                for item in read_jsonl(sample_dirs[sample_id] / "words.jsonl")
            }
        word = word_maps[sample_id].get(word_id)
        if word is None:
            raise ValueError(f"{sample_id}: word_id {word_id!r} does not exist in words.jsonl")
        if row["raw_word"] != str(word.get("raw_word", "")):
            raise ValueError(f"{sample_id}/{word_id}: raw_word does not match words.jsonl")
        if row["confidence"] not in {"1", "2", "3"}:
            raise ValueError(f"{sample_id}/{word_id}: confidence must be 1, 2, or 3")
        if not row["evidence_note"]:
            raise ValueError(f"{sample_id}/{word_id}: evidence_note is required")

        start_text = row["gold_start"]
        end_text = row["gold_end"]
        if not start_text and not end_text:
            if row["confidence"] != "1" or "unverifiable" not in row["evidence_note"].lower():
                raise ValueError(
                    f"{sample_id}/{word_id}: blank times require confidence=1 and an unverifiable note"
                )
        elif not start_text or not end_text:
            raise ValueError(f"{sample_id}/{word_id}: gold_start and gold_end must both be present")
        else:
            start = _parse_float(start_text, "gold_start", sample_id)
            end = _parse_float(end_text, "gold_end", sample_id)
            if sample_id not in durations:
                durations[sample_id] = float(read_json(sample_dirs[sample_id] / "media.json")["duration"])
            if start < 0 or end <= start or end > durations[sample_id] + 1e-6:
                raise ValueError(
                    f"{sample_id}/{word_id}: gold interval [{start}, {end}] is outside duration "
                    f"{durations[sample_id]}"
                )
        normalized.append(row)

    if expected_keys is not None:
        expected = set(expected_keys)
        if seen != expected:
            missing = sorted(expected - seen)
            extra = sorted(seen - expected)
            raise ValueError(f"anchor key mismatch; missing={missing}, extra={extra}")
    return normalized


def _risk_quintiles(clean_ids: Sequence[str], risk_scores: Mapping[str, float]) -> dict[str, int]:
    ordered = sorted(clean_ids, key=lambda sample_id: (risk_scores.get(sample_id, 0.0), sample_id))
    total = len(ordered)
    if total == 0:
        return {}
    return {sample_id: min(5, (rank * 5) // total + 1) for rank, sample_id in enumerate(ordered)}


def select_anchor_samples(
    run_dir: Path,
    risk_csv: Path,
    baseline_alignment_csv: Path,
    *,
    seed: int = 2026,
) -> list[dict[str, Any]]:
    risk_rows = _read_csv(risk_csv)
    risk_scores = {
        row["sample_id"]: float(row["risk_score"])
        for row in risk_rows
        if row.get("sample_id") and row.get("risk_score") not in {None, ""}
    }
    alert_types = {"suspected_text_audio_mismatch", "low_aligned_word_fraction"}
    alert_ids = {
        row["sample_id"]
        for row in _read_csv(baseline_alignment_csv)
        if row.get("issue_type") in alert_types
    }
    quality_rows = _read_csv(run_dir / "reports" / "quality.csv")
    clean_ids = sorted(
        row["sample_id"] for row in quality_rows if row.get("pairing_status") == "no_issue_detected"
    )
    if len(alert_ids) != 23:
        raise ValueError(f"expected 23 existing text-alert samples, found {len(alert_ids)}")
    if len(clean_ids) != 49:
        raise ValueError(f"expected 49 clean samples, found {len(clean_ids)}")

    quintiles = _risk_quintiles(clean_ids, risk_scores)
    video_ids = _sample_video_ids(run_dir)
    rng = random.Random(seed)
    selected_controls: list[str] = []
    used_videos: set[str] = set()
    for quintile in range(1, 6):
        candidates = [sample_id for sample_id in clean_ids if quintiles[sample_id] == quintile]
        candidates.sort()
        rng.shuffle(candidates)
        while candidates and sum(quintiles[item] == quintile for item in selected_controls) < 2:
            unused = [item for item in candidates if video_ids[item] not in used_videos]
            chosen = unused[0] if unused else candidates[0]
            selected_controls.append(chosen)
            used_videos.add(video_ids[chosen])
            candidates.remove(chosen)

    rows: list[dict[str, Any]] = []
    for sample_id in sorted(alert_ids):
        rows.append(
            {
                "sample_id": sample_id,
                "video_id": video_ids[sample_id],
                "selection_group": "existing_text_alert",
                "risk_quintile": "",
                "risk_score": f"{risk_scores.get(sample_id, 0.0):.8f}",
                "existing_ctc_alert": "true",
            }
        )
    for sample_id in sorted(selected_controls, key=lambda item: (quintiles[item], item)):
        rows.append(
            {
                "sample_id": sample_id,
                "video_id": video_ids[sample_id],
                "selection_group": "clean_stratified_control",
                "risk_quintile": str(quintiles[sample_id]),
                "risk_score": f"{risk_scores.get(sample_id, 0.0):.8f}",
                "existing_ctc_alert": "false",
            }
        )
    if len(rows) != 33:
        raise ValueError(f"expected 33 anchor samples, selected {len(rows)}")
    return rows


def _preselect_words(words: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    candidates = [row for row in words if str(row.get("raw_word", "")).strip()]
    if len(candidates) <= 5:
        return candidates
    positions = (0.10, 0.30, 0.50, 0.70, 0.90)
    indices: list[int] = []
    for position in positions:
        index = int(round((len(candidates) - 1) * position))
        if index not in indices:
            indices.append(index)
    if len(indices) < 5:
        for index in range(len(candidates)):
            if index not in indices:
                indices.append(index)
            if len(indices) == 5:
                break
    return [candidates[index] for index in sorted(indices[:5])]


def prepare_annotation_templates(
    run_dir: Path,
    output_dir: Path,
    risk_csv: Path,
    baseline_alignment_csv: Path,
    *,
    seed: int = 2026,
    force: bool = False,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "reviewer_a_segments": output_dir / "reviewer_a_segments.csv",
        "reviewer_b_segments": output_dir / "reviewer_b_segments.csv",
        "reviewer_a_anchors": output_dir / "reviewer_a_anchors.csv",
        "reviewer_b_anchors": output_dir / "reviewer_b_anchors.csv",
        "anchor_selection": output_dir / "anchor_sample_selection.csv",
    }
    existing = [path for path in paths.values() if path.exists()]
    if existing and not force:
        raise FileExistsError(f"annotation forms already exist: {existing}")

    archive_dir = output_dir / "template_archive"
    archived: list[str] = []
    for name in ("reviewer_a.csv", "reviewer_b.csv"):
        source = output_dir / name
        if not source.exists():
            continue
        rows = _read_csv(source)
        filled = any(
            str(row.get(field, "")).strip()
            for row in rows
            for field in (
                "text_audio_status",
                "visual_status",
                "reviewer_confidence",
                "reviewer_notes",
            )
        )
        if filled:
            raise ValueError(f"refusing to archive and replace a non-empty review form: {source}")
        archive_dir.mkdir(parents=True, exist_ok=True)
        target = archive_dir / f"{source.stem}.pre_multisegment.csv"
        if not target.exists():
            shutil.copy2(source, target)
            archived.append(str(target))

    selection = select_anchor_samples(run_dir, risk_csv, baseline_alignment_csv, seed=seed)
    sample_dirs = _sample_dirs(run_dir)
    anchors: list[dict[str, str]] = []
    for selected in selection:
        sample_id = str(selected["sample_id"])
        for word in _preselect_words(read_jsonl(sample_dirs[sample_id] / "words.jsonl")):
            anchors.append(
                {
                    "sample_id": sample_id,
                    "word_id": str(word.get("word_id", "")),
                    "raw_word": str(word.get("raw_word", "")),
                    "gold_start": "",
                    "gold_end": "",
                    "confidence": "",
                    "evidence_note": "",
                }
            )

    write_csv(paths["reviewer_a_segments"], [], VISUAL_SEGMENT_FIELDS)
    write_csv(paths["reviewer_b_segments"], [], VISUAL_SEGMENT_FIELDS)
    write_csv(paths["reviewer_a_anchors"], anchors, ANCHOR_FIELDS)
    write_csv(paths["reviewer_b_anchors"], anchors, ANCHOR_FIELDS)
    write_csv(paths["anchor_selection"], selection, ANCHOR_SELECTION_FIELDS)
    manifest = {
        "seed": seed,
        "anchor_sample_count": len(selection),
        "anchor_row_count": len(anchors),
        "existing_text_alert_count": sum(row["existing_ctc_alert"] == "true" for row in selection),
        "clean_control_count": sum(row["existing_ctc_alert"] == "false" for row in selection),
        "forms": {name: str(path) for name, path in paths.items()},
        "archived_empty_summary_forms": archived,
        "blinding": {
            "reviewer_forms_contain_alert_status": False,
            "reviewer_forms_contain_auxiliary_scores": False,
            "anchor_selection_admin_only": True,
        },
    }
    write_json(output_dir / "annotation_manifest.json", manifest)
    return manifest


def _group_by_sample(rows: Sequence[Mapping[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["sample_id"])].append(dict(row))
    return dict(grouped)


def segment_sets_equivalent(
    left: Sequence[Mapping[str, str]],
    right: Sequence[Mapping[str, str]],
    *,
    endpoint_tolerance: float = 0.20,
    tiou_threshold: float = 0.80,
) -> tuple[bool, list[float]]:
    if len(left) != len(right):
        return False, []
    candidates: list[tuple[float, int, int]] = []
    for left_index, left_row in enumerate(left):
        for right_index, right_row in enumerate(right):
            if (
                left_row["visual_label"] != right_row["visual_label"]
                or left_row["face_id"] != right_row["face_id"]
                or left_row["episode_id"] != right_row["episode_id"]
            ):
                continue
            left_start = float(left_row["start"])
            left_end = float(left_row["end"])
            right_start = float(right_row["start"])
            right_end = float(right_row["end"])
            iou = temporal_iou(left_start, left_end, right_start, right_end)
            if (
                abs(left_start - right_start) <= endpoint_tolerance
                and abs(left_end - right_end) <= endpoint_tolerance
                and iou >= tiou_threshold
            ):
                candidates.append((iou, left_index, right_index))
    matched_left: set[int] = set()
    matched_right: set[int] = set()
    matched_ious: list[float] = []
    for iou, left_index, right_index in sorted(candidates, reverse=True):
        if left_index in matched_left or right_index in matched_right:
            continue
        matched_left.add(left_index)
        matched_right.add(right_index)
        matched_ious.append(iou)
    return len(matched_left) == len(left), matched_ious


def analyze_annotation_agreement(
    run_dir: Path,
    segments_a_path: Path,
    segments_b_path: Path,
    anchors_a_path: Path,
    anchors_b_path: Path,
    output_dir: Path,
    *,
    endpoint_tolerance: float = 0.20,
    tiou_threshold: float = 0.80,
) -> dict[str, Any]:
    sample_dirs = _sample_dirs(run_dir)
    expected_samples = set(sample_dirs)
    zero_face_samples = {
        sample_id
        for sample_id, sample_dir in sample_dirs.items()
        if int(read_json(sample_dir / "vision_diagnostic.json", {}).get("successful_candidate_count", 0) or 0) == 0
    }
    segments_a = validate_visual_segments(
        _read_csv(segments_a_path), run_dir, expected_sample_ids=expected_samples
    )
    segments_b = validate_visual_segments(
        _read_csv(segments_b_path), run_dir, expected_sample_ids=expected_samples
    )
    anchors_a_raw = _read_csv(anchors_a_path)
    anchor_keys = {(row["sample_id"], row["word_id"]) for row in anchors_a_raw}
    anchors_a = validate_anchor_rows(anchors_a_raw, run_dir, expected_keys=anchor_keys)
    anchors_b = validate_anchor_rows(_read_csv(anchors_b_path), run_dir, expected_keys=anchor_keys)

    grouped_a = _group_by_sample(segments_a)
    grouped_b = _group_by_sample(segments_b)
    segment_required: list[dict[str, str]] = []
    all_ious: list[float] = []
    equivalent_count = 0
    for sample_id in sorted(expected_samples):
        equivalent, ious = segment_sets_equivalent(
            grouped_a.get(sample_id, []),
            grouped_b.get(sample_id, []),
            endpoint_tolerance=endpoint_tolerance,
            tiou_threshold=tiou_threshold,
        )
        all_ious.extend(ious)
        low_confidence = any(
            row["confidence"] == "1" or row["visual_label"] in {"unverifiable", "multiple_ambiguous"}
            for row in grouped_a.get(sample_id, []) + grouped_b.get(sample_id, [])
        )
        requires_no_face_adjudication = sample_id in zero_face_samples
        if equivalent and not low_confidence and not requires_no_face_adjudication:
            equivalent_count += 1
        else:
            reasons = []
            if not equivalent:
                reasons.append("segment_disagreement")
            if low_confidence:
                reasons.append("low_confidence_or_unverifiable")
            if requires_no_face_adjudication:
                reasons.append("zero_face_requires_adjudication")
            segment_required.append({"sample_id": sample_id, "reason": ";".join(reasons)})

    anchors_b_map = {(row["sample_id"], row["word_id"]): row for row in anchors_b}
    anchor_required: list[dict[str, str]] = []
    endpoint_errors: list[float] = []
    for left in anchors_a:
        key = (left["sample_id"], left["word_id"])
        right = anchors_b_map[key]
        reasons = []
        left_blank = not left["gold_start"]
        right_blank = not right["gold_start"]
        if left_blank != right_blank:
            reasons.append("verifiability_disagreement")
        elif not left_blank:
            start_error = abs(float(left["gold_start"]) - float(right["gold_start"]))
            end_error = abs(float(left["gold_end"]) - float(right["gold_end"]))
            endpoint_errors.extend([start_error, end_error])
            if start_error > endpoint_tolerance or end_error > endpoint_tolerance:
                reasons.append("endpoint_disagreement")
        if left["confidence"] == "1" or right["confidence"] == "1":
            reasons.append("low_confidence")
        if reasons:
            anchor_required.append(
                {"sample_id": key[0], "word_id": key[1], "reason": ";".join(sorted(set(reasons)))}
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "adjudication_segments_required.csv", segment_required, ["sample_id", "reason"])
    write_csv(
        output_dir / "adjudication_anchors_required.csv",
        anchor_required,
        ["sample_id", "word_id", "reason"],
    )
    segment_form = output_dir / "adjudication_segments.csv"
    if not segment_form.exists():
        write_csv(segment_form, [], VISUAL_SEGMENT_FIELDS)
    anchor_form = output_dir / "adjudication_anchors.csv"
    if not anchor_form.exists():
        anchors_a_map = {(row["sample_id"], row["word_id"]): row for row in anchors_a}
        anchor_template = []
        for required in anchor_required:
            source = anchors_a_map[(required["sample_id"], required["word_id"])]
            anchor_template.append({
                "sample_id": source["sample_id"],
                "word_id": source["word_id"],
                "raw_word": source["raw_word"],
                "gold_start": "",
                "gold_end": "",
                "confidence": "",
                "evidence_note": "",
            })
        write_csv(anchor_form, anchor_template, ANCHOR_FIELDS)
    metrics = {
        "segment_sample_count": len(expected_samples),
        "segment_agreed_without_escalation": equivalent_count,
        "segment_adjudication_required": len(segment_required),
        "segment_raw_agreement": equivalent_count / len(expected_samples),
        "matched_segment_mean_tiou": sum(all_ious) / len(all_ious) if all_ious else None,
        "anchor_count": len(anchors_a),
        "anchor_adjudication_required": len(anchor_required),
        "anchor_mean_absolute_endpoint_difference": (
            sum(endpoint_errors) / len(endpoint_errors) if endpoint_errors else None
        ),
        "endpoint_tolerance_seconds": endpoint_tolerance,
        "tiou_threshold": tiou_threshold,
    }
    write_json(output_dir / "annotation_agreement.json", metrics)
    return metrics


def _average_anchor(left: Mapping[str, str], right: Mapping[str, str]) -> dict[str, str]:
    result = dict(left)
    if left["gold_start"] and right["gold_start"]:
        result["gold_start"] = f"{(float(left['gold_start']) + float(right['gold_start'])) / 2:.6f}"
        result["gold_end"] = f"{(float(left['gold_end']) + float(right['gold_end'])) / 2:.6f}"
    result["confidence"] = str(min(int(left["confidence"]), int(right["confidence"])))
    result["evidence_note"] = f"reviewer_a: {left['evidence_note']} | reviewer_b: {right['evidence_note']}"
    return result


def finalize_annotation_gold(
    run_dir: Path,
    final_decisions_path: Path,
    segments_a_path: Path,
    segments_b_path: Path,
    anchors_a_path: Path,
    anchors_b_path: Path,
    adjudicated_segments_path: Path,
    adjudicated_anchors_path: Path,
    agreement_dir: Path,
    output_dir: Path,
    *,
    endpoint_tolerance: float = 0.20,
    tiou_threshold: float = 0.80,
) -> dict[str, Any]:
    analyze_annotation_agreement(
        run_dir,
        segments_a_path,
        segments_b_path,
        anchors_a_path,
        anchors_b_path,
        agreement_dir,
        endpoint_tolerance=endpoint_tolerance,
        tiou_threshold=tiou_threshold,
    )
    expected_samples = set(_sample_dirs(run_dir))
    final_decisions = _read_csv(final_decisions_path)
    decision_ids = {row.get("sample_id", "") for row in final_decisions}
    if len(final_decisions) != len(expected_samples) or decision_ids != expected_samples:
        raise ValueError("final_decisions must contain exactly one row for every sample")
    for row in final_decisions:
        sample_id = row["sample_id"]
        if row.get("final_transcript_status", "") not in TRANSCRIPT_STATUSES:
            raise ValueError(f"{sample_id}: invalid final_transcript_status")
        if row.get("final_visual_status", "") not in VISUAL_STATUSES:
            raise ValueError(f"{sample_id}: invalid final_visual_status")
        if not row.get("adjudicator", "").strip() or not row.get("evidence_note", "").strip():
            raise ValueError(f"{sample_id}: final decision requires adjudicator and evidence")
    segments_a = validate_visual_segments(_read_csv(segments_a_path), run_dir)
    segments_b = validate_visual_segments(_read_csv(segments_b_path), run_dir)
    anchors_a = validate_anchor_rows(_read_csv(anchors_a_path), run_dir)
    anchor_keys = {(row["sample_id"], row["word_id"]) for row in anchors_a}
    anchors_b = validate_anchor_rows(_read_csv(anchors_b_path), run_dir, expected_keys=anchor_keys)

    segment_required = {
        row["sample_id"] for row in _read_csv(agreement_dir / "adjudication_segments_required.csv")
    }
    anchor_required = {
        (row["sample_id"], row["word_id"])
        for row in _read_csv(agreement_dir / "adjudication_anchors_required.csv")
    }
    adjudicated_segments = validate_visual_segments(
        _read_csv(adjudicated_segments_path),
        run_dir,
        expected_sample_ids=segment_required,
        require_sample_coverage=True,
    ) if segment_required else []
    if {row["sample_id"] for row in adjudicated_segments} != segment_required:
        raise ValueError("adjudicated visual segments must replace every required sample and no others")
    adjudicated_anchors = validate_anchor_rows(
        _read_csv(adjudicated_anchors_path), run_dir, expected_keys=anchor_required
    ) if anchor_required else []

    grouped_a = _group_by_sample(segments_a)
    grouped_adjudicated = _group_by_sample(adjudicated_segments)
    final_segments: list[dict[str, str]] = []
    for sample_id in sorted(expected_samples):
        source = grouped_adjudicated[sample_id] if sample_id in segment_required else grouped_a[sample_id]
        final_segments.extend(source)

    anchors_b_map = {(row["sample_id"], row["word_id"]): row for row in anchors_b}
    adjudicated_anchor_map = {
        (row["sample_id"], row["word_id"]): row for row in adjudicated_anchors
    }
    final_anchors: list[dict[str, str]] = []
    for left in anchors_a:
        key = (left["sample_id"], left["word_id"])
        if key in anchor_required:
            final_anchors.append(adjudicated_anchor_map[key])
        else:
            final_anchors.append(_average_anchor(left, anchors_b_map[key]))

    validate_visual_segments(final_segments, run_dir)
    final_segment_groups = _group_by_sample(final_segments)
    required_label = {
        "active_face_identified": "active_speaking",
        "no_visible_target": "no_visible_target",
        "multiple_ambiguous": "multiple_ambiguous",
        "unverifiable": "unverifiable",
    }
    for decision in final_decisions:
        sample_id = decision["sample_id"]
        labels = {row["visual_label"] for row in final_segment_groups[sample_id]}
        expected_label = required_label[decision["final_visual_status"]]
        if expected_label not in labels:
            raise ValueError(
                f"{sample_id}: final visual decision {decision['final_visual_status']} "
                f"requires at least one {expected_label} segment"
            )
        if decision["final_visual_status"] != "active_face_identified" and "active_speaking" in labels:
            raise ValueError(f"{sample_id}: quarantined visual decision cannot contain active_speaking")
    validate_anchor_rows(final_anchors, run_dir, expected_keys=anchor_keys)
    output_dir.mkdir(parents=True, exist_ok=True)
    final_segments_path = output_dir / "final_visual_segments.csv"
    final_anchors_path = output_dir / "final_anchor_words.csv"
    write_csv(final_segments_path, final_segments, VISUAL_SEGMENT_FIELDS)
    write_csv(final_anchors_path, final_anchors, ANCHOR_FIELDS)
    hashes = {
        "final_decisions": _sha256(final_decisions_path),
        "final_visual_segments": _sha256(final_segments_path),
        "final_anchor_words": _sha256(final_anchors_path),
    }
    lock = {
        "status": "independent_human_gold_locked",
        "sample_count": len(expected_samples),
        "visual_segment_count": len(final_segments),
        "anchor_count": len(final_anchors),
        "segment_adjudication_sample_count": len(segment_required),
        "anchor_adjudication_count": len(anchor_required),
        "sha256": hashes,
        "auxiliary_models_unlocked": True,
    }
    write_json(output_dir / "gold_lock.json", lock)
    return lock


def export_multisegment_proposed_configs(
    final_decisions_csv: Path,
    final_segments_csv: Path,
    baseline_alignment_csv: Path,
    output_dir: Path,
    run_dir: Path,
) -> dict[str, Any]:
    result = export_proposed_configs(
        final_decisions_csv,
        baseline_alignment_csv,
        output_dir,
    )
    decisions = {row["sample_id"]: row for row in _read_csv(final_decisions_csv)}
    segments = validate_visual_segments(_read_csv(final_segments_csv), run_dir)
    active_by_sample: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in segments:
        if row["visual_label"] == "active_speaking":
            active_by_sample[row["sample_id"]].append(row)

    face_rows: list[dict[str, str]] = []
    for sample_id, decision in sorted(decisions.items()):
        visual_status = decision["final_visual_status"]
        active = active_by_sample.get(sample_id, [])
        if visual_status == "active_face_identified" and not active:
            raise ValueError(f"{sample_id}: active_face_identified requires an active_speaking segment")
        if visual_status != "active_face_identified" and active:
            raise ValueError(
                f"{sample_id}: {visual_status} cannot export active_speaking target-face segments"
            )
        for segment in active:
            face_rows.append(
                {
                    "sample_id": sample_id,
                    "start": segment["start"],
                    "end": segment["end"],
                    "face_id": segment["face_id"],
                    "episode_id": segment["episode_id"],
                    "reviewer": decision["adjudicator"],
                    "evidence_note": (
                        f"segment_id={segment['segment_id']}; {segment['evidence_note']}; "
                        f"adjudication={decision['evidence_note']}"
                    ),
                }
            )
    fields = ["sample_id", "start", "end", "face_id", "episode_id", "reviewer", "evidence_note"]
    write_csv(output_dir / "target_face_segments.proposed.csv", face_rows, fields)
    result["face_segments"] = len(face_rows)
    result["multi_segment_export"] = True
    return result
