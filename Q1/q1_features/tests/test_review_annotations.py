from __future__ import annotations

import csv
from pathlib import Path

import pytest

from q1_features.review_annotations import (
    ANCHOR_FIELDS,
    VISUAL_SEGMENT_FIELDS,
    export_multisegment_proposed_configs,
    segment_sets_equivalent,
    validate_anchor_rows,
    validate_visual_segments,
)
from q1_features.review_experiment import ALIGNMENT_FIELDS
from q1_features.storage import write_csv, write_json, write_jsonl


def _run(tmp_path: Path) -> Path:
    run = tmp_path / "run"
    write_jsonl(
        run / "manifest.jsonl",
        [{"sample_index": 0, "sample_id": "sample-1", "video_id": "video-1"}],
    )
    sample = run / "samples" / "000000"
    write_json(sample / "media.json", {"duration": 10.0})
    write_json(sample / "vision_diagnostic.json", {"successful_candidate_count": 10})
    write_jsonl(
        sample / "vision_rows.jsonl",
        [
            {"face_id": 0, "episode_id": 0, "start": 0.0, "end": 4.0},
            {"face_id": 0, "episode_id": 1, "start": 5.0, "end": 9.0},
        ],
    )
    write_jsonl(
        sample / "words.jsonl",
        [
            {"word_id": 0, "raw_word": "hello"},
            {"word_id": 1, "raw_word": "world"},
        ],
    )
    return run


def _segment(
    segment_id: str = "s1",
    start: str = "0.0",
    end: str = "4.0",
    episode_id: str = "0",
) -> dict[str, str]:
    return {
        "sample_id": "sample-1",
        "segment_id": segment_id,
        "start": start,
        "end": end,
        "visual_label": "active_speaking",
        "face_id": "0",
        "episode_id": episode_id,
        "confidence": "3",
        "evidence_note": "watched the complete interval",
    }


def test_visual_segments_accept_multiple_cross_episode_rows(tmp_path: Path) -> None:
    run = _run(tmp_path)
    rows = [_segment(), _segment("s2", "5.0", "9.0", "1")]
    assert len(validate_visual_segments(rows, run)) == 2


def test_visual_segments_reject_out_of_range_interval(tmp_path: Path) -> None:
    run = _run(tmp_path)
    with pytest.raises(ValueError, match="outside duration"):
        validate_visual_segments([_segment(end="10.5")], run)


def test_visual_segments_reject_duplicate_and_overlap(tmp_path: Path) -> None:
    run = _run(tmp_path)
    with pytest.raises(ValueError, match="duplicate visual segment"):
        validate_visual_segments([_segment(), _segment()], run)
    with pytest.raises(ValueError, match="overlapping segments"):
        validate_visual_segments([_segment(), _segment("s2", "3.0", "4.5")], run)


def test_visual_segments_reject_unknown_face_episode(tmp_path: Path) -> None:
    run = _run(tmp_path)
    with pytest.raises(ValueError, match="unknown OpenFace pair"):
        validate_visual_segments([_segment(episode_id="99")], run)


def test_anchor_word_must_exist_in_words_jsonl(tmp_path: Path) -> None:
    run = _run(tmp_path)
    row = {
        "sample_id": "sample-1",
        "word_id": "99",
        "raw_word": "missing",
        "gold_start": "1.0",
        "gold_end": "1.2",
        "confidence": "3",
        "evidence_note": "word boundary heard twice",
    }
    with pytest.raises(ValueError, match="does not exist"):
        validate_anchor_rows([row], run)


def test_anchor_unverifiable_requires_explicit_low_confidence_note(tmp_path: Path) -> None:
    run = _run(tmp_path)
    row = {
        "sample_id": "sample-1",
        "word_id": "0",
        "raw_word": "hello",
        "gold_start": "",
        "gold_end": "",
        "confidence": "1",
        "evidence_note": "unverifiable because speech is masked by noise",
    }
    assert validate_anchor_rows([row], run) == [row]
    row["evidence_note"] = "unclear"
    with pytest.raises(ValueError, match="unverifiable note"):
        validate_anchor_rows([row], run)


def test_segment_equivalence_uses_endpoint_and_tiou_thresholds() -> None:
    left = [_segment()]
    close = [_segment(start="0.1", end="4.1")]
    far = [_segment(start="0.3", end="4.3")]
    assert segment_sets_equivalent(left, close)[0] is True
    assert segment_sets_equivalent(left, far)[0] is False


def test_multisegment_export_writes_every_active_segment(tmp_path: Path) -> None:
    run = _run(tmp_path)
    final_decisions = tmp_path / "final_decisions.csv"
    final_fields = [
        "sample_index",
        "sample_id",
        "video_id",
        "final_transcript_status",
        "transcript_start",
        "transcript_end",
        "final_visual_status",
        "visual_start",
        "visual_end",
        "face_id",
        "episode_id",
        "adjudicator",
        "evidence_note",
        "decision_source",
    ]
    write_csv(
        final_decisions,
        [
            {
                "sample_index": 0,
                "sample_id": "sample-1",
                "video_id": "video-1",
                "final_transcript_status": "match",
                "transcript_start": "",
                "transcript_end": "",
                "final_visual_status": "active_face_identified",
                "visual_start": "0.0",
                "visual_end": "4.0",
                "face_id": "0",
                "episode_id": "0",
                "adjudicator": "reviewer_c",
                "evidence_note": "independent adjudication completed",
                "decision_source": "adjudication",
            }
        ],
        final_fields,
    )
    segments_path = tmp_path / "segments.csv"
    write_csv(
        segments_path,
        [_segment(), _segment("s2", "5.0", "9.0", "1")],
        VISUAL_SEGMENT_FIELDS,
    )
    baseline = tmp_path / "alignment.csv"
    write_csv(baseline, [], ALIGNMENT_FIELDS)

    output = tmp_path / "proposed"
    result = export_multisegment_proposed_configs(
        final_decisions, segments_path, baseline, output, run
    )
    with (output / "target_face_segments.proposed.csv").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))

    assert result["face_segments"] == 2
    assert [row["episode_id"] for row in rows] == ["0", "1"]
    assert all(row["reviewer"] == "reviewer_c" for row in rows)
