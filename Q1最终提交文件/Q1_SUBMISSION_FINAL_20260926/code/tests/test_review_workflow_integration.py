from __future__ import annotations

import csv
from pathlib import Path

import pytest

from q1_features.review_experiment import (
    ADJUDICATION_FIELDS,
    ALIGNMENT_FIELDS,
    REVIEW_FIELDS,
    export_proposed_configs,
    finalize_decisions,
)
from q1_features.storage import write_csv, write_json, write_jsonl


def _review(sample_id: str, reviewer: str) -> dict[str, str | int]:
    return {
        "review_order": 1,
        "sample_index": 0,
        "sample_id": sample_id,
        "video_id": "video-1",
        "video_path": "/data/video.mp4",
        "audio_path": "/run/audio.wav",
        "raw_text": "hello",
        "words_path": "/run/words.jsonl",
        "vision_rows_path": "/run/vision_rows.jsonl",
        "frames_dir": "/run/frames",
        "duration_s": 1.0,
        "reviewer_id": reviewer,
        "transcript_status": "match",
        "transcript_start": "",
        "transcript_end": "",
        "visual_status": "no_visible_target",
        "visual_start": "0.0",
        "visual_end": "1.0",
        "face_id": "",
        "episode_id": "",
        "confidence": "3",
        "evidence_note": "watched full clip; no visible speaker",
        "review_seconds": "20",
        "submitted_at_utc": "2026-09-24T00:00:00Z",
    }


def _zero_face_run(tmp_path: Path) -> Path:
    run = tmp_path / "run"
    write_jsonl(run / "manifest.jsonl", [{
        "sample_index": 0,
        "sample_id": "sample-1",
        "video_id": "video-1",
    }])
    write_json(
        run / "samples" / "000000" / "vision_diagnostic.json",
        {"successful_candidate_count": 0},
    )
    return run


def test_zero_face_agreement_still_requires_adjudication(tmp_path: Path) -> None:
    run = _zero_face_run(tmp_path)
    reviewer_a = tmp_path / "reviewer_a.csv"
    reviewer_b = tmp_path / "reviewer_b.csv"
    write_csv(reviewer_a, [_review("sample-1", "reviewer_a")], REVIEW_FIELDS)
    write_csv(reviewer_b, [_review("sample-1", "reviewer_b")], REVIEW_FIELDS)
    adjudication = tmp_path / "adjudication.csv"
    write_csv(adjudication, [], ADJUDICATION_FIELDS)

    with pytest.raises(ValueError, match="automatic_zero_face"):
        finalize_decisions(reviewer_a, reviewer_b, adjudication, run, tmp_path / "final.csv")


def test_final_decisions_export_no_face_once(tmp_path: Path) -> None:
    final_path = tmp_path / "final.csv"
    final_fields = [
        "sample_index", "sample_id", "video_id", "final_transcript_status",
        "transcript_start", "transcript_end", "final_visual_status", "visual_start",
        "visual_end", "face_id", "episode_id", "adjudicator", "evidence_note",
        "decision_source",
    ]
    write_csv(final_path, [{
        "sample_index": 0,
        "sample_id": "sample-1",
        "video_id": "video-1",
        "final_transcript_status": "match",
        "transcript_start": "",
        "transcript_end": "",
        "final_visual_status": "no_visible_target",
        "visual_start": "0.0",
        "visual_end": "1.0",
        "face_id": "",
        "episode_id": "",
        "adjudicator": "reviewer_c",
        "evidence_note": "full clip confirms natural visual absence",
        "decision_source": "adjudication",
    }], final_fields)
    baseline = tmp_path / "alignment_review.csv"
    write_csv(baseline, [{
        "sample_id": "sample-1",
        "issue_type": "vision_no_face",
        "start": "0.0",
        "end": "1.0",
        "review_status": "unresolved",
        "reviewer": "baseline",
        "evidence": "automatic zero-face result",
        "action": "keep_unresolved",
    }], ALIGNMENT_FIELDS)

    result = export_proposed_configs(final_path, baseline, tmp_path / "proposed")
    with (tmp_path / "proposed" / "alignment_review.proposed.csv").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))

    assert result["alignment_rows"] == 1
    assert rows[0]["issue_type"] == "vision_no_face"
    assert rows[0]["review_status"] == "confirmed_mismatch"
    assert rows[0]["action"] == "quarantine_pairing"

