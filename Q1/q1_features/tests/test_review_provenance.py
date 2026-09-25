from __future__ import annotations

from q1_features.review_experiment import validate_review_rows


def _row() -> dict[str, str]:
    return {
        "sample_id": "sample-1",
        "transcript_status": "match",
        "transcript_start": "",
        "transcript_end": "",
        "visual_status": "no_visible_target",
        "visual_start": "0.0",
        "visual_end": "1.0",
        "face_id": "",
        "episode_id": "",
        "confidence": "3",
        "evidence_note": "watched and listened to the full clip",
        "reviewer_id": "",
        "review_seconds": "",
        "submitted_at_utc": "",
    }


def test_raw_review_requires_provenance_and_timing() -> None:
    errors = validate_review_rows([_row()], expected_ids={"sample-1"})
    assert any("reviewer_id is required" in error for error in errors)
    assert any("review_seconds" in error for error in errors)
    assert any("submitted_at_utc" in error for error in errors)


def test_final_decision_validation_can_use_separate_provenance_schema() -> None:
    assert validate_review_rows(
        [_row()], expected_ids={"sample-1"}, require_provenance=False
    ) == []


def test_unverifiable_transcript_requires_localized_interval() -> None:
    row = _row()
    row.update({
        "reviewer_id": "reviewer_a",
        "review_seconds": "12",
        "submitted_at_utc": "2026-09-24T00:00:00Z",
        "transcript_status": "unverifiable",
    })
    errors = validate_review_rows([row], expected_ids={"sample-1"})
    assert any("unverifiable requires transcript interval" in error for error in errors)

