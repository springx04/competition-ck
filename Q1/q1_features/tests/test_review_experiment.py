from __future__ import annotations

from q1_features.review_experiment import (
    cohen_kappa, grouped_bootstrap_agreement, krippendorff_alpha_nominal, validate_review_rows,
)


def complete_row() -> dict[str, str]:
    return {
        "sample_id": "sample-1", "transcript_status": "match",
        "transcript_start": "", "transcript_end": "",
        "visual_status": "active_face_identified", "visual_start": "0.0",
        "visual_end": "1.0", "face_id": "0", "episode_id": "0",
        "confidence": "3", "evidence_note": "watched full clip",
        "reviewer_id": "reviewer_a", "review_seconds": "20",
        "submitted_at_utc": "2026-09-24T00:00:00Z",
    }


def test_perfect_agreement_metrics_are_one() -> None:
    labels = ["match", "mismatch", "match", "unverifiable"]
    assert cohen_kappa(labels, labels) == 1.0
    assert krippendorff_alpha_nominal(labels, labels) == 1.0


def test_disagreement_metrics_are_below_one() -> None:
    left = ["match", "match", "mismatch", "unverifiable"]
    right = ["match", "mismatch", "mismatch", "match"]
    assert cohen_kappa(left, right) < 1.0
    assert krippendorff_alpha_nominal(left, right) < 1.0


def test_review_validation_accepts_evidence_bound_face() -> None:
    assert validate_review_rows([complete_row()], expected_ids={"sample-1"}) == []


def test_review_validation_rejects_unbounded_mismatch() -> None:
    row = complete_row()
    row["transcript_status"] = "mismatch"
    errors = validate_review_rows([row], expected_ids={"sample-1"})
    assert any("mismatch requires transcript interval" in error for error in errors)



def test_grouped_bootstrap_is_deterministic_and_grouped() -> None:
    left = [
        {"sample_id": "a", "video_id": "v1", "transcript_status": "match"},
        {"sample_id": "b", "video_id": "v1", "transcript_status": "match"},
        {"sample_id": "c", "video_id": "v2", "transcript_status": "mismatch"},
    ]
    right = [
        {"sample_id": "a", "video_id": "v1", "transcript_status": "match"},
        {"sample_id": "b", "video_id": "v1", "transcript_status": "match"},
        {"sample_id": "c", "video_id": "v2", "transcript_status": "mismatch"},
    ]
    first = grouped_bootstrap_agreement(left, right, "transcript_status", iterations=20)
    second = grouped_bootstrap_agreement(left, right, "transcript_status", iterations=20)
    assert first == second
    assert first["group_count"] == 2
    assert first["cohen_kappa_95ci"] == [1.0, 1.0]
