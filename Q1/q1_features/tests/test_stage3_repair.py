from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from q1_features.stage3_repair import (
    build_stage3_visual_timeline,
    classify_mapping_evidence,
    classify_resolution_status,
    mapping_proposal_row,
    assert_candidate_path_safe,
    assert_semantic_audio_mask,
    assert_source_traceability,
    verify_expected_sha256,
)


def window(start: float, logits: dict[int, float | None], end: float | None = None) -> dict:
    return {
        "sample_id": "s", "start": start, "end": start + 0.5 if end is None else end,
        "cluster_logits": json.dumps({str(key): value for key, value in logits.items()}),
    }


def source(source_id: int, cluster: int, start: float, end: float) -> dict:
    return {
        "source_id": source_id, "cluster_id": cluster, "start": start, "end": end,
        "confidence": 0.95,
    }


def test_silent_pause_between_talknet_anchors_is_recovered_by_identity_continuity() -> None:
    timeline = [
        window(0.0, {0: 0.8}), window(0.25, {0: 0.6}),
        window(0.50, {0: -0.4}), window(0.75, {0: 0.5}),
    ]
    rows, _ = build_stage3_visual_timeline(
        sample_id="s", duration=1.0, stage2_timeline=timeline,
        source_rows=[source(0, 0, 0.0, 1.0)], partition_stable=True, av_percentiles={},
    )
    assert rows[2]["segment_status"] == "VERIFIED_CONTINUITY"
    assert rows[2]["propagated_identity"] == 0


def test_competing_active_identity_stops_previous_identity_propagation() -> None:
    timeline = [
        window(0.0, {0: 0.8}), window(0.25, {0: 0.7}),
        window(0.50, {0: -0.3, 1: 0.9}), window(0.75, {0: -0.4}),
    ]
    rows, _ = build_stage3_visual_timeline(
        sample_id="s", duration=1.0, stage2_timeline=timeline,
        source_rows=[source(0, 0, 0.0, 1.0), source(1, 1, 0.50, 0.70)],
        partition_stable=True, av_percentiles={},
    )
    assert rows[2]["segment_status"] == "VERIFIED_ACTIVE"
    assert rows[2]["selected_identity"] == 1
    assert rows[3]["segment_status"] != "VERIFIED_CONTINUITY"


def test_arcface_identity_jump_does_not_propagate_old_anchor() -> None:
    timeline = [window(0.0, {0: 0.9}), window(0.25, {0: 0.7}), window(0.50, {1: -0.2})]
    rows, _ = build_stage3_visual_timeline(
        sample_id="s", duration=0.75, stage2_timeline=timeline,
        source_rows=[source(0, 0, 0.0, 0.50), source(1, 1, 0.50, 0.75)],
        partition_stable=True, av_percentiles={},
    )
    assert rows[2]["segment_status"] == "UNRESOLVED"


def test_no_face_always_remains_missing() -> None:
    rows, _ = build_stage3_visual_timeline(
        sample_id="s", duration=0.5, stage2_timeline=[window(0.0, {})],
        source_rows=[], partition_stable=True, av_percentiles={},
    )
    assert rows[0]["segment_status"] == "MISSING"


def test_propagation_cannot_cross_identity_gap_over_limit() -> None:
    timeline = [window(0.0, {0: 0.9}), window(0.25, {0: 0.8}), window(2.0, {0: -0.2})]
    rows, _ = build_stage3_visual_timeline(
        sample_id="s", duration=2.25, stage2_timeline=timeline,
        source_rows=[source(0, 0, 0.0, 0.50), source(1, 0, 2.0, 2.25)],
        partition_stable=True, av_percentiles={}, max_identity_gap=1.0,
    )
    assert rows[-1]["segment_status"] == "UNRESOLVED"


def test_mapping_strong_is_proposal_only_and_never_modifies_manifest() -> None:
    score_pairs = {name: (0.2, 0.7) for name in ("whisper_token", "whisper_char", "ctc_token", "ctc_char")}
    assignments = {name: "candidate" for name in score_pairs}
    status, reason = classify_mapping_evidence(
        sample_id="original", candidate_id="candidate", assignments=assignments,
        score_pairs=score_pairs, global_true_rank=8, global_candidate_rank=1,
    )
    assert status == "MAPPING_STRONG"
    proposal = mapping_proposal_row("original", "candidate", status, reason)
    assert proposal["proposal_only"] == 1
    assert proposal["manifest_modified"] == 0


def test_resolved_conflict_is_not_counted_as_unresolved() -> None:
    status, _ = classify_resolution_status(
        semantic_status="CONFLICT", temporal_status="NOT_APPLICABLE",
        visual_status="PARTIALLY_VERIFIED",
    )
    assert status == "RESOLVED_CONFLICT"


def test_resolved_missing_is_not_counted_as_unresolved() -> None:
    status, _ = classify_resolution_status(
        semantic_status="MATCH_STRONG", temporal_status="VERIFIED", visual_status="MISSING",
    )
    assert status == "RESOLVED_MISSING"


def test_unstable_mapping_remains_truly_unresolved() -> None:
    status, _ = classify_resolution_status(
        semantic_status="MATCH_STRONG", temporal_status="VERIFIED",
        visual_status="FULLY_VERIFIED", mapping_status="MAPPING_WEAK",
    )
    assert status == "UNRESOLVED"


def test_mapping_strong_becomes_resolved_conflict_even_if_semantic_axis_was_unresolved() -> None:
    status, _ = classify_resolution_status(
        semantic_status="UNRESOLVED", temporal_status="NOT_APPLICABLE",
        visual_status="FULLY_VERIFIED", mapping_status="MAPPING_STRONG",
    )
    assert status == "RESOLVED_CONFLICT"


def test_confirmed_no_face_is_resolved_missing_without_erasing_other_axes() -> None:
    status, _ = classify_resolution_status(
        semantic_status="UNRESOLVED", temporal_status="NOT_APPLICABLE",
        visual_status="MISSING", mapping_status="MAPPING_UNRESOLVED",
    )
    assert status == "RESOLVED_MISSING"



def test_semantic_conflict_audio_mask_must_stay_zero() -> None:
    assert_semantic_audio_mask(["MATCH_STRONG", "CONFLICT", "UNRESOLVED"], np.array([1, 0, 0], np.uint8))
    with pytest.raises(ValueError, match="audio mask"):
        assert_semantic_audio_mask(["MATCH_STRONG", "CONFLICT"], np.array([1, 1], np.uint8))


def test_candidate_path_cannot_overwrite_stage2_or_formal_input(tmp_path) -> None:
    formal = tmp_path / "formal.npz"
    stage2 = tmp_path / "stage2.npz"
    with pytest.raises(ValueError, match="overwrite protected"):
        assert_candidate_path_safe(stage2, [formal, stage2])
    assert_candidate_path_safe(tmp_path / "stage3.npz", [formal, stage2])


def test_frozen_hash_detects_baseline_mutation(tmp_path) -> None:
    path = tmp_path / "baseline.bin"
    path.write_bytes(b"frozen")
    expected = hashlib.sha256(b"frozen").hexdigest()
    assert verify_expected_sha256(path, expected)
    path.write_bytes(b"changed")
    assert not verify_expected_sha256(path, expected)


def test_stage3_source_ids_must_trace_to_frozen_raw_rows() -> None:
    rows = [{"sample_index": 3, "source_id": 0}, {"sample_index": 3, "source_id": 1}]
    assert_source_traceability([3, 3], [0, 1], rows)
    with pytest.raises(ValueError, match="untraceable"):
        assert_source_traceability([3], [2], rows)
