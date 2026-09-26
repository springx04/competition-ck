from __future__ import annotations

import numpy as np
import pytest

from scripts.finalize_alignment_repair_stage2 import _raw_vision_candidate_series, _verify_baseline
from q1_features.extractors.vision import VISION_COLUMNS

from q1_features.alignment_repair import (
    build_talknet_identity_timeline,
    classify_semantic_pairing,
    classify_temporal_alignment,
    decide_active_speaker_window,
    mapping_candidate_rows,
    merge_visual_timeline,
    official_alignment_segment,
    usable_word_interval,
)


def metrics(f1: float, chars: float | None = None) -> dict[str, float]:
    return {"token_f1": f1, "character_similarity": f1 if chars is None else chars}


def test_high_ctc_diagnostic_wer_cannot_veto_semantic_match() -> None:
    status, _ = classify_semantic_pairing(
        chain_ok=True, mapping_suspected=False,
        official_whisper=metrics(0.9), official_ctc=metrics(0.0), ctc_whisper=metrics(0.0),
        true_pair_rank=1, top1_minus_true_margin=0.0,
    )
    assert status == "MATCH_STRONG"


def test_diagnostic_wer_does_not_determine_temporal_status() -> None:
    status, _ = classify_temporal_alignment(
        "MATCH_STRONG", forced_alignment_success=True, forced_alignment_coverage=1.0,
        common_coverage=0.9, midpoint_median_s=0.2, midpoint_p90_s=0.5,
    )
    assert status == "VERIFIED"


def test_semantic_conflict_has_no_usable_word_audio_interval() -> None:
    assert usable_word_interval("CONFLICT", {"start": 0.1, "end": 0.4}) is None
    status, _ = classify_temporal_alignment(
        "CONFLICT", forced_alignment_success=True, forced_alignment_coverage=1.0,
        common_coverage=1.0, midpoint_median_s=0.0, midpoint_p90_s=0.0,
    )
    assert status == "NOT_APPLICABLE"


def test_forced_alignment_segment_uses_official_text() -> None:
    assert official_alignment_segment("OFFICIAL WORDS", 2.5) == {
        "start": 0.0, "end": 2.5, "text": "OFFICIAL WORDS"
    }


def test_off_diagonal_clip_permutation_is_detected() -> None:
    rows = mapping_candidate_rows(["clip1", "clip2"], np.asarray([[0.2, 0.9], [0.8, 0.1]]), off_diagonal_margin=0.1)
    assert [row["mapping_suspected"] for row in rows] == [1, 1]


def test_arcface_identity_alone_does_not_select_active_speaker() -> None:
    assert decide_active_speaker_window([0], {0: None}) == ("no_active", None)


def test_talknet_logit_changes_segment_active_speaker_status() -> None:
    assert decide_active_speaker_window([0], {0: -0.1}) == ("no_active", None)
    assert decide_active_speaker_window([0], {0: 0.1}) == ("verified", 0)


def test_two_active_clusters_are_ambiguous() -> None:
    assert decide_active_speaker_window([0, 1], {0: 0.2, 1: 0.3}) == ("ambiguous", None)


def test_no_face_window_stays_missing() -> None:
    assert decide_active_speaker_window([], {}) == ("missing", None)



def test_timeline_ignores_rows_without_face_identity() -> None:
    rows = build_talknet_identity_timeline(
        sample_id="s", duration=0.5, episode_to_cluster={},
        visible_rows=[{"face_id": None, "episode_id": None, "start": 0.0, "end": 0.5}],
        talknet_rows=[], av_percentiles={},
    )
    assert rows[0]["window_status"] == "missing"


def test_local_visual_ambiguity_does_not_zero_whole_sample() -> None:
    timeline = [
        {"start": 0.0, "window_status": "verified", "active_cluster": 0, "av_sync_evidence": "support", "quality_flag": ""},
        {"start": 0.25, "window_status": "ambiguous", "active_cluster": "", "av_sync_evidence": "neutral", "quality_flag": ""},
        {"start": 0.50, "window_status": "verified", "active_cluster": 0, "av_sync_evidence": "support", "quality_flag": ""},
    ]
    segments = merge_visual_timeline(timeline, duration=0.75, stride_s=0.25)
    assert [row["status"] for row in segments] == ["verified", "ambiguous", "verified"]
    assert sum(row["end"] - row["start"] for row in segments if row["status"] == "verified") == pytest.approx(0.5)


def test_timeline_combines_arcface_clusters_with_talknet_logits() -> None:
    visible = [
        {"face_id": 0, "episode_id": 0, "start": 0.0, "end": 1.0},
        {"face_id": 1, "episode_id": 0, "start": 0.0, "end": 1.0},
    ]
    talknet = [
        {"face_id": 0, "episode_id": 0, "start": 0.0, "end": 0.04, "class1_logit": 0.5},
        {"face_id": 1, "episode_id": 0, "start": 0.0, "end": 0.04, "class1_logit": -0.5},
    ]
    rows = build_talknet_identity_timeline(
        sample_id="s", duration=0.5, episode_to_cluster={"0:0": 0, "1:0": 1},
        visible_rows=visible, talknet_rows=talknet, av_percentiles={0: 0.9, 1: 0.1},
    )
    assert rows[0]["window_status"] == "verified"
    assert rows[0]["active_cluster"] == 0
    assert rows[0]["av_sync_evidence"] == "support"


def test_invalid_duration_rejected_for_official_alignment() -> None:
    with pytest.raises(ValueError):
        official_alignment_segment("text", 0.0)


def test_raw_openface_candidate_series_preserves_provenance_and_local_masking() -> None:
    record = {
        "image_index": 3, "face_id": 1, "episode_id": 2,
        "start": 0.25, "end": 0.29, "confidence": 0.95,
        **{column: float(index) for index, column in enumerate(VISION_COLUMNS)},
    }
    values, intervals, eligible, source_ids, sources = _raw_vision_candidate_series(
        [record], {"1:2": 7},
        [{"start": 0.0, "end": 0.5, "status": "verified", "active_cluster": 7}],
    )
    assert values.shape == (1, 22)
    assert intervals.tolist() == [[0.25, 0.29]]
    assert source_ids.tolist() == [0]
    assert eligible.tolist() == [1]
    assert sources[0]["cluster_id"] == 7
    assert sources[0]["eligible"] == 1

    _, _, ambiguous, _, _ = _raw_vision_candidate_series(
        [record], {"1:2": 7},
        [{"start": 0.0, "end": 0.5, "status": "ambiguous", "active_cluster": ""}],
    )
    assert ambiguous.tolist() == [0]


def test_frozen_baseline_verification_detects_changes(tmp_path) -> None:
    project = tmp_path / "project"
    reports = tmp_path / "reports"
    project.mkdir()
    reports.mkdir()
    target = project / "formal.bin"
    target.write_bytes(b"frozen")
    import hashlib
    expected = hashlib.sha256(b"frozen").hexdigest()
    (reports / "frozen_baseline_sha256.tsv").write_text(
        f"role\tpath\tbytes\tsha256\nformal\tformal.bin\t6\t{expected}\n",
        encoding="utf-8",
    )
    assert _verify_baseline(project, reports)["ok"] is True
    target.write_bytes(b"changed")
    result = _verify_baseline(project, reports)
    assert result["ok"] is False
    assert result["changed"][0]["path"] == "formal.bin"
