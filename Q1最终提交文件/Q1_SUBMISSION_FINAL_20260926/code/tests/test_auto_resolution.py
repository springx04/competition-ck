from __future__ import annotations

import json

import numpy as np
import pytest

from q1_features.auto_resolution import (
    build_problem_partition,
    build_text_permutation_test,
    classify_text_audio_evidence,
    normalize_text,
    text_similarity,
)
from q1_features.auto_vision import connected_components, robust_centroid
from q1_features.storage import write_csv


def test_normalization_and_similarity_are_deterministic() -> None:
    assert normalize_text("We're, ready!") == "WE'RE READY"
    assert text_similarity("A short test", "a short test") == 1.0
    assert text_similarity("A short test", "different words") < 1.0


def test_text_evidence_preserves_semantic_only_distinction() -> None:
    whisper = {
        "word_overlap_f1": 0.9,
        "ctc_diagnostic_wer": 0.8,
        "ctc_accepted_word_fraction": 0.5,
        "common_timed_word_coverage": 0.4,
        "median_midpoint_difference_s": 0.2,
        "p90_midpoint_difference_s": 0.4,
    }
    permutation = {"true_rank": 1, "empirical_percentile": 0.975}
    level, _, flags = classify_text_audio_evidence(whisper, permutation)
    assert level == "TA_SEMANTIC_ONLY"
    assert flags == {"semantic_support": True, "ctc_support": False, "boundary_support": False}


def test_text_permutation_controls_exclude_same_video() -> None:
    manifest = [
        {"sample_id": "s0", "video_id": "v0", "raw_text": "alpha beta", "duration": 2.0},
        {"sample_id": "s1", "video_id": "v0", "raw_text": "same video", "duration": 2.0},
        {"sample_id": "s2", "video_id": "v1", "raw_text": "gamma delta", "duration": 2.0},
        {"sample_id": "s3", "video_id": "v2", "raw_text": "epsilon zeta", "duration": 2.0},
    ]
    whisper = [
        {"sample_id": row["sample_id"], "normalized_whisperx_transcript": row["raw_text"]}
        for row in manifest
    ]
    rows = build_text_permutation_test(manifest, whisper, negative_count=2)
    chosen = json.loads(next(row for row in rows if row["sample_id"] == "s0")["negative_sample_ids_json"])
    assert "s1" not in chosen
    assert set(chosen) == {"s2", "s3"}


def test_partition_gate_fails_closed_on_unexpected_counts(tmp_path) -> None:
    quality = tmp_path / "quality.csv"
    issues = tmp_path / "issues.csv"
    write_csv(quality, [{"sample_id": "s0", "paired_use": 0, "vision_episode_count": 2}], ["sample_id", "paired_use", "vision_episode_count"])
    write_csv(issues, [{"sample_id": "s0", "issue_type": "identity_unknown"}], ["sample_id", "issue_type"])
    with pytest.raises(ValueError, match="expected 51"):
        build_problem_partition(quality, issues)


def test_arcface_components_use_frozen_threshold() -> None:
    keys = ["0:0", "1:1", "2:2"]
    similarities = {("0:0", "1:1"): 0.61, ("1:1", "2:2"): 0.59, ("0:0", "2:2"): 0.20}
    assert connected_components(keys, similarities, 0.60) == [["0:0", "1:1"], ["2:2"]]


def test_robust_centroid_is_finite_and_normalized() -> None:
    embeddings = np.asarray([[1.0, 0.0], [0.99, 0.01], [-1.0, 0.0]], dtype=np.float32)
    centroid, dispersion, retained = robust_centroid(embeddings, trim_fraction=0.34)
    assert np.isfinite(centroid).all()
    assert np.linalg.norm(centroid) == pytest.approx(1.0)
    assert dispersion >= 0.0
    assert retained >= 1
