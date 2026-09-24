from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from q1_features.manifest import Sample
from q1_features.quality import audit_pairing
from q1_features.runner import collect_run, stage_pool, validate_run
from q1_features.storage import load_npz, write_json, write_jsonl, write_npz


def sample(index: int) -> Sample:
    return Sample(index, f"v$_${index}", "v", str(index), f"v/{index}.mp4", "hello", "label", index + 2, "ok")


def compact(index: int, value: float) -> dict[str, np.ndarray]:
    return {
        "text": np.full((50, 768), value, np.float32),
        "audio": np.full((50, 25), value, np.float32),
        "vision": np.full((50, 22), value, np.float32),
        "valid_mask": np.ones(50, np.uint8),
        "observed_mask": np.ones((50, 3), np.uint8),
        "perturb_mask": np.zeros((50, 3), np.uint8),
        "coverage": np.ones((50, 3), np.float32),
        "time_intervals": np.column_stack((np.arange(50), np.arange(1, 51))).astype(np.float64),
        "duration": np.asarray(50.0, np.float64),
        "valid_length": np.asarray(50, np.int64),
        "sample_index": np.asarray(index, np.int64),
        "paired_use": np.asarray(1, np.uint8),
    }


def test_collect_preserves_failed_middle_manifest_row(tmp_path: Path) -> None:
    manifest = [sample(0), sample(1), sample(2)]
    for item in (manifest[0], manifest[2]):
        directory = tmp_path / "samples" / f"{item.sample_index:06d}"
        write_npz(directory / "compact50.npz", **compact(item.sample_index, float(item.sample_index + 1)))
    collect_run(tmp_path, manifest)
    result = load_npz(tmp_path / "features" / "q1_compact50.npz")
    assert result["text"].shape == (3, 50, 768)
    assert np.all(result["text"][1] == 0)
    assert result["valid_length"].tolist() == [50, 0, 50]
    assert result["sample_index"].tolist() == [0, 1, 2]


def test_suspected_whole_segment_pairing_keeps_native_bert_but_hides_text_bins(tmp_path: Path) -> None:
    item = sample(0)
    sample_dir = tmp_path / "samples" / "000000"
    write_json(sample_dir / "status.json", {
        "sample_id": item.sample_id, "sample_index": 0, "stages": {},
        "processing_status": "partial", "pairing_status": "suspected", "paired_use": False,
    })
    write_json(sample_dir / "media.json", {"duration": 1.0})
    write_jsonl(sample_dir / "words.jsonl", [{
        "word_id": 0, "raw_word": "hello", "accepted_interval": [0.0, 1.0],
        "alignment_reasons": [],
    }])
    write_npz(sample_dir / "bert_words.npz", word_features=np.ones((1, 768), np.float32), word_ids=np.array([0], np.int64))
    cfg = {"pool": {"angle_resultant_min": 1e-6, "bins": 50}}
    stage_pool(cfg, tmp_path, [item])
    pooled = load_npz(sample_dir / "compact50.npz")
    native = load_npz(sample_dir / "native.npz")
    assert np.all(pooled["observed_mask"][:, 0] == 0)
    assert np.all(pooled["text"] == 0)
    assert np.all(native["word_features"] == 1)
    assert native["word_eligible"].tolist() == [0]


def test_validator_recomputes_native_csr_and_union_coverage(tmp_path: Path) -> None:
    item = sample(0)
    sample_dir = tmp_path / "samples" / "000000"
    write_json(sample_dir / "status.json", {
        "sample_id": item.sample_id, "sample_index": 0, "stages": {},
        "processing_status": "partial", "pairing_status": "no_issue_detected",
        "paired_use": True, "text_time_policy": "accept",
    })
    write_json(sample_dir / "media.json", {"duration": 1.0})
    write_jsonl(sample_dir / "words.jsonl", [{
        "word_id": 0, "raw_word": "hello", "accepted_interval": [0.0, 0.03],
        "alignment_reasons": [],
    }])
    write_npz(sample_dir / "bert_words.npz", word_features=np.full((1, 768), 2.0, np.float32), word_ids=np.array([0], np.int64))
    cfg = {"pool": {"angle_resultant_min": 1e-6, "bins": 50}}
    stage_pool(cfg, tmp_path, [item])
    collect_run(tmp_path, [item])
    assert validate_run(tmp_path, [item], {item.sample_id}) == []
    report = json.loads((tmp_path / "reports" / "validation.json").read_text(encoding="utf-8"))
    assert report["recomputed_observed_window_count"] == 2
    assert report["seed_2026_recomputed_examples"]


def test_automatic_mismatch_alarm_is_quarantined_until_review() -> None:
    result = audit_pairing(
        diagnostic_wer=0.9, aligned_word_fraction=0.4,
        automatic_issues=[], reviews=[], wer_warn=0.65, aligned_fraction_warn=0.80,
    )
    assert result["pairing_status"] == "suspected"
    assert result["paired_use"] is False
    assert {issue["issue_type"] for issue in result["issues"]} == {
        "suspected_text_audio_mismatch", "low_aligned_word_fraction"
    }
    assert result["text_time_policy"] == "quarantine_all"


def test_audio_video_mismatch_does_not_destroy_accepted_text_times() -> None:
    result = audit_pairing(
        diagnostic_wer=0.1, aligned_word_fraction=1.0, automatic_issues=[],
        reviews=[{
            "issue_type": "audio_video_mismatch", "start": "", "end": "",
            "review_status": "confirmed_mismatch", "action": "quarantine_pairing",
            "evidence": "different scene",
        }],
    )
    assert result["paired_use"] is False
    assert result["text_time_policy"] == "accept"


def test_local_text_mismatch_yields_explicit_quarantine_interval() -> None:
    result = audit_pairing(
        diagnostic_wer=0.1, aligned_word_fraction=1.0, automatic_issues=[],
        reviews=[{
            "issue_type": "text_audio_mismatch", "start": "1.25", "end": "2.0",
            "review_status": "confirmed_mismatch", "action": "quarantine_pairing",
            "evidence": "one substituted segment",
        }],
    )
    assert result["text_time_policy"] == "quarantine_intervals"
    assert result["text_quarantine_intervals"] == [[1.25, 2.0]]


def test_review_must_release_every_automatic_issue_before_pairing_is_restored() -> None:
    result = audit_pairing(
        diagnostic_wer=0.9, aligned_word_fraction=0.4, automatic_issues=[],
        reviews=[{
            "issue_type": "suspected_text_audio_mismatch", "start": "", "end": "",
            "review_status": "confirmed_match", "action": "release_pairing",
            "evidence": "full clip checked",
        }],
    )
    assert result["pairing_status"] == "suspected"
    assert result["paired_use"] is False
    assert result["text_time_policy"] == "quarantine_all"


def test_complete_matching_review_releases_automatic_text_alarm() -> None:
    result = audit_pairing(
        diagnostic_wer=0.9, aligned_word_fraction=1.0, automatic_issues=[],
        reviews=[{
            "issue_type": "suspected_text_audio_mismatch", "start": "", "end": "",
            "review_status": "confirmed_match", "action": "release_pairing",
            "evidence": "full clip checked",
        }],
    )
    assert result["pairing_status"] == "reviewed_match"
    assert result["paired_use"] is True
    assert result["text_time_policy"] == "accept"


def test_invalid_review_status_action_pair_is_rejected() -> None:
    with pytest.raises(ValueError, match="status/action"):
        audit_pairing(
            diagnostic_wer=0.1, aligned_word_fraction=1.0, automatic_issues=[],
            reviews=[{
                "issue_type": "text_audio_mismatch", "start": "", "end": "",
                "review_status": "confirmed_match", "action": "quarantine_pairing",
                "evidence": "contradictory row",
            }],
        )
