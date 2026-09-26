from __future__ import annotations

import numpy as np

from q1_features.finalization import (
    EXPECTED_RESOLUTION_COUNTS,
    validate_csr,
    validate_masked_features,
    validate_source_sample_membership,
    validate_status_rows,
    verify_freeze_manifest,
)


def status_rows() -> list[dict]:
    rows = []
    index = 0
    for status, count in EXPECTED_RESOLUTION_COUNTS.items():
        for _ in range(count):
            paired = int(index < 68 and status in {"RESOLVED_VALID", "RESOLVED_PARTIAL"})
            rows.append({"sample_id": f"s{index}", "resolution_status": status, "paired_use": paired})
            index += 1
    # The construction above may not allocate all 68 safe rows because of status order.
    for row in rows:
        row["paired_use"] = 0
    for row in [item for item in rows if item["resolution_status"] in {"RESOLVED_VALID", "RESOLVED_PARTIAL"}][:68]:
        row["paired_use"] = 1
    return rows


def test_final_status_fixed_counts_and_paired_use() -> None:
    assert validate_status_rows(status_rows()) == []


def test_unsafe_resolution_cannot_be_paired() -> None:
    rows = status_rows()
    target = next(row for row in rows if row["resolution_status"] == "UNRESOLVED")
    donor = next(row for row in rows if row["paired_use"] == 1)
    target["paired_use"] = 1
    donor["paired_use"] = 0
    assert any("unsafe_paired_use" in error for error in validate_status_rows(rows))


def test_csr_validation_catches_nonmonotonic_and_negative_overlap() -> None:
    errors = validate_csr(
        np.array([0, 2, 1]), np.array([0]), np.array([-0.1]), np.array([1.0])
    )
    assert "nonmonotonic_indptr" in errors
    assert "negative_overlap" in errors
    assert "csr_length_mismatch" not in errors


def test_masked_features_must_be_zero_and_unmasked_finite() -> None:
    values = np.array([[1.0, 2.0], [0.0, 0.0]], np.float32)
    mask = np.array([1, 0], np.uint8)
    assert validate_masked_features(values, mask, "audio") == []
    values[1, 0] = 1.0
    assert "nonzero_masked_audio" in validate_masked_features(values, mask, "audio")


def test_source_mapping_cannot_cross_sample_boundary() -> None:
    indptr = np.array([0, 2, 3])
    word_samples = np.array([4, 5])
    source_samples = np.array([4, 4, 5])
    assert validate_source_sample_membership(indptr, word_samples, source_samples, "vision") == []
    source_samples[-1] = 4
    assert validate_source_sample_membership(indptr, word_samples, source_samples, "vision")


def test_frozen_manifest_preserves_intentional_missing_state(tmp_path) -> None:
    rows = [{"path": "missing.csv", "exists": 0, "bytes": 0, "sha256": ""}]
    assert verify_freeze_manifest(tmp_path, rows)["ok"]
    (tmp_path / "missing.csv").write_text("unexpected")
    result = verify_freeze_manifest(tmp_path, rows)
    assert not result["ok"]
    assert result["changed"][0]["expected_exists"] is False
