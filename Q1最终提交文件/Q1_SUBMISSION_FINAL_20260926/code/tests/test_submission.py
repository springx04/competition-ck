from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from q1_features.submission import inspect_submission, verify_manifest, write_manifest


PROJECT = Path(__file__).resolve().parents[1]
FINAL_BUNDLE = PROJECT / "runs/q1_final_20260926/submission_q1_final"


def test_final_submission_quick_and_full() -> None:
    quick = inspect_submission(FINAL_BUNDLE, mode="quick")
    full = inspect_submission(FINAL_BUNDLE, mode="full")
    assert quick["ok"] is True
    assert full["ok"] is True
    assert full["sample_count"] == 100
    assert full["word_count"] == 1932
    assert full["paired_use"] == 68


def test_manifest_detects_tampering(tmp_path: Path) -> None:
    payload = tmp_path / "payload.txt"
    payload.write_text("frozen\n", encoding="utf-8")
    write_manifest(tmp_path, [("payload.txt", "test")])
    assert verify_manifest(tmp_path)["ok"] is True
    payload.write_text("modified\n", encoding="utf-8")
    result = verify_manifest(tmp_path)
    assert result["ok"] is False
    assert any("mismatch" in error for error in result["errors"])


def test_manifest_rejects_unexpected_file(tmp_path: Path) -> None:
    payload = tmp_path / "payload.txt"
    payload.write_text("frozen\n", encoding="utf-8")
    write_manifest(tmp_path, [("payload.txt", "test")])
    (tmp_path / "extra.txt").write_text("undeclared\n", encoding="utf-8")
    result = verify_manifest(tmp_path)
    assert result["ok"] is False
    assert any("unexpected files" in error for error in result["errors"])


def test_submission_size_guard(tmp_path: Path) -> None:
    script = PROJECT / "scripts/package_q1_final_submission.py"
    spec = importlib.util.spec_from_file_location("package_q1_final_submission", script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    archive = tmp_path / "release.zip"
    archive.write_bytes(b"12345")
    module.ensure_size_limit(archive, 5)
    with pytest.raises(ValueError, match="exceeds size limit"):
        module.ensure_size_limit(archive, 4)
