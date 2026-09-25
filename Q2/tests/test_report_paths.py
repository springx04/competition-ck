from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from q2.report_builder import test_report_directory as resolve_test_report


def test_report_reads_completed_test_directory_and_keeps_legacy_fallback(tmp_path):
    test_root = tmp_path / "reports/test"
    test_root.mkdir(parents=True)
    assert resolve_test_report(tmp_path) == test_root
    (test_root / "latest.json").write_text(json.dumps({"directory": "reports/test/run_time"}))
    assert resolve_test_report(tmp_path) == test_root / "run_time"
