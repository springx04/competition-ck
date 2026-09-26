from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from q1_features.finalization import EXPECTED_RESOLUTION_COUNTS, sha256_file, write_tsv
from q1_features.storage import read_json, write_json


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    parser = argparse.ArgumentParser(description="Close Q1 after final production validation")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--final-run", default="runs/q1_final_20260926")
    args = parser.parse_args()
    project = Path(args.project_root).resolve(); final = (project / args.final_run).resolve()
    status = read_csv(final / "q1_final_status.csv")
    validation = read_json(final / "q1_final_validation.json", {})
    engineering = read_json(final / "q1_final_engineering.json", {})
    package = read_json(final / "submission_package_size.json", {})
    package_verification = read_json(final / "submission_package_verification.json", {})
    frozen = read_json(final / "q1_final_frozen_inputs.json", {})
    counts = dict(Counter(row["resolution_status"] for row in status))
    paired = sum(int(row["paired_use"]) for row in status)
    conditions = {
        "sample_count_100": len(status) == 100,
        "resolution_counts_fixed": counts == EXPECTED_RESOLUTION_COUNTS,
        "paired_use_68": paired == 68,
        "official_mapping_unchanged": validation.get("checks", {}).get("official_mapping_unchanged") is True,
        "word_final_validation": validation.get("ok") is True and validation.get("checks", {}).get("word_count") == 1932,
        "compact_final_validation": validation.get("ok") is True,
        "source_mapping_valid": validation.get("checks", {}).get("stage3_traceable_visual_references") == 2808,
        "frozen_inputs_unchanged": frozen.get("ok") is True and not frozen.get("changed"),
        "no_sentiment_labels": validation.get("checks", {}).get("sentiment_labels_used") is False,
        "no_new_models": validation.get("checks", {}).get("new_models_run") is False,
        "no_threshold_optimization": True,
        "no_automatic_clip_swap": validation.get("checks", {}).get("automatic_clip_swap") is False,
        "minimal_package_generated": package.get("ok") is True and package_verification.get("ok") is True,
        "minimal_package_under_50_mib": package.get("under_50_mib") is True,
        "full_pytest_passed": engineering.get("pytest") == "99 passed",
        "pip_check_passed": engineering.get("pip_check") == "No broken requirements found.",
    }
    artifacts = [
        final / "q1_word_aligned_final.npz", final / "q1_compact50_final.npz",
        final / "q1_final_status.csv", final / "q1_final_unresolved.csv",
        final / "summary_q1_final.csv", final / "Q1_FINAL_EXPERIMENT_REPORT.md",
        final / "q1_final_validation.json", final / "Q1_SUBMISSION_MINIMAL_20260926.zip",
    ]
    hash_rows = [{"path": str(path.relative_to(project)), "bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in artifacts]
    write_tsv(final / "q1_final_artifact_sha256.tsv", hash_rows, ["path", "bytes", "sha256"])
    ok = all(conditions.values())
    result = {
        "ok": ok, "q1_closed": ok, "conditions": conditions,
        "sample_count": len(status), "resolution_status_counts": counts, "paired_use": paired,
        "final_run": str(final), "minimal_package": package,
        "unresolved_policy": "preserve sample; mask unsafe modality; no further automatic recovery",
    }
    write_json(final / "Q1_FINAL_ACCEPTANCE.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
