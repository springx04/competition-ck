from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize Q1 automatic-resolution changes against the frozen baseline")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run = Path(args.run_dir).resolve()
    auto = run / "reports" / "auto_resolution"

    quality = read_csv(run / "reports" / "quality.csv")
    candidates = read_csv(auto / "auto_resolution_candidates.csv")
    text_evidence = {row["sample_id"]: row for row in read_csv(auto / "text_audio_evidence.csv")}
    arc = [
        row for row in read_csv(auto / "arcface_identity_clusters.csv")
        if abs(float(row["threshold"]) - 0.60) < 1e-9
    ]
    arc_by_id = {row["sample_id"]: row for row in arc}
    summary = json.loads((auto / "auto_resolution_summary.json").read_text(encoding="utf-8"))

    baseline_paired = sum(int(row["paired_use"]) == 1 for row in quality)
    baseline_isolated = len(quality) - baseline_paired
    identity_rows = [row for row in candidates if int(row["original_identity_issue"]) == 1]
    ta_rows = [row for row in candidates if int(row["original_ta_issue"]) == 1]
    no_face_rows = [row for row in candidates if int(row["original_no_face"]) == 1]
    candidate_distribution = Counter(row["trimodal_candidate_status"] for row in candidates)
    identity_distribution = Counter(row["vision_evidence_level"] for row in identity_rows)
    ta_distribution = Counter(row["ta_evidence_level"] for row in ta_rows)

    paired_ids = [row["sample_id"] for row in quality if int(row["paired_use"]) == 1]
    paired_ta_distribution = Counter(text_evidence[sample_id]["ta_evidence_level"] for sample_id in paired_ids)
    potential_verified = candidate_distribution["verified"]
    result = {
        "comparison_scope": "frozen production baseline versus candidate-only automatic evidence",
        "not_an_accuracy_comparison": True,
        "reason_no_accuracy_delta": "No independent human gold standard exists; production pairing was intentionally not rewritten.",
        "baseline": {
            "samples": len(quality),
            "formal_paired_use_true": baseline_paired,
            "isolated_unverifiable": baseline_isolated,
            "identity_unresolved": len(identity_rows),
            "text_audio_unresolved": len(ta_rows),
            "vision_no_face": len(no_face_rows),
        },
        "candidate_stage": {
            "trimodal": dict(candidate_distribution),
            "identity_risk": dict(identity_distribution),
            "text_audio_alerts": {
                "TA_STRONG": ta_distribution["TA_STRONG"],
                "TA_SEMANTIC_ONLY": ta_distribution["TA_SEMANTIC_ONLY"],
                "TA_CONFLICT": ta_distribution["TA_CONFLICT"],
            },
            "no_face_as_missing": sum(row["vision_evidence_level"] == "V_MISSING" for row in no_face_rows),
            "original_paired_49_text_audio": {
                "TA_STRONG": paired_ta_distribution["TA_STRONG"],
                "TA_SEMANTIC_ONLY": paired_ta_distribution["TA_SEMANTIC_ONLY"],
                "TA_CONFLICT": paired_ta_distribution["TA_CONFLICT"],
            },
            "new_text_audio_risk_ids": summary["new_text_audio_risk_candidate_ids"],
        },
        "change": {
            "formal_paired_sample_change": 0,
            "fully_uncertain_change": candidate_distribution["uncertain"] - baseline_isolated,
            "fully_uncertain_reduction_count": baseline_isolated - candidate_distribution["uncertain"],
            "fully_uncertain_reduction_fraction_of_51": (baseline_isolated - candidate_distribution["uncertain"]) / baseline_isolated,
            "new_strong_candidate_count": potential_verified,
            "new_strong_candidate_fraction_of_51": potential_verified / baseline_isolated,
            "new_partial_candidate_count": candidate_distribution["partial"],
            "new_partial_candidate_fraction_of_51": candidate_distribution["partial"] / baseline_isolated,
            "conditional_formal_paired_if_two_verified_are_approved": baseline_paired + potential_verified,
            "conditional_paired_absolute_percentage_point_change": 100 * potential_verified / len(quality),
            "conditional_paired_relative_count_change": potential_verified / baseline_paired,
            "identity_fully_uncertain_reduction_count": len(identity_rows) - identity_distribution["V_UNCERTAIN"],
            "identity_fully_uncertain_reduction_fraction": (len(identity_rows) - identity_distribution["V_UNCERTAIN"]) / len(identity_rows),
            "text_audio_strong_resolution_count": ta_distribution["TA_STRONG"],
            "text_audio_semantic_only_diagnosis_count": ta_distribution["TA_SEMANTIC_ONLY"],
        },
        "independent_alignment_diagnostics": {
            "whisperx_completed": summary["whisperx_completion"]["ok"],
            "whisperx_total": summary["whisperx_completion"]["total"],
            "ctc_whisperx_midpoint_sample_n": summary["ctc_whisperx_median_midpoint_difference_s"]["n"],
            "ctc_whisperx_midpoint_median_s": summary["ctc_whisperx_median_midpoint_difference_s"]["median"],
            "ctc_whisperx_midpoint_p90_s": summary["ctc_whisperx_median_midpoint_difference_s"]["p90"],
            "production_ctc_replaced": False,
        },
        "arcface_identity_cluster_distribution_for_37": dict(Counter(
            int(arc_by_id[row["sample_id"]]["identity_cluster_count"]) for row in identity_rows
        )),
        "production_write": False,
    }
    target = Path(args.output).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
