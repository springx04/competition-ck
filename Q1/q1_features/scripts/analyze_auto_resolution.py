from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from q1_features.auto_resolution import (
    build_text_permutation_test,
    classify_text_audio_evidence,
    read_csv,
    summarize_whisperx_sample,
    write_problem_partition,
)
from q1_features.storage import read_json, read_jsonl, write_csv, write_json


WHISPER_FIELDS = [
    "sample_id", "video_id", "status", "normalized_official_text",
    "normalized_whisperx_transcript", "whisperx_word_timestamps_json",
    "whisperx_wer", "whisperx_cer", "word_overlap_count",
    "word_overlap_precision", "word_overlap_recall", "word_overlap_f1",
    "word_overlap_jaccard", "ctc_diagnostic_wer",
    "ctc_accepted_word_fraction", "common_timed_word_count",
    "common_timed_word_coverage", "median_start_difference_s",
    "p90_start_difference_s", "median_end_difference_s",
    "p90_end_difference_s", "median_midpoint_difference_s",
    "p90_midpoint_difference_s", "coverage",
]
WORD_COMPARISON_FIELDS = [
    "sample_id", "word_id", "raw_word", "ctc_start", "ctc_end",
    "whisperx_start", "whisperx_end", "start_abs_difference_s",
    "end_abs_difference_s", "midpoint_abs_difference_s", "whisperx_score",
]
PERMUTATION_FIELDS = [
    "sample_id", "video_id", "negative_count", "true_similarity",
    "negative_similarity_median", "negative_similarity_p95", "true_rank",
    "empirical_percentile", "p_like_score", "negative_sample_ids_json",
]
EVIDENCE_FIELDS = [
    "sample_id", "ta_evidence_level", "text_audio_status", "semantic_support",
    "ctc_support", "boundary_support", "evidence_reason",
]


def command_partition(args: argparse.Namespace) -> dict:
    return write_problem_partition(args.quality_csv, args.alignment_issues_csv, args.output)


def command_text(args: argparse.Namespace) -> dict:
    run = Path(args.run_dir).resolve()
    output = Path(args.output_dir).resolve()
    raw_dir = output / "whisperx_raw"
    manifest = read_jsonl(run / "manifest.jsonl")
    if len(manifest) != 100:
        raise ValueError(f"expected 100 manifest rows, found {len(manifest)}")
    enriched_manifest = []
    whisper_rows = []
    word_rows = []
    failures = []
    for item in manifest:
        index = int(item["sample_index"])
        sample_dir = run / "samples" / f"{index:06d}"
        raw_path = raw_dir / f"{index:06d}.json"
        if not raw_path.is_file():
            raise FileNotFoundError(f"missing WhisperX result: {raw_path}")
        raw = read_json(raw_path)
        if raw.get("sample_id") != item["sample_id"]:
            raise ValueError(f"WhisperX sample mismatch for {raw_path}")
        if raw.get("status") != "ok":
            failures.append({"sample_id": item["sample_id"], "error": raw.get("error", "")})
        diagnostic = read_json(sample_dir / "diagnostic.json", {})
        words = read_jsonl(sample_dir / "words.jsonl")
        summary, comparisons = summarize_whisperx_sample(
            item["sample_id"],
            item["raw_text"],
            words,
            raw,
            float(diagnostic.get("diagnostic_wer", 0.0) or 0.0),
        )
        summary["video_id"] = item["video_id"]
        whisper_rows.append(summary)
        word_rows.extend(comparisons)
        media = read_json(sample_dir / "media.json", {})
        enriched_manifest.append({**item, "duration": float(media["duration"])})

    write_csv(output / "whisperx_results.csv", whisper_rows, WHISPER_FIELDS)
    write_csv(output / "ctc_whisperx_word_comparison.csv", word_rows, WORD_COMPARISON_FIELDS)
    permutation_rows = build_text_permutation_test(
        enriched_manifest,
        whisper_rows,
        negative_count=args.negative_count,
    )
    write_csv(output / "text_audio_permutation_test.csv", permutation_rows, PERMUTATION_FIELDS)
    permutation_by_id = {row["sample_id"]: row for row in permutation_rows}
    evidence_rows = []
    for row in whisper_rows:
        level, reason, flags = classify_text_audio_evidence(row, permutation_by_id[row["sample_id"]])
        evidence_rows.append(
            {
                "sample_id": row["sample_id"],
                "ta_evidence_level": level,
                "text_audio_status": {
                    "TA_STRONG": "verified",
                    "TA_SEMANTIC_ONLY": "timing_partial",
                    "TA_CONFLICT": "uncertain",
                }[level],
                **{key: int(value) for key, value in flags.items()},
                "evidence_reason": reason,
            }
        )
    write_csv(output / "text_audio_evidence.csv", evidence_rows, EVIDENCE_FIELDS)
    result = {
        "sample_count": len(whisper_rows),
        "whisperx_ok": sum(row["status"] == "ok" for row in whisper_rows),
        "whisperx_failed": len(failures),
        "word_comparison_rows": len(word_rows),
        "negative_count": args.negative_count,
        "ta_levels": {},
        "failures": failures,
    }
    for row in evidence_rows:
        result["ta_levels"][row["ta_evidence_level"]] = result["ta_levels"].get(row["ta_evidence_level"], 0) + 1
    write_json(output / "text_audio_analysis_summary.json", result)
    return result


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Q1 automatic multi-evidence analysis")
    sub = root.add_subparsers(dest="command", required=True)
    partition = sub.add_parser("partition")
    partition.add_argument("--quality-csv", required=True)
    partition.add_argument("--alignment-issues-csv", required=True)
    partition.add_argument("--output", required=True)
    text = sub.add_parser("text")
    text.add_argument("--run-dir", required=True)
    text.add_argument("--output-dir", required=True)
    text.add_argument("--negative-count", type=int, default=20)
    return root


def main() -> int:
    args = parser().parse_args()
    result = command_partition(args) if args.command == "partition" else command_text(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
