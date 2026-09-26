from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from q1_features.alignment_repair import (
    classify_conflict_cause,
    classify_semantic_pairing,
    mapping_candidate_rows,
    sha256_file,
    text_pair_metrics,
)
from q1_features.auto_resolution import read_csv
from q1_features.storage import read_json, read_jsonl, write_csv, write_json


def _hash_text(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def _write_tsv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _frozen_files(project: Path, baseline: Path) -> list[tuple[str, Path]]:
    files = [
        ("source_revision", project / "SOURCE_REVISION"),
        ("main_config", project / "configs" / "q1.yaml"),
        ("formal_alignment_review", project / "configs" / "alignment_review.csv"),
        ("formal_target_face_segments", project / "configs" / "target_face_segments.csv"),
        ("compact50", baseline / "features" / "q1_compact50.npz"),
        ("manifest", baseline / "manifest.jsonl"),
        ("sample_index", baseline / "sample_index.csv"),
        ("quality", baseline / "reports" / "quality.csv"),
        ("alignment_issues", baseline / "reports" / "alignment_issues.csv"),
        ("validation", baseline / "reports" / "validation.json"),
    ]
    for index in range(100):
        sample = baseline / "samples" / f"{index:06d}"
        files.extend([
            ("ctc_alignment", sample / "words.jsonl"),
            ("bert_word_features", sample / "bert_words.npz"),
            ("audio_native_features", sample / "audio_native.npz"),
            ("vision_native_features", sample / "vision_native.npz"),
            ("openface_raw_csv", sample / "openface" / "features.csv"),
            ("video_frame_timeline", sample / "video_frames.jsonl"),
        ])
    return files


def _freeze_baseline(project: Path, baseline: Path, output: Path) -> dict[str, object]:
    rows = []
    for role, path in _frozen_files(project, baseline):
        if not path.is_file() and role != "openface_raw_csv":
            raise FileNotFoundError(path)
        rows.append({
            "role": role,
            "path": str(path.relative_to(project)),
            "bytes": path.stat().st_size if path.is_file() else 0,
            "sha256": sha256_file(path) if path.is_file() else "",
        })
    _write_tsv(output / "frozen_baseline_sha256.tsv", rows, ["role", "path", "bytes", "sha256"])
    combined = hashlib.sha256("\n".join(f"{row['sha256']}  {row['path']}" for row in rows).encode()).hexdigest()
    summary = {"file_count": len(rows), "combined_sha256": combined, "production_write": False}
    write_json(output / "frozen_baseline_summary.json", summary)
    return summary


def _xlsx_rows(path: Path) -> dict[int, dict[str, str]]:
    frame = pd.read_excel(path, sheet_name="label", usecols=["video_id", "clip_id", "text"])
    return {
        excel_row: {
            "video_id": str(row.video_id),
            "clip_id": str(row.clip_id),
            "text": str(row.text),
        }
        for excel_row, row in enumerate(frame.itertuples(index=False), start=2)
    }


def _data_chain_audit(project: Path, data_root: Path, baseline: Path, manifest: list[dict], output: Path) -> list[dict]:
    workbook = data_root / "附件1-数据集原始多模态样本" / "MOSEI数据集部分原始视频-100条" / "label-100.xlsx"
    xlsx = _xlsx_rows(workbook)
    rows = []
    for item in manifest:
        index = int(item["sample_index"])
        excel_row = int(item["excel_row"])
        source = xlsx.get(excel_row)
        mp4 = data_root / str(item["video_relpath"])
        wav = baseline / "samples" / f"{index:06d}" / "audio_16k.wav"
        reasons = []
        if source is None:
            reasons.append("xlsx_row_missing")
        else:
            if source["video_id"] != str(item["video_id"]):
                reasons.append("xlsx_video_id_mismatch")
            if source["clip_id"] != str(item["clip_id"]):
                reasons.append("xlsx_clip_id_mismatch")
            if source["text"].strip() != str(item["raw_text"]).strip():
                reasons.append("xlsx_official_text_mismatch")
        if not mp4.is_file():
            reasons.append("mp4_missing")
        if not wav.is_file():
            reasons.append("wav_missing")
        mp4_hash = sha256_file(mp4) if mp4.is_file() else ""
        wav_hash = sha256_file(wav) if wav.is_file() else ""
        rows.append({
            "sample_id": item["sample_id"],
            "video_id": item["video_id"],
            "clip_id": item["clip_id"],
            "xlsx_row": excel_row,
            "official_text_hash": _hash_text(item["raw_text"]),
            "mp4_path": str(mp4),
            "mp4_sha256": mp4_hash,
            "wav_path": str(wav),
            "wav_sha256": wav_hash,
            "ctc_input": str(wav),
            "whisperx_input": str(wav),
            "chain_ok": int(not reasons),
            "reason": ";".join(reasons),
        })
    write_csv(output / "data_chain_audit.csv", rows, list(rows[0]))
    return rows


def _transcripts_and_metrics(baseline: Path, manifest: list[dict], auto: Path, output: Path) -> list[dict]:
    whisper_results = {row["sample_id"]: row for row in read_csv(auto / "whisperx_results.csv")}
    rows = []
    pairs = (("official_ctc", "official", "ctc"), ("official_whisperx", "official", "whisperx"), ("ctc_whisperx", "ctc", "whisperx"))
    for item in manifest:
        index = int(item["sample_index"])
        diagnostic = read_json(baseline / "samples" / f"{index:06d}" / "diagnostic.json")
        official = str(item["raw_text"])
        ctc = str(diagnostic.get("diagnostic_transcript", ""))
        whisper = str(whisper_results[item["sample_id"]].get("normalized_whisperx_transcript", ""))
        texts = {"official": official, "ctc": ctc, "whisperx": whisper}
        row = {
            "sample_id": item["sample_id"], "video_id": item["video_id"], "clip_id": item["clip_id"],
            "official_transcript": official, "ctc_diagnostic_transcript": ctc, "whisperx_transcript": whisper,
            "ctc_diagnostic_wer_legacy": diagnostic.get("diagnostic_wer", ""),
        }
        for prefix, left, right in pairs:
            for name, value in text_pair_metrics(texts[left], texts[right]).items():
                row[f"{prefix}_{name}"] = value
        rows.append(row)
    write_csv(output / "asr_transcripts_all.csv", rows, list(rows[0]))
    return rows


def _within_video_mapping(manifest: list[dict], transcripts: list[dict], output: Path, margin: float) -> list[dict]:
    by_video: dict[str, list[dict]] = defaultdict(list)
    transcript_by_id = {row["sample_id"]: row for row in transcripts}
    for item in manifest:
        by_video[str(item["video_id"])].append(item)
    matrix_rows = []
    candidate_rows = []
    for video_id, items in sorted(by_video.items()):
        items.sort(key=lambda row: int(row["sample_index"]))
        ids = [str(row["sample_id"]) for row in items]
        matrix = np.zeros((len(ids), len(ids)), dtype=float)
        for i, official_id in enumerate(ids):
            official = transcript_by_id[official_id]["official_transcript"]
            for j, audio_id in enumerate(ids):
                whisper = transcript_by_id[audio_id]["whisperx_transcript"]
                metrics = text_pair_metrics(official, whisper)
                matrix[i, j] = metrics["score"]
                matrix_rows.append({
                    "video_id": video_id,
                    "official_sample_id": official_id,
                    "audio_sample_id": audio_id,
                    "is_diagonal": int(i == j),
                    **metrics,
                })
        for row in mapping_candidate_rows(ids, matrix, off_diagonal_margin=margin):
            row["video_id"] = video_id
            candidate_rows.append(row)
    write_csv(output / "within_video_mapping_matrix.csv", matrix_rows, list(matrix_rows[0]))
    write_csv(output / "within_video_mapping_candidates.csv", candidate_rows, list(candidate_rows[0]))
    return candidate_rows


def _global_retrieval(manifest: list[dict], transcripts: list[dict], output: Path) -> list[dict]:
    transcript_by_id = {row["sample_id"]: row for row in transcripts}
    ids = [str(row["sample_id"]) for row in manifest]
    manifest_by_id = {str(row["sample_id"]): row for row in manifest}
    rows = []
    for sample_id in ids:
        official = transcript_by_id[sample_id]["official_transcript"]
        scored = []
        for audio_id in ids:
            score = text_pair_metrics(official, transcript_by_id[audio_id]["whisperx_transcript"])["score"]
            scored.append((score, audio_id))
        scored.sort(key=lambda value: (-value[0], value[1]))
        true_score = next(score for score, audio_id in scored if audio_id == sample_id)
        true_rank = next(i for i, (_, audio_id) in enumerate(scored, start=1) if audio_id == sample_id)
        top_score, top_id = scored[0]
        rows.append({
            "sample_id": sample_id,
            "video_id": manifest_by_id[sample_id]["video_id"],
            "true_pair_rank": true_rank,
            "true_pair_score": true_score,
            "top1_sample_id": top_id,
            "top1_video_id": manifest_by_id[top_id]["video_id"],
            "top1_score": top_score,
            "top1_minus_true_margin": top_score - true_score,
            "top5": json.dumps([{"sample_id": audio_id, "score": score} for score, audio_id in scored[:5]], separators=(",", ":")),
        })
    write_csv(output / "global_text_audio_retrieval.csv", rows, list(rows[0]))
    return rows


def _semantic_status(
    transcripts: list[dict], chain: list[dict], mappings: list[dict], retrieval: list[dict],
    auto: Path, output: Path,
) -> list[dict]:
    chain_by_id = {row["sample_id"]: row for row in chain}
    mapping_by_id = {row["sample_id"]: row for row in mappings}
    retrieval_by_id = {row["sample_id"]: row for row in retrieval}
    old_evidence = {row["sample_id"]: row for row in read_csv(auto / "text_audio_evidence.csv")}
    rows = []
    for transcript in transcripts:
        sample_id = transcript["sample_id"]
        def pair(prefix: str) -> dict[str, float]:
            return {name: float(transcript[f"{prefix}_{name}"]) for name in ("token_f1", "character_similarity")}
        retrieval_row = retrieval_by_id[sample_id]
        status, reason = classify_semantic_pairing(
            chain_ok=bool(int(chain_by_id[sample_id]["chain_ok"])),
            mapping_suspected=bool(int(mapping_by_id[sample_id]["mapping_suspected"])),
            official_whisper=pair("official_whisperx"),
            official_ctc=pair("official_ctc"),
            ctc_whisper=pair("ctc_whisperx"),
            true_pair_rank=int(retrieval_row["true_pair_rank"]),
            top1_minus_true_margin=float(retrieval_row["top1_minus_true_margin"]),
        )
        cause = classify_conflict_cause(
            chain_ok=bool(int(chain_by_id[sample_id]["chain_ok"])),
            mapping_suspected=bool(int(mapping_by_id[sample_id]["mapping_suspected"])),
            official_whisper_f1=float(transcript["official_whisperx_token_f1"]),
            official_ctc_f1=float(transcript["official_ctc_token_f1"]),
            ctc_whisper_f1=float(transcript["ctc_whisperx_token_f1"]),
        )
        rows.append({
            "sample_id": sample_id,
            "semantic_pairing_status": status,
            "semantic_reason": reason,
            "conflict_cause": cause,
            "legacy_ta_evidence_level": old_evidence[sample_id]["ta_evidence_level"],
            "legacy_ctc_diagnostic_wer": transcript["ctc_diagnostic_wer_legacy"],
            "official_whisper_token_f1": transcript["official_whisperx_token_f1"],
            "official_ctc_token_f1": transcript["official_ctc_token_f1"],
            "ctc_whisper_token_f1": transcript["ctc_whisperx_token_f1"],
            "true_pair_rank": retrieval_row["true_pair_rank"],
            "mapping_suspected": mapping_by_id[sample_id]["mapping_suspected"],
            "chain_ok": chain_by_id[sample_id]["chain_ok"],
        })
    write_csv(output / "semantic_pairing_status.csv", rows, list(rows[0]))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare candidate-only Q1 alignment repair stage 2")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--baseline-run", default="runs/q1_full_20260924")
    parser.add_argument("--output-run", default="runs/q1_alignment_repair_stage2")
    args = parser.parse_args()

    project = Path(args.project_root).resolve()
    data_root = Path(args.data_root).resolve()
    baseline = (project / args.baseline_run).resolve()
    stage = (project / args.output_run).resolve()
    reports = stage / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    shutil.copy2(project / "configs" / "q1_alignment_repair_stage2.yaml", stage / "config.snapshot.yaml")

    manifest = read_jsonl(baseline / "manifest.jsonl")
    if len(manifest) != 100 or len({row["sample_id"] for row in manifest}) != 100:
        raise ValueError("formal manifest must contain 100 unique samples")
    freeze = _freeze_baseline(project, baseline, reports)
    chain = _data_chain_audit(project, data_root, baseline, manifest, reports)
    auto = baseline / "reports" / "auto_resolution"
    transcripts = _transcripts_and_metrics(baseline, manifest, auto, reports)
    mappings = _within_video_mapping(manifest, transcripts, reports, margin=0.10)
    retrieval = _global_retrieval(manifest, transcripts, reports)
    semantic = _semantic_status(transcripts, chain, mappings, retrieval, auto, reports)

    old_conflicts = {row["sample_id"] for row in read_csv(auto / "text_audio_evidence.csv") if row["ta_evidence_level"] == "TA_CONFLICT"}
    if len(old_conflicts) != 20:
        # Stage 1 contains 19 original alerts plus one newly discovered risk.
        raise ValueError(f"expected 20 total TA_CONFLICT rows, found {len(old_conflicts)}")
    original_alerts = {row["sample_id"] for row in read_csv(auto / "problem_partition.csv") if int(row["ta_issue"]) == 1}
    original_conflicts = old_conflicts & original_alerts
    if len(original_conflicts) != 19:
        raise ValueError(f"expected 19 original TA_CONFLICT rows, found {len(original_conflicts)}")
    semantic_by_id = {row["sample_id"]: row for row in semantic}
    summary = {
        "sample_count": len(manifest),
        "data_chain_ok": sum(int(row["chain_ok"]) for row in chain),
        "mapping_suspected": sum(int(row["mapping_suspected"]) for row in mappings),
        "semantic_status_all100": dict(Counter(row["semantic_pairing_status"] for row in semantic)),
        "original_19_conflict_causes": dict(Counter(semantic_by_id[sample_id]["conflict_cause"] for sample_id in original_conflicts)),
        "frozen_baseline": freeze,
        "candidate_only": True,
        "production_write": False,
    }
    write_json(reports / "stage2_prepare_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
