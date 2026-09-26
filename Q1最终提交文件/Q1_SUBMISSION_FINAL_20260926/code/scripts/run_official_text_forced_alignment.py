from __future__ import annotations

import argparse
import json
import os
import platform
from difflib import SequenceMatcher
from importlib.metadata import version
from pathlib import Path

from q1_features.alignment_repair import normalize_text, official_alignment_segment, sha256_file
from q1_features.storage import read_json, read_jsonl, write_csv, write_json, write_jsonl


def _flatten_words(aligned: dict) -> list[dict]:
    if aligned.get("word_segments"):
        return [dict(row) for row in aligned["word_segments"]]
    return [dict(word) for segment in aligned.get("segments", []) for word in segment.get("words", [])]


def _map_to_official(official_words: list[dict], aligned_words: list[dict]) -> list[dict]:
    left = [normalize_text(str(row.get("raw_word", ""))) for row in official_words]
    right = [normalize_text(str(row.get("word", ""))) for row in aligned_words]
    mapped: dict[int, dict] = {}
    for block in SequenceMatcher(a=left, b=right, autojunk=False).get_matching_blocks():
        for offset in range(block.size):
            mapped[block.a + offset] = aligned_words[block.b + offset]
    rows = []
    for position, official in enumerate(official_words):
        aligned = mapped.get(position, {})
        start, end = aligned.get("start"), aligned.get("end")
        valid = start is not None and end is not None and float(end) > float(start)
        rows.append({
            "word_id": int(official["word_id"]),
            "raw_word": str(official.get("raw_word", "")),
            "start": float(start) if valid else None,
            "end": float(end) if valid else None,
            "score": aligned.get("score") if valid else None,
            "aligned": int(valid),
            "reason": "" if valid else "official_word_not_located_by_forced_alignment",
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Align the official Q1 transcript directly to each frozen audio file")
    parser.add_argument("--baseline-run", required=True)
    parser.add_argument("--output-run", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-name", default="WAV2VEC2_ASR_LARGE_LV60K_960H")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--nltk-data", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    os.environ["NLTK_DATA"] = str(Path(args.nltk_data).resolve())
    import torch
    import whisperx

    baseline = Path(args.baseline_run).resolve()
    output = Path(args.output_run).resolve() / "reports"
    raw_dir = output / "official_forced_alignment_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(args.checkpoint).resolve()
    expected_name = "wav2vec2_fairseq_large_lv60k_asr_ls960.pth"
    if checkpoint.name != expected_name or not checkpoint.is_file():
        raise FileNotFoundError(f"expected official checkpoint named {expected_name}: {checkpoint}")

    manifest = read_jsonl(baseline / "manifest.jsonl")
    if len(manifest) != 100:
        raise ValueError(f"expected 100 samples, found {len(manifest)}")
    model, metadata = whisperx.load_align_model(
        language_code="en",
        device=args.device,
        model_name=args.model_name,
        model_dir=str(checkpoint.parent),
    )

    completed = failed = 0
    for item in manifest:
        index = int(item["sample_index"])
        target = raw_dir / f"{index:06d}.json"
        if args.resume and target.is_file():
            existing = read_json(target, {})
            if existing.get("sample_id") == item["sample_id"] and existing.get("status") == "ok":
                completed += 1
                continue
        sample_dir = baseline / "samples" / f"{index:06d}"
        audio_path = sample_dir / "audio_16k.wav"
        duration = float(read_json(sample_dir / "media.json")["duration"])
        official_words = read_jsonl(sample_dir / "words.jsonl")
        try:
            audio = whisperx.load_audio(str(audio_path))
            aligned = whisperx.align(
                [official_alignment_segment(str(item["raw_text"]), duration)],
                model,
                metadata,
                audio,
                args.device,
                return_char_alignments=False,
            )
            word_rows = _map_to_official(official_words, _flatten_words(aligned))
            timed = sum(int(row["aligned"]) for row in word_rows)
            status = "ok" if timed else "failed"
            payload = {
                "sample_index": index,
                "sample_id": item["sample_id"],
                "status": status,
                "transcript_source": "official_manifest_raw_text",
                "official_text": item["raw_text"],
                "duration": duration,
                "word_count": len(word_rows),
                "timed_word_count": timed,
                "aligned_word_fraction": timed / max(1, len(word_rows)),
                "words": word_rows,
                "error_type": "" if timed else "NoTimedWords",
                "error": "" if timed else "alignment returned no reliable official-word intervals",
            }
        except Exception as exc:
            payload = {
                "sample_index": index,
                "sample_id": item["sample_id"],
                "status": "failed",
                "transcript_source": "official_manifest_raw_text",
                "official_text": item["raw_text"],
                "duration": duration,
                "word_count": len(official_words),
                "timed_word_count": 0,
                "aligned_word_fraction": 0.0,
                "words": [
                    {"word_id": int(row["word_id"]), "raw_word": row.get("raw_word", ""), "start": None, "end": None,
                     "score": None, "aligned": 0, "reason": "forced_alignment_failed"}
                    for row in official_words
                ],
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        write_json(target, payload)
        completed += int(payload["status"] == "ok")
        failed += int(payload["status"] != "ok")
        print(json.dumps({"sample_index": index, "sample_id": item["sample_id"], "status": payload["status"], "coverage": payload["aligned_word_fraction"]}), flush=True)

    records = [read_json(raw_dir / f"{int(item['sample_index']):06d}.json") for item in manifest]
    write_jsonl(output / "official_text_forced_alignment.jsonl", records)
    summary_rows = [{
        "sample_id": row["sample_id"],
        "sample_index": row["sample_index"],
        "status": row["status"],
        "transcript_source": row["transcript_source"],
        "word_count": row["word_count"],
        "timed_word_count": row["timed_word_count"],
        "aligned_word_fraction": row["aligned_word_fraction"],
        "error_type": row["error_type"],
        "error": row["error"],
    } for row in records]
    write_csv(output / "official_text_forced_alignment_summary.csv", summary_rows, list(summary_rows[0]))
    metadata_out = {
        "whisperx_version": version("whisperx"),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": args.device,
        "model_name": args.model_name,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "transcript_source": "official_manifest_raw_text",
        "sample_count": len(records),
        "completed": sum(row["status"] == "ok" for row in records),
        "failed": sum(row["status"] != "ok" for row in records),
        "production_write": False,
    }
    write_json(output / "official_text_forced_alignment_metadata.json", metadata_out)
    print(json.dumps(metadata_out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
