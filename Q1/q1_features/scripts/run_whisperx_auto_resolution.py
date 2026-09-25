from __future__ import annotations

import argparse
import json
import platform
import sys
from importlib.metadata import version
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run frozen WhisperX diagnostics for all Q1 samples")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="large-v2")
    parser.add_argument("--align-model", default="WAV2VEC2_ASR_LARGE_LV60K_960H")
    parser.add_argument("--language", default="en")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compute-type", default="float16")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--vad-method", choices=["silero", "pyannote"], default="silero")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    import torch
    import whisperx

    run_dir = Path(args.run_dir).resolve()
    output = Path(args.output_dir).resolve()
    raw_dir = output / "whisperx_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest = [json.loads(line) for line in (run_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(manifest) != 100:
        raise ValueError(f"expected 100 samples, found {len(manifest)}")

    model = whisperx.load_model(
        args.model,
        args.device,
        compute_type=args.compute_type,
        language=args.language,
        vad_method=args.vad_method,
    )
    align_model, align_metadata = whisperx.load_align_model(
        language_code=args.language,
        device=args.device,
        model_name=args.align_model,
    )
    completed = 0
    failed = 0
    for item in manifest:
        index = int(item["sample_index"])
        target = raw_dir / f"{index:06d}.json"
        if args.resume and target.is_file():
            existing = json.loads(target.read_text(encoding="utf-8"))
            if existing.get("sample_id") == item["sample_id"] and existing.get("status") == "ok":
                completed += 1
                continue
        audio_path = run_dir / "samples" / f"{index:06d}" / "audio_16k.wav"
        try:
            audio = whisperx.load_audio(str(audio_path))
            transcription = model.transcribe(
                audio,
                batch_size=args.batch_size,
                language=args.language,
            )
            aligned = whisperx.align(
                transcription["segments"],
                align_model,
                align_metadata,
                audio,
                args.device,
                return_char_alignments=False,
            )
            segments = aligned.get("segments", [])
            words = []
            for segment in segments:
                for word in segment.get("words", []):
                    if str(word.get("word", "")).strip():
                        words.append(
                            {
                                "word": str(word.get("word", "")).strip(),
                                "start": word.get("start"),
                                "end": word.get("end"),
                                "score": word.get("score"),
                            }
                        )
            transcript = " ".join(str(segment.get("text", "")).strip() for segment in segments).strip()
            payload = {
                "sample_index": index,
                "sample_id": item["sample_id"],
                "status": "ok",
                "language": transcription.get("language", args.language),
                "transcript": transcript,
                "segments": segments,
                "words": words,
            }
            completed += 1
        except Exception as exc:
            payload = {
                "sample_index": index,
                "sample_id": item["sample_id"],
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "transcript": "",
                "segments": [],
                "words": [],
            }
            failed += 1
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"sample_index": index, "sample_id": item["sample_id"], "status": payload["status"]}, ensure_ascii=False), flush=True)

    metadata = {
        "whisperx_version": version("whisperx"),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "model": args.model,
        "align_model": args.align_model,
        "language": args.language,
        "device": args.device,
        "compute_type": args.compute_type,
        "batch_size": args.batch_size,
        "vad_method": args.vad_method,
        "sample_count": len(manifest),
        "completed": completed,
        "failed": failed,
        "argv": sys.argv,
    }
    (output / "whisperx_run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
