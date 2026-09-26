from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate candidate-only Q1 automatic-resolution stage 1")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    run = Path(args.run_dir).resolve()
    out = Path(args.output_dir).resolve()
    errors: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    partition = read_csv(out / "problem_partition.csv")
    require(len(partition) == 51, f"problem_partition rows={len(partition)} != 51")
    groups = Counter(row["group"] for row in partition)
    expected_groups = {
        "identity_only": 28,
        "text_audio_only_single_episode": 10,
        "identity_and_text_audio": 9,
        "vision_no_face_and_text_audio": 4,
    }
    require(dict(groups) == expected_groups, f"problem partition mismatch: {dict(groups)}")

    whisper = read_csv(out / "whisperx_results.csv")
    require(len(whisper) == 100, f"WhisperX rows={len(whisper)} != 100")
    require(len({row["sample_id"] for row in whisper}) == 100, "WhisperX sample IDs are not unique")
    require(all(row["status"] == "ok" for row in whisper), "WhisperX contains non-ok rows")
    raw = sorted((out / "whisperx_raw").glob("*.json"))
    require(len(raw) == 100, f"WhisperX raw files={len(raw)} != 100")

    permutation = read_csv(out / "text_audio_permutation_test.csv")
    require(len(permutation) == 100, f"permutation rows={len(permutation)} != 100")
    require(all(int(row["negative_count"]) >= 20 for row in permutation), "permutation has fewer than 20 negatives")
    text_evidence = read_csv(out / "text_audio_evidence.csv")
    require(len(text_evidence) == 100, f"text evidence rows={len(text_evidence)} != 100")

    clusters = read_csv(out / "arcface_identity_clusters.csv")
    primary = [row for row in clusters if abs(float(row["threshold"]) - 0.60) < 1e-9]
    require(len(primary) == 100, f"ArcFace primary rows={len(primary)} != 100")
    require(len(clusters) == 300, f"ArcFace sensitivity rows={len(clusters)} != 300")

    av = read_csv(out / "av_sync_scores.csv")
    av_null = read_csv(out / "av_sync_null_test.csv")
    require(len(av) == 137, f"AV-sync cluster rows={len(av)} != 137")
    require(len(av_null) == 3041, f"AV-sync null rows={len(av_null)} != 3041")
    for row in av:
        for key in ("real_sync_score", "empirical_percentile"):
            if row[key] != "":
                value = float(row[key])
                require(math.isfinite(value), f"nonfinite AV {key} for {row['sample_id']}")

    talknet_episode = read_csv(out / "talknet_episode_scores.csv")
    talknet_frame = read_csv(out / "talknet_frame_logits.csv")
    require(len(talknet_episode) == 191, f"TalkNet episodes={len(talknet_episode)} != 191")
    require(sum(row["status"] == "ok" for row in talknet_episode) == 165, "TalkNet ok episode count mismatch")
    require(len(talknet_frame) == 19067, f"TalkNet frame logits={len(talknet_frame)} != 19067")
    for row in talknet_frame:
        logit = float(row["class1_logit"])
        require(math.isfinite(logit), f"nonfinite TalkNet logit for {row['sample_id']}")
        require(int(row["is_speaking"]) == int(logit >= 0.0), f"TalkNet threshold mismatch for {row['sample_id']}")

    candidates = read_csv(out / "auto_resolution_candidates.csv")
    require(len(candidates) == 51, f"candidate rows={len(candidates)} != 51")
    require(len({row["sample_id"] for row in candidates}) == 51, "candidate sample IDs are not unique")
    for row in candidates:
        release = float(row["releasable_visual_time_ratio"])
        masked = float(row["masked_visual_time_ratio"])
        require(0.0 <= release <= 1.0 and 0.0 <= masked <= 1.0, f"invalid ratios for {row['sample_id']}")
        require(abs(release + masked - 1.0) <= 1e-6, f"ratios do not sum to 1 for {row['sample_id']}")
        if int(row["original_no_face"]):
            require(row["vision_evidence_level"] == "V_MISSING", f"no-face not V_MISSING: {row['sample_id']}")
            require(row["vision_candidate_status"] == "missing", f"no-face not missing: {row['sample_id']}")
            require(release == 0.0 and masked == 1.0, f"no-face visual time not fully masked: {row['sample_id']}")

    summary = read_json(out / "auto_resolution_summary.json")
    require(summary.get("candidate_only") is True, "summary is not candidate-only")
    require(summary.get("production_write") is False, "summary claims production write")
    require(summary.get("trimodal_candidate_distribution") == {"uncertain": 39, "partial": 10, "verified": 2}, "trimodal distribution mismatch")
    require(summary.get("new_text_audio_risk_candidate_count") == 2, "new TA risk count mismatch")
    new_risks = read_csv(out / "new_text_audio_risk_candidates.csv")
    require(len(new_risks) == 2, f"new TA risks={len(new_risks)} != 2")

    word_validation = read_json(out / "q1_word_aligned_candidate.validation.json")
    require(word_validation.get("ok") is True and not word_validation.get("errors"), "word-aligned candidate validation failed")
    require(word_validation.get("sample_count") == 100, "word-aligned sample count mismatch")
    baseline = read_json(out / "frozen_baseline_verification.json")
    require(baseline.get("ok") is True, "frozen production baseline changed")

    result = {
        "ok": not errors,
        "errors": errors,
        "counts": {
            "partition": dict(groups), "whisperx": len(whisper),
            "permutation": len(permutation), "arcface_primary": len(primary),
            "av_sync": len(av), "av_sync_null": len(av_null),
            "talknet_episodes": len(talknet_episode), "talknet_frames": len(talknet_frame),
            "candidates": len(candidates), "new_text_audio_risks": len(new_risks),
        },
        "candidate_only": True,
        "production_write": False,
    }
    target = out / "auto_resolution_acceptance.json"
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
