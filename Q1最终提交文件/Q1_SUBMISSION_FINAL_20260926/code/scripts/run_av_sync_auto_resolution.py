from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf

from q1_features.auto_resolution import read_csv
from q1_features.auto_vision import assign_openface_episodes
from q1_features.storage import read_json, read_jsonl, write_csv, write_json


SCORE_FIELDS = [
    "sample_id", "video_id", "cluster_id", "episode_keys_json", "visual_signal_basis",
    "valid_frame_count", "real_sync_score", "best_lag_s", "null_count",
    "null_median", "null_p95", "empirical_percentile", "p_like_score",
    "rank_within_sample", "top1_top2_margin",
]
NULL_FIELDS = [
    "sample_id", "video_id", "cluster_id", "null_type", "null_id",
    "shift_s", "source_sample_id", "sync_score",
]


def safe_correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    mask = np.isfinite(left) & np.isfinite(right)
    if int(mask.sum()) < 5:
        return None
    a, b = left[mask].astype(np.float64), right[mask].astype(np.float64)
    if float(a.std()) < 1e-8 or float(b.std()) < 1e-8:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def best_local_correlation(visual: np.ndarray, audio: np.ndarray, max_lag: int) -> tuple[float | None, int | None]:
    best_score = None
    best_lag = None
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            left, right = visual[-lag:], audio[:lag]
        elif lag > 0:
            left, right = visual[:-lag], audio[lag:]
        else:
            left, right = visual, audio
        score = safe_correlation(left, right)
        if score is not None and (best_score is None or score > best_score):
            best_score, best_lag = score, lag
    return best_score, best_lag


def shifted(values: np.ndarray, frames: int) -> np.ndarray:
    result = np.full_like(values, np.nan, dtype=np.float64)
    if frames < 0:
        result[:frames] = values[-frames:]
    elif frames > 0:
        result[frames:] = values[:-frames]
    else:
        result[:] = values
    return result


def audio_energy(sample_dir: Path, frame_map: list[dict]) -> np.ndarray:
    waveform, sample_rate = sf.read(sample_dir / "audio_16k.wav", dtype="float32", always_2d=False)
    waveform = np.asarray(waveform, dtype=np.float64)
    values = []
    for frame in frame_map:
        start = max(0, int(round(float(frame["start"]) * sample_rate)))
        end = min(len(waveform), int(round(float(frame["end"]) * sample_rate)))
        chunk = waveform[start:end]
        rms = math.sqrt(float(np.mean(chunk * chunk))) if len(chunk) else 0.0
        values.append(math.log(rms + 1e-8))
    return np.asarray(values, dtype=np.float64)


def mouth_value(row: dict) -> tuple[float | None, str]:
    try:
        inner = math.hypot(float(row["x_66"]) - float(row["x_62"]), float(row["y_66"]) - float(row["y_62"]))
        width = math.hypot(float(row["x_54"]) - float(row["x_48"]), float(row["y_54"]) - float(row["y_48"]))
        value = inner / width if width > 1e-6 else float("nan")
        if math.isfinite(value):
            return value, "mouth_landmarks_62_66_over_48_54"
    except (KeyError, TypeError, ValueError):
        pass
    try:
        value = float(row["AU25_r"]) + float(row["AU26_r"])
        if math.isfinite(value):
            return value, "AU25_r_plus_AU26_r_fallback"
    except (KeyError, TypeError, ValueError):
        pass
    return None, "unavailable"


def resample_normalized(values: np.ndarray, length: int) -> np.ndarray:
    if len(values) == length:
        return values.copy()
    if len(values) < 2 or length < 1:
        return np.full(length, np.nan)
    return np.interp(np.linspace(0, 1, length), np.linspace(0, 1, len(values)), values)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute Q1 mouth-motion/audio-energy sync with null controls")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--cluster-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--threshold", type=float, default=0.60)
    parser.add_argument("--local-lag-s", type=float, default=0.20)
    parser.add_argument("--cross-null-count", type=int, default=20)
    args = parser.parse_args()

    run = Path(args.run_dir).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = read_jsonl(run / "manifest.jsonl")
    by_id = {str(item["sample_id"]): item for item in manifest}
    cluster_rows = [
        row for row in read_csv(args.cluster_csv)
        if abs(float(row["threshold"]) - args.threshold) < 1e-9
    ]
    if len(cluster_rows) != 100:
        raise ValueError(f"expected 100 primary-threshold cluster rows, found {len(cluster_rows)}")

    frame_maps = {}
    audio_by_sample = {}
    duration_by_sample = {}
    for item in manifest:
        sample_id = str(item["sample_id"])
        sample_dir = run / "samples" / f"{int(item['sample_index']):06d}"
        frames = read_jsonl(sample_dir / "video_frames.jsonl")
        frame_maps[sample_id] = frames
        audio_by_sample[sample_id] = audio_energy(sample_dir, frames)
        duration_by_sample[sample_id] = float(read_json(sample_dir / "media.json")["duration"])

    score_rows = []
    null_rows = []
    max_lag_frames = int(round(args.local_lag_s * 25.0))
    temporal_shifts = [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]
    for cluster_row in cluster_rows:
        sample_id = cluster_row["sample_id"]
        item = by_id[sample_id]
        sample_dir = run / "samples" / f"{int(item['sample_index']):06d}"
        records = assign_openface_episodes(sample_dir)
        by_episode = {}
        for row in records:
            by_episode.setdefault(f"{int(row['face_id'])}:{int(row['episode_id'])}", []).append(row)
        components = json.loads(cluster_row["clusters_json"])
        frames = frame_maps[sample_id]
        frame_index = {int(frame["image_index"]): index for index, frame in enumerate(frames)}
        audio = audio_by_sample[sample_id]
        for cluster_id, episode_keys in enumerate(components):
            visual = np.full(len(frames), np.nan, dtype=np.float64)
            values_at_frame = {}
            bases = []
            for episode_key in episode_keys:
                for row in by_episode.get(episode_key, []):
                    value, basis = mouth_value(row)
                    if value is not None:
                        values_at_frame.setdefault(int(row["image_index"]), []).append(value)
                        bases.append(basis)
            for image_index, values in values_at_frame.items():
                if image_index in frame_index:
                    visual[frame_index[image_index]] = float(np.mean(values))
            real_score, best_lag = best_local_correlation(visual, audio, max_lag_frames)
            null_scores = []
            for shift_s in temporal_shifts:
                shifted_audio = shifted(audio, int(round(shift_s * 25.0)))
                score, _ = best_local_correlation(visual, shifted_audio, max_lag_frames)
                if score is not None:
                    null_scores.append(score)
                    null_rows.append(
                        {
                            "sample_id": sample_id,
                            "video_id": item["video_id"],
                            "cluster_id": cluster_id,
                            "null_type": "temporal_shift",
                            "null_id": f"shift_{shift_s:+.1f}",
                            "shift_s": shift_s,
                            "source_sample_id": sample_id,
                            "sync_score": score,
                        }
                    )
            controls = []
            for other in manifest:
                other_id = str(other["sample_id"])
                if other_id == sample_id or str(other["video_id"]) == str(item["video_id"]):
                    continue
                distance = abs(duration_by_sample[other_id] - duration_by_sample[sample_id]) / max(
                    duration_by_sample[other_id], duration_by_sample[sample_id], 1e-9
                )
                controls.append((distance, other_id))
            controls.sort(key=lambda value: (value[0], value[1]))
            for control_index, (_, other_id) in enumerate(controls[: args.cross_null_count]):
                other_audio = resample_normalized(audio_by_sample[other_id], len(audio))
                score, _ = best_local_correlation(visual, other_audio, max_lag_frames)
                if score is not None:
                    null_scores.append(score)
                    null_rows.append(
                        {
                            "sample_id": sample_id,
                            "video_id": item["video_id"],
                            "cluster_id": cluster_id,
                            "null_type": "cross_sample_audio",
                            "null_id": f"cross_{control_index:02d}",
                            "shift_s": "",
                            "source_sample_id": other_id,
                            "sync_score": score,
                        }
                    )
            if real_score is None:
                percentile = p_like = ""
            else:
                percentile = sum(value < real_score for value in null_scores) / len(null_scores) if null_scores else ""
                p_like = (1 + sum(value >= real_score for value in null_scores)) / (len(null_scores) + 1) if null_scores else ""
            score_rows.append(
                {
                    "sample_id": sample_id,
                    "video_id": item["video_id"],
                    "cluster_id": cluster_id,
                    "episode_keys_json": json.dumps(episode_keys, separators=(",", ":")),
                    "visual_signal_basis": Counter(bases).most_common(1)[0][0] if bases else "unavailable",
                    "valid_frame_count": int(np.isfinite(visual).sum()),
                    "real_sync_score": real_score if real_score is not None else "",
                    "best_lag_s": best_lag / 25.0 if best_lag is not None else "",
                    "null_count": len(null_scores),
                    "null_median": float(np.median(null_scores)) if null_scores else "",
                    "null_p95": float(np.quantile(null_scores, 0.95)) if null_scores else "",
                    "empirical_percentile": percentile,
                    "p_like_score": p_like,
                    "rank_within_sample": "",
                    "top1_top2_margin": "",
                }
            )

    by_sample_scores = {}
    for row in score_rows:
        score = float(row["real_sync_score"]) if row["real_sync_score"] != "" else float("-inf")
        by_sample_scores.setdefault(row["sample_id"], []).append((score, row))
    for values in by_sample_scores.values():
        values.sort(key=lambda value: (-value[0], int(value[1]["cluster_id"])))
        second = values[1][0] if len(values) > 1 else None
        for rank, (score, row) in enumerate(values, start=1):
            row["rank_within_sample"] = rank
            if rank == 1 and math.isfinite(score):
                row["top1_top2_margin"] = score - second if second is not None and math.isfinite(second) else ""

    write_csv(output / "av_sync_scores.csv", score_rows, SCORE_FIELDS)
    write_csv(output / "av_sync_null_test.csv", null_rows, NULL_FIELDS)
    metadata = {
        "sample_count": len(manifest),
        "cluster_count": len(score_rows),
        "null_row_count": len(null_rows),
        "threshold": args.threshold,
        "local_lag_s": args.local_lag_s,
        "temporal_shifts_s": temporal_shifts,
        "cross_sample_null_count": args.cross_null_count,
        "visual_signal": "normalized mouth landmark distance with AU25/AU26 fallback",
        "audio_signal": "PCM log RMS on frozen video-frame intervals",
    }
    write_json(output / "av_sync_run_metadata.json", metadata)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
