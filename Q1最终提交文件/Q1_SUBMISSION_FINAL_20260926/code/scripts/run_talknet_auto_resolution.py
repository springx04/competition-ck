from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from q1_features.auto_vision import assign_openface_episodes, landmark_bbox
from q1_features.storage import read_jsonl, write_csv, write_json


FRAME_FIELDS = [
    "sample_id", "video_id", "face_id", "episode_id", "episode_frame_index",
    "source_image_index", "start", "end", "class1_logit", "is_speaking",
]
EPISODE_FIELDS = [
    "sample_id", "video_id", "face_id", "episode_id", "status", "reason",
    "start", "end", "input_openface_rows", "talknet_frame_count",
    "mean_class1_logit", "median_class1_logit", "max_class1_logit",
    "speaking_frame_fraction",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _smooth(values: np.ndarray, kernel: int = 13) -> np.ndarray:
    from scipy.signal import medfilt

    if len(values) < 3:
        return values
    size = min(kernel, len(values) if len(values) % 2 else len(values) - 1)
    return medfilt(values, kernel_size=max(1, size)) if size >= 3 else values


def crop_episode(sample_dir: Path, rows: list[dict], fps: float = 25.0, crop_scale: float = 0.40):
    import cv2

    ordered = sorted(rows, key=lambda row: float(row["start"]))
    times = np.asarray([(float(row["start"]) + float(row["end"])) / 2 for row in ordered])
    boxes = []
    valid_rows = []
    for row in ordered:
        box = landmark_bbox(row)
        if box is not None and np.isfinite(box).all():
            boxes.append(box)
            valid_rows.append(row)
    if not valid_rows:
        raise ValueError("no finite landmark bounding boxes")
    ordered = valid_rows
    times = np.asarray([(float(row["start"]) + float(row["end"])) / 2 for row in ordered])
    boxes = np.asarray(boxes, dtype=np.float64)
    centers_x = _smooth((boxes[:, 0] + boxes[:, 2]) / 2)
    centers_y = _smooth((boxes[:, 1] + boxes[:, 3]) / 2)
    half_sizes = _smooth(np.maximum(boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]) / 2)
    start = float(min(float(row["start"]) for row in ordered))
    end = float(max(float(row["end"]) for row in ordered))
    frame_count = max(1, int(math.floor((end - start) * fps + 1e-6)))
    query_times = start + (np.arange(frame_count, dtype=np.float64) + 0.5) / fps
    nearest = np.abs(times[:, None] - query_times[None, :]).argmin(axis=0)
    faces, source_indices = [], []
    for row_index in nearest:
        row = ordered[int(row_index)]
        image = cv2.imread(str(sample_dir / "frames" / str(row["filename"])))
        if image is None:
            raise FileNotFoundError(str(sample_dir / "frames" / str(row["filename"])))
        bs = float(half_sizes[int(row_index)])
        bsi = max(1, int(bs * (1 + 2 * crop_scale)))
        padded = np.pad(image, ((bsi, bsi), (bsi, bsi), (0, 0)), "constant", constant_values=110)
        my = float(centers_y[int(row_index)]) + bsi
        mx = float(centers_x[int(row_index)]) + bsi
        top, bottom = int(my - bs), int(my + bs * (1 + 2 * crop_scale))
        left, right = int(mx - bs * (1 + crop_scale)), int(mx + bs * (1 + crop_scale))
        face = padded[max(0, top):max(1, bottom), max(0, left):max(1, right)]
        if face.size == 0:
            raise ValueError(f"empty face crop at image {row['image_index']}")
        gray_224 = cv2.cvtColor(cv2.resize(face, (224, 224)), cv2.COLOR_BGR2GRAY)
        # The released demo writes 224x224 tracks and then center-crops 112x112
        # immediately before inference.
        gray = gray_224[56:168, 56:168]
        faces.append(gray)
        source_indices.append(int(row["image_index"]))
    return start, np.asarray(faces, dtype=np.float32), np.asarray(source_indices, dtype=np.int64)


def episode_mfcc(audio: np.ndarray, sample_rate: int, start: float, video_frames: int):
    import python_speech_features

    sample_start = max(0, int(round(start * sample_rate)))
    sample_end = min(len(audio), sample_start + int(round(video_frames / 25.0 * sample_rate)))
    segment = audio[sample_start:sample_end]
    features = python_speech_features.mfcc(
        segment, sample_rate, numcep=13, winlen=0.025, winstep=0.010
    )
    usable_video = min(video_frames, len(features) // 4)
    if usable_video <= 0:
        raise ValueError("episode too short for synchronized MFCC and video")
    return features[: usable_video * 4].astype(np.float32), usable_video


def infer_logits(network, audio_features: np.ndarray, video_features: np.ndarray, durations: tuple[int, ...]):
    import torch

    frame_count = len(video_features)
    predictions = []
    with torch.no_grad():
        for duration in durations:
            chunk_frames = max(1, int(duration * 25))
            values = []
            for offset in range(0, frame_count, chunk_frames):
                stop = min(frame_count, offset + chunk_frames)
                input_a = torch.from_numpy(audio_features[offset * 4: stop * 4]).unsqueeze(0).cuda()
                input_v = torch.from_numpy(video_features[offset:stop]).unsqueeze(0).cuda()
                embed_a = network.model.forward_audio_frontend(input_a)
                embed_v = network.model.forward_visual_frontend(input_v)
                embed_a, embed_v = network.model.forward_cross_attention(embed_a, embed_v)
                output = network.model.forward_audio_visual_backend(embed_a, embed_v)
                values.extend(network.lossAV.forward(output, labels=None).tolist())
            if len(values) != frame_count:
                raise ValueError(f"TalkNet output length {len(values)} != {frame_count}")
            predictions.append(values)
    return np.mean(np.asarray(predictions, dtype=np.float32), axis=0)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run official TalkNet on frozen OpenFace episodes")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--durations", default="1,2,3,4,5,6")
    parser.add_argument("--max-samples", type=int)
    args = parser.parse_args()

    import torch
    from scipy.io import wavfile

    run = Path(args.run_dir).resolve()
    output = Path(args.output_dir).resolve()
    source = Path(args.source_dir).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(source))
    from talkNet import talkNet

    durations = tuple(sorted({int(value) for value in args.durations.split(",") if int(value) > 0}))
    network = talkNet()
    network.loadParameters(str(checkpoint))
    network.eval()

    frame_rows, episode_rows = [], []
    manifest = read_jsonl(run / "manifest.jsonl")
    if args.max_samples is not None:
        manifest = manifest[: args.max_samples]
    for item in manifest:
        sample_id = str(item["sample_id"])
        sample_dir = run / "samples" / f"{int(item['sample_index']):06d}"
        records = assign_openface_episodes(sample_dir)
        grouped: dict[int, list[dict]] = defaultdict(list)
        for row in records:
            grouped[int(row["episode_id"])].append(row)
        sample_rate, audio = wavfile.read(sample_dir / "audio_16k.wav")
        for episode_id, rows in sorted(grouped.items()):
            face_id = int(rows[0]["face_id"])
            episode_start = min(float(row["start"]) for row in rows)
            episode_end = max(float(row["end"]) for row in rows)
            summary = {
                "sample_id": sample_id, "video_id": item["video_id"], "face_id": face_id,
                "episode_id": episode_id, "status": "error", "reason": "",
                "start": episode_start, "end": episode_end, "input_openface_rows": len(rows),
                "talknet_frame_count": 0, "mean_class1_logit": "",
                "median_class1_logit": "", "max_class1_logit": "",
                "speaking_frame_fraction": "",
            }
            try:
                start, video, source_indices = crop_episode(sample_dir, rows)
                audio_features, usable = episode_mfcc(audio, sample_rate, start, len(video))
                video = video[:usable]
                source_indices = source_indices[:usable]
                logits = infer_logits(network, audio_features, video, durations)
                for index, (source_index, logit) in enumerate(zip(source_indices, logits)):
                    frame_rows.append({
                        "sample_id": sample_id, "video_id": item["video_id"], "face_id": face_id,
                        "episode_id": episode_id, "episode_frame_index": index,
                        "source_image_index": int(source_index), "start": start + index / 25.0,
                        "end": start + (index + 1) / 25.0, "class1_logit": float(logit),
                        "is_speaking": int(float(logit) >= 0.0),
                    })
                summary.update({
                    "status": "ok", "talknet_frame_count": len(logits),
                    "mean_class1_logit": float(np.mean(logits)),
                    "median_class1_logit": float(np.median(logits)),
                    "max_class1_logit": float(np.max(logits)),
                    "speaking_frame_fraction": float(np.mean(logits >= 0.0)),
                })
            except Exception as exc:
                summary["reason"] = f"{type(exc).__name__}: {exc}"
            episode_rows.append(summary)
            print(json.dumps({"sample_id": sample_id, "episode_id": episode_id, "status": summary["status"], "reason": summary["reason"]}, ensure_ascii=False), flush=True)

    write_csv(output / "talknet_frame_logits.csv", frame_rows, FRAME_FIELDS)
    write_csv(output / "talknet_episode_scores.csv", episode_rows, EPISODE_FIELDS)
    metadata = {
        "implementation": "official TalkNet checkpoint and model; frozen OpenFace episode crops",
        "official_repository": "https://github.com/TaoRuijie/TalkNet-ASD",
        "official_google_drive_file_id": "1AbN9fCf9IexMxEKXLQY2KYBlb-IhSEea",
        "checkpoint": str(checkpoint), "checkpoint_bytes": checkpoint.stat().st_size,
        "checkpoint_sha256": sha256_file(checkpoint), "source_dir": str(source),
        "torch": torch.__version__, "cuda": torch.cuda.get_device_name(0),
        "durations_s": list(durations), "threshold": "class-1 logit >= 0",
        "sample_count": len(manifest), "episode_count": len(episode_rows),
        "ok_episode_count": sum(row["status"] == "ok" for row in episode_rows),
        "failed_episode_count": sum(row["status"] != "ok" for row in episode_rows),
        "frame_count": len(frame_rows),
    }
    write_json(output / "talknet_run_metadata.json", metadata)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0 if metadata["failed_episode_count"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
