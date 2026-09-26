from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .storage import read_json, read_jsonl, write_csv, write_json, write_npz


def _read_openface(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return []
        reader.fieldnames = [name.strip() for name in reader.fieldnames]
        return [{str(key).strip(): str(value).strip() for key, value in row.items()} for row in reader]


def landmark_bbox(row: Mapping[str, Any]) -> tuple[float, float, float, float] | None:
    xs, ys = [], []
    for index in range(68):
        try:
            x, y = float(row[f"x_{index}"]), float(row[f"y_{index}"])
        except (KeyError, TypeError, ValueError):
            continue
        if np.isfinite(x) and np.isfinite(y):
            xs.append(x)
            ys.append(y)
    return (min(xs), min(ys), max(xs), max(ys)) if xs and ys else None


def _bbox_iou(a: tuple[float, ...] | None, b: tuple[float, ...] | None) -> float:
    if a is None or b is None:
        return 1.0
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def assign_openface_episodes(
    sample_dir: Path,
    *,
    trajectory_gap_s: float = 0.20,
    bbox_iou_split: float = 0.10,
    bbox_center_jump_fraction: float = 0.25,
    scene_hist_l1_split: float = 0.65,
) -> list[dict[str, Any]]:
    from PIL import Image

    candidates = _read_openface(sample_dir / "openface" / "features.csv")
    frame_map = read_jsonl(sample_dir / "video_frames.jsonl")
    frame_info = {int(row["image_index"]): row for row in frame_map}
    scene_break_after: set[int] = set()
    previous_hist = None
    previous_index = None
    for row in sorted(frame_map, key=lambda item: int(item["image_index"])):
        image_index = int(row["image_index"])
        path = sample_dir / "frames" / str(row["filename"])
        if not path.is_file():
            continue
        image = np.asarray(Image.open(path).convert("RGB").resize((64, 64)), dtype=np.uint8)
        hist = np.concatenate(
            [np.histogram(image[..., channel], bins=16, range=(0, 256))[0] for channel in range(3)]
        ).astype(np.float64).reshape(3, 16)
        hist /= np.maximum(hist.sum(axis=1, keepdims=True), 1.0)
        if previous_hist is not None:
            distance = float(np.abs(hist - previous_hist).sum(axis=1).mean())
            if distance > scene_hist_l1_split:
                scene_break_after.add(int(previous_index))
        previous_hist = hist
        previous_index = image_index

    successful = [row for row in candidates if int(float(row.get("success", 0) or 0)) == 1]
    tracks: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in successful:
        tracks[int(float(row["face_id"]))].append(row)
    records: list[dict[str, Any]] = []
    episode_count = 0
    for face_id, track in sorted(tracks.items()):
        track.sort(key=lambda row: int(float(row["frame"])))
        previous = None
        episode_id = -1
        for row in track:
            image_index = int(float(row["frame"]))
            start_new = previous is None
            if previous is not None:
                previous_index = int(float(previous["frame"]))
                current_time = float(frame_info[image_index]["start"])
                previous_time = float(frame_info[previous_index]["start"])
                gap = current_time - previous_time
                a, b = landmark_bbox(previous), landmark_bbox(row)
                width = float(frame_info[image_index].get("display_width", 0) or 0)
                height = float(frame_info[image_index].get("display_height", 0) or 0)
                diagonal = math.hypot(width, height)
                if diagonal <= 0 and a is not None and b is not None:
                    diagonal = max(math.hypot(a[2] - a[0], a[3] - a[1]), 1.0)
                center_distance = 0.0
                if a is not None and b is not None:
                    center_distance = math.hypot(
                        (a[0] + a[2] - b[0] - b[2]) / 2,
                        (a[1] + a[3] - b[1] - b[3]) / 2,
                    )
                if (
                    gap > trajectory_gap_s
                    or (_bbox_iou(a, b) < bbox_iou_split and center_distance > diagonal * bbox_center_jump_fraction)
                    or previous_index in scene_break_after
                ):
                    start_new = True
            if start_new:
                episode_id = episode_count
                episode_count += 1
            frame = frame_info[image_index]
            records.append(
                {
                    **row,
                    "face_id": face_id,
                    "episode_id": episode_id,
                    "image_index": image_index,
                    "start": float(frame["start"]),
                    "end": float(frame["end"]),
                    "filename": str(frame["filename"]),
                }
            )
            previous = row
    diagnostic = read_json(sample_dir / "vision_diagnostic.json", {})
    expected = int(diagnostic.get("episode_count", 0) or 0)
    if expected != episode_count:
        raise ValueError(f"{sample_dir.name}: episode reconstruction mismatch {episode_count} != {expected}")
    return records


def five_landmarks(row: Mapping[str, Any]) -> np.ndarray:
    def point(index: int) -> np.ndarray:
        return np.asarray([float(row[f"x_{index}"]), float(row[f"y_{index}"])], dtype=np.float32)

    left_eye = np.mean([point(index) for index in range(36, 42)], axis=0)
    right_eye = np.mean([point(index) for index in range(42, 48)], axis=0)
    return np.stack([left_eye, right_eye, point(30), point(48), point(54)]).astype(np.float32)


def robust_centroid(embeddings: np.ndarray, trim_fraction: float = 0.20) -> tuple[np.ndarray, float, int]:
    values = np.asarray(embeddings, dtype=np.float32)
    values /= np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)
    centroid = values.mean(axis=0)
    centroid /= max(float(np.linalg.norm(centroid)), 1e-12)
    keep_count = len(values)
    if len(values) >= 5 and trim_fraction > 0:
        similarities = values @ centroid
        keep_count = max(3, int(math.ceil(len(values) * (1.0 - trim_fraction))))
        keep = np.argsort(similarities)[-keep_count:]
        centroid = values[keep].mean(axis=0)
        centroid /= max(float(np.linalg.norm(centroid)), 1e-12)
        values = values[keep]
    # Numerical round-off can make cosine distance a few ulps negative.
    dispersion = max(0.0, float(np.median(1.0 - values @ centroid)))
    return centroid.astype(np.float32), dispersion, keep_count


def connected_components(keys: Sequence[str], similarities: Mapping[tuple[str, str], float], threshold: float) -> list[list[str]]:
    adjacency = {key: set() for key in keys}
    for (left, right), similarity in similarities.items():
        if similarity >= threshold:
            adjacency[left].add(right)
            adjacency[right].add(left)
    components = []
    seen = set()
    for key in sorted(keys):
        if key in seen:
            continue
        stack = [key]
        component = []
        seen.add(key)
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in sorted(adjacency[current]):
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        components.append(sorted(component))
    return sorted(components, key=lambda component: component[0])


def partition_signature(components: Sequence[Sequence[str]]) -> str:
    return "|".join(",".join(sorted(component)) for component in sorted(components, key=lambda value: sorted(value)[0]))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
