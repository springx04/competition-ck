from __future__ import annotations

import csv
import math
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from q1_features.pooling import NativeSeries

VISION_COLUMNS = [
    "AU01_r", "AU02_r", "AU04_r", "AU05_r", "AU06_r", "AU07_r",
    "AU09_r", "AU10_r", "AU12_r", "AU14_r", "AU15_r", "AU17_r",
    "AU20_r", "AU23_r", "AU25_r", "AU26_r", "AU45_r",
    "pose_Rx", "pose_Ry", "pose_Rz", "gaze_angle_x", "gaze_angle_y",
]


class VisionExtractionError(RuntimeError):
    pass


def run_openface(
    openface_bin: str | Path, openface_env: str | Path, frames_dir: str | Path,
    out_dir: str | Path, width: int, height: int, timeout_s: int = 1800,
) -> subprocess.CompletedProcess[str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fx, fy, cx, cy = 500.0 * width / 640.0, 500.0 * height / 480.0, width / 2.0, height / 2.0
    command = [
        "micromamba", "run", "-p", str(openface_env), str(openface_bin),
        "-fdir", str(frames_dir), "-out_dir", str(out), "-of", "features.csv",
        "-au_static", "-aus", "-pose", "-gaze", "-2Dfp",
        "-fx", str(fx), "-fy", str(fy), "-cx", str(cx), "-cy", str(cy),
    ]
    return subprocess.run(command, check=True, capture_output=True, text=True, timeout=timeout_s, cwd=Path(openface_bin).parent)


def read_openface_csv(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise VisionExtractionError("OpenFace CSV has no header")
        reader.fieldnames = [name.strip() for name in reader.fieldnames]
        missing = [name for name in VISION_COLUMNS + ["frame", "face_id", "success", "confidence"] if name not in reader.fieldnames]
        if missing:
            raise VisionExtractionError("missing OpenFace columns: " + ", ".join(missing))
        return [{str(k).strip(): v for k, v in row.items()} for row in reader]


def select_vision_rows(
    frame_map: list[dict[str, Any]], candidates: list[dict[str, Any]], *,
    confidence_min: float = 0.8, reviews: list[dict[str, Any]] | None = None,
    frames_dir: str | Path | None = None, trajectory_gap_s: float = 0.20,
    bbox_iou_split: float = 0.10, bbox_center_jump_fraction: float = 0.25,
    scene_hist_l1_split: float = 0.65,
) -> tuple[NativeSeries, list[dict[str, Any]], dict[str, Any]]:
    """Conservatively choose a target; multi-candidate clips need review."""
    by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        try:
            by_frame[int(float(row["frame"]))].append(row)
        except (TypeError, ValueError):
            continue
    frame_info = {int(row["image_index"]): row for row in frame_map}

    def bbox(row: dict[str, Any]) -> tuple[float, float, float, float] | None:
        xs, ys = [], []
        for key, value in row.items():
            name = str(key).strip()
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if name.startswith("x_") and np.isfinite(number): xs.append(number)
            elif name.startswith("y_") and np.isfinite(number): ys.append(number)
        if not xs or not ys:
            return None
        return min(xs), min(ys), max(xs), max(ys)

    def iou(a, b) -> float:
        if a is None or b is None: return 1.0
        left, top, right, bottom = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
        inter = max(0.0, right - left) * max(0.0, bottom - top)
        area_a, area_b = max(0.0, a[2]-a[0]) * max(0.0, a[3]-a[1]), max(0.0, b[2]-b[0]) * max(0.0, b[3]-b[1])
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    scene_break_after: set[int] = set()
    if frames_dir is not None:
        from PIL import Image
        previous = None
        previous_index = None
        for row in sorted(frame_map, key=lambda x: int(x["image_index"])):
            image_index = int(row["image_index"])
            path = Path(frames_dir) / str(row["filename"])
            if not path.is_file():
                continue
            image = np.asarray(Image.open(path).convert("RGB").resize((64, 64)), dtype=np.uint8)
            hist = np.concatenate([np.histogram(image[..., c], bins=16, range=(0, 256), density=False)[0] for c in range(3)]).astype(np.float64)
            hist = hist.reshape(3, 16)
            hist /= np.maximum(hist.sum(axis=1, keepdims=True), 1.0)
            if previous is not None:
                distance = float(np.abs(hist - previous).sum(axis=1).mean())
                if distance > scene_hist_l1_split:
                    scene_break_after.add(int(previous_index))
            previous, previous_index = hist, image_index

    successful = [r for r in candidates if int(float(r.get("success", 0))) == 1]
    tracks: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in successful:
        tracks[int(float(row["face_id"]))].append(row)
    episode_lookup: dict[int, int] = {}
    episode_count = 0
    for face_id, track in sorted(tracks.items()):
        track.sort(key=lambda r: int(float(r["frame"])))
        previous_row = None
        episode = -1
        for row in track:
            image_index = int(float(row["frame"]))
            start_new = previous_row is None
            if previous_row is not None:
                prev_index = int(float(previous_row["frame"]))
                current_time = float(frame_info.get(image_index, {}).get("start", 0.0))
                previous_time = float(frame_info.get(prev_index, {}).get("start", 0.0))
                gap = current_time - previous_time
                a, b = bbox(previous_row), bbox(row)
                diagonal = math.hypot(
                    float(frame_info.get(image_index, {}).get("display_width", 0) or 0),
                    float(frame_info.get(image_index, {}).get("display_height", 0) or 0),
                )
                if diagonal <= 0 and a is not None and b is not None:
                    diagonal = max(math.hypot(a[2]-a[0], a[3]-a[1]), math.hypot(b[2]-b[0], b[3]-b[1]), 1.0)
                center_distance = 0.0
                if a is not None and b is not None:
                    center_distance = math.hypot((a[0]+a[2]-b[0]-b[2])/2, (a[1]+a[3]-b[1]-b[3])/2)
                if gap > trajectory_gap_s or (iou(a, b) < bbox_iou_split and center_distance > diagonal * bbox_center_jump_fraction) or prev_index in scene_break_after:
                    start_new = True
            if start_new:
                episode = episode_count
                episode_count += 1
            episode_lookup[id(row)] = episode
            previous_row = row
    automatic_single = episode_count == 1 and bool(successful)
    review_rows = reviews or []
    values = np.zeros((len(frame_map), 22), dtype=np.float32)
    intervals = np.zeros((len(frame_map), 2), dtype=np.float64)
    eligible = np.zeros(len(frame_map), dtype=bool)
    rows: list[dict[str, Any]] = []
    for i, frame in enumerate(frame_map):
        image_index = int(frame["image_index"])
        start, end = float(frame["start"]), float(frame["end"])
        intervals[i] = (start, end)
        choices = by_frame.get(image_index, [])
        identity_basis = "assumed_single_visible" if automatic_single else "identity_unknown"
        chosen: dict[str, Any] | None = None
        if automatic_single and len(choices) == 1:
            chosen = choices[0]
        else:
            applicable = [r for r in review_rows if float(r["start"]) <= start < float(r["end"])]
            if len(applicable) == 1:
                face_id = int(applicable[0]["face_id"])
                requested_episode = str(applicable[0].get("episode_id", "")).strip()
                selected = [r for r in choices if int(float(r["face_id"])) == face_id and (not requested_episode or episode_lookup.get(id(r)) == int(requested_episode))]
                if len(selected) == 1:
                    chosen = selected[0]
                    identity_basis = "manual_review"
        reasons: list[str] = []
        if chosen is None:
            reasons.append("no_face" if not choices else "identity_unknown")
            face_id = None
            success = 0
            confidence = 0.0
        else:
            face_id = int(float(chosen["face_id"]))
            episode_id = episode_lookup.get(id(chosen))
            success = int(float(chosen["success"]))
            confidence = float(chosen["confidence"])
            try:
                vector = np.asarray([float(chosen[name]) for name in VISION_COLUMNS], dtype=np.float32)
            except (TypeError, ValueError):
                vector = np.zeros(22, dtype=np.float32)
                reasons.append("invalid_output")
            if success != 1:
                reasons.append("no_face")
            if confidence < confidence_min:
                reasons.append("low_confidence")
            if not np.isfinite(vector).all():
                reasons.append("invalid_output")
            if not reasons:
                values[i] = vector
                eligible[i] = True
        rows.append({
            "vision_row_id": i, "image_index": image_index,
            "source_frame_index": frame.get("source_frame_index"), "start": start, "end": end,
            "face_id": face_id, "episode_id": episode_id if chosen is not None else None, "identity_basis": identity_basis,
            "success": success, "confidence": confidence,
            "eligible": bool(eligible[i]), "reasons": reasons,
        })
    diagnostics = {"successful_candidate_count": len(successful), "automatic_single": automatic_single, "episode_count": episode_count, "scene_break_count": len(scene_break_after)}
    return NativeSeries(values, intervals, eligible, np.arange(len(frame_map), dtype=np.int64)), rows, diagnostics
