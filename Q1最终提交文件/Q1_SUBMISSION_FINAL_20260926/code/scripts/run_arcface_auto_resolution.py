from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from q1_features.auto_vision import (
    assign_openface_episodes,
    connected_components,
    five_landmarks,
    partition_signature,
    robust_centroid,
    sha256_file,
)
from q1_features.storage import read_jsonl, write_csv, write_json, write_npz


SIMILARITY_FIELDS = [
    "sample_id", "video_id", "episode_a", "face_id_a", "episode_b", "face_id_b",
    "cosine_similarity", "episode_a_valid_embeddings", "episode_b_valid_embeddings",
    "episode_a_dispersion", "episode_b_dispersion",
]
CLUSTER_FIELDS = [
    "sample_id", "video_id", "threshold", "episode_count",
    "valid_embedding_episode_count", "identity_cluster_count", "clusters_json",
    "cluster_time_coverage_json", "partition_signature",
    "stable_partition_0_50_0_60_0_70", "simultaneous_different_identity_clusters",
    "episode_quality_json",
]


def interval_union(intervals: list[tuple[float, float]]) -> float:
    if not intervals:
        return 0.0
    ordered = sorted(intervals)
    total = 0.0
    start, end = ordered[0]
    for next_start, next_end in ordered[1:]:
        if next_start <= end:
            end = max(end, next_end)
        else:
            total += end - start
            start, end = next_start, next_end
    return total + end - start


def main() -> int:
    parser = argparse.ArgumentParser(description="Run frozen buffalo_l ArcFace episode diagnostics")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--max-frames", type=int, default=12)
    parser.add_argument("--confidence-min", type=float, default=0.80)
    parser.add_argument("--trim-fraction", type=float, default=0.20)
    args = parser.parse_args()

    import cv2
    import onnxruntime

    run = Path(args.run_dir).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    model_root = Path(args.model_root).resolve()
    recognition_paths = sorted(model_root.rglob("w600k_r50.onnx"))
    if len(recognition_paths) != 1:
        raise RuntimeError(f"expected one buffalo_l w600k_r50.onnx, found {recognition_paths}")
    recognition_path = recognition_paths[0]
    session = onnxruntime.InferenceSession(
        str(recognition_path), providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name
    reference_landmarks = np.asarray(
        [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
         [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32
    )

    def embedding_from_openface(image, landmarks):
        transform, _ = cv2.estimateAffinePartial2D(
            np.asarray(landmarks, dtype=np.float32), reference_landmarks, method=cv2.LMEDS
        )
        if transform is None:
            raise RuntimeError("five-point similarity transform failed")
        crop = cv2.warpAffine(image, transform, (112, 112), borderValue=0.0)
        blob = cv2.dnn.blobFromImage(
            crop, scalefactor=1.0 / 127.5, size=(112, 112),
            mean=(127.5, 127.5, 127.5), swapRB=True,
        )
        return np.asarray(session.run(None, {input_name: blob})[0]).reshape(-1).astype(np.float32)

    manifest = read_jsonl(run / "manifest.jsonl")
    episode_quality = []
    embedding_keys = []
    embedding_values = []
    similarity_rows = []
    cluster_rows = []
    for item in manifest:
        sample_id = str(item["sample_id"])
        sample_dir = run / "samples" / f"{int(item['sample_index']):06d}"
        records = assign_openface_episodes(sample_dir)
        grouped: dict[int, list[dict]] = defaultdict(list)
        for row in records:
            grouped[int(row["episode_id"])].append(row)
        centroids: dict[str, np.ndarray] = {}
        quality_by_key = {}
        intervals_by_key = {}
        for episode_id, rows in sorted(grouped.items()):
            rows = sorted(rows, key=lambda row: int(row["image_index"]))
            face_id = int(rows[0]["face_id"])
            key = f"{face_id}:{episode_id}"
            eligible = [
                row for row in rows
                if float(row.get("confidence", 0.0) or 0.0) >= args.confidence_min
            ]
            if len(eligible) > args.max_frames:
                indices = np.linspace(0, len(eligible) - 1, args.max_frames).round().astype(int)
                selected = [eligible[index] for index in sorted(set(indices.tolist()))]
            else:
                selected = eligible
            embeddings = []
            failures = []
            for row in selected:
                image = cv2.imread(str(sample_dir / "frames" / row["filename"]))
                if image is None:
                    failures.append(f"missing_image:{row['filename']}")
                    continue
                try:
                    embedding = embedding_from_openface(image, five_landmarks(row))
                    embedding /= max(float(np.linalg.norm(embedding)), 1e-12)
                    if np.isfinite(embedding).all():
                        embeddings.append(embedding)
                    else:
                        failures.append(f"nonfinite_embedding:{row['filename']}")
                except Exception as exc:
                    failures.append(f"{type(exc).__name__}:{row['filename']}")
            dispersion = ""
            retained = 0
            if embeddings:
                centroid, dispersion_value, retained = robust_centroid(
                    np.stack(embeddings), trim_fraction=args.trim_fraction
                )
                centroids[key] = centroid
                embedding_keys.append(f"{sample_id}|{key}")
                embedding_values.append(centroid)
                dispersion = dispersion_value
            intervals = [(float(row["start"]), float(row["end"])) for row in rows]
            intervals_by_key[key] = intervals
            quality = {
                "sample_id": sample_id,
                "video_id": item["video_id"],
                "episode_key": key,
                "face_id": face_id,
                "episode_id": episode_id,
                "start": min(value[0] for value in intervals),
                "end": max(value[1] for value in intervals),
                "frame_count": len(rows),
                "selected_frame_count": len(selected),
                "valid_embedding_count": len(embeddings),
                "robust_retained_count": retained,
                "dispersion": dispersion,
                "failures_json": json.dumps(failures, ensure_ascii=False, separators=(",", ":")),
            }
            episode_quality.append(quality)
            quality_by_key[key] = quality

        keys = sorted(grouped and [f"{int(rows[0]['face_id'])}:{episode_id}" for episode_id, rows in grouped.items()] or [])
        similarities = {}
        for left_index, left in enumerate(keys):
            for right in keys[left_index + 1:]:
                similarity = float(centroids[left] @ centroids[right]) if left in centroids and right in centroids else float("nan")
                if np.isfinite(similarity):
                    similarities[(left, right)] = similarity
                left_quality, right_quality = quality_by_key[left], quality_by_key[right]
                similarity_rows.append(
                    {
                        "sample_id": sample_id,
                        "video_id": item["video_id"],
                        "episode_a": left.split(":")[1],
                        "face_id_a": left.split(":")[0],
                        "episode_b": right.split(":")[1],
                        "face_id_b": right.split(":")[0],
                        "cosine_similarity": similarity if np.isfinite(similarity) else "",
                        "episode_a_valid_embeddings": left_quality["valid_embedding_count"],
                        "episode_b_valid_embeddings": right_quality["valid_embedding_count"],
                        "episode_a_dispersion": left_quality["dispersion"],
                        "episode_b_dispersion": right_quality["dispersion"],
                    }
                )
        threshold_components = {
            threshold: connected_components(keys, similarities, threshold)
            for threshold in (0.50, 0.60, 0.70)
        }
        signatures = {threshold: partition_signature(value) for threshold, value in threshold_components.items()}
        stable = len(set(signatures.values())) == 1
        for threshold, components in threshold_components.items():
            cluster_intervals = []
            simultaneous = False
            for cluster_index, component in enumerate(components):
                intervals = [interval for key in component for interval in intervals_by_key[key]]
                cluster_intervals.append(
                    {
                        "cluster_id": cluster_index,
                        "episodes": component,
                        "coverage_s": interval_union(intervals),
                        "start": min(value[0] for value in intervals) if intervals else None,
                        "end": max(value[1] for value in intervals) if intervals else None,
                    }
                )
            for left_index, left_cluster in enumerate(cluster_intervals):
                for right_cluster in cluster_intervals[left_index + 1:]:
                    for left_key in left_cluster["episodes"]:
                        for right_key in right_cluster["episodes"]:
                            if any(
                                min(a_end, b_end) > max(a_start, b_start)
                                for a_start, a_end in intervals_by_key[left_key]
                                for b_start, b_end in intervals_by_key[right_key]
                            ):
                                simultaneous = True
            cluster_rows.append(
                {
                    "sample_id": sample_id,
                    "video_id": item["video_id"],
                    "threshold": threshold,
                    "episode_count": len(keys),
                    "valid_embedding_episode_count": len(centroids),
                    "identity_cluster_count": len(components),
                    "clusters_json": json.dumps(components, separators=(",", ":")),
                    "cluster_time_coverage_json": json.dumps(cluster_intervals, separators=(",", ":")),
                    "partition_signature": signatures[threshold],
                    "stable_partition_0_50_0_60_0_70": int(stable),
                    "simultaneous_different_identity_clusters": int(simultaneous),
                    "episode_quality_json": json.dumps(
                        [quality_by_key[key] for key in keys], ensure_ascii=False, separators=(",", ":")
                    ),
                }
            )
        print(json.dumps({"sample_id": sample_id, "episodes": len(keys), "embedded": len(centroids)}), flush=True)

    write_csv(output / "arcface_episode_similarity.csv", similarity_rows, SIMILARITY_FIELDS)
    write_csv(output / "arcface_identity_clusters.csv", cluster_rows, CLUSTER_FIELDS)
    write_csv(output / "arcface_episode_quality.csv", episode_quality, list(episode_quality[0]) if episode_quality else [])
    if embedding_values:
        write_npz(
            output / "arcface_episode_embeddings.npz",
            episode_keys=np.asarray(embedding_keys),
            embeddings=np.stack(embedding_values).astype(np.float32),
        )
    model_files = sorted(model_root.rglob("*.onnx"))
    metadata = {
        "implementation": "InsightFace buffalo_l ArcFace ONNX direct inference",
        "insightface_model_release": "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip",
        "onnxruntime_version": getattr(onnxruntime, "__version__", "unknown"),
        "providers_available": onnxruntime.get_available_providers(),
        "model_pack": "buffalo_l",
        "thresholds": [0.50, 0.60, 0.70],
        "sample_count": len(manifest),
        "samples_with_episodes": len({row["sample_id"] for row in episode_quality}),
        "episode_count": len(episode_quality),
        "valid_embedding_episode_count": len(embedding_values),
        "model_files": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in model_files
        ],
    }
    write_json(output / "arcface_run_metadata.json", metadata)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
