from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from q1_features.finalization import (
    read_tsv,
    validate_csr,
    validate_masked_features,
    validate_source_sample_membership,
    validate_status_rows,
    verify_freeze_manifest,
)
from q1_features.storage import load_npz, read_jsonl, write_json


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate final frozen Q1 artifacts")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--formal-run", default="runs/q1_full_20260924")
    parser.add_argument("--stage3-run", default="runs/q1_alignment_repair_stage3")
    parser.add_argument("--final-run", default="runs/q1_final_20260926")
    args = parser.parse_args()
    project = Path(args.project_root).resolve(); formal = (project / args.formal_run).resolve()
    stage3 = (project / args.stage3_run).resolve(); final = (project / args.final_run).resolve()
    errors: list[str] = []
    checks: dict[str, object] = {}

    manifest = sorted(read_jsonl(formal / "manifest.jsonl"), key=lambda row: int(row["sample_index"]))
    status = read_csv(final / "q1_final_status.csv")
    errors.extend(validate_status_rows(status))
    manifest_ids = [row["sample_id"] for row in manifest]
    status_ids = [row["sample_id"] for row in status]
    if status_ids != manifest_ids: errors.append("final_status_sample_order_or_ids_changed")
    for item, row in zip(manifest, status, strict=True):
        if row["video_id"] != str(item["video_id"]) or row["clip_id"] != str(item["clip_id"]):
            errors.append(f"official_mapping_changed:{item['sample_id']}")
    forbidden = {"sentiment", "sentiment_label", "emotion", "emotion_label", "label"}
    if status and forbidden & set(status[0]): errors.append("sentiment_or_emotion_label_column_present")

    word = load_npz(final / "q1_word_aligned_final.npz")
    stage3_word = load_npz(stage3 / "reports/q1_word_aligned_stage3_candidate.npz")
    n = len(word["word_id"])
    if n != 1932: errors.append(f"word_count={n}")
    if len(word["sample_indptr"]) != 101 or int(word["sample_indptr"][0]) != 0 or int(word["sample_indptr"][-1]) != n:
        errors.append("word_sample_indptr")
    if np.any(np.diff(word["sample_indptr"]) < 0): errors.append("word_sample_indptr_nonmonotonic")
    for sample_index, sample_id in enumerate(manifest_ids):
        left, right = int(word["sample_indptr"][sample_index]), int(word["sample_indptr"][sample_index + 1])
        if np.any(word["sample_index"][left:right] != sample_index) or np.any(word["sample_id"][left:right] != sample_id):
            errors.append(f"word_sample_membership:{sample_id}")
    if word["text"].shape != (n, 768) or word["audio"].shape != (n, 25) or word["vision"].shape != (n, 22):
        errors.append("word_feature_shapes")
    for name in ("text", "audio", "vision"):
        errors.extend(validate_masked_features(word[name], word[f"{name}_mask"], name))
    if not np.array_equal(word["masks"], np.column_stack([word["text_mask"], word["audio_mask"], word["vision_mask"]])):
        errors.append("combined_word_masks")
    for name in ("audio", "vision"):
        errors.extend(f"{name}:{item}" for item in validate_csr(
            word[f"{name}_indptr"], word[f"{name}_source_ids"],
            word[f"{name}_overlap_s"], word[f"{name}_weights"]))
        errors.extend(validate_source_sample_membership(
            word[f"{name}_indptr"], word["sample_index"], word[f"{name}_source_sample_index"], name))
        if np.any(word[f"{name}_source_ids"] < 0): errors.append(f"negative_{name}_source_id")
    for name in ("audio_coverage", "vision_coverage"):
        if np.any(word[name] < 0) or np.any(word[name] > 1) or not np.isfinite(word[name]).all():
            errors.append(f"invalid_{name}")
    for key in stage3_word:
        if not np.array_equal(word[key], stage3_word[key]): errors.append(f"word_changed_from_stage3:{key}")

    compact = load_npz(final / "q1_compact50_final.npz")
    stage3_compact = load_npz(stage3 / "reports/q1_compact50_stage3_candidate.npz")
    if compact["text"].shape != (100, 50, 768) or compact["audio"].shape != (100, 50, 25) or compact["vision"].shape != (100, 50, 22):
        errors.append("compact_feature_shapes")
    if not set(np.unique(compact["valid_mask"]).tolist()) <= {0, 1}: errors.append("nonbinary_valid_mask")
    if not set(np.unique(compact["observed_mask"]).tolist()) <= {0, 1}: errors.append("nonbinary_observed_mask")
    for index, name in enumerate(("text", "audio", "vision")):
        errors.extend(validate_masked_features(compact[name], compact["observed_mask"][:, :, index], f"compact_{name}"))
    if np.any(compact["coverage"] < 0) or np.any(compact["coverage"] > 1) or not np.isfinite(compact["coverage"]).all():
        errors.append("invalid_compact_coverage")
    for key in stage3_compact:
        if not np.array_equal(compact[key], stage3_compact[key]): errors.append(f"compact_changed_from_stage3:{key}")
    paired = np.asarray([int(row["paired_use"]) for row in status], np.uint8)
    if not np.array_equal(word["sample_paired_use"], paired) or not np.array_equal(compact["paired_use"], paired):
        errors.append("paired_use_not_frozen_status")

    no_face_indices = {int(row["sample_index"]) for row in status if row["resolution_status"] == "RESOLVED_MISSING"}
    for index in no_face_indices:
        left, right = int(word["sample_indptr"][index]), int(word["sample_indptr"][index + 1])
        if np.any(word["vision_mask"][left:right]) or np.any(compact["observed_mask"][index, :, 2]):
            errors.append(f"no_face_has_visual_observation:{index}")
    unsafe_semantic = np.isin(word["semantic_status"], ["CONFLICT", "UNRESOLVED"])
    if np.any(word["audio_mask"][unsafe_semantic]): errors.append("semantic_conflict_audio_released")
    for row in status:
        if row["resolution_status"] in {"UNRESOLVED", "RESOLVED_CONFLICT", "RESOLVED_MISSING"} and int(row["paired_use"]):
            errors.append(f"unsafe_status_released:{row['sample_id']}")

    raw_rows = read_jsonl(stage3 / "reports/raw_vision_stage3_sources.jsonl")
    known_indices = {int(row["sample_index"]) for row in raw_rows}
    raw_map = {(int(row["sample_index"]), int(row["source_id"])): int(row["stage3_eligible"]) for row in raw_rows}
    traceable = 0
    for sample_index, source_id in zip(word["vision_source_sample_index"], word["vision_source_ids"], strict=True):
        pair = (int(sample_index), int(source_id))
        if int(sample_index) in known_indices:
            traceable += 1
            if raw_map.get(pair) != 1:
                errors.append(f"untraceable_stage3_visual_source:{pair}")
                break
    if traceable != 2808: errors.append(f"stage3_traceable_visual_references={traceable}")

    freeze = verify_freeze_manifest(project, read_tsv(final / "q1_final_frozen_inputs.tsv"))
    if not freeze["ok"]: errors.append("frozen_inputs_changed")
    checks.update({
        "sample_count": len(status), "unique_sample_count": len(set(status_ids)), "word_count": n,
        "resolution_status_counts": dict(Counter(row["resolution_status"] for row in status)),
        "paired_use_count": sum(int(row["paired_use"]) for row in status),
        "word_shapes": {"text": list(word["text"].shape), "audio": list(word["audio"].shape), "vision": list(word["vision"].shape)},
        "compact_shapes": {"text": list(compact["text"].shape), "audio": list(compact["audio"].shape), "vision": list(compact["vision"].shape)},
        "stage3_traceable_visual_references": traceable,
        "official_mapping_unchanged": not any(item.startswith("official_mapping_changed") for item in errors),
        "frozen_hashes": freeze,
        "sentiment_labels_used": False, "new_models_run": False, "automatic_clip_swap": False,
    })
    result = {"ok": not errors, "errors": errors, "checks": checks}
    write_json(final / "q1_final_validation.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
