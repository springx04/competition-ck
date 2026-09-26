from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np


EXPECTED_STATUS_COUNTS = {
    "RESOLVED_VALID": 57,
    "RESOLVED_PARTIAL": 19,
    "RESOLVED_CONFLICT": 5,
    "RESOLVED_MISSING": 4,
    "UNRESOLVED": 15,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _binary(array: np.ndarray, name: str) -> None:
    values = set(np.unique(array).tolist())
    _require(values <= {0, 1}, f"{name} is not binary: {sorted(values)}")


def _manifest_rows(root: Path) -> list[dict[str, str]]:
    path = root / "MANIFEST.tsv"
    _require(path.is_file(), "missing MANIFEST.tsv")
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def verify_manifest(root: str | Path) -> dict[str, Any]:
    root = Path(root).resolve()
    rows = _manifest_rows(root)
    errors: list[str] = []
    seen: set[str] = set()
    for row in rows:
        relative = row.get("relative_path", "")
        if not relative or relative in seen:
            errors.append(f"invalid or duplicate manifest path: {relative!r}")
            continue
        seen.add(relative)
        candidate = (root / relative).resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            errors.append(f"manifest path escapes package: {relative}")
            continue
        if not candidate.is_file():
            errors.append(f"missing manifest file: {relative}")
            continue
        actual_bytes = candidate.stat().st_size
        if str(actual_bytes) != row.get("bytes"):
            errors.append(f"size mismatch: {relative}")
        if sha256_file(candidate) != row.get("sha256"):
            errors.append(f"sha256 mismatch: {relative}")

    declared = seen | {"MANIFEST.tsv", "SHA256SUMS"}
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    }
    unexpected = sorted(actual - declared)
    missing = sorted(declared - actual)
    if unexpected:
        errors.append(f"unexpected files: {unexpected}")
    if missing:
        errors.append(f"undeclared control files: {missing}")
    return {"ok": not errors, "file_count": len(rows), "errors": errors}


def _check_word_package(path: Path, *, full: bool) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        required = {
            "sample_indptr", "sample_id", "sample_index", "word_id", "raw_word",
            "start", "end", "text", "audio", "vision", "text_mask", "audio_mask",
            "vision_mask", "audio_coverage", "vision_coverage", "audio_indptr",
            "audio_source_ids", "audio_source_sample_index", "audio_overlap_s",
            "audio_weights", "vision_indptr", "vision_source_ids",
            "vision_source_sample_index", "vision_overlap_s", "vision_weights",
            "masks", "sample_paired_use", "sample_resolution_status",
        }
        missing = sorted(required - set(data.files))
        _require(not missing, f"word NPZ missing keys: {missing}")
        words = len(data["word_id"])
        _require(words == 1932, f"word count is {words}, expected 1932")
        _require(data["sample_indptr"].shape == (101,), "sample_indptr shape mismatch")
        _require(data["sample_indptr"][0] == 0, "sample_indptr must start at zero")
        _require(data["sample_indptr"][-1] == words, "sample_indptr endpoint mismatch")
        _require(np.all(np.diff(data["sample_indptr"]) >= 0), "sample_indptr is not monotonic")
        _require(data["text"].shape == (1932, 768), "word text shape mismatch")
        _require(data["audio"].shape == (1932, 25), "word audio shape mismatch")
        _require(data["vision"].shape == (1932, 22), "word vision shape mismatch")
        _require(data["masks"].shape == (1932, 3), "word masks shape mismatch")
        for name in ("text_mask", "audio_mask", "vision_mask", "masks", "sample_paired_use"):
            _binary(data[name], name)
        stacked = np.column_stack((data["text_mask"], data["audio_mask"], data["vision_mask"]))
        _require(np.array_equal(data["masks"], stacked), "combined word masks do not match modality masks")
        sample_ids = np.asarray(data["sample_id"]).astype(str)
        unique_ids = list(dict.fromkeys(sample_ids.tolist()))
        _require(len(unique_ids) == 100, "word NPZ does not contain 100 unique sample IDs")
        _require(len(data["sample_paired_use"]) == 100, "sample_paired_use length mismatch")
        _require(int(data["sample_paired_use"].sum()) == 68, "word paired_use count mismatch")
        status_counts = dict(Counter(np.asarray(data["sample_resolution_status"]).astype(str).tolist()))
        _require(status_counts == EXPECTED_STATUS_COUNTS, f"word status counts mismatch: {status_counts}")

        if full:
            for name in ("text", "audio", "vision"):
                values = data[name]
                mask = data[f"{name}_mask"].astype(bool)
                _require(np.isfinite(values).all(), f"{name} contains NaN/Inf")
                _require(np.count_nonzero(values[~mask]) == 0, f"{name} has nonzero values where mask=0")
            for name in ("audio_coverage", "vision_coverage"):
                values = data[name]
                _require(np.isfinite(values).all(), f"{name} contains NaN/Inf")
                _require(bool(np.all((values >= 0) & (values <= 1))), f"{name} is outside [0,1]")
            starts, ends = data["start"], data["end"]
            _require(np.isfinite(starts).all() and np.isfinite(ends).all(), "word times contain NaN/Inf")
            _require(bool(np.all(ends >= starts)), "word interval has end < start")
            for modality in ("audio", "vision"):
                indptr = data[f"{modality}_indptr"]
                source_ids = data[f"{modality}_source_ids"]
                source_samples = data[f"{modality}_source_sample_index"]
                overlap = data[f"{modality}_overlap_s"]
                weights = data[f"{modality}_weights"]
                _require(indptr.shape == (words + 1,), f"{modality} indptr shape mismatch")
                _require(indptr[0] == 0 and np.all(np.diff(indptr) >= 0), f"{modality} indptr invalid")
                _require(indptr[-1] == len(source_ids), f"{modality} CSR endpoint mismatch")
                _require(len(source_ids) == len(source_samples) == len(overlap) == len(weights), f"{modality} CSR lengths differ")
                _require(np.all(source_ids >= 0), f"{modality} source ID is negative")
                _require(np.isfinite(overlap).all() and np.all(overlap >= 0), f"{modality} overlap invalid")
                _require(np.isfinite(weights).all() and np.all(weights >= 0), f"{modality} weights invalid")
                owner_rows = np.repeat(np.arange(words), np.diff(indptr))
                expected_samples = data["sample_index"][owner_rows]
                _require(np.array_equal(source_samples, expected_samples), f"{modality} source sample mapping mismatch")

        return {
            "word_count": words,
            "sample_count": len(unique_ids),
            "paired_use": int(data["sample_paired_use"].sum()),
            "status_counts": status_counts,
            "sample_ids": unique_ids,
        }


def _check_compact_package(path: Path, *, full: bool) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        required = {
            "text", "audio", "vision", "valid_mask", "observed_mask", "perturb_mask",
            "coverage", "time_intervals", "duration", "valid_length", "sample_index",
            "paired_use", "resolution_status",
        }
        missing = sorted(required - set(data.files))
        _require(not missing, f"compact NPZ missing keys: {missing}")
        _require(data["text"].shape == (100, 50, 768), "compact text shape mismatch")
        _require(data["audio"].shape == (100, 50, 25), "compact audio shape mismatch")
        _require(data["vision"].shape == (100, 50, 22), "compact vision shape mismatch")
        _require(data["observed_mask"].shape == (100, 50, 3), "compact observed_mask shape mismatch")
        _require(data["coverage"].shape == (100, 50, 3), "compact coverage shape mismatch")
        _require(data["time_intervals"].shape == (100, 50, 2), "compact time_intervals shape mismatch")
        for name in ("valid_mask", "observed_mask", "perturb_mask", "paired_use"):
            _binary(data[name], name)
        _require(np.array_equal(data["sample_index"], np.arange(100)), "compact sample_index mismatch")
        _require(int(data["paired_use"].sum()) == 68, "compact paired_use count mismatch")
        status_counts = dict(Counter(np.asarray(data["resolution_status"]).astype(str).tolist()))
        _require(status_counts == EXPECTED_STATUS_COUNTS, f"compact status counts mismatch: {status_counts}")
        if full:
            for name in ("text", "audio", "vision", "coverage", "time_intervals", "duration"):
                _require(np.isfinite(data[name]).all(), f"compact {name} contains NaN/Inf")
            _require(bool(np.all((data["coverage"] >= 0) & (data["coverage"] <= 1))), "compact coverage outside [0,1]")
            _require(bool(np.all(data["time_intervals"][..., 1] >= data["time_intervals"][..., 0])), "compact interval has end < start")
            for index, name in enumerate(("text", "audio", "vision")):
                mask = data["observed_mask"][..., index].astype(bool)
                _require(np.count_nonzero(data[name][~mask]) == 0, f"compact {name} nonzero where observed_mask=0")
        return {
            "sample_count": 100,
            "window_count": 50,
            "paired_use": int(data["paired_use"].sum()),
            "status_counts": status_counts,
        }


def inspect_submission(root: str | Path, *, mode: str = "quick") -> dict[str, Any]:
    if mode not in {"quick", "full"}:
        raise ValueError("mode must be 'quick' or 'full'")
    root = Path(root).resolve()
    required = [
        "README.md", "SOURCE_REVISION", "MANIFEST.tsv", "SHA256SUMS",
        "features/q1_word_aligned_final.npz", "features/q1_compact50_final.npz",
        "metadata/summary_q1_final.csv", "metadata/q1_final_status.csv",
        "metadata/q1_final_validation.json",
    ]
    missing = [name for name in required if not (root / name).is_file()]
    _require(not missing, f"missing submission files: {missing}")
    revision = (root / "SOURCE_REVISION").read_text(encoding="utf-8").strip()
    _require(revision == "da6c8e5bd553d0480ca941384ae20f279929872f", "SOURCE_REVISION mismatch")

    manifest = verify_manifest(root)
    _require(manifest["ok"], f"manifest verification failed: {manifest['errors']}")
    word = _check_word_package(root / "features/q1_word_aligned_final.npz", full=mode == "full")
    compact = _check_compact_package(root / "features/q1_compact50_final.npz", full=mode == "full")

    summary = read_csv_rows(root / "metadata/summary_q1_final.csv")
    status = read_csv_rows(root / "metadata/q1_final_status.csv")
    _require(len(summary) == len(status) == 100, "summary/status row count mismatch")
    _require(len({row["sample_id"] for row in summary}) == 100, "summary sample IDs are not unique")
    _require({row["sample_id"] for row in summary} == set(word["sample_ids"]), "summary and word sample IDs differ")
    _require(sum(int(row["word_count"]) for row in summary) == 1932, "summary word count mismatch")
    _require(sum(int(row["paired_use"]) for row in summary) == 68, "summary paired_use mismatch")
    status_counts = dict(Counter(row["resolution_status"] for row in status))
    _require(status_counts == EXPECTED_STATUS_COUNTS, f"CSV status counts mismatch: {status_counts}")
    forbidden_columns = {"sentiment", "sentiment_label", "emotion", "emotion_label", "label"}
    _require(not (forbidden_columns & set(summary[0])), "summary contains an emotion-label column")
    _require(not (forbidden_columns & set(status[0])), "status contains an emotion-label column")

    with (root / "metadata/q1_final_validation.json").open(encoding="utf-8") as handle:
        validation = json.load(handle)
    _require(validation.get("ok") is True and validation.get("errors") == [], "final validation is not successful")
    _require(word["status_counts"] == compact["status_counts"] == status_counts, "status counts differ across artifacts")

    return {
        "ok": True,
        "mode": mode,
        "source_revision": revision,
        "manifest_files": manifest["file_count"],
        "sample_count": 100,
        "word_count": 1932,
        "paired_use": 68,
        "resolution_status_counts": status_counts,
        "word_shapes": {"text": [1932, 768], "audio": [1932, 25], "vision": [1932, 22]},
        "compact_shapes": {"text": [100, 50, 768], "audio": [100, 50, 25], "vision": [100, 50, 22]},
    }


def write_manifest(root: str | Path, entries: Iterable[tuple[str, str]]) -> list[dict[str, Any]]:
    root = Path(root).resolve()
    rows: list[dict[str, Any]] = []
    for relative, role in sorted(entries):
        path = root / relative
        rows.append({
            "relative_path": relative,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "role": role,
        })
    manifest = root / "MANIFEST.tsv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["relative_path", "bytes", "sha256", "role"], delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    sums = root / "SHA256SUMS"
    sums.write_text("".join(f"{row['sha256']}  {row['relative_path']}\n" for row in rows), encoding="utf-8")
    return rows
