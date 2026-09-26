from __future__ import annotations

import csv
import hashlib
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


EXPECTED_RESOLUTION_COUNTS = {
    "RESOLVED_VALID": 57,
    "RESOLVED_PARTIAL": 19,
    "RESOLVED_CONFLICT": 5,
    "RESOLVED_MISSING": 4,
    "UNRESOLVED": 15,
}
UNSAFE_PAIRED_STATUSES = {"RESOLVED_CONFLICT", "RESOLVED_MISSING", "UNRESOLVED"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_tsv(path: Path, rows: Iterable[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def build_freeze_manifest(project: Path, stage2: Path, stage3: Path, policy: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(role: str, path: Path, *, expected_missing: bool = False) -> None:
        relative = str(path.resolve().relative_to(project.resolve()))
        if relative in seen:
            return
        if expected_missing:
            if path.exists():
                raise ValueError(f"frozen missing artifact unexpectedly exists: {relative}")
            seen.add(relative)
            rows.append({"role": role, "path": relative, "exists": 0, "bytes": 0, "sha256": ""})
            return
        if not path.is_file():
            raise FileNotFoundError(path)
        seen.add(relative)
        rows.append({
            "role": role,
            "path": relative,
            "exists": 1,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })

    # Inherit and independently re-hash the 610-item formal/raw evidence freeze.
    formal_manifest = read_tsv(stage2 / "reports" / "frozen_baseline_sha256.tsv")
    for item in formal_manifest:
        path = project / item["path"]
        expected_missing = not item["sha256"]
        if expected_missing and path.exists():
            raise ValueError(f"formal frozen missing state changed before Stage4: {item['path']}")
        if not expected_missing and (not path.is_file() or sha256_file(path) != item["sha256"]):
            raise ValueError(f"formal frozen artifact changed before Stage4: {item['path']}")
        add(f"formal:{item['role']}", path, expected_missing=expected_missing)

    for root, role in ((stage2, "stage2"), (stage3, "stage3")):
        for path in sorted(root.rglob("*")):
            if path.is_file():
                add(role, path)
    add("stage4_policy", policy)
    return rows


def verify_freeze_manifest(project: Path, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    checks = []
    for row in rows:
        path = project / str(row["path"])
        expected_exists = bool(int(row.get("exists", 1 if row.get("sha256") else 0)))
        actual_exists = path.is_file()
        actual = sha256_file(path) if actual_exists else ""
        checks.append({
            "path": str(row["path"]),
            "expected_exists": expected_exists,
            "actual_exists": actual_exists,
            "expected_sha256": str(row["sha256"]),
            "actual_sha256": actual,
            "unchanged": actual_exists == expected_exists and actual == str(row["sha256"]),
        })
    return {
        "ok": all(item["unchanged"] for item in checks),
        "file_count": len(checks),
        "changed": [item for item in checks if not item["unchanged"]],
    }


def validate_status_rows(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    errors: list[str] = []
    if len(rows) != 100:
        errors.append(f"status_row_count={len(rows)}")
    ids = [str(row["sample_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        errors.append("duplicate_sample_id")
    counts = Counter(str(row["resolution_status"]) for row in rows)
    if dict(counts) != EXPECTED_RESOLUTION_COUNTS:
        errors.append(f"resolution_counts={dict(counts)}")
    paired = sum(int(row["paired_use"]) for row in rows)
    if paired != 68:
        errors.append(f"paired_use={paired}")
    for row in rows:
        status = str(row["resolution_status"])
        paired_use = int(row["paired_use"])
        if status in UNSAFE_PAIRED_STATUSES and paired_use:
            errors.append(f"unsafe_paired_use:{row['sample_id']}:{status}")
    return errors


def validate_csr(indptr: np.ndarray, source_ids: np.ndarray, overlap: np.ndarray, weights: np.ndarray) -> list[str]:
    errors: list[str] = []
    if indptr.ndim != 1 or len(indptr) == 0 or int(indptr[0]) != 0:
        errors.append("invalid_indptr_start")
    if np.any(np.diff(indptr) < 0):
        errors.append("nonmonotonic_indptr")
    if int(indptr[-1]) != len(source_ids) or len(source_ids) != len(overlap) or len(overlap) != len(weights):
        errors.append("csr_length_mismatch")
    if np.any(overlap < 0):
        errors.append("negative_overlap")
    if not np.isfinite(overlap).all() or not np.isfinite(weights).all():
        errors.append("nonfinite_csr_values")
    return errors


def validate_masked_features(features: np.ndarray, mask: np.ndarray, name: str) -> list[str]:
    errors: list[str] = []
    if not set(np.unique(mask).tolist()) <= {0, 1}:
        errors.append(f"nonbinary_{name}_mask")
    if not np.isfinite(features[mask == 1]).all():
        errors.append(f"nonfinite_unmasked_{name}")
    if np.count_nonzero(features[mask == 0]):
        errors.append(f"nonzero_masked_{name}")
    return errors


def validate_source_sample_membership(
    indptr: np.ndarray,
    word_sample_index: np.ndarray,
    source_sample_index: np.ndarray,
    name: str,
) -> list[str]:
    errors: list[str] = []
    for word_index, sample_index in enumerate(word_sample_index):
        left, right = int(indptr[word_index]), int(indptr[word_index + 1])
        if np.any(source_sample_index[left:right] != sample_index):
            errors.append(f"{name}_source_crosses_sample:{word_index}")
            break
    return errors
