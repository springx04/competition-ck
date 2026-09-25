from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from q1_features.storage import read_json, write_json


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify that Q1 frozen production assets were not modified")
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--baseline-json", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    baseline = read_json(args.baseline_json)
    rows = []
    for relative_path, expected in baseline["files"].items():
        path = root / relative_path
        actual = sha256(path) if path.is_file() else None
        rows.append({
            "path": relative_path,
            "exists": path.is_file(),
            "expected_sha256": expected,
            "actual_sha256": actual,
            "unchanged": actual == expected,
        })
    result = {"ok": all(row["unchanged"] for row in rows), "files": rows}
    write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
