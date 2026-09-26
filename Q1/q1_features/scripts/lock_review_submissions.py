from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from q1_features.storage import write_json


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create an immutable SHA256 snapshot of independent Q1 review submissions"
    )
    parser.add_argument("--reviewer-a-summary", required=True)
    parser.add_argument("--reviewer-b-summary", required=True)
    parser.add_argument("--reviewer-a-segments", required=True)
    parser.add_argument("--reviewer-b-segments", required=True)
    parser.add_argument("--reviewer-a-anchors", required=True)
    parser.add_argument("--reviewer-b-anchors", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    sources = {
        "reviewer_a_summary": Path(args.reviewer_a_summary).resolve(),
        "reviewer_b_summary": Path(args.reviewer_b_summary).resolve(),
        "reviewer_a_segments": Path(args.reviewer_a_segments).resolve(),
        "reviewer_b_segments": Path(args.reviewer_b_segments).resolve(),
        "reviewer_a_anchors": Path(args.reviewer_a_anchors).resolve(),
        "reviewer_b_anchors": Path(args.reviewer_b_anchors).resolve(),
    }
    missing = [str(path) for path in sources.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing review submissions: {missing}")

    output = Path(args.output_dir).resolve()
    manifest_path = output / "submission_lock.json"
    if manifest_path.exists() or (output.exists() and any(output.iterdir())):
        raise FileExistsError(f"lock directory must be new and empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    records = {}
    for name, source in sources.items():
        target = output / f"{name}.csv"
        shutil.copy2(source, target)
        source_hash = sha256(source)
        target_hash = sha256(target)
        if source_hash != target_hash:
            raise RuntimeError(f"hash mismatch while locking {source}")
        target.chmod(0o444)
        records[name] = {
            "source": str(source),
            "locked_copy": str(target),
            "sha256": target_hash,
            "bytes": target.stat().st_size,
        }

    manifest = {
        "status": "independent_reviewer_submissions_locked",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": records,
        "auxiliary_outputs_were_not_inputs": True,
    }
    write_json(manifest_path, manifest)
    manifest_path.chmod(0o444)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
