from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

from q1_features.submission import inspect_submission, sha256_file, write_manifest


ARCHIVE_ROOT = "Q1_SUBMISSION_FINAL_20260926"
OLD_MINIMAL_SHA256 = "5dce03b5758ca631759f02afacf5e1bc84a9440da4019969a78f0eeb521dcb12"
TEXT_SUFFIXES = {".csv", ".json", ".md", ".py", ".sh", ".toml", ".tsv", ".txt", ".yaml", ".yml"}
PRIVATE_PATTERNS = {
    "home_path": re.compile(r"/home/[A-Za-z0-9._-]+(?:/|\b)"),
    "server_ip": re.compile(r"\b192\.168\.3\.28\b"),
    "proxy": re.compile(r"\b127\.0\.0\.1:7890\b"),
    "ssh_port": re.compile(r"\b" + "22" + "001" + r"\b"),
}


def _copy_file(source: Path, target: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def _copy_tree(source: Path, target: Path) -> None:
    if not source.is_dir():
        raise FileNotFoundError(source)
    shutil.copytree(
        source,
        target,
        ignore=shutil.ignore_patterns(".pytest_cache", "__pycache__", "*.pyc", "*.pyo", "*.part", "*.tmp"),
    )


def _sanitize_string(value: str, project: Path) -> str:
    replacements = [
        (str(project), "<PROJECT_ROOT>"),
        (str(project.parent / "E题数据"), "<DATA_ROOT>"),
        ("/" + "home" + "/conda/feedstock_root", "<CONDA_BUILD_ROOT>"),
        (str(project.parents[1]), "<USER_HOME>"),
    ]
    result = value
    for source, target in replacements:
        result = result.replace(source, target)
    return result


def _sanitize_json(value: Any, project: Path) -> Any:
    if isinstance(value, dict):
        return {key: _sanitize_json(item, project) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize_json(item, project) for item in value]
    if isinstance(value, str):
        return _sanitize_string(value, project)
    return value


def _copy_sanitized_json(source: Path, target: Path, project: Path) -> None:
    with source.open(encoding="utf-8") as handle:
        value = json.load(handle)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(_sanitize_json(value, project), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def role_for(relative: str) -> str:
    if relative.startswith("features/"):
        return "final_feature"
    if relative.startswith("metadata/"):
        return "audit_metadata"
    if relative.startswith("examples/"):
        return "paper_example"
    if relative.startswith("code/tests/"):
        return "regression_test"
    if relative.startswith("code/configs/"):
        return "configuration"
    if relative.startswith("code/"):
        return "source_code"
    if relative.startswith("documentation/"):
        return "documentation"
    return "release_control"


def scan_private_information(root: Path) -> list[str]:
    errors: list[str] = []
    forbidden_names = {"labels.csv", "sentiment_labels.csv", "emotion_labels.csv"}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        if path.name.lower() in forbidden_names:
            errors.append(f"forbidden label file: {relative}")
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for label, pattern in PRIVATE_PATTERNS.items():
            if pattern.search(text):
                errors.append(f"{label} found in {relative}")
    return errors


def ensure_size_limit(path: Path, max_bytes: int) -> None:
    if path.stat().st_size > max_bytes:
        raise ValueError(f"archive exceeds size limit: {path.stat().st_size} > {max_bytes}")


def build_submission(project: Path, final: Path, output: Path, max_bytes: int) -> dict[str, Any]:
    project = project.resolve()
    final = final.resolve()
    output = output.resolve()
    if not final.is_dir():
        raise FileNotFoundError(final)
    if output.exists() or Path(str(output) + ".sha256").exists():
        raise FileExistsError(f"refusing to overwrite existing release: {output}")

    validation = json.loads((final / "q1_final_validation.json").read_text(encoding="utf-8"))
    if validation.get("ok") is not True or validation.get("errors") != []:
        raise ValueError("final validation must be ok=true with errors=[] before packaging")

    old_zip = final / "Q1_SUBMISSION_MINIMAL_20260926.zip"
    if sha256_file(old_zip) != OLD_MINIMAL_SHA256:
        raise ValueError("historical minimal ZIP checksum changed")

    bundle = final / "submission_q1_final"
    if bundle.exists():
        raise FileExistsError(f"refusing to overwrite existing bundle: {bundle}")
    temporary = final / "submission_q1_final.building"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)

    formal = project / "runs/q1_full_20260924"
    try:
        direct_copies = {
            final / "q1_word_aligned_final.npz": temporary / "features/q1_word_aligned_final.npz",
            final / "q1_compact50_final.npz": temporary / "features/q1_compact50_final.npz",
            formal / "features/audio_feature_names.json": temporary / "features/feature_names/audio_feature_names.json",
            formal / "features/feature_names.json": temporary / "features/feature_names/feature_names.json",
            formal / "features/vision_feature_names.json": temporary / "features/feature_names/vision_feature_names.json",
            final / "summary_q1_final.csv": temporary / "metadata/summary_q1_final.csv",
            final / "q1_final_status.csv": temporary / "metadata/q1_final_status.csv",
            final / "q1_final_unresolved.csv": temporary / "metadata/q1_final_unresolved.csv",
            final / "q1_final_validation.json": temporary / "metadata/q1_final_validation.json",
            final / "q1_final_build_summary.json": temporary / "metadata/q1_final_build_summary.json",
            final / "q1_final_frozen_inputs.json": temporary / "metadata/q1_final_frozen_inputs.json",
            final / "q1_final_frozen_inputs.tsv": temporary / "metadata/q1_final_frozen_inputs.tsv",
            final / "q1_final_coverage_terms.csv": temporary / "metadata/q1_final_coverage_terms.csv",
            final / "q1_final_artifact_sha256.tsv": temporary / "metadata/q1_final_artifact_sha256.tsv",
            final / "config.snapshot.yaml": temporary / "metadata/config.snapshot.yaml",
            final / "paper_assets/q1_typical_alignment.csv": temporary / "examples/q1_typical_alignment.csv",
            final / "paper_assets/q1_typical_alignment_timeline.png": temporary / "examples/q1_typical_alignment_timeline.png",
            final / "paper_assets/q1_abnormal_mask_example.csv": temporary / "examples/q1_abnormal_mask_example.csv",
            project / "SOURCE_REVISION": temporary / "SOURCE_REVISION",
            project / "pyproject.toml": temporary / "code/pyproject.toml",
            project / "env/requirements-q1.txt": temporary / "code/requirements-q1.txt",
            project / "THIRD_PARTY_NOTICES.md": temporary / "documentation/THIRD_PARTY_NOTICES.md",
            project / "docs/reports/Q1第一问最终报告总结_20260926.md": temporary / "documentation/Q1第一问最终报告总结_20260926.md",
            project / "docs/delivery/Q1最终提交包README_20260926.md": temporary / "README.md",
            final / "submission_q1_minimal/ALIGNMENT_SOURCE_MAPPING.md": temporary / "documentation/ALIGNMENT_SOURCE_MAPPING.md",
        }
        for source, target in direct_copies.items():
            _copy_file(source, target)

        for name in ("src", "scripts", "tests", "configs"):
            _copy_tree(project / name, temporary / "code" / name)

        _copy_sanitized_json(final / "Q1_FINAL_ACCEPTANCE.json", temporary / "metadata/Q1_FINAL_ACCEPTANCE.json", project)
        _copy_sanitized_json(final / "q1_final_engineering.json", temporary / "metadata/q1_final_engineering.json", project)
        _copy_sanitized_json(
            final / "submission_q1_minimal/MODEL_TOOL_VERSIONS.json",
            temporary / "metadata/MODEL_TOOL_VERSIONS.json",
            project,
        )

        inspector = """from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "code" / "src"))
from q1_features.submission import inspect_submission

parser = argparse.ArgumentParser(description="Inspect the Q1 final submission without network access")
parser.add_argument("--mode", choices=("quick", "full"), default="quick")
args = parser.parse_args()
print(json.dumps(inspect_submission(ROOT, mode=args.mode), ensure_ascii=False, indent=2))
"""
        (temporary / "inspect_q1_submission.py").write_text(inspector, encoding="utf-8")

        privacy_errors = scan_private_information(temporary)
        if privacy_errors:
            raise ValueError("privacy/data scan failed: " + "; ".join(privacy_errors))

        entries = []
        for path in sorted(item for item in temporary.rglob("*") if item.is_file()):
            relative = path.relative_to(temporary).as_posix()
            if relative not in {"MANIFEST.tsv", "SHA256SUMS"}:
                entries.append((relative, role_for(relative)))
        manifest_rows = write_manifest(temporary, entries)
        inspection = inspect_submission(temporary, mode="full")

        temporary.rename(bundle)
        output.parent.mkdir(parents=True, exist_ok=True)
        zip_temporary = Path(str(output) + ".building")
        if zip_temporary.exists():
            zip_temporary.unlink()
        with ZipFile(zip_temporary, "w", compression=ZIP_DEFLATED, compresslevel=9, allowZip64=True) as archive:
            for path in sorted(item for item in bundle.rglob("*") if item.is_file()):
                archive.write(path, f"{ARCHIVE_ROOT}/{path.relative_to(bundle).as_posix()}")
        with ZipFile(zip_temporary) as archive:
            corrupt = archive.testzip()
            if corrupt is not None:
                raise ValueError(f"ZIP CRC failure: {corrupt}")
            with tempfile.TemporaryDirectory(prefix="q1-submission-check-") as temp_dir:
                archive.extractall(temp_dir)
                extracted_inspection = inspect_submission(Path(temp_dir) / ARCHIVE_ROOT, mode="full")
        ensure_size_limit(zip_temporary, max_bytes)
        zip_temporary.rename(output)

        zip_sha = sha256_file(output)
        sha_path = Path(str(output) + ".sha256")
        sha_path.write_text(f"{zip_sha}  {output.name}\n", encoding="utf-8")
        result = {
            "ok": True,
            "bundle": str(bundle),
            "zip_path": str(output),
            "bytes": output.stat().st_size,
            "mib": output.stat().st_size / (1024 * 1024),
            "sha256": zip_sha,
            "file_count": len(manifest_rows) + 2,
            "under_50_mib": output.stat().st_size <= 50 * 1024 * 1024,
            "old_minimal_sha256_unchanged": True,
            "inspection": inspection,
            "extracted_inspection": extracted_inspection,
            "labels_included": False,
            "weights_included": False,
            "raw_media_included": False,
        }
        (final / "q1_final_submission_package.json").write_text(
            json.dumps(_sanitize_json(result, project), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return result
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the versioned Q1 final competition submission")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--final-run", default="runs/q1_final_20260926")
    parser.add_argument("--output", default="runs/q1_final_20260926/Q1_SUBMISSION_FINAL_20260926.zip")
    parser.add_argument("--max-bytes", type=int, default=50 * 1024 * 1024)
    args = parser.parse_args()
    project = Path(args.project_root).resolve()
    result = build_submission(project, project / args.final_run, project / args.output, args.max_bytes)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
