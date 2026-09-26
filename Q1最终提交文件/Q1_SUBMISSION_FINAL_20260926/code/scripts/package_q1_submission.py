from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from q1_features.finalization import sha256_file
from q1_features.storage import read_json, write_json


INSPECT_SCRIPT = r'''from pathlib import Path
import csv, hashlib, numpy as np

root = Path(__file__).resolve().parent
with np.load(root / "q1_word_aligned_final.npz", allow_pickle=False) as data:
    assert len(data["sample_indptr"]) == 101
    assert len(data["word_id"]) == 1932
    assert data["text"].shape == (1932, 768)
    assert data["audio"].shape == (1932, 25)
    assert data["vision"].shape == (1932, 22)
    for name in ("text", "audio", "vision"):
        mask = data[f"{name}_mask"]
        assert set(np.unique(mask).tolist()) <= {0, 1}
        assert np.isfinite(data[name][mask == 1]).all()
        assert np.count_nonzero(data[name][mask == 0]) == 0
    for name in ("audio", "vision"):
        indptr = data[f"{name}_indptr"]
        assert indptr[0] == 0 and np.all(np.diff(indptr) >= 0)
        assert indptr[-1] == len(data[f"{name}_source_ids"])
with (root / "summary_q1_final.csv").open(encoding="utf-8-sig") as handle:
    rows = list(csv.DictReader(handle))
assert len(rows) == 100 and sum(int(row["paired_use"]) for row in rows) == 68
print("Q1 minimal submission: structural checks passed")
'''


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Q1 minimal competition submission bundle")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--formal-run", default="runs/q1_full_20260924")
    parser.add_argument("--stage3-run", default="runs/q1_alignment_repair_stage3")
    parser.add_argument("--final-run", default="runs/q1_final_20260926")
    args = parser.parse_args()
    project = Path(args.project_root).resolve(); formal = (project / args.formal_run).resolve()
    stage3 = (project / args.stage3_run).resolve(); final = (project / args.final_run).resolve()
    validation = read_json(final / "q1_final_validation.json")
    if not validation or not validation.get("ok"):
        raise ValueError("final validation must pass before packaging")
    bundle = final / "submission_q1_minimal"; bundle.mkdir(parents=True, exist_ok=True)
    copies = {
        final / "q1_word_aligned_final.npz": bundle / "q1_word_aligned_final.npz",
        final / "summary_q1_final.csv": bundle / "summary_q1_final.csv",
        final / "q1_final_status.csv": bundle / "q1_final_status.csv",
        final / "q1_final_validation.json": bundle / "q1_final_validation.json",
        project / "configs/q1_finalization_policy.yaml": bundle / "q1_finalization_policy.yaml",
        formal / "features/audio_feature_names.json": bundle / "feature_names/audio_feature_names.json",
        formal / "features/feature_names.json": bundle / "feature_names/feature_names.json",
        formal / "features/vision_feature_names.json": bundle / "feature_names/vision_feature_names.json",
    }
    for source, target in copies.items():
        if not source.is_file(): raise FileNotFoundError(source)
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(source, target)

    source_revision = (project / "SOURCE_REVISION").read_text(encoding="utf-8").strip()
    versions = {
        "source_revision": source_revision,
        "formal_environment": read_json(formal / "environment.json"),
        "stage1_model_provenance": read_json(formal / "reports/auto_resolution/auto_resolution_model_provenance.json"),
        "stage3_acceptance": read_json(stage3 / "reports/stage3_acceptance.json"),
        "new_models_run_in_stage4": False,
    }
    write_json(bundle / "MODEL_TOOL_VERSIONS.json", versions)
    (bundle / "inspect_q1_submission.py").write_text(INSPECT_SCRIPT, encoding="utf-8")
    (bundle / "ALIGNMENT_SOURCE_MAPPING.md").write_text(
        "# Q1词级对齐与来源映射\n\n"
        "主产物以 `sample_indptr` 划分100条样本，以 `audio_indptr`/`vision_indptr` 构成词到原生来源的CSR映射。"
        "`*_source_sample_index` 必须等于该词的 `sample_index`；`*_overlap_s` 非负；mask=0时对应特征为零。"
        "词级 start/end 来自已冻结时间对齐，语义冲突或未决样本不会伪造音频时间。"
        "视觉仅保留 VERIFIED_ACTIVE 或 VERIFIED_CONTINUITY 来源；no-face视觉始终mask。\n",
        encoding="utf-8")
    (bundle / "README.md").write_text(
        "# Q1 最小提交包\n\n"
        "主产物：`q1_word_aligned_final.npz`；100条样本、1932词，维度为text=768、audio=25、vision=22。\n\n"
        "`summary_q1_final.csv` 是全部100条结果汇总；`q1_final_status.csv` 保留semantic/temporal/visual/mapping/resolution多轴状态。"
        "本包不含情感标签、模型权重、原始媒体或研究阶段中间文件。`q1_compact50_final.npz` 是辅助固定长度表示，保留在完整研究目录，未放入最小包。\n\n"
        "验证：使用项目锁定环境运行 `python inspect_q1_submission.py`。Coverage表示经过mask后的可用性，不是accuracy。\n",
        encoding="utf-8")

    files = sorted(path for path in bundle.rglob("*") if path.is_file() and path.name != "Q1_SUBMISSION_MANIFEST.txt")
    lines = ["sha256  bytes  relative_path"]
    for path in files:
        lines.append(f"{sha256_file(path)}  {path.stat().st_size}  {path.relative_to(bundle)}")
    manifest = "\n".join(lines) + "\n"
    (bundle / "Q1_SUBMISSION_MANIFEST.txt").write_text(manifest, encoding="utf-8")
    (final / "Q1_SUBMISSION_MANIFEST.txt").write_text(manifest, encoding="utf-8")

    zip_path = final / "Q1_SUBMISSION_MINIMAL_20260926.zip"
    with ZipFile(zip_path, "w", compression=ZIP_DEFLATED, compresslevel=6, allowZip64=True) as archive:
        for path in sorted(item for item in bundle.rglob("*") if item.is_file()):
            archive.write(path, f"submission_q1_minimal/{path.relative_to(bundle).as_posix()}")
    result = {
        "ok": True, "zip_path": str(zip_path), "bytes": zip_path.stat().st_size,
        "mib": zip_path.stat().st_size / (1024 * 1024), "sha256": sha256_file(zip_path),
        "file_count": sum(path.is_file() for path in bundle.rglob("*")),
        "under_10_mib": zip_path.stat().st_size <= 10 * 1024 * 1024,
        "under_50_mib": zip_path.stat().st_size <= 50 * 1024 * 1024,
        "compact50_included": False, "labels_included": False, "weights_included": False,
    }
    write_json(final / "submission_package_size.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
