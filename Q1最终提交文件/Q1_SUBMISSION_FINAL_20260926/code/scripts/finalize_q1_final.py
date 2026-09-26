from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import matplotlib.pyplot as plt
import numpy as np

from q1_features.finalization import (
    EXPECTED_RESOLUTION_COUNTS,
    build_freeze_manifest,
    read_tsv,
    sha256_file,
    validate_status_rows,
    verify_freeze_manifest,
    write_tsv,
)
from q1_features.storage import load_npz, read_json, read_jsonl, write_csv, write_json, write_npz


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def freeze_inputs(project: Path, stage2: Path, stage3: Path, policy: Path, output: Path) -> dict:
    manifest_path = output / "q1_final_frozen_inputs.tsv"
    if manifest_path.is_file():
        rows = read_tsv(manifest_path)
    else:
        rows = build_freeze_manifest(project, stage2, stage3, policy)
        write_tsv(manifest_path, rows, ["role", "path", "exists", "bytes", "sha256"])
    verification = verify_freeze_manifest(project, rows)
    digest = hashlib.sha256(
        "\n".join(f"{row['sha256']}  {row['path']}" for row in rows).encode()
    ).hexdigest()
    summary = {**verification, "combined_sha256": digest}
    write_json(output / "q1_final_frozen_inputs.json", summary)
    if not verification["ok"]:
        raise ValueError("frozen Stage1/2/3 input changed")
    return summary


def build_final_status(
    manifest: list[dict], stage3_rows: list[dict[str, str]], proposals: list[dict[str, str]], output: Path
) -> list[dict]:
    status = {row["sample_id"]: row for row in stage3_rows}
    proposed = {row["sample_id"]: row for row in proposals}
    rows = []
    for item in manifest:
        sid = item["sample_id"]
        source, proposal = status[sid], proposed[sid]
        resolution = source["resolution_status"]
        paired = int(proposal["proposed_paired_use"])
        if resolution in {"RESOLVED_CONFLICT", "RESOLVED_MISSING", "UNRESOLVED"}:
            paired = 0
        rows.append({
            "sample_index": int(item["sample_index"]),
            "sample_id": sid,
            "video_id": item["video_id"],
            "clip_id": item["clip_id"],
            "semantic_status": source["semantic_status"],
            "temporal_status": source["temporal_status"],
            "visual_status": source["visual_status"],
            "mapping_status": source["mapping_status"],
            "resolution_status": resolution,
            "paired_use": paired,
            "text_usable_fraction": float(source["usable_text"]),
            "audio_usable_fraction": float(source["usable_audio_fraction"]),
            "vision_usable_fraction": float(source["usable_visual_fraction"]),
            "trimodal_usable_fraction": float(source["usable_trimodal_fraction"]),
            "reason": source["reason"],
        })
    errors = validate_status_rows(rows)
    if errors:
        raise ValueError("final status policy violation: " + ";".join(errors))
    strong = next(row for row in rows if row["sample_id"] == "-iRBcNs9oI8$_$6")
    if strong["mapping_status"] != "MAPPING_STRONG" or strong["resolution_status"] != "RESOLVED_CONFLICT" or strong["paired_use"]:
        raise ValueError("MAPPING_STRONG fixed SAFE_MASK policy violated")
    write_csv(output / "q1_final_status.csv", rows, list(rows[0]))
    return rows


def write_unresolved(rows: list[dict], output: Path) -> list[dict]:
    result = []
    for row in rows:
        if row["resolution_status"] != "UNRESOLVED":
            continue
        if row["mapping_status"] in {"MAPPING_WEAK", "MAPPING_UNRESOLVED"}:
            category = "mapping unstable"
        elif row["semantic_status"] == "UNRESOLVED":
            category = "semantic pairing unresolved"
        else:
            category = "visual identity unresolved"
        result.append({
            "sample_id": row["sample_id"], "category": category, "reason": row["reason"],
            "sample_preserved": 1, "unsafe_modality_masked": 1,
            "automatic_recovery_attempted": 1, "forced_repair": 0,
        })
    counts = Counter(row["category"] for row in result)
    expected = {"visual identity unresolved": 10, "mapping unstable": 2, "semantic pairing unresolved": 3}
    if dict(counts) != expected:
        raise ValueError(f"unresolved categories changed: {dict(counts)}")
    write_csv(output / "q1_final_unresolved.csv", result, list(result[0]))
    return result


def freeze_npz(stage3_reports: Path, output: Path, rows: list[dict]) -> tuple[dict, dict]:
    paired = np.asarray([int(row["paired_use"]) for row in rows], np.uint8)
    status = np.asarray([row["resolution_status"] for row in rows])
    word = load_npz(stage3_reports / "q1_word_aligned_stage3_candidate.npz")
    if len(word["sample_indptr"]) != 101 or len(word["word_id"]) != 1932:
        raise ValueError("Stage3 word structure changed")
    word["sample_paired_use"] = paired
    word["sample_resolution_status"] = status
    word["production_policy"] = np.asarray(["q1-production-freeze-20260926"])
    write_npz(output / "q1_word_aligned_final.npz", **word)

    compact = load_npz(stage3_reports / "q1_compact50_stage3_candidate.npz")
    if compact["text"].shape != (100, 50, 768) or compact["audio"].shape != (100, 50, 25) or compact["vision"].shape != (100, 50, 22):
        raise ValueError("Stage3 compact structure changed")
    compact["paired_use"] = paired
    compact["production_resolution_status"] = status
    compact["production_policy"] = np.asarray(["q1-production-freeze-20260926"])
    write_npz(output / "q1_compact50_final.npz", **compact)
    return word, compact


def write_summary(manifest: list[dict], rows: list[dict], word: dict, compact: dict, output: Path) -> list[dict]:
    result = []
    for item, status in zip(manifest, rows, strict=True):
        index = int(item["sample_index"])
        result.append({
            "sample_index": index, "sample_id": item["sample_id"],
            "duration": float(compact["duration"][index]),
            "word_count": int(word["sample_indptr"][index + 1] - word["sample_indptr"][index]),
            "text_dim": 768, "audio_dim": 25, "vision_dim": 22,
            "alignment_granularity": "word",
            "text_usable_fraction": status["text_usable_fraction"],
            "audio_usable_fraction": status["audio_usable_fraction"],
            "vision_usable_fraction": status["vision_usable_fraction"],
            "trimodal_usable_fraction": status["trimodal_usable_fraction"],
            "resolution_status": status["resolution_status"], "paired_use": status["paired_use"],
            "abnormal_reason": status["reason"],
        })
    write_csv(output / "summary_q1_final.csv", result, list(result[0]))
    return result


def _source_interval_map(native: dict, prefix: str) -> dict[int, tuple[float, float]]:
    return {
        int(source_id): (float(interval[0]), float(interval[1]))
        for source_id, interval in zip(native[f"{prefix}_source_ids"], native[f"{prefix}_intervals"], strict=True)
    }


def write_paper_assets(
    formal: Path, manifest: list[dict], rows: list[dict], word: dict, output: Path
) -> dict:
    assets = output / "paper_assets"; assets.mkdir(parents=True, exist_ok=True)
    candidates = [row for row in rows if row["resolution_status"] == "RESOLVED_VALID"
                  and row["semantic_status"] == "MATCH_STRONG" and row["temporal_status"] == "VERIFIED"
                  and row["visual_status"] == "FULLY_VERIFIED"]
    candidates.sort(key=lambda row: (-float(row["trimodal_usable_fraction"]), row["sample_id"]))
    chosen = candidates[0]; sample_index = int(chosen["sample_index"])
    left, right = int(word["sample_indptr"][sample_index]), int(word["sample_indptr"][sample_index + 1])
    good = [i for i in range(left, right) if np.all(word["masks"][i] == 1)
            and word["audio_indptr"][i + 1] > word["audio_indptr"][i]
            and word["vision_indptr"][i + 1] > word["vision_indptr"][i]]
    runs: list[list[int]] = []
    for index in good:
        if not runs or index != runs[-1][-1] + 1: runs.append([index])
        else: runs[-1].append(index)
    selected = max(runs, key=lambda run: (len(run), -run[0]))[:8]
    native = load_npz(formal / "samples" / f"{sample_index:06d}" / "native.npz")
    audio_intervals = _source_interval_map(native, "audio")
    vision_intervals = _source_interval_map(native, "vision")
    frames = read_jsonl(formal / "samples" / f"{sample_index:06d}" / "video_frames.jsonl")

    def describe_sources(index: int, prefix: str, interval_map: dict[int, tuple[float, float]]) -> tuple[str, str]:
        begin, end = int(word[f"{prefix}_indptr"][index]), int(word[f"{prefix}_indptr"][index + 1])
        source_ids = [int(value) for value in word[f"{prefix}_source_ids"][begin:end]]
        intervals = [interval_map[source_id] for source_id in source_ids]
        return json.dumps(source_ids, separators=(",", ":")), json.dumps(intervals, separators=(",", ":"))

    typical = []
    for index in selected:
        audio_ids, audio_times = describe_sources(index, "audio", audio_intervals)
        vision_ids, vision_times = describe_sources(index, "vision", vision_intervals)
        vmid = (float(word["start"][index]) + float(word["end"][index])) / 2
        frame = min(frames, key=lambda item: abs((float(item["start"]) + float(item["end"])) / 2 - vmid))
        typical.append({
            "sample_id": chosen["sample_id"], "word_id": int(word["word_id"][index]),
            "word": str(word["raw_word"][index]), "start": float(word["start"][index]), "end": float(word["end"][index]),
            "audio_source_ids": audio_ids, "audio_source_intervals": audio_times,
            "vision_source_ids": vision_ids, "vision_source_intervals": vision_times,
            "vision_frame_index": int(frame["source_frame_index"]), "vision_timestamp": float(frame["start"]),
            "text_mask": int(word["text_mask"][index]), "audio_mask": int(word["audio_mask"][index]),
            "vision_mask": int(word["vision_mask"][index]), "audio_coverage": float(word["audio_coverage"][index]),
            "vision_coverage": float(word["vision_coverage"][index]),
        })
    write_csv(assets / "q1_typical_alignment.csv", typical, list(typical[0]))

    abnormal = sorted((row for row in rows if row["resolution_status"] in {"RESOLVED_MISSING", "RESOLVED_CONFLICT"}),
                      key=lambda row: (0 if row["resolution_status"] == "RESOLVED_MISSING" else 1, row["sample_id"]))[0]
    ai = int(abnormal["sample_index"]); al, ar = int(word["sample_indptr"][ai]), int(word["sample_indptr"][ai + 1])
    abnormal_rows = [{
        "sample_id": abnormal["sample_id"], "resolution_status": abnormal["resolution_status"],
        "word_id": int(word["word_id"][i]), "word": str(word["raw_word"][i]),
        "start": float(word["start"][i]), "end": float(word["end"][i]),
        "text_mask": int(word["text_mask"][i]), "audio_mask": int(word["audio_mask"][i]),
        "vision_mask": int(word["vision_mask"][i]), "audio_coverage": float(word["audio_coverage"][i]),
        "vision_coverage": float(word["vision_coverage"][i]), "reason": abnormal["reason"],
    } for i in range(al, min(ar, al + 12))]
    write_csv(assets / "q1_abnormal_mask_example.csv", abnormal_rows, list(abnormal_rows[0]))

    fig, ax = plt.subplots(figsize=(10, 3.8))
    y = {"text": 2, "audio": 1, "vision": 0}; colors = {"text": "#4477AA", "audio": "#EE6677", "vision": "#228833"}
    for item in typical:
        start, width = item["start"], item["end"] - item["start"]
        for modality in y:
            alpha = 0.85 if item[f"{modality}_mask"] else 0.12
            ax.broken_barh([(start, width)], (y[modality] - 0.32, 0.64), facecolors=colors[modality], alpha=alpha)
        ax.text(start + width / 2, 2.38, item["word"], ha="center", va="bottom", fontsize=8, rotation=25)
    ax.set_yticks([0, 1, 2], ["Vision", "Audio", "Text"]); ax.set_xlabel("Time (s)")
    safe_sample_id = chosen["sample_id"].replace("$", r"\$")
    ax.set_title(f"Typical word-aligned multimodal trace: {safe_sample_id}")
    ax.grid(axis="x", alpha=0.2); fig.tight_layout()
    fig.savefig(assets / "q1_typical_alignment_timeline.png", dpi=180); plt.close(fig)
    return {"typical_sample_id": chosen["sample_id"], "typical_word_count": len(typical),
            "abnormal_sample_id": abnormal["sample_id"], "selection_uses_sentiment_label": False}


def write_coverage_terms(stage3_reports: Path, output: Path) -> list[dict]:
    rows = read_csv(stage3_reports / "coverage_comparison_stage3.csv")
    selected = []
    for row in rows:
        if row["group"] != "all100" or row["version"] not in {"formal", "stage2", "stage3"}: continue
        coverage_type = "raw_observed_coverage" if row["version"] == "formal" else "verified_usable_coverage"
        selected.append({"scope": "all100", "source": row["version"], "coverage_type": coverage_type,
            "text": row["text_observed_fraction"], "audio": row["audio_observed_fraction"],
            "vision": row["vision_observed_fraction"], "trimodal": row["trimodal_observed_fraction"],
            "accuracy_metric": 0})
    write_csv(output / "q1_final_coverage_terms.csv", selected, list(selected[0]))
    return selected


def write_report(project: Path, output: Path, freeze: dict, paper: dict) -> None:
    validation = read_json(output / "q1_final_validation.json", {})
    engineering = read_json(output / "q1_final_engineering.json", {})
    package = read_json(output / "submission_package_size.json", {})
    lines = [
        "# Q1 最终生产冻结与实验报告", "",
        "## 1. 结论", "",
        "Q1 已按固定自动策略冻结。最终产物保留100条样本，以词级对齐NPZ为主产物、50-bin NPZ为辅助表示。未使用情感标签、人工Gold或新模型，未执行clip交换。", "",
        "- 最终状态：RESOLVED_VALID=57、RESOLVED_PARTIAL=19、RESOLVED_CONFLICT=5、RESOLVED_MISSING=4、UNRESOLVED=15。",
        "- paired-use：Formal 49 → Final 68。Stage 3 相对 Stage 2 没有继续增加释放样本；主要变化是已验证身份轨迹内的视觉时间覆盖。",
        "- 15条证据不足样本全部保留，危险模态保持mask；不再运行新的恢复算法。", "",
        "## 2. 方法演进", "",
        "1. Formal：完成BERT、CTC、openSMILE、OpenFace和50-bin正式基线，paired-use=49。",
        "2. Stage 1：使用WhisperX、ArcFace、TalkNet与AV-sync生成独立自动证据，不直接写回。",
        "3. Stage 2：拆分semantic pairing与temporal alignment，使用official-text forced alignment和局部TalkNet视觉mask。",
        "4. Stage 3：TalkNet只建立speaker anchor；ArcFace在同身份、无竞争说话者且轨迹间隔受限时传播视觉身份。",
        "5. Stage 4：不再优化，只按固定状态策略冻结Stage 3候选。", "",
        "## 3. Coverage术语与结果", "",
        "`raw_observed_coverage`表示正式基线中工具可提取/保留的观测；`verified_usable_coverage`表示经过语义、时间和身份质量控制后允许跨模态使用的观测。二者都不是accuracy。",
        "- Stage 3 Vision verified usable=76.04%，Trimodal verified usable=52.14%。",
        "- Audio从Formal约99.50%降至Stage 3的78.28%，原因是语义冲突/未决位置被主动mask，不是音频提取器性能下降。",
        "- paired-use从49增至68发生在Stage 2；Stage 3主要把原37条视觉风险的平均可用视觉比例从33.55%提高到51.44%。", "",
        "## 4. 视觉连续性参数", "",
        "固定 `max_identity_gap_seconds=1.0`。敏感性：0.5s→50.25%，1.0s→51.44%，1.5s→51.44%。样本级状态在该范围稳定；1.0s是预先固定主配置，不宣称为最优参数。", "",
        "## 5. Mapping冻结", "",
        "所有100条继续使用官方sample/video/clip mapping。唯一MAPPING_STRONG `-iRBcNs9oI8$_$6 → -iRBcNs9oI8$_$9` 仅作为诊断证据保留，最终动作是RESOLVED_CONFLICT + SAFE_MASK，没有实施mapping correction。", "",
        "## 6. 证据边界", "",
        "没有人工Gold，因此文中的verified只表示满足冻结自动规则，不表示人工确认或真实准确率。RESOLVED_CONFLICT表示安全mask方案明确，不表示冲突被修复；RESOLVED_MISSING不插值、不借用人物或片段。", "",
        "## 7. 论文附件材料", "",
        f"- 典型样本：`{paper['typical_sample_id']}`，按无标签固定规则选择，展示{paper['typical_word_count']}个连续词及source mapping。",
        f"- 异常示例：`{paper['abnormal_sample_id']}`，只展示mask和可追踪性，不宣称修复。", "",
        "## 8. 工程验收", "",
        f"- 最终validation：ok={str(validation.get('ok', False)).lower()}，errors={validation.get('errors', [])}。",
        f"- pytest：{engineering.get('pytest', '待运行')}；pip check：{engineering.get('pip_check', '待运行')}。",
        f"- 冻结输入：{freeze['file_count']}项，ok={str(freeze['ok']).lower()}。",
        f"- 最小提交包：{package.get('zip_path', '待生成')}，{package.get('bytes', '待统计')} bytes。", "",
        "## 9. 最终停止", "",
        "通过Q1_FINAL_ACCEPTANCE后，Q1关闭；不再针对15条UNRESOLVED追加模型、阈值或自动恢复算法。",
    ]
    (output / "Q1_FINAL_EXPERIMENT_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze Stage3 as final Q1 production artifacts")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--formal-run", default="runs/q1_full_20260924")
    parser.add_argument("--stage2-run", default="runs/q1_alignment_repair_stage2")
    parser.add_argument("--stage3-run", default="runs/q1_alignment_repair_stage3")
    parser.add_argument("--output-run", default="runs/q1_final_20260926")
    args = parser.parse_args()
    project = Path(args.project_root).resolve(); formal = (project / args.formal_run).resolve()
    stage2 = (project / args.stage2_run).resolve(); stage3 = (project / args.stage3_run).resolve()
    output = (project / args.output_run).resolve(); output.mkdir(parents=True, exist_ok=True)
    policy = project / "configs/q1_finalization_policy.yaml"
    shutil.copy2(policy, output / "config.snapshot.yaml")
    freeze = freeze_inputs(project, stage2, stage3, policy, output)
    manifest = sorted(read_jsonl(formal / "manifest.jsonl"), key=lambda row: int(row["sample_index"]))
    if len(manifest) != 100: raise ValueError("manifest must contain 100 samples")
    stage3_reports = stage3 / "reports"
    rows = build_final_status(manifest, read_csv(stage3_reports / "final_resolution_status_stage3.csv"),
                              read_csv(stage3_reports / "production_writeback_proposal_stage3.csv"), output)
    unresolved = write_unresolved(rows, output)
    word, compact = freeze_npz(stage3_reports, output, rows)
    summary = write_summary(manifest, rows, word, compact, output)
    paper = write_paper_assets(formal, manifest, rows, word, output)
    coverage = write_coverage_terms(stage3_reports, output)
    write_json(output / "q1_final_build_summary.json", {
        "sample_count": len(rows), "word_count": len(word["word_id"]),
        "resolution_status": dict(Counter(row["resolution_status"] for row in rows)),
        "paired_use": sum(int(row["paired_use"]) for row in rows),
        "unresolved_categories": dict(Counter(row["category"] for row in unresolved)),
        "paper_assets": paper, "coverage_rows": coverage,
        "new_model_run": False, "sentiment_labels_used": False, "clip_swap": False,
    })
    write_report(project, output, freeze, paper)
    print(json.dumps(read_json(output / "q1_final_build_summary.json"), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
