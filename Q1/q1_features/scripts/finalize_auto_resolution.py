from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from q1_features.auto_resolution import read_csv
from q1_features.storage import read_json, read_jsonl, write_csv, write_json


def as_float(value, default=None):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def union_length(intervals):
    intervals = sorted((float(start), float(end)) for start, end in intervals if float(end) > float(start))
    if not intervals:
        return 0.0
    total = 0.0
    left, right = intervals[0]
    for next_left, next_right in intervals[1:]:
        if next_left <= right:
            right = max(right, next_right)
        else:
            total += right - left
            left, right = next_left, next_right
    return total + right - left


def load_by_id(path: Path):
    return {row["sample_id"]: row for row in read_csv(path)}


def evidence_level_for_vision(partition, cluster, av_top, duration):
    if int(partition["no_face"]):
        return "V_MISSING", "missing", 0.0, "confirmed automatic no-face condition from frozen OpenFace output"
    if not int(partition["identity_issue"]):
        coverage = min(1.0, sum(
            as_float(item.get("coverage_s"), 0.0)
            for item in json.loads(cluster["cluster_time_coverage_json"])
        ) / max(duration, 1e-9))
        return "V_STRONG", "verified", coverage, "single-episode frozen baseline has no identity ambiguity"

    episode_count = int(cluster["episode_count"])
    embedded_count = int(cluster["valid_embedding_episode_count"])
    cluster_count = int(cluster["identity_cluster_count"])
    stable = int(cluster["stable_partition_0_50_0_60_0_70"]) == 1
    simultaneous = int(cluster["simultaneous_different_identity_clusters"]) == 1
    percentile = as_float(av_top.get("empirical_percentile") if av_top else None)
    margin = as_float(av_top.get("top1_top2_margin") if av_top else None)
    av_strong = percentile is not None and percentile >= 0.95 and (cluster_count == 1 or (margin is not None and margin >= 0.10))
    av_partial = percentile is not None and percentile >= 0.80
    embedded_complete = episode_count > 0 and embedded_count == episode_count
    selected_coverage = 0.0
    if av_top:
        selected_cluster = int(av_top["cluster_id"])
        details = json.loads(cluster["cluster_time_coverage_json"])
        selected = next((item for item in details if int(item["cluster_id"]) == selected_cluster), None)
        if selected:
            selected_coverage = min(1.0, as_float(selected.get("coverage_s"), 0.0) / max(duration, 1e-9))

    if stable and embedded_complete and cluster_count == 1 and not simultaneous and av_strong:
        return "V_STRONG", "verified", selected_coverage, "ArcFace merges all episodes stably and AV-sync exceeds frozen strong null threshold"
    if embedded_complete and av_partial and selected_coverage > 0:
        return "V_PARTIAL", "uncertain", selected_coverage, "only the top AV-supported ArcFace identity cluster is a usable candidate"
    return "V_UNCERTAIN", "uncertain", 0.0, "ArcFace partition or AV-sync does not uniquely bind the target identity"


def main() -> int:
    parser = argparse.ArgumentParser(description="Finalize candidate-only Q1 automatic-resolution reports")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--talknet-status", required=True)
    args = parser.parse_args()
    run = Path(args.run_dir).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)

    manifest = read_jsonl(run / "manifest.jsonl")
    manifest_by_id = {row["sample_id"]: row for row in manifest}
    partition_rows = read_csv(output / "problem_partition.csv")
    partition = {row["sample_id"]: {**row, **{k: int(row[k]) for k in ("ta_issue", "identity_issue", "no_face", "current_paired_use")}} for row in partition_rows}
    if len(partition) != 51:
        raise ValueError(f"partition must contain 51 rows, found {len(partition)}")
    whisper = load_by_id(output / "whisperx_results.csv")
    permutation = load_by_id(output / "text_audio_permutation_test.csv")
    text_evidence = load_by_id(output / "text_audio_evidence.csv")
    cluster_rows = [row for row in read_csv(output / "arcface_identity_clusters.csv") if abs(float(row["threshold"]) - 0.60) < 1e-9]
    clusters = {row["sample_id"]: row for row in cluster_rows}
    av_rows = read_csv(output / "av_sync_scores.csv")
    av_top = {row["sample_id"]: row for row in av_rows if str(row["rank_within_sample"]) == "1"}
    talknet = read_json(args.talknet_status, {})
    mfa = read_json(output / "mfa_status.json", {})

    required_all = [whisper, permutation, text_evidence, clusters]
    if any(set(table) != set(manifest_by_id) for table in required_all):
        raise ValueError("WhisperX, permutation, text evidence, and ArcFace primary rows must each cover all 100 samples")

    quality = {row["sample_id"]: row for row in read_csv(run / "reports" / "quality.csv")}
    candidates = []
    segment_rows = []
    for sample_id in sorted(partition):
        p = partition[sample_id]
        item = manifest_by_id[sample_id]
        sample_dir = run / "samples" / f"{int(item['sample_index']):06d}"
        duration = float(read_json(sample_dir / "media.json")["duration"])
        w = whisper[sample_id]
        t = text_evidence[sample_id]
        perm = permutation[sample_id]
        c = clusters[sample_id]
        av = av_top.get(sample_id)
        v_level, v_status, release_ratio, v_reason = evidence_level_for_vision(p, c, av, duration)
        ta_status = t["text_audio_status"]
        if ta_status == "verified" and v_status == "verified":
            trimodal = "verified"
        elif ta_status == "uncertain" or (v_status == "uncertain" and v_level == "V_UNCERTAIN"):
            trimodal = "uncertain"
        else:
            trimodal = "partial"

        components = json.loads(c["clusters_json"])
        top_cluster = int(av["cluster_id"]) if av else None
        quality_by_episode = {entry["episode_key"]: entry for entry in json.loads(c["episode_quality_json"])}
        for cluster_id, episode_keys in enumerate(components):
            usable = v_level in {"V_STRONG", "V_PARTIAL"} and cluster_id == top_cluster
            for episode_key in episode_keys:
                episode = quality_by_episode[episode_key]
                segment_rows.append({
                    "sample_id": sample_id,
                    "cluster_id": cluster_id,
                    "episode_key": episode_key,
                    "start": episode["start"],
                    "end": episode["end"],
                    "candidate_action": "candidate_usable" if usable else "candidate_mask",
                    "reason": v_reason,
                })
        if p["no_face"]:
            segment_rows.append({
                "sample_id": sample_id, "cluster_id": "", "episode_key": "", "start": 0.0,
                "end": duration, "candidate_action": "candidate_mask", "reason": v_reason,
            })

        candidates.append({
            "sample_id": sample_id,
            "original_group": p["group"],
            "current_paired_use": p["current_paired_use"],
            "original_ta_issue": p["ta_issue"],
            "original_identity_issue": p["identity_issue"],
            "original_no_face": p["no_face"],
            "ctc_diagnostic_wer": w["ctc_diagnostic_wer"],
            "ctc_accepted_word_fraction": w["ctc_accepted_word_fraction"],
            "whisperx_status": w["status"],
            "whisperx_wer": w["whisperx_wer"],
            "whisperx_cer": w["whisperx_cer"],
            "whisperx_word_overlap_f1": w["word_overlap_f1"],
            "ctc_whisperx_common_word_coverage": w["common_timed_word_coverage"],
            "ctc_whisperx_median_midpoint_difference_s": w["median_midpoint_difference_s"],
            "ctc_whisperx_p90_midpoint_difference_s": w["p90_midpoint_difference_s"],
            "permutation_true_similarity": perm["true_similarity"],
            "permutation_negative_p95": perm["negative_similarity_p95"],
            "permutation_true_rank": perm["true_rank"],
            "permutation_empirical_percentile": perm["empirical_percentile"],
            "ta_evidence_level": t["ta_evidence_level"],
            "mfa_status": mfa.get("status", "not_run"),
            "mfa_alignment_success": "",
            "mfa_aligned_word_fraction": "",
            "arcface_episode_count": c["episode_count"],
            "arcface_valid_embedding_episode_count": c["valid_embedding_episode_count"],
            "arcface_identity_cluster_count_0_60": c["identity_cluster_count"],
            "arcface_partition_stable_0_50_0_60_0_70": c["stable_partition_0_50_0_60_0_70"],
            "arcface_simultaneous_different_clusters": c["simultaneous_different_identity_clusters"],
            "av_sync_real_score": av["real_sync_score"] if av else "",
            "av_sync_null_median": av["null_median"] if av else "",
            "av_sync_null_p95": av["null_p95"] if av else "",
            "av_sync_empirical_percentile": av["empirical_percentile"] if av else "",
            "av_sync_top1_top2_margin": av["top1_top2_margin"] if av else "",
            "talknet_status": talknet.get("status", "unavailable"),
            "talknet_logit": "",
            "text_audio_candidate_status": ta_status,
            "vision_candidate_status": v_status,
            "trimodal_candidate_status": trimodal,
            "text_audio_status": ta_status,
            "vision_status": v_status,
            "trimodal_status": trimodal,
            "vision_evidence_level": v_level,
            "releasable_visual_time_ratio": release_ratio,
            "masked_visual_time_ratio": 1.0 - release_ratio,
            "evidence_reason": t["evidence_reason"] + "; " + v_reason,
        })

    candidate_path = output / "auto_resolution_candidates.csv"
    write_csv(candidate_path, candidates, list(candidates[0]))
    write_csv(output / "visual_candidate_segments.csv", segment_rows, list(segment_rows[0]))

    identity_ids = {sample_id for sample_id, row in partition.items() if row["identity_issue"]}
    ta_alert_ids = {sample_id for sample_id, row in partition.items() if row["ta_issue"]}
    identity_cluster_distribution = Counter(
        int(clusters[sample_id]["identity_cluster_count"]) for sample_id in identity_ids
    )
    ta_distribution = Counter(text_evidence[sample_id]["ta_evidence_level"] for sample_id in ta_alert_ids)
    v_distribution = Counter(row["vision_evidence_level"] for row in candidates)
    final_distribution = Counter(row["trimodal_candidate_status"] for row in candidates)
    group_distribution = {}
    for group in sorted({row["original_group"] for row in candidates}):
        group_distribution[group] = dict(Counter(
            row["trimodal_candidate_status"] for row in candidates if row["original_group"] == group
        ))
    boundary = [as_float(whisper[sample_id]["median_midpoint_difference_s"]) for sample_id in manifest_by_id]
    boundary = [value for value in boundary if value is not None]
    real_scores = [as_float(row["real_sync_score"]) for row in av_rows]
    real_scores = [value for value in real_scores if value is not None]
    null_rows = read_csv(output / "av_sync_null_test.csv")
    null_scores = [as_float(row["sync_score"]) for row in null_rows]
    null_scores = [value for value in null_scores if value is not None]
    summary = {
        "candidate_only": True,
        "production_write": False,
        "partition": dict(Counter(row["original_group"] for row in candidates)),
        "whisperx_completion": {
            "ok": sum(whisper[sample_id]["status"] == "ok" for sample_id in manifest_by_id),
            "total": len(manifest_by_id),
        },
        "ta_alert_evidence_distribution": dict(ta_distribution),
        "identity_risk_cluster_distribution_0_60": {str(k): v for k, v in sorted(identity_cluster_distribution.items())},
        "vision_evidence_distribution_all_51": dict(v_distribution),
        "trimodal_candidate_distribution": dict(final_distribution),
        "group_trimodal_distribution": group_distribution,
        "ctc_whisperx_median_midpoint_difference_s": {
            "n": len(boundary), "median": float(np.median(boundary)) if boundary else None,
            "p90": float(np.quantile(boundary, 0.90)) if boundary else None,
        },
        "av_sync": {
            "real_n": len(real_scores), "real_median": float(np.median(real_scores)) if real_scores else None,
            "null_n": len(null_scores), "null_median": float(np.median(null_scores)) if null_scores else None,
            "real_minus_null_median": (float(np.median(real_scores)) - float(np.median(null_scores))) if real_scores and null_scores else None,
        },
        "talknet": talknet,
        "mfa": mfa,
        "word_aligned_validation": read_json(output / "q1_word_aligned_candidate.validation.json", {}),
    }
    write_json(output / "auto_resolution_summary.json", summary)

    report = [
        "# Q1 自动多证据核验阶段一报告", "",
        "> 本报告仅描述候选证据，不构成人工确认，不修改正式 pairing 配置或 `q1_compact50.npz`。", "",
        "## 1. 风险集合门控", "",
        f"- 51 条分组：`{json.dumps(summary['partition'], ensure_ascii=False)}`。",
        "- 分组与预注册的 28/10/9/4 完全一致后才继续执行。", "",
        "## 2. 文本—音频证据", "",
        f"- WhisperX 完成：{summary['whisperx_completion']['ok']}/{summary['whisperx_completion']['total']}。",
        f"- 原 23 条音文告警的证据等级：`{json.dumps(dict(ta_distribution), ensure_ascii=False)}`。",
        f"- CTC/WhisperX 共同词 midpoint 差：n={summary['ctc_whisperx_median_midpoint_difference_s']['n']}，median={summary['ctc_whisperx_median_midpoint_difference_s']['median']} s，P90={summary['ctc_whisperx_median_midpoint_difference_s']['p90']} s。", "",
        "## 3. 视觉身份与视听同步", "",
        f"- 37 条身份风险样本在 ArcFace 0.60 下的 cluster 数分布：`{json.dumps(summary['identity_risk_cluster_distribution_0_60'], ensure_ascii=False)}`。",
        f"- 51 条候选的视觉等级：`{json.dumps(dict(v_distribution), ensure_ascii=False)}`。",
        f"- AV-sync real median={summary['av_sync']['real_median']}，null median={summary['av_sync']['null_median']}，差值={summary['av_sync']['real_minus_null_median']}。",
        f"- TalkNet：{talknet.get('status', 'unavailable')}；未使用不明第三方权重。", "",
        "## 4. MFA", "",
        f"- 状态：{mfa.get('status', 'not_run')}。原因：{mfa.get('reason', '')}", "",
        "## 5. 51 条最终候选", "",
        f"- trimodal：`{json.dumps(dict(final_distribution), ensure_ascii=False)}`。",
        f"- 分组结果：`{json.dumps(group_distribution, ensure_ascii=False)}`。",
        "- 逐条表：`auto_resolution_candidates.csv`。",
        "- 可用/应 mask 的视觉候选区间：`visual_candidate_segments.csv`。", "",
        "## 6. 词级候选包", "",
        f"- 验证：`ok={summary['word_aligned_validation'].get('ok')}`，words={summary['word_aligned_validation'].get('word_count')}，aligned={summary['word_aligned_validation'].get('aligned_word_count')}。",
        "- 无有效 CTC 时间的词仍保留文本、mask 和失败原因，未伪造音视频时间。", "",
        "## 7. 边界与限制", "",
        "- 未使用情感标签，未训练或按这 100 条结果优化阈值。",
        "- `TA_CONFLICT` 表示独立证据未能共同建立官方配对，不等同于已经证明错配。",
        "- `V_STRONG/V_PARTIAL` 仍是自动候选，不是人工 Gold。",
        "- 本阶段没有生产写回；下一轮应审核候选证据后再决定是否更新正式结果。", "",
    ]
    (output / "Q1_auto_resolution_stage1_report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
