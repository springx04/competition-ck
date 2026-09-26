"""Build Q2 technical and paper tables from completed artifacts only."""
import csv
import json
from pathlib import Path
import statistics

import numpy as np


def is_main_scenario(scenario):
    parts = scenario.split("_")
    return (len(parts) == 3 and parts[0] in ("T", "A", "V", "TA", "TV", "AV")
            and parts[1] in ("front", "middle", "rear")
            and parts[2] in ("0.2", "0.4", "0.6", "0.8"))


def rows(path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def metric_line(row):
    pcc = f"{float(row['pearson']):.4f}" if row["pearson"] else "未定义：" + row["pearson_reason"]
    return (f"| {row['scenario']} | {row['n']} | {float(row['accuracy']):.4f} | "
            f"{float(row['macro_f1']):.4f} | {float(row['weighted_f1']):.4f} | "
            f"{float(row['mae']):.4f} | {pcc} |")


def main_summaries(metrics, title, deletion=None):
    main = [r for r in metrics if r["scope"] == "all_samples" and
            is_main_scenario(r["scenario"]) and not r["duplicate_of"]]
    rates = {r["scenario"]: r for r in (deletion or [])}
    result = [f"## {title}", "", "名义跨度是官方aligned位置比例，非秒数；以下按场景等权汇总。", "",
              "| 变量 | 条件 | 场景数 | Acc | Macro-F1 | MAE | ΔF1 | ΔMAE | 平均有效删除样本数 | 平均实际删除率 |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for factor, ix, labels in (("模态", 0, ("T", "A", "V", "TA", "TV", "AV")),
                               ("位置", 1, ("front", "middle", "rear")),
                               ("跨度", 2, ("0.2", "0.4", "0.6", "0.8"))):
        for label in labels:
            subset = [r for r in main if r["scenario"].split("_")[ix] == label]
            if not subset:
                continue
            actual = [float(rates[r["scenario"]]["actual_rate_mean"]) for r in subset
                      if r["scenario"] in rates and rates[r["scenario"]]["actual_rate_mean"]]
            rate_text = f"{statistics.mean(actual):.4f}" if actual else "未计算"
            result.append(f"| {factor} | {label} | {len(subset)} | "
                          f"{statistics.mean(float(r['accuracy']) for r in subset):.4f} | "
                          f"{statistics.mean(float(r['macro_f1']) for r in subset):.4f} | "
                          f"{statistics.mean(float(r['mae']) for r in subset):.4f} | "
                          f"{statistics.mean(float(r['delta_macro_f1']) for r in subset):+.4f} | "
                          f"{statistics.mean(float(r['delta_mae']) for r in subset):+.4f} | "
                          f"{statistics.mean(int(r['n'])-int(r['no_new_damage']) for r in subset):.1f} | "
                          f"{rate_text} |")
    return result + [""]


def stress_table(metrics):
    stress = [r for r in metrics if r["scope"] == "all_samples" and r["scenario"] != "clean"
              and not is_main_scenario(r["scenario"])]
    return ["| 场景 | N | Acc | Macro-F1 | Weighted-F1 | MAE | PCC |",
            "|---|---:|---:|---:|---:|---:|---:|"] + [metric_line(r) for r in stress] + [""]


def main_table(metrics):
    main = [r for r in metrics if r["scope"] == "all_samples" and
            is_main_scenario(r["scenario"])]
    return ["| 主场景 | N | Acc | Macro-F1 | Weighted-F1 | MAE | PCC |",
            "|---|---:|---:|---:|---:|---:|---:|"] + [metric_line(r) for r in main] + [""]


def visual_zero_table(metrics):
    names = ("clean", "V_middle_0.2", "V_middle_0.4", "V_middle_0.8",
             "AV_middle_0.4", "TV_middle_0.4", "TAV_middle_0.4")
    chosen = [r for r in metrics if r["scope"] == "original_visual_empty" and
              r["scenario"] in names]
    return (["### 原视觉全零子集", "",
             "这些样本没有原视觉恢复目标；视觉补偿不解释为视觉原始贡献。", ""] +
            ["| 场景 | N | Acc | Macro-F1 | Weighted-F1 | MAE | PCC |",
             "|---|---:|---:|---:|---:|---:|---:|"] +
            [metric_line(r) for r in chosen] + [""]) if chosen else []


def paired_summary(directory):
    paired = rows(directory / "paired_conditions.csv")
    if not paired:
        return []
    result = ["### 共同受损样本上的配对条件", "",
              "每组条件取共同产生新增删除的样本，再对组等权汇总。", "",
              "| 变量 | 条件 | 组数 | 平均共同样本数 | Macro-F1 | MAE | 实际删除率 |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for factor in ("modality", "position", "span"):
        conditions = sorted({r["condition"] for r in paired if r["factor"] == factor})
        for condition in conditions:
            chosen = [r for r in paired if r["factor"] == factor and
                      r["condition"] == condition and r["macro_f1"]]
            if chosen:
                result.append(f"| {factor} | {condition} | {len(chosen)} | "
                              f"{statistics.mean(int(r['n']) for r in chosen):.1f} | "
                              f"{statistics.mean(float(r['macro_f1']) for r in chosen):.4f} | "
                              f"{statistics.mean(float(r['mae']) for r in chosen):.4f} | "
                              f"{statistics.mean(float(r['mean_actual_rate']) for r in chosen):.4f} |")
    return result + [""]


def deletion_strata(directory):
    matched = rows(directory / "matched_deletion_rates.csv")
    dual = rows(directory / "dual_both_damaged.csv")
    result = []
    if matched:
        result += ["### 同实际删除率分层", "",
                   "| 模态比较 | 实际率箱 | 模态 | 有效组数 | 平均共同N | Macro-F1 | MAE |",
                   "|---|---|---|---:|---:|---:|---:|"]
        for comparison in ("T_vs_A", "T_vs_V", "A_vs_V"):
            for rate_bin in ("(0,0.2]", "(0.2,0.4]", "(0.4,0.6]", "(0.6,0.8]", "(0.8,1.0]"):
                for pattern in comparison.split("_vs_"):
                    chosen = [r for r in matched if r["comparison"] == comparison and
                              r["rate_bin"] == rate_bin and r["pattern"] == pattern and r["macro_f1"]]
                    if chosen:
                        result.append(f"| {comparison} | {rate_bin} | {pattern} | {len(chosen)} | "
                                      f"{statistics.mean(int(r['n']) for r in chosen):.1f} | "
                                      f"{statistics.mean(float(r['macro_f1']) for r in chosen):.4f} | "
                                      f"{statistics.mean(float(r['mae']) for r in chosen):.4f} |")
        result.append("")
    if dual:
        result += ["### 双模态均实际受损样本", "",
                   "| 模态组合 | 有效场景数 | 平均N | Macro-F1 | MAE |",
                   "|---|---:|---:|---:|---:|"]
        for pattern in ("TA", "TV", "AV"):
            chosen = [r for r in dual if r["scenario"].startswith(pattern + "_") and r["macro_f1"]]
            if chosen:
                result.append(f"| {pattern} | {len(chosen)} | "
                              f"{statistics.mean(int(r['both_modalities_damaged_n']) for r in chosen):.1f} | "
                              f"{statistics.mean(float(r['macro_f1']) for r in chosen):.4f} | "
                              f"{statistics.mean(float(r['mae']) for r in chosen):.4f} |")
        result.append("")
    return result


def special_table(predictions):
    return (["| 样本 | 类别 | 连续值 | P(Neg) | P(Neu) | P(Pos) |",
             "|---|---|---:|---:|---:|---:|"] +
            [f"| {r['sample_id']} | {r['pred_label']} | {float(r['pred_score']):.5f} | "
             f"{float(r['prob_negative']):.4f} | {float(r['prob_neutral']):.4f} | "
             f"{float(r['prob_positive']):.4f} |" for r in predictions] + [""])


def _failures(predictions):
    clean = {r["sample_id"]: r for r in predictions if r["scenario"] == "clean"}
    candidates = [r for r in predictions if r["scenario"] != "clean"
                  and is_main_scenario(r["scenario"])
                  and clean[r["sample_id"]]["pred_class"] == r["class_id"]
                  and r["pred_class"] != r["class_id"]]
    candidates.sort(key=lambda r: abs(float(r["pred_score"])-float(r["true_score"])), reverse=True)
    seen, chosen = set(), []
    for row in candidates:
        if row["sample_id"] not in seen:
            seen.add(row["sample_id"])
            chosen.append(row)
        if len(chosen) == 5:
            break
    return chosen


def _figures(metrics, directory):
    if not metrics:
        return []
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    main = [r for r in metrics if r["scope"] == "all_samples" and
            is_main_scenario(r["scenario"]) and not r["duplicate_of"]]
    for factor, ix, labels in (("modality", 0, ("T", "A", "V", "TA", "TV", "AV")),
                               ("position", 1, ("front", "middle", "rear")),
                               ("span", 2, ("0.2", "0.4", "0.6", "0.8"))):
        values = [statistics.mean(float(r["macro_f1"]) for r in main
                                  if r["scenario"].split("_")[ix] == label) for label in labels]
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.plot(labels, values, marker="o")
        ax.set(xlabel=factor + " (official aligned positions)", ylabel="Macro-F1")
        ax.grid(alpha=.3)
        fig.tight_layout()
        path = directory / (factor + "_macro_f1.png")
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(path)
    return paths


def test_report_directory(root):
    root = Path(root)
    pointer = root / "reports/test/latest.json"
    if pointer.exists():
        return root / json.loads(pointer.read_text(encoding="utf-8"))["directory"]
    return root / "reports/test"


def build_reports(root):
    root = Path(root)
    report_dir = root / "reports"
    from .reporting import suite_progress, package_sizes
    progress = suite_progress(root)
    selection = json.loads((report_dir / "selection.json").read_text(encoding="utf-8")) if (report_dir / "selection.json").exists() else None
    special = rows(root / "outputs/q2_predictions_aligned.csv")
    special_state = rows(root / "outputs/q2_special_state.csv")
    package = root / "delivery/q2_inference"
    package_bytes = package_sizes(package, report_dir / "package_sizes.csv") if package.exists() else None
    zip_path = root / "delivery/q2_inference.zip"
    selected_dir = (report_dir / "valid" / selection["variant"] / "seed_1111") if selection else None
    valid = rows(selected_dir / "metrics_per_scenario.csv") if selected_dir else []
    valid_deletion = rows(selected_dir / "deletion_rates.csv") if selected_dir else []
    valid_preds = rows(selected_dir / "predictions.csv") if selected_dir else []
    test_dir = test_report_directory(root)
    test = rows(test_dir / "metrics_per_scenario.csv")
    test_deletion = rows(test_dir / "deletion_rates.csv")
    inventory = json.loads((report_dir / "data_inventory.json").read_text(encoding="utf-8")) if (report_dir / "data_inventory.json").exists() else None
    normalizer_path = root / "data/processed/normalizer.npz"
    with np.load(normalizer_path) as saved:
        normalization = {"audio_count": int(saved["audio_count"]),
                         "vision_count": int(saved["vision_count"]),
                         "class_prior": saved["class_prior"].tolist(),
                         "score_prior": float(saved["score_prior"])}
    verify = json.loads((report_dir / "verification.json").read_text(encoding="utf-8")) if (report_dir / "verification.json").exists() else None
    exported = json.loads((report_dir / "export_verification.json").read_text(encoding="utf-8")) if (report_dir / "export_verification.json").exists() else None
    figures = _figures(valid, report_dir / "figures")

    technical = ["# E题 Q2 技术报告", "", "## 实际完成状态", "",
                 f"- 60轮run：{sum(r['status']=='complete' for r in progress)}/30；"
                 f"最终变体：{selection['variant'] if selection else '未选定'}；固定部署seed=1111。",
                 f"- 附件3预测：{len(special)}/30；离线推理验证：{exported.get('passed') if exported else '未执行'}。",
                 f"- Q2推理目录：{package_bytes if package_bytes is not None else '未知'}字节；"
                 f"ZIP：{zip_path.stat().st_size if zip_path.exists() else '未知'}字节。", "",
                 "## 实际环境与来源", "",
                 "```text", (report_dir / "environment.txt").read_text(encoding="utf-8").strip()
                 if (report_dir / "environment.txt").exists() else "未记录", "```", "",
                 "EBMC MSD原类与许可证逐字节对应；上游提交见 `THIRD_PARTY.md`。"
                 "文本模型为 google/bert_uncased_L-4_H-256_A-4，11,104,768参数，训练前FP16存储、FP32计算。", "",
                 "## 数据与词表", ""]
    if inventory:
        technical += ["| split | 样本 | 全零视觉 | 类别0/1/2 | 标签符号不一致 |",
                      "|---|---:|---:|---|---:|"]
        technical += [f"| {split} | {r['samples']} | {r['visual_all_zero']} | {r['class_counts']} | "
                      f"{r['label_sign_mismatch']} |" for split, r in inventory["splits"].items()]
        technical += ["", f"video_id跨划分重叠：{inventory['video_overlap']}。", ""]
    technical += [(report_dir / "tokenizer_compatibility.md").read_text(encoding="utf-8")
                  if (report_dir / "tokenizer_compatibility.md").exists() else "词表尚未核验", "",
                  f"音频标准化分母为train有效行{normalization['audio_count']}，视觉为"
                  f"{normalization['vision_count']}；类别先验{normalization['class_prior']}，"
                  f"强度空观测先验{normalization['score_prior']:.4f}。原全零行仍为零占位。", "",
                  "## 主方法与验证", "", "```text",
                  "当前输入 → 编码前遮蔽 → 三路编码 → MSD → 原可用来源单轮补偿",
                  "→ 误差估计与内容融合 → 独立三分类/连续强度输出", "```", "",
                  "损失：clean与corrupt双任务CE+Huber，0.05 MSD，ramp后0.2 span、0.1误差校准、0.2预测一致性。"
                  "前5轮仅clean与MSD；EMA只在训练侧。", "",
                  f"语义与实际CUDA验证：{verify if verify else '未执行'}。", ""]
    if selection:
        technical += ["## 30个run及消融", "",
                      "| 变体 | seed | best epoch | clean F1 | clean MAE | missing F1 | missing MAE | 训练秒 | 峰值显存MB | 训练参数量 | 推理参数量 |",
                      "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        technical += [f"| {r['variant']} | {r['seed']} | {r['best_epoch']} | {r['clean_macro_f1']:.4f} | "
                      f"{r['clean_mae']:.4f} | {r['missing_macro_f1']:.4f} | {r['missing_mae']:.4f} | "
                      f"{r['training_seconds']:.1f} | {r['peak_memory_bytes']/1e6:.1f} | "
                      f"{r['parameter_count']} | {r['deployed_parameters']} |"
                      for r in selection["runs"]]
        technical += ["", "full/seed1111曾中断续训，其训练秒数采用history创建至第60轮结束的墙钟跨度，包含中断间隔；其余run采用训练进程计时。"]
        technical += ["", "三种子均值/样本标准差：", "",
                      "| 变体 | missing F1均值±SD | missing MAE均值±SD | clean F1均值±SD | clean MAE均值±SD |",
                      "|---|---:|---:|---:|---:|"]
        technical += [f"| {r['variant']} | {r['missing_macro_f1']:.4f}±{r['missing_macro_f1_std']:.4f} | "
                      f"{r['missing_mae']:.4f}±{r['missing_mae_std']:.4f} | "
                      f"{r['clean_macro_f1']:.4f}±{r['clean_macro_f1_std']:.4f} | "
                      f"{r['clean_mae']:.4f}±{r['clean_mae_std']:.4f} |" for r in selection["candidates"]]
        technical += ["", f"支配关系：{selection['dominated_by']}；未支配排序：{selection['ordered_survivors']}。"
                      "选模应仅使用valid；test观察历史以实验记录为准，本报告不自动认定测试独立性。", ""]
    if valid:
        technical += main_summaries(valid, "valid模态、位置与跨度规律", valid_deletion)
        technical += visual_zero_table(valid)
        technical += paired_summary(selected_dir)
        technical += deletion_strata(selected_dir)
        technical += ["实际删除率与同删除率分层，见所选run的 `paired_conditions.csv`、"
                      "`deletion_rates.csv`、`matched_deletion_rates.csv`、`dual_both_damaged.csv`。", "",
                      "### valid压力场景", ""] + stress_table(valid)
        technical += ["### valid完整主网格", ""] + main_table(valid)
    if figures:
        technical += ["valid曲线：" + "、".join(str(p.relative_to(root)) for p in figures) + "；数值来源为场景CSV。", ""]
    if selected_dir and (selected_dir / "reliability.json").exists():
        technical += ["## 误差估计诊断", "", (selected_dir / "reliability.json").read_text(encoding="utf-8"), ""]
    if (report_dir / "bootstrap_comparisons.json").exists():
        technical += ["## 来源video_id配对bootstrap", "", (report_dir / "bootstrap_comparisons.json").read_text(encoding="utf-8"), ""]
    failures = _failures(valid_preds)
    if failures:
        technical += ["## valid失败案例：clean正确、受损错误", "",
                      "| 样本 | 场景 | 真类 | 受损预测 | 真强度 | 受损预测强度 |",
                      "|---|---|---:|---:|---:|---:|"]
        technical += [f"| {r['sample_id']} | {r['scenario']} | {r['class_id']} | {r['pred_class']} | "
                      f"{float(r['true_score']):.3f} | {float(r['pred_score']):.3f} |" for r in failures]
        technical += [""]
    if test:
        technical += ["## test评估结果", "", f"数据目录：`{test_dir}`；独立性说明见实验记录。", "clean及8个压力场景：", "",
                      "| 场景 | N | Acc | Macro-F1 | Weighted-F1 | MAE | PCC |",
                      "|---|---:|---:|---:|---:|---:|---:|"]
        technical += [metric_line(r) for r in test if r["scope"] == "all_samples" and
                      (r["scenario"] == "clean" or not is_main_scenario(r["scenario"]))]
        technical += [""] + main_summaries(test, "test主网格规律", test_deletion)
        technical += visual_zero_table(test)
        technical += paired_summary(test_dir)
        technical += deletion_strata(test_dir)
    if special:
        technical += ["## 附件3全部30条预测", ""] + special_table(special)
        zero_vision = sum(int(r["observed_visual_positions"]) == 0 for r in special_state)
        technical += [f"专项中当前视觉全零样本{zero_vision}条；状态逐样本见 `outputs/q2_special_state.csv`。",
                      "专项无真实标签，不计算准确率；全零当前值不等于已知人工缺失真相。", ""]
    technical += ["## 导出与限制", "", f"离线验证：{exported if exported else '未执行'}。",
                  f"Q2目录实际{package_bytes if package_bytes is not None else '未知'}字节；"
                  f"距离50,000,000字节尚余{50_000_000-package_bytes if package_bytes is not None else '未知'}字节。",
                  "逐文件体积见 `package_sizes.csv`。Q1/Q3的实际合包体积尚未确认时，"
                  "不宣称全题50,000,000字节已满足。", "",
                  "Q3复用同一学生及官方aligned索引、当前U/J、原观测/补偿来源与局部融合量。"
                  "没有精确秒级映射，不能将50位置伪称视频秒数。", ""]
    (report_dir / "report.md").write_text("\n".join(technical), encoding="utf-8")

    paper = ["# E题 Q2 论文结果材料", "", "## 方法实例", "",
             "冻结小型BERT、三路128维时序编码、EBMC MSD、共享单轮补偿、监督误差系数融合；"
             "三分类与连续强度联合监督。", "", "## 开发集三种子对照", ""]
    if selection:
        paper += ["| 变体 | 推理参数量 | clean F1均值±SD | clean MAE均值±SD | missing F1均值±SD | missing MAE均值±SD |",
                  "|---|---:|---:|---:|---:|---:|"]
        paper += [f"| {r['variant']} | {r['deployed_parameters']} | "
                  f"{r['clean_macro_f1']:.4f}±{r['clean_macro_f1_std']:.4f} | "
                  f"{r['clean_mae']:.4f}±{r['clean_mae_std']:.4f} | "
                  f"{r['missing_macro_f1']:.4f}±{r['missing_macro_f1_std']:.4f} | "
                  f"{r['missing_mae']:.4f}±{r['missing_mae_std']:.4f} |" for r in selection["candidates"]]
        paper += ["", f"valid按预定规则选择{selection['variant']}；部署seed1111 best学生。", ""]
    if valid:
        paper += main_summaries(valid, "valid缺失规律", valid_deletion)
        paper += visual_zero_table(valid)
        paper += paired_summary(selected_dir)
        paper += deletion_strata(selected_dir)
        paper += ["### valid全部主网格", ""] + main_table(valid)
    if selected_dir and (selected_dir / "reliability.json").exists():
        paper += ["## 误差估计诊断", "", (selected_dir / "reliability.json").read_text(encoding="utf-8"), ""]
    if failures:
        paper += ["## clean正确而缺失后错误的样本", "",
                  "| 样本 | 场景 | 真类 | 受损预测 | 真强度 | 预测强度 |",
                  "|---|---|---:|---:|---:|---:|"]
        paper += [f"| {r['sample_id']} | {r['scenario']} | {r['class_id']} | {r['pred_class']} | "
                  f"{float(r['true_score']):.3f} | {float(r['pred_score']):.3f} |" for r in failures]
        paper += [""]
    if test:
        paper += main_summaries(test, "test留出规律", test_deletion) + ["## test压力", ""] + stress_table(test)
        paper += ["### test完整主网格", ""] + main_table(test)
    if special:
        paper += ["## 无真值专项应用", ""] + special_table(special)
    paper += ["## 结论边界", "", "valid用于开发选模，test只在选模后报告；专项没有真值。"
              "跨度均在官方位置轴定义，不换算秒数；误差估计的可信度需同时看排序诊断与任务消融。", "",
              "所有表格来源为本轮 `reports/valid`、`reports/test` 下的CSV；Macro-F1为三类F1等权均值，"
              "MAE为预测强度的平均绝对误差，PCC为Pearson相关；定义或未定义原因见原始场景CSV。", ""]
    (report_dir / "paper_q2_results.md").write_text("\n".join(paper), encoding="utf-8")
    return {"completed_runs": sum(r["status"] == "complete" for r in progress),
            "special_rows": len(special), "package_bytes": package_bytes}
