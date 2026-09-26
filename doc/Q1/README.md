# Q1 文档索引

本目录整理 Q1 的方法方案、实施说明、实验记录和交付文档。Formal 生产结果、Stage 1 诊断证据、Stage 2/3 候选修复仍分别保留在 `Q1/q1_features/` 的对应运行目录；本目录中的方法和索引只引用已经存在的文件与真实统计。

新版 Stage 2/3 对齐修复已合入 `Q1/q1_features/runs/q1_alignment_repair_stage2/` 和 `q1_alignment_repair_stage3/`。这些结果仍标记为 candidate-only，不替换正式 `q1_full_20260924`；候选状态、覆盖变化、映射疑点和未决样本见[论文结果与数据索引](reports/论文结果与数据索引.md)。

## 最终入口

- [Q1_最终方法.md](Q1_最终方法.md)：以已完成生产运行和真实配置为准的最终方法、输入输出实例、公式、审计规则与失败边界。
- [reports/论文结果与数据索引.md](reports/论文结果与数据索引.md)：论文可用的结果、中间数据、诊断/候选实验、失败方案及作图字段索引。
- [reports/Q1_合并说明与最新结果.md](reports/Q1_合并说明与最新结果.md)：Q1_new 合并范围、Formal/Stage 1/2/3/4 分层、最新候选统计和复核边界。
- [reports/全100条特征汇总.md](reports/全100条特征汇总.md)：由 `sample_index.csv` 与 `quality.csv` 逐行复算的 100 条样本 ID、有效时长、模态维度、粒度和 49/51 配对边界。

Stage 3 候选可复核入口：

- `Q1/q1_features/runs/q1_alignment_repair_stage2/reports/Q1_ALIGNMENT_REPAIR_STAGE2_REPORT.md`
- `Q1/q1_features/runs/q1_alignment_repair_stage3/reports/Q1_ALIGNMENT_REPAIR_STAGE3_REPORT.md`
- `Q1/q1_features/runs/q1_alignment_repair_stage3/reports/stage3_acceptance.json`
- `Q1/q1_features/runs/q1_alignment_repair_stage3/reports/q1_word_aligned_stage3_candidate.validation.json`
- `Q1/q1_features/runs/q1_alignment_repair_stage3/reports/q1_compact50_stage3_candidate.validation.json`

`Q1/q1_features/docs/repair_history/Q1_FINAL_EXPERIMENT_REPORT.md` 是 Stage 4 冻结报告。它没有附带新的 Stage 4 最终 NPZ 或独立验收文件，因此不能替代上述 Stage 3 本地候选验证。

## 方法与实施方案

- `E题_Q1_多模态特征构建方法_v3_审核稿.md`
- `E题_Q1_多模态特征构建方法与实施方案_v2_MMSA-FET.md`
- `E题_Q1_服务器具体实现说明_v1_Agent执行版.md`
- `E题_Q1_实现后测试_结果报告与收尾判定方案_v1.md`

## 实验报告

见 `reports/`：实验结果、验收记录、待解决问题、数据质量策略实施记录、下一阶段实验实施报告和 Q1_new 合并说明。

## 交接与交付

见 `delivery/`：GPT 研究交接说明、上传说明和代码索引；Stage 2/3 交接原文与包元数据见 `Q1/q1_features/docs/repair_history/`。

## 论文图目录

论文图统一位于 `doc/paper_figures/figures/` 下的 `Q1/`、`Q2/`、`Q3/` 子目录。Q1_01–Q1_10 保留为 Formal 历史图；Q1_11–Q1_22 为 Stage 2/3 合并后新增的 12 组图，图注应区分候选 coverage、状态和 Formal 生产结果，不能将候选状态写成人工 Gold。
