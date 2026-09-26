# Q1 项目入口

Q1 是 100 条真实样本的文本–音频–视觉特征构建与对齐审计项目。当前目录保留 Formal 生产运行、Stage 1 诊断证据、Stage 2/3 候选修复和旧交付交接材料；候选修复不自动替换 Formal 结果。

## 当前结论

- Formal 生产基线：`q1_features/runs/q1_full_20260924/`，100 个样本、37 个源视频，七个提取阶段均为 100 成功、0 失败；`validation.json` 为 `ok=true`、`errors=[]`。
- Formal 配对边界：`paired_use=true/false=49/51`。51 条问题样本继续保持 `unverifiable` 和 `paired_use=false`，不能写成已人工确认。
- Stage 2：官方文本强制对齐完成 100/100 条；`q1_alignment_repair_stage2/reports/` 中的音频与视觉结果均为候选，`candidate_only=true`、`production_write=false`。
- Stage 3：在 Stage 2 基础上进行四指标 mapping 核验、TalkNet speaker anchor 和 ArcFace identity continuity；候选状态为 `57/19/5/4/15`（`RESOLVED_VALID/PARTIAL/CONFLICT/MISSING/UNRESOLVED`），未执行写回。
- Stage 4：`q1_features/docs/repair_history/Q1_FINAL_EXPERIMENT_REPORT.md` 只有冻结叙述；本地没有与其对应的新 Stage 4 最终 NPZ 或独立验收文件。可复核的本地候选是 Stage 3 的两个 NPZ 和其 validation JSON。

## 目录

| 路径 | 内容 |
|---|---|
| [`q1_features/README.md`](q1_features/README.md) | 代码、配置、运行、候选和维护边界 |
| `q1_features/configs/` | Formal、Stage 2/3 和审核配置 |
| `q1_features/src/q1_features/` | 特征提取、池化、审计及修复候选实现 |
| `q1_features/runs/q1_full_20260924/` | Formal 生产结果和 Stage 1 自动证据 |
| `q1_features/runs/q1_alignment_repair_stage2/` | Stage 2 候选报告、候选 NPZ 与中间证据 |
| `q1_features/runs/q1_alignment_repair_stage3/` | Stage 3 候选报告、状态、来源映射与候选 NPZ |
| `q1_features/docs/repair_history/` | Stage 2/3 交接、Stage 4 报告、包元数据和原始说明 |
| `q1_features/history/pre_stage3_20260925/` | 合并时被新版覆盖的 4 个旧文件 |
| `../doc/Q1/` | 最终方法、论文结果索引、合并说明和历史方案 |

## 推荐阅读

先读 [`../doc/Q1/Q1_最终方法.md`](../doc/Q1/Q1_最终方法.md)，再读 [`../doc/Q1/reports/论文结果与数据索引.md`](../doc/Q1/reports/论文结果与数据索引.md) 和 [`../doc/Q1/reports/Q1_合并说明与最新结果.md`](../doc/Q1/reports/Q1_合并说明与最新结果.md)。Stage 2/3 原始报告位于各自 `runs/*/reports/` 目录；报告中的 coverage 是候选可用性指标，不是准确率。

## 隔离规则

Formal `q1_compact50.npz`、manifest、正式 pairing 配置和 Stage 2 输入保持冻结。Stage 3 的 `production_writeback_proposal_stage3.csv`、`q1_word_aligned_stage3_candidate.npz` 与 `q1_compact50_stage3_candidate.npz` 只能用于复核；没有负责人批准，不得覆盖正式 NPZ、交换 clip 或释放 unresolved 样本。
