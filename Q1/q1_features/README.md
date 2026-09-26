# 数学建模 Q1 多模态特征提取

本项目完成 100 条真实样本的文本、音频和视觉特征提取、时间对齐、50 窗口聚合、质量审核与结构验证，为后续 Q2/Q3 建模提供可追溯的多模态输入。项目不训练情感分类器，也不改变原始数据集。Stage 2/3 对齐修复只生成候选，不覆盖 Formal 生产结果。

## 当前状态

- 基础源码版本：`da6c8e5bd553d0480ca941384ae20f279929872f`，以 [`SOURCE_REVISION`](SOURCE_REVISION) 为准。
- 最终运行目录：`runs/q1_full_20260924/`。
- 100 条样本的 `media`、`text`、`align`、`audio`、`vision`、`audit`、`pool` 阶段均已完成。
- 最终验证：`ok=true`、`errors=[]`、`selected_only=false`。
- 51 条样本仍为人工审核未解决状态，均保持 `paired_use=false`；这不属于提取失败。
- Formal 生产结果保持 `paired_use=true/false=49/51`。
- Stage 2 候选位于 `runs/q1_alignment_repair_stage2/`：完成 100/100 条官方文本强制对齐，`paired-use` 候选为 68，未写回 Formal。
- Stage 3 候选位于 `runs/q1_alignment_repair_stage3/`：包含词级和 50-bin 候选 NPZ、来源映射和逐样本状态；`candidate_only=true`、`production_write=false`。
- Stage 3 验收记录为 93 项测试通过、候选 validation 均 `ok=true`、冻结输入未变化；这不是 Stage 4 最终交付验收。`docs/repair_history/Q1_FINAL_EXPERIMENT_REPORT.md` 只有 Stage 4 冻结报告，未附新的最终 NPZ 或独立 Stage 4 验收文件。

完整结论见[实验结果报告](docs/reports/实验结果报告.md)、[待解决问题报告](docs/reports/Q1待解决问题报告.md)和[服务器验收记录](docs/reports/FINAL_ACCEPTANCE.md)。

数据质量双盲复核的总体规则见[Q1 双盲审核与辅助诊断操作指南](docs/operations/Q1双盲审核与辅助诊断操作指南.md)，多片段与锚点词的逐步命令见[Q1 多片段与锚点词审核执行清单](docs/operations/Q1多片段与锚点词审核执行清单.md)。当前状态见[Q1 下一阶段实验实施报告](docs/reports/Q1下一阶段实验实施报告.md)，历史基础设施记录见[Q1 数据质量策略实施记录](docs/reports/Q1数据质量策略实施记录.md)。

## 项目导航

| 路径 | 用途 |
|---|---|
| `configs/` | 主配置及人工审核表 |
| `src/q1_features/` | 特征提取、审核、聚合、验证与报告代码 |
| `scripts/` | 固定模型下载及准备脚本 |
| `tests/` | 不依赖在线模型的单元测试 |
| `models/` | 锁定版本的 BERT、CTC 模型及离线缓存 |
| `runs/q1_full_20260924/` | 100 条样本的最终结果、状态、报告和审核证据 |
| `runs/q1_alignment_repair_stage2/` | Stage 2 官方文本强制对齐和局部视觉候选；只读候选 |
| `runs/q1_alignment_repair_stage3/` | Stage 3 映射核验、TalkNet anchor、ArcFace continuity 和候选 NPZ；只读候选 |
| `docs/repair_history/` | Stage 2/3 交接、Stage 4 报告和包元数据；历史/分析材料 |
| `history/pre_stage3_20260925/` | 4 个被新版覆盖的旧文件的历史快照 |
| `third_party/` | OpenFace、dlib 和 MMSA-FET 源码/构建材料 |
| `env/` | 环境清单、构建日志和真实运行证据 |
| `tools/` | 项目本地工具链 |
| `docs/` | 需求、运行指南、实验报告和交付记录 |

详细目录职责与维护边界见[项目布局说明](docs/PROJECT_LAYOUT.md)。

## 环境、模型、数据和结果

```text
外部原始数据（只读）
        │
        ▼
configs/q1.yaml ──► src/q1_features ──► runs/q1_full_20260924
        │                   ▲
        ├── models/         │
        ├── third_party/    │
        ├── env/            │
        └── tools/ ─────────┘
```

修复候选链路保持隔离：

```text
runs/q1_full_20260924 (Formal, frozen)
        ├──► runs/q1_alignment_repair_stage2 (candidate only)
        │          └──► runs/q1_alignment_repair_stage3 (candidate only)
        └──► q1_full_20260924/reports/auto_resolution (Stage 1 evidence)
```

实际验收使用服务器已有的 Python 3.10 环境；原始、可复现的锁定环境安装方法保留在[服务器运行指南](docs/operations/服务器运行指南.md)中。现有模型、OpenFace 二进制和运行目录的相对位置属于运行契约，请勿随意移动。

## 快速检查

在项目根目录执行；`Q1_PY` 应指向已经通过验收的 Python 解释器：

```bash
export Q1_ROOT="$PWD"
export Q1_PY="/home/jqy/miniconda3/envs/shumo/bin/python"

"$Q1_PY" -m pip check
"$Q1_PY" -m pytest tests -q
"$Q1_PY" -m q1_features --help
```

重新验证现有完整结果：

```bash
"$Q1_PY" -m q1_features validate \
  --config configs/q1.yaml \
  --run-dir runs/q1_full_20260924
```

断点续跑前应先确认配置、源码版本和模型记录均未改变：

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

"$Q1_PY" -m q1_features run \
  --config configs/q1.yaml \
  --run-dir runs/q1_full_20260924 \
  --resume
```

完整安装、冒烟测试、人工审核及 `vision → audit → pool → collect → validate → report` 顺序见[服务器运行指南](docs/operations/服务器运行指南.md)。

## 关键产物

- 聚合特征：`runs/q1_full_20260924/features/q1_compact50.npz`
- 最终验证：`runs/q1_full_20260924/reports/validation.json`
- 质量报告：`runs/q1_full_20260924/reports/quality.csv`
- 对齐问题：`runs/q1_full_20260924/reports/alignment_issues.csv`
- 审核证据：`runs/q1_full_20260924/reports/manual_review_evidence/`
- Stage 2 报告：`runs/q1_alignment_repair_stage2/reports/Q1_ALIGNMENT_REPAIR_STAGE2_REPORT.md`
- Stage 3 报告：`runs/q1_alignment_repair_stage3/reports/Q1_ALIGNMENT_REPAIR_STAGE3_REPORT.md`
- Stage 3 词级候选：`runs/q1_alignment_repair_stage3/reports/q1_word_aligned_stage3_candidate.npz`
- Stage 3 50-bin 候选：`runs/q1_alignment_repair_stage3/reports/q1_compact50_stage3_candidate.npz`
- 交付归档登记：[交付归档说明](docs/delivery/交付归档说明.md)

## 维护约束

- 不删除、移动或覆盖原始数据。
- 不手工修改 `runs/` 中的特征数组、索引、状态或证据。
- Stage 2/3 产物只能作为候选审计材料；没有负责人批准，不得把候选 NPZ、mapping proposal 或 `paired_use` 候选写回 Formal。
- `MAPPING_STRONG`、`FULLY_VERIFIED` 和 `RESOLVED_VALID` 均是固定自动规则下的候选状态，不是人工 Gold 或准确率证明。
- 不临时替换已锁定模型或 OpenFace 版本。
- 未实际完成视听审核的记录必须保持 `unresolved` 或 `unverifiable`。
- 调整配置或代码后只能在版本和配置一致的运行目录中使用 `--resume`。
- 新交付版本使用新的文件名、内容清单和 SHA256，不覆盖历史归档。

## Stage 2/3 对齐修复候选（2026-09-26）

`runs/q1_alignment_repair_stage2/` 与 `runs/q1_alignment_repair_stage3/` 保存了 `Q1_new` 合入的自动候选、逐段证据和验证表。Stage 3 使用 TalkNet 建立说话者锚点，并只在同一 ArcFace 身份、无竞争身份且间隔不超过 1.0 秒时传播视觉状态；它没有交换 clip、修改正式配置或写回正式 NPZ。

Stage 3 报告中的 100 条候选状态为 `RESOLVED_VALID=57`、`RESOLVED_PARTIAL=19`、`RESOLVED_CONFLICT=5`、`RESOLVED_MISSING=4`、`UNRESOLVED=15`；这些是自动规则下的候选状态，不是人工 Gold。正式生产基线仍是 `runs/q1_full_20260924/`，候选特征的覆盖变化和失败边界见仓库根目录 `doc/Q1/reports/论文结果与数据索引.md` 及 `doc/paper_figures/README.md`。

Stage 3 运行记录中 `candidate_only=true`、`production_write=false`；任何 production writeback proposal 都必须经过人工批准，不能把候选 68 条 paired-use 直接当作正式发布数量。
