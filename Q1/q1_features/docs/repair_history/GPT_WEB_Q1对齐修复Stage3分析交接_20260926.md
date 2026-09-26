# Q1 对齐修复 Stage 3：网页版 GPT 分析交接说明

## 1. 请网页版 GPT 完成的任务

请基于压缩包内的冻结基线、Stage 1/2 上游证据、Stage 3 代码、候选数组和验证记录，独立审查本次“TalkNet speaker anchor + ArcFace identity continuity + clip mapping stability”实验，并回答：

1. 4 条 `mapping_suspected` 的多指标 Hungarian 证据是否足以支持当前 `MAPPING_STRONG/WEAK/UNRESOLVED` 分类，尤其是唯一的 `MAPPING_STRONG` 是否仍需额外媒体证据；
2. TalkNet 只负责说话者锚点、ArcFace 只负责有界身份连续传播的职责划分是否合理；传播停止条件是否足够保守；
3. 0.5/1.0/1.5 秒敏感性结果能否支持固定使用 1.0 秒，而没有数据内选优；
4. 原 37 条视觉风险从 Stage 2 的 8/14/15（full/partial/unresolved）变化为 Stage 3 的 16/6/5/10（full/partial/ambiguous/unresolved）是否得到逐窗证据支持；
5. `RESOLVED_VALID/PARTIAL/CONFLICT/MISSING/UNRESOLVED` 的优先级与解释是否合理，是否存在一个总状态遮蔽多轴问题的风险；
6. 15 条真正 `UNRESOLVED` 是否分类正确，并为它们设计下一步不使用情感标签、不追求最大恢复率的验证方案；
7. Stage 3 视觉覆盖上升是否完全来自可追踪的冻结 OpenFace 原始行；候选 NPZ 的 mask、CSR、source mapping 和安全隔离是否充分；
8. `production_writeback_proposal_stage3.csv` 中哪些行具备进入人工批准的条件，哪些必须继续隔离；
9. 给出下一阶段最小实验计划、停止条件和生产写回前验收清单。

不要把 coverage 上升解释为准确率提升；本实验没有人工金标准。不要建议直接交换 clip 或覆盖正式 Q1 结果。

## 2. 实验边界

- 正式基线：`runs/q1_full_20260924`，保持冻结。
- Stage 2 输入：`runs/q1_alignment_repair_stage2`，保持冻结。
- Stage 3：`runs/q1_alignment_repair_stage3`，仅生成候选。
- 未使用人工审核或情感标签。
- 未训练、微调或按结果选择模型阈值。
- 未自动交换任何 clip。
- 未重新运行 BERT、openSMILE 或 OpenFace。
- 未修改 manifest、正式 pairing 配置、正式 NPZ 或 Stage 2 NPZ。
- `writeback_executed=0`；所有生产建议都仍需负责人批准。

## 3. 建议阅读顺序

1. `README_FIRST.md`：本说明。
2. `reports_stage3/Q1_ALIGNMENT_REPAIR_STAGE3_REPORT.md`。
3. `reports_stage3/stage3_acceptance.json` 与 `stage3_engineering_verification.json`。
4. `reports_stage3/final_resolution_status_stage3.csv`。
5. `reports_stage3/mapping_validation_stage3.csv`。
6. `reports_stage3/vision_summary_stage3.csv`、`vision_segment_stage3.csv`、`visual_gap_sensitivity_stage3.csv`。
7. `reports_stage3/coverage_comparison_stage3.csv`。
8. `reports_stage3/*.validation.json` 与 `stage3_baseline_verification.json`。
9. `reports_stage2/`：Stage 3 的冻结输入和对照。
10. `results_stage1_auto_resolution/`：WhisperX、ArcFace、TalkNet、AV-sync 上游证据。
11. `core_code/`、`scripts/`、`tests/`：实现与保护测试。

## 4. 关键结果

### 映射核验

- `-iRBcNs9oI8$_$6 → -iRBcNs9oI8$_$9`：`MAPPING_STRONG`；四种 Hungarian 匹配一致，WhisperX/CTC token 0→0.5，全局排名 48→1。
- `-iRBcNs9oI8$_$3 → -iRBcNs9oI8$_$7`：`MAPPING_WEAK`；token 与 character 指标给出不同候选。
- `-mJ2ud6oKI8$_$1 → -mJ2ud6oKI8$_$8`：`MAPPING_UNRESOLVED`；低绝对相似度且指标不一致。
- `-ri04Z7vwnc$_$2 → -ri04Z7vwnc$_$5`：`MAPPING_UNRESOLVED`；方向一致但绝对相似度不足。

没有执行 clip 交换。

### 视觉连续性

- Stage 2 原 37 条：8 full、14 partial、15 unresolved。
- Stage 3 原 37 条：16 full、6 partial、5 ambiguous、10 unresolved。
- TalkNet 直接确认：100.233 秒。
- ArcFace 连续性新增恢复：54.250 秒。
- 歧义 mask：19.950 秒；missing：18.631 秒；unresolved：91.240 秒。
- 原 37 条平均视觉可用比例：33.55%→51.44%。
- 0.5/1.0/1.5 秒的样本级分类相同；可用比例分别为 50.25%、51.44%、51.44%。主结果固定使用 1.0 秒。

### 最终候选状态

- `RESOLVED_VALID`：57
- `RESOLVED_PARTIAL`：19
- `RESOLVED_CONFLICT`：5
- `RESOLVED_MISSING`：4
- `UNRESOLVED`：15

候选 paired-use 为 68，与 Stage 2 相同；Stage 3 的主要增益是已可用样本内部的视觉时间覆盖，不是增加释放样本数。

### 覆盖率

- 全 100 条：Vision 57.46%→70.10%→76.04%，Trimodal 39.06%→49.16%→52.14%。
- 原 51 条隔离：Vision 18.24%→43.02%→54.67%，Trimodal 0%→22.75%→28.59%。
- 原 37 条视觉风险：Vision 0%→34.16%→50.22%，Trimodal 0%→31.35%→39.41%。

顺序均为 Formal→Stage 2→Stage 3。Coverage 仅表示候选可用性，不是准确率。

## 5. 15 条真正未决

- 视觉身份链未决（10）：`-9y-fZ3swSY$_$8`、`-HwX2H8Z4hY$_$2`、`-HwX2H8Z4hY$_$5`、`-HwX2H8Z4hY$_$6`、`-hnBHBN8p5A$_$6`、`-hnBHBN8p5A$_$7`、`-iRBcNs9oI8$_$9`、`-mJ2ud6oKI8$_$9`、`-s9qJ7ATP7w$_$0`、`-s9qJ7ATP7w$_$7`。
- 映射证据不稳定（2）：`-iRBcNs9oI8$_$3`、`-ri04Z7vwnc$_$2`。
- 语义配对未决（3）：`-mJ2ud6oKI8$_$6`、`-mJ2ud6oKI8$_$8`、`-vxjVxOeScU$_$4`。

## 6. 工程验证

- 完整测试：93 passed；其中 Stage 3 专项回归测试 15 passed。
- `pip check`：No broken requirements found。
- 正式 `validation.json.ok=true`。
- 词级、50-bin 候选 validation 均 `ok=true`。
- Stage 3 候选中 2,808 个视觉来源引用可追踪。
- 11 项正式/Stage 2 冻结文件哈希全部未变化。
- 生产建议 100 行，`writeback_executed` 全部为 0。

## 7. 重要解释边界

- `MAPPING_STRONG` 是候选证据，不是允许自动换片。
- `FULLY_VERIFIED` 是固定自动规则下的候选状态，不是人工金标准。
- ArcFace continuity 只能恢复同一已锚定身份的短暂停顿，不能跨身份跳变、竞争活跃身份、聚类不稳定、长间隔或 no-face。
- `RESOLVED_CONFLICT` 表示安全处置已明确为 mask，不表示跨模态内容被修复。
- `RESOLVED_MISSING` 表示自然缺失已明确，不允许补造视觉特征。
- 正式结果仍是 `runs/q1_full_20260924`；Stage 3 NPZ 不得覆盖正式或 Stage 2 NPZ。

## 8. 包内未包含内容

为保护数据并控制体积，本包不含：原始视频/音频/帧、情感标签、模型权重、Hugging Face 缓存、Python/OpenFace 环境、OpenFace 构建目录及历史大体积交付包。
