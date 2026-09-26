# Q1 对齐修复 Stage 2：网页版 GPT 分析交接说明

## 1. 请网页版 GPT 完成的任务

请基于本压缩包内的代码、配置和真实实验输出，独立审查 Q1 对齐修复 Stage 2，并回答：

1. 当前“语义配对”和“时间对齐”拆分是否合理，固定阈值是否有明显方法学缺陷；
2. 19 条原 TA_CONFLICT 的五类归因是否得到数据支持，4 条 `mapping_suspected` 应如何进一步验证；
3. official-text forced alignment 的结果是否被正确使用，是否存在“技术成功被误写成语义正确”的风险；
4. ArcFace、TalkNet、AV-sync 的职责划分与局部 mask 是否正确，是否还有身份泄漏、循环评价或来源映射问题；
5. 新旧 coverage 的变化应如何解释，尤其是视觉/三模态上升而音频覆盖下降；
6. 两个候选 NPZ 的结构、mask、CSR、source mapping 是否足以支撑后续建模；
7. 对剩余 12 条 semantic CONFLICT、7 条 semantic UNRESOLVED、15 条原视觉风险 unresolved，提出不依赖情感标签、不以恢复最多样本为目标的下一阶段实验方案；
8. 明确哪些结论可以自动采纳，哪些必须保持候选或需要额外证据。

请不要把候选 `paired_use=68` 解释成正式恢复 19 条，也不要建议直接覆盖正式配置或正式 `q1_compact50.npz`。

## 2. 实验边界

- 正式基线：`runs/q1_full_20260924`。
- Stage 2 候选目录：`runs/q1_alignment_repair_stage2`。
- 未使用情感标签。
- 未进行人工审核。
- 未训练或微调模型。
- 未自动交换 clip。
- 未重新运行 OpenFace/BERT/openSMILE。
- 未写回正式 `alignment_review.csv`、`target_face_segments.csv` 或 `q1_compact50.npz`。
- MFA 因没有锁定环境、声学模型和词典而未运行；它是可选 fallback。

## 3. 建议阅读顺序

1. `README_FIRST.md`：本说明。
2. `reports_stage2/Q1_ALIGNMENT_REPAIR_STAGE2_REPORT.md`：完整实验结论和新旧对比。
3. `reports_stage2/stage2_acceptance.json`：机器可读的最终状态。
4. `reports_stage2/alignment_repair_status.csv`：100 条样本最终候选状态。
5. `reports_stage2/semantic_pairing_status.csv`、`ctc_vs_official_forced_alignment.csv`：音文与时间对齐证据。
6. `reports_stage2/vision_sample_summary.csv`、`vision_segment_candidates.csv`、`visual_status_transition.csv`：视觉局部判定。
7. `reports_stage2/compact50_coverage_comparison.csv`：正式基线与候选覆盖率。
8. `reports_stage2/*.validation.json` 与 `frozen_baseline_verification.json`：结构和不可变性验证。
9. `results_stage1_auto_resolution/`：Stage 1 的 WhisperX、ArcFace、TalkNet、AV-sync 上游证据。
10. `core_code/`、`scripts/`、`tests/`：实现和回归测试。

## 4. 关键结果

### 数据链和语义

- 100/100 条数据链完整。
- 4 条同 video 的 clip 映射可疑，但没有自动交换。
- 原 23 条音文风险：
  - MATCH_STRONG：3
  - MATCH_WEAK：2
  - CONFLICT：12
  - UNRESOLVED：6
- 原 19 条 TA_CONFLICT 归因：
  - data-chain error：0
  - clip mapping suspected：4
  - 两个 ASR 一致但均与 Official 冲突：8
  - ASR 自身不稳定：5
  - 其他未决：2

### 时间对齐

- official-text forced alignment 技术成功：100/100。
- 全 100 条：71 VERIFIED、10 PARTIAL、19 NOT_APPLICABLE。
- 原 23 条风险：1 VERIFIED、4 PARTIAL、18 NOT_APPLICABLE。
- 100% 技术成功只表示对齐器产生了结果，不能证明音频真的说了官方文本；因此语义 CONFLICT/UNRESOLVED 的生成时间不会进入候选音频特征。
- 固定主阈值：共同词覆盖≥0.80、midpoint median≤0.30s、P90≤0.60s。敏感性结果只报告，不用于选择最大恢复率阈值。

### 视觉

- 原 37 条身份风险：8 fully_verified、14 partially_verified、15 unresolved。
- ArcFace 只进行 identity clustering；TalkNet 在 0.5 秒窗口、0.25 秒步长上决定 active cluster；AV-sync 仅作为辅助质量标记。
- 原 37 条视觉风险的时间比例：verified 35.26%、ambiguous/no-active 59.16%、missing 5.59%。
- 4 条 no-face 继续保持 missing。

### 覆盖变化

- 原 51 条隔离样本视觉观测率：18.24% → 43.02%。
- 原 51 条隔离样本三模态观测率：0% → 22.75%。
- 原 37 条视觉风险三模态观测率：0% → 31.35%。
- 全 100 条三模态观测率：39.06% → 49.16%。
- 候选 paired-use：49 → 68，仅用于描述候选覆盖。
- 音频覆盖下降来自对语义冲突/未决样本主动置 mask，不应简单解释为性能退化。

## 5. 重要实现纠错

正式流水线为安全隔离，会将 `identity_unknown` 样本的生产视觉张量置零，并移除正式视觉行中的 face/episode 标识。Stage 2 初版若直接复用这些安全张量，会错误地得到“37 条视觉风险全部不可恢复”。

最终实现已改为：

- 从冻结的 `openface/features.csv` 读取原始22维 AU/pose/gaze；
- 使用项目既有的 `assign_openface_episodes` 重建 episode；
- 使用固定 ArcFace cluster 与 TalkNet 局部窗口决定 eligible 行；
- 在 `raw_vision_candidate_sources.jsonl` 保存 source_id、帧、face、episode、cluster、时间与 eligible；
- 将100份 OpenFace CSV及100份帧时间轴纳入正式基线哈希；
- 逐项检查候选 NPZ 的 vision source reference 可追溯性。

网页端分析时应重点审查这一设计是否充分，以及局部窗口/轨迹边界处是否仍有偏差。

## 6. 验证结果

- `pytest`：78 passed。
- `pip check`：No broken requirements found。
- 正式 `validation.json.ok=true`。
- 两个候选 NPZ 验证均为 `ok=true`。
- 正式基线 610 项 SHA256/缺失态核验：changed=0。
- 词级候选：1932 词，text observed=1932，audio observed=1700，vision observed=1351。
- 原始视觉候选来源行：7657；原37条风险使用的视觉来源引用：2006。

## 7. 包内未包含内容

为控制体积并保护数据，压缩包不含：

- 原始视频、音频和帧图像；
- 模型权重和 Hugging Face 缓存；
- Python、OpenFace、Mamba 环境；
- OpenFace 构建目录；
- 情感标签；
- 8.6 GB 历史最终交付包。

包内的 manifest/sample index 只包含样本ID、视频/clip映射、官方文本和数据路径，不包含情感标签。

## 8. 结果使用限制

当前 Stage 2 只形成候选证据。除非下一阶段完成独立验证并经过负责人批准，否则不得：

- 将 `alignment_repair_status.csv` 自动写回生产配置；
- 用候选 NPZ 覆盖正式 NPZ；
- 将 PARTIAL 或 UNRESOLVED 描述为已确认；
- 自动交换4条 mapping suspected clip；
- 将 official forced alignment 的100%技术成功率描述为100%语义正确。
