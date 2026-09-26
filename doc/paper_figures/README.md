# 论文数据图与可复算材料

本目录包含 **83组数据图**，每组同时提供220 dpi PNG、矢量PDF和可编辑SVG，以及对应CSV。全部来自保存的实际结果，不生成流程图或方法主图。

打开 [图册](index.html) 可按问题筛选和查看大图；论文排版优先使用PDF，修改标签使用SVG。图内使用简短英文，以下提供中文图意与可用结论。

## 使用与证据边界

- Q2 test曾影响后续结构探索，图中test结果只能作为已观察测试集上的探索结果，不能写成独立留出确认。
- 三种子训练曲线保留波动；消融不声称统计显著。valid为开发集，附件3/4无标签。
- 历史训练只保存epoch平均损失和间隔验证，没有逐batch梯度/损失；全量逐轮明细见 [Q2_all_epoch_history.csv](data/Q2_all_epoch_history.csv)。未插值或伪造逐step记录。
- 对接近评分使用点图及有刻度的局部坐标；比例热图固定0–1，效应跨量级时分面且标明不同尺度。零线、负增益与反例均保留。
- Q3 block3历史汇总存在实际成本不一致（T/A各22，V26），不作为严格同成本优势证据；word无此成本问题。
- Q1接受率与可用率是流程指标，非人工准确率；Q3忠实性不等于情感预测正确或人类因果解释。

## 建议论文选图

正文优先：Q1_02、Q1_03；Q2_01、Q2_04、Q2_06、Q2_07、Q2_12、Q2_14；Q3_01、Q3_02、Q3_06、Q3_10，以及01/14/18号局部解释。其余图用于补充实验和错误归因；不要将几十张图全部挤入正文。

## 图表清单

| 图号 | 内容 | 数据行数 | 图与数据 |
|---|---|---:|---|
| Q1_01_duration | 原始样本长度与可配对状态 | 100 | [PNG](figures/Q1/Q1_01_duration.png) · [PDF](figures/Q1/Q1_01_duration.pdf) · [SVG](figures/Q1/Q1_01_duration.svg) · [CSV](data/Q1_01_duration.csv) |
| Q1_02_coverage | 文本定位与视觉有效覆盖率 | 100 | [PNG](figures/Q1/Q1_02_coverage.png) · [PDF](figures/Q1/Q1_02_coverage.pdf) · [SVG](figures/Q1/Q1_02_coverage.svg) · [CSV](data/Q1_02_coverage.csv) |
| Q1_03_clock | 声画时钟起点与终点差异 | 100 | [PNG](figures/Q1/Q1_03_clock.png) · [PDF](figures/Q1/Q1_03_clock.pdf) · [SVG](figures/Q1/Q1_03_clock.svg) · [CSV](data/Q1_03_clock.csv) |
| Q1_04_quality_rank | 逐样本质量差异与视觉片段数 | 100 | [PNG](figures/Q1/Q1_04_quality_rank.png) · [PDF](figures/Q1/Q1_04_quality_rank.pdf) · [SVG](figures/Q1/Q1_04_quality_rank.svg) · [CSV](data/Q1_04_quality_rank.csv) |
| Q1_05_problem_partition | 自动补查样本的问题分区 | 4 | [PNG](figures/Q1/Q1_05_problem_partition.png) · [PDF](figures/Q1/Q1_05_problem_partition.pdf) · [SVG](figures/Q1/Q1_05_problem_partition.svg) · [CSV](data/Q1_05_problem_partition.csv) |
| Q1_06_identity_similarity | 人脸片段身份相似度与嵌入离散程度 | 236 | [PNG](figures/Q1/Q1_06_identity_similarity.png) · [PDF](figures/Q1/Q1_06_identity_similarity.pdf) · [SVG](figures/Q1/Q1_06_identity_similarity.svg) · [CSV](data/Q1_06_identity_similarity.csv) |
| Q1_07_sync_null | 视听同步得分与错位随机参照 | 137 | [PNG](figures/Q1/Q1_07_sync_null.png) · [PDF](figures/Q1/Q1_07_sync_null.pdf) · [SVG](figures/Q1/Q1_07_sync_null.svg) · [CSV](data/Q1_07_sync_null.csv) |
| Q1_08_identity_quality | 身份嵌入的成功数量与稳定性 | 191 | [PNG](figures/Q1/Q1_08_identity_quality.png) · [PDF](figures/Q1/Q1_08_identity_quality.pdf) · [SVG](figures/Q1/Q1_08_identity_quality.svg) · [CSV](data/Q1_08_identity_quality.csv) |
| Q1_09_word_alignment | 逐词CTC分数与接受区间时长 | 1932 | [PNG](figures/Q1/Q1_09_word_alignment.png) · [PDF](figures/Q1/Q1_09_word_alignment.pdf) · [SVG](figures/Q1/Q1_09_word_alignment.svg) · [CSV](data/Q1_09_word_alignment.csv) |
| Q1_10_transcription_errors | 转写误差类型分解 | 300 | [PNG](figures/Q1/Q1_10_transcription_errors.png) · [PDF](figures/Q1/Q1_10_transcription_errors.pdf) · [SVG](figures/Q1/Q1_10_transcription_errors.svg) · [CSV](data/Q1_10_transcription_errors.csv) |
| Q1_11_stage_coverage | 正式提取与两阶段候选覆盖变化 | 21 | [PNG](figures/Q1/Q1_11_stage_coverage.png) · [PDF](figures/Q1/Q1_11_stage_coverage.pdf) · [SVG](figures/Q1/Q1_11_stage_coverage.svg) · [CSV](data/Q1_11_stage_coverage.csv) |
| Q1_12_paired_coverage | 全部样本覆盖变化分布 | 300 | [PNG](figures/Q1/Q1_12_paired_coverage.png) · [PDF](figures/Q1/Q1_12_paired_coverage.pdf) · [SVG](figures/Q1/Q1_12_paired_coverage.svg) · [CSV](data/Q1_12_paired_coverage.csv) |
| Q1_13_visual_composition | 视觉身份风险样本的可用与隔离成分 | 37 | [PNG](figures/Q1/Q1_13_visual_composition.png) · [PDF](figures/Q1/Q1_13_visual_composition.pdf) · [SVG](figures/Q1/Q1_13_visual_composition.svg) · [CSV](data/Q1_13_visual_composition.csv) |
| Q1_14_visual_transitions | 视觉风险样本状态转移 | 37 | [PNG](figures/Q1/Q1_14_visual_transitions.png) · [PDF](figures/Q1/Q1_14_visual_transitions.pdf) · [SVG](figures/Q1/Q1_14_visual_transitions.svg) · [CSV](data/Q1_14_visual_transitions.csv) |
| Q1_15_gap_sensitivity | 视觉身份连续间隔敏感性 | 3 | [PNG](figures/Q1/Q1_15_gap_sensitivity.png) · [PDF](figures/Q1/Q1_15_gap_sensitivity.pdf) · [SVG](figures/Q1/Q1_15_gap_sensitivity.svg) · [CSV](data/Q1_15_gap_sensitivity.csv) |
| Q1_16_mapping_evidence | 四条映射疑点的交叉证据 | 4 | [PNG](figures/Q1/Q1_16_mapping_evidence.png) · [PDF](figures/Q1/Q1_16_mapping_evidence.pdf) · [SVG](figures/Q1/Q1_16_mapping_evidence.svg) · [CSV](data/Q1_16_mapping_evidence.csv) |
| Q1_17_alignment_agreement | CTC与官方文本强制对齐的一致性 | 100 | [PNG](figures/Q1/Q1_17_alignment_agreement.png) · [PDF](figures/Q1/Q1_17_alignment_agreement.pdf) · [SVG](figures/Q1/Q1_17_alignment_agreement.svg) · [CSV](data/Q1_17_alignment_agreement.csv) |
| Q1_18_alignment_thresholds | 时间一致性阈值敏感性 | 18 | [PNG](figures/Q1/Q1_18_alignment_thresholds.png) · [PDF](figures/Q1/Q1_18_alignment_thresholds.pdf) · [SVG](figures/Q1/Q1_18_alignment_thresholds.svg) · [CSV](data/Q1_18_alignment_thresholds.csv) |
| Q1_19_resolution_axes | 最终候选状态与独立语义轴 | 100 | [PNG](figures/Q1/Q1_19_resolution_axes.png) · [PDF](figures/Q1/Q1_19_resolution_axes.pdf) · [SVG](figures/Q1/Q1_19_resolution_axes.svg) · [CSV](data/Q1_19_resolution_axes.csv) |
| Q1_20_semantic_masking | 语义冲突与未决音频的安全屏蔽 | 300 | [PNG](figures/Q1/Q1_20_semantic_masking.png) · [PDF](figures/Q1/Q1_20_semantic_masking.pdf) · [SVG](figures/Q1/Q1_20_semantic_masking.svg) · [CSV](data/Q1_20_semantic_masking.csv) |
| Q1_21_word_sources | 词级候选CSR来源与覆盖分布 | 3864 | [PNG](figures/Q1/Q1_21_word_sources.png) · [PDF](figures/Q1/Q1_21_word_sources.pdf) · [SVG](figures/Q1/Q1_21_word_sources.svg) · [CSV](data/Q1_21_word_sources.csv) |
| Q1_22_visual_timelines | 受影响样本的视觉状态时间分布 | 1243 | [PNG](figures/Q1/Q1_22_visual_timelines.png) · [PDF](figures/Q1/Q1_22_visual_timelines.pdf) · [SVG](figures/Q1/Q1_22_visual_timelines.svg) · [CSV](data/Q1_22_visual_timelines.csv) |
| Q2_01_training | 最终模型训练损失与验证曲线 | 20 | [PNG](figures/Q2/Q2_01_training.png) · [PDF](figures/Q2/Q2_01_training.pdf) · [SVG](figures/Q2/Q2_01_training.svg) · [CSV](data/Q2_01_training.csv) |
| Q2_02_seed_training | 固定结构三种子训练波动 | 60 | [PNG](figures/Q2/Q2_02_seed_training.png) · [PDF](figures/Q2/Q2_02_seed_training.pdf) · [SVG](figures/Q2/Q2_02_seed_training.svg) · [CSV](data/Q2_02_seed_training.csv) |
| Q2_03_schedule | 学习率与延长训练的验证表现 | 80 | [PNG](figures/Q2/Q2_03_schedule.png) · [PDF](figures/Q2/Q2_03_schedule.pdf) · [SVG](figures/Q2/Q2_03_schedule.svg) · [CSV](data/Q2_03_schedule.csv) |
| Q2_04_ablations | 最终结构的三个直接消融 | 8 | [PNG](figures/Q2/Q2_04_ablations.png) · [PDF](figures/Q2/Q2_04_ablations.pdf) · [SVG](figures/Q2/Q2_04_ablations.svg) · [CSV](data/Q2_04_ablations.csv) |
| Q2_05_exploration | 保存的无偏置探索结果全景 | 54 | [PNG](figures/Q2/Q2_05_exploration.png) · [PDF](figures/Q2/Q2_05_exploration.pdf) · [SVG](figures/Q2/Q2_05_exploration.svg) · [CSV](data/Q2_05_exploration.csv) |
| Q2_06_grid | 验证与测试72场景完整热图 | 144 | [PNG](figures/Q2/Q2_06_grid.png) · [PDF](figures/Q2/Q2_06_grid.pdf) · [SVG](figures/Q2/Q2_06_grid.svg) · [CSV](data/Q2_06_grid.csv) |
| Q2_07_robustness | 按模态分解的缺失强度曲线 | 48 | [PNG](figures/Q2/Q2_07_robustness.png) · [PDF](figures/Q2/Q2_07_robustness.pdf) · [SVG](figures/Q2/Q2_07_robustness.svg) · [CSV](data/Q2_07_robustness.csv) |
| Q2_08_positions | 缺失位置对分类和回归的影响 | 24 | [PNG](figures/Q2/Q2_08_positions.png) · [PDF](figures/Q2/Q2_08_positions.pdf) · [SVG](figures/Q2/Q2_08_positions.svg) · [CSV](data/Q2_08_positions.csv) |
| Q2_09_actual_deletion | 名义缺失跨度与真实删除率 | 72 | [PNG](figures/Q2/Q2_09_actual_deletion.png) · [PDF](figures/Q2/Q2_09_actual_deletion.pdf) · [SVG](figures/Q2/Q2_09_actual_deletion.svg) · [CSV](data/Q2_09_actual_deletion.csv) |
| Q2_10_paired_conditions | 共同受损样本上的配对条件比较 | 216 | [PNG](figures/Q2/Q2_10_paired_conditions.png) · [PDF](figures/Q2/Q2_10_paired_conditions.pdf) · [SVG](figures/Q2/Q2_10_paired_conditions.svg) · [CSV](data/Q2_10_paired_conditions.csv) |
| Q2_11_stress | 整模态与多模态压力场景 | 16 | [PNG](figures/Q2/Q2_11_stress.png) · [PDF](figures/Q2/Q2_11_stress.pdf) · [SVG](figures/Q2/Q2_11_stress.svg) · [CSV](data/Q2_11_stress.csv) |
| Q2_12_class_metrics | 验证与测试类别瓶颈 | 6 | [PNG](figures/Q2/Q2_12_class_metrics.png) · [PDF](figures/Q2/Q2_12_class_metrics.pdf) · [SVG](figures/Q2/Q2_12_class_metrics.svg) · [CSV](data/Q2_12_class_metrics.csv) |
| Q2_13_error_distribution | 回归残差小提琴与绝对误差分布 | 727 | [PNG](figures/Q2/Q2_13_error_distribution.png) · [PDF](figures/Q2/Q2_13_error_distribution.pdf) · [SVG](figures/Q2/Q2_13_error_distribution.svg) · [CSV](data/Q2_13_error_distribution.csv) |
| Q2_14_prediction_diagnostics | 测试混淆矩阵与回归密度 | 727 | [PNG](figures/Q2/Q2_14_prediction_diagnostics.png) · [PDF](figures/Q2/Q2_14_prediction_diagnostics.pdf) · [SVG](figures/Q2/Q2_14_prediction_diagnostics.svg) · [CSV](data/Q2_14_prediction_diagnostics.csv) |
| Q2_15_special_availability | 附件3全部样本的当前可观测状态 | 30 | [PNG](figures/Q2/Q2_15_special_availability.png) · [PDF](figures/Q2/Q2_15_special_availability.pdf) · [SVG](figures/Q2/Q2_15_special_availability.svg) · [CSV](data/Q2_15_special_availability.csv) |
| Q2_16_valid_diagnostics | 验证集真实重载的混淆与回归密度 | 728 | [PNG](figures/Q2/Q2_16_valid_diagnostics.png) · [PDF](figures/Q2/Q2_16_valid_diagnostics.pdf) · [SVG](figures/Q2/Q2_16_valid_diagnostics.svg) · [CSV](data/Q2_16_valid_diagnostics.csv) |
| Q2_17_valid_residuals | 验证集分情感类别残差与误差分布 | 728 | [PNG](figures/Q2/Q2_17_valid_residuals.png) · [PDF](figures/Q2/Q2_17_valid_residuals.pdf) · [SVG](figures/Q2/Q2_17_valid_residuals.svg) · [CSV](data/Q2_17_valid_residuals.csv) |
| Q2_18_valid_confidence | 验证集置信度与错误关系 | 728 | [PNG](figures/Q2/Q2_18_valid_confidence.png) · [PDF](figures/Q2/Q2_18_valid_confidence.pdf) · [SVG](figures/Q2/Q2_18_valid_confidence.svg) · [CSV](data/Q2_18_valid_confidence.csv) |
| Q2_19_valid_groups | 验证集按有效文本长度分层 | 4 | [PNG](figures/Q2/Q2_19_valid_groups.png) · [PDF](figures/Q2/Q2_19_valid_groups.pdf) · [SVG](figures/Q2/Q2_19_valid_groups.svg) · [CSV](data/Q2_19_valid_groups.csv) |
| Q2_20_valid_error_flows | 验证集全部误分类流向与置信度 | 728 | [PNG](figures/Q2/Q2_20_valid_error_flows.png) · [PDF](figures/Q2/Q2_20_valid_error_flows.pdf) · [SVG](figures/Q2/Q2_20_valid_error_flows.svg) · [CSV](data/Q2_20_valid_error_flows.csv) |
| Q2_21_valid_intensity_bias | 验证集情感强度分层与有符号回归偏差 | 728 | [PNG](figures/Q2/Q2_21_valid_intensity_bias.png) · [PDF](figures/Q2/Q2_21_valid_intensity_bias.pdf) · [SVG](figures/Q2/Q2_21_valid_intensity_bias.svg) · [CSV](data/Q2_21_valid_intensity_bias.csv) |
| Q2_22_valid_missing_profile | 验证集局部缺失的模态、位置与跨度规律 | 72 | [PNG](figures/Q2/Q2_22_valid_missing_profile.png) · [PDF](figures/Q2/Q2_22_valid_missing_profile.pdf) · [SVG](figures/Q2/Q2_22_valid_missing_profile.svg) · [CSV](data/Q2_22_valid_missing_profile.csv) |
| Q3_01_faithfulness | 三预算忠实性增益与视频簇区间 | 12 | [PNG](figures/Q3/Q3_01_faithfulness.png) · [PDF](figures/Q3/Q3_01_faithfulness.pdf) · [SVG](figures/Q3/Q3_01_faithfulness.svg) · [CSV](data/Q3_01_faithfulness.csv) |
| Q3_02_gain_violin | 20%预算逐样本忠实性增益分布 | 2868 | [PNG](figures/Q3/Q3_02_gain_violin.png) · [PDF](figures/Q3/Q3_02_gain_violin.pdf) · [SVG](figures/Q3/Q3_02_gain_violin.svg) · [CSV](data/Q3_02_gain_violin.csv) |
| Q3_03_gain_ecdf | 忠实性增益累计分布与负增益比例 | 2868 | [PNG](figures/Q3/Q3_03_gain_ecdf.png) · [PDF](figures/Q3/Q3_03_gain_ecdf.pdf) · [SVG](figures/Q3/Q3_03_gain_ecdf.svg) · [CSV](data/Q3_03_gain_ecdf.csv) |
| Q3_04_control_coverage | 严格随机对照的可比较覆盖率 | 12 | [PNG](figures/Q3/Q3_04_control_coverage.png) · [PDF](figures/Q3/Q3_04_control_coverage.pdf) · [SVG](figures/Q3/Q3_04_control_coverage.svg) · [CSV](data/Q3_04_control_coverage.csv) |
| Q3_05_random_counts | 逐样本严格随机集合数量 | 2868 | [PNG](figures/Q3/Q3_05_random_counts.png) · [PDF](figures/Q3/Q3_05_random_counts.pdf) · [SVG](figures/Q3/Q3_05_random_counts.svg) · [CSV](data/Q3_05_random_counts.csv) |
| Q3_06_keep | 保留实验的配对增益与反例 | 8 | [PNG](figures/Q3/Q3_06_keep.png) · [PDF](figures/Q3/Q3_06_keep.pdf) · [SVG](figures/Q3/Q3_06_keep.svg) · [CSV](data/Q3_06_keep.csv) |
| Q3_07_granularity | 整词和分块选择的粒度敏感性 | 4 | [PNG](figures/Q3/Q3_07_granularity.png) · [PDF](figures/Q3/Q3_07_granularity.pdf) · [SVG](figures/Q3/Q3_07_granularity.svg) · [CSV](data/Q3_07_granularity.csv) |
| Q3_08_modality_shapley | 全验证集整模态影响与双输出Shapley | 2184 | [PNG](figures/Q3/Q3_08_modality_shapley.png) · [PDF](figures/Q3/Q3_08_modality_shapley.pdf) · [SVG](figures/Q3/Q3_08_modality_shapley.svg) · [CSV](data/Q3_08_modality_shapley.csv) |
| Q3_09_modality_ecdf | 整模态影响全范围分布 | 2184 | [PNG](figures/Q3/Q3_09_modality_ecdf.png) · [PDF](figures/Q3/Q3_09_modality_ecdf.pdf) · [SVG](figures/Q3/Q3_09_modality_ecdf.svg) · [CSV](data/Q3_09_modality_ecdf.csv) |
| Q3_10_special_modality | 附件4全部20条模态作用与双输出贡献 | 60 | [PNG](figures/Q3/Q3_10_special_modality.png) · [PDF](figures/Q3/Q3_10_special_modality.pdf) · [SVG](figures/Q3/Q3_10_special_modality.svg) · [CSV](data/Q3_10_special_modality.csv) |
| Q3_11_direction_failures | 方向重测的全部验证集失败样本 | 9 | [PNG](figures/Q3/Q3_11_direction_failures.png) · [PDF](figures/Q3/Q3_11_direction_failures.pdf) · [SVG](figures/Q3/Q3_11_direction_failures.svg) · [CSV](data/Q3_11_direction_failures.csv) |
| Q3_12_low_impact | 关键集合与低影响集合对照 | 9 | [PNG](figures/Q3/Q3_12_low_impact.png) · [PDF](figures/Q3/Q3_12_low_impact.pdf) · [SVG](figures/Q3/Q3_12_low_impact.svg) · [CSV](data/Q3_12_low_impact.csv) |
| Q3_13_stratification | 正确错误与中性类分层忠实性 | 15 | [PNG](figures/Q3/Q3_13_stratification.png) · [PDF](figures/Q3/Q3_13_stratification.pdf) · [SVG](figures/Q3/Q3_13_stratification.svg) · [CSV](data/Q3_13_stratification.csv) |
| Q3_14_task_after_deletion | 关键删除与随机删除后的任务表现 | 15 | [PNG](figures/Q3/Q3_14_task_after_deletion.png) · [PDF](figures/Q3/Q3_14_task_after_deletion.pdf) · [SVG](figures/Q3/Q3_14_task_after_deletion.svg) · [CSV](data/Q3_14_task_after_deletion.csv) |
| Q3_15_subset_predictions | 20条样本八个模态保留子集的实际输出 | 160 | [PNG](figures/Q3/Q3_15_subset_predictions.png) · [PDF](figures/Q3/Q3_15_subset_predictions.pdf) · [SVG](figures/Q3/Q3_15_subset_predictions.svg) · [CSV](data/Q3_15_subset_predictions.csv) |
| Q3_16_budget_geometry | 实际预算可达性与连续片段数 | 80 | [PNG](figures/Q3/Q3_16_budget_geometry.png) · [PDF](figures/Q3/Q3_16_budget_geometry.pdf) · [SVG](figures/Q3/Q3_16_budget_geometry.svg) · [CSV](data/Q3_16_budget_geometry.csv) |
| Q3_17_local_concentration | 验证集逐位置影响集中度 | 2169 | [PNG](figures/Q3/Q3_17_local_concentration.png) · [PDF](figures/Q3/Q3_17_local_concentration.pdf) · [SVG](figures/Q3/Q3_17_local_concentration.svg) · [CSV](data/Q3_17_local_concentration.csv) |
| Q3_18_signed_local | 验证集正负单位置影响 | 2169 | [PNG](figures/Q3/Q3_18_signed_local.png) · [PDF](figures/Q3/Q3_18_signed_local.pdf) · [SVG](figures/Q3/Q3_18_signed_local.svg) · [CSV](data/Q3_18_signed_local.csv) |
| Q3_19_equal_cost_granularity | 排除成本不一致后的粒度敏感性 | 8 | [PNG](figures/Q3/Q3_19_equal_cost_granularity.png) · [PDF](figures/Q3/Q3_19_equal_cost_granularity.pdf) · [SVG](figures/Q3/Q3_19_equal_cost_granularity.svg) · [CSV](data/Q3_19_equal_cost_granularity.csv) |
| Q3_case_01 | 附件4样本01的三模态逐位置重要性 | 66 | [PNG](figures/Q3/Q3_case_01.png) · [PDF](figures/Q3/Q3_case_01.pdf) · [SVG](figures/Q3/Q3_case_01.svg) · [CSV](data/Q3_case_01.csv) |
| Q3_case_02 | 附件4样本02的三模态逐位置重要性 | 57 | [PNG](figures/Q3/Q3_case_02.png) · [PDF](figures/Q3/Q3_case_02.pdf) · [SVG](figures/Q3/Q3_case_02.svg) · [CSV](data/Q3_case_02.csv) |
| Q3_case_03 | 附件4样本03的三模态逐位置重要性 | 99 | [PNG](figures/Q3/Q3_case_03.png) · [PDF](figures/Q3/Q3_case_03.pdf) · [SVG](figures/Q3/Q3_case_03.svg) · [CSV](data/Q3_case_03.csv) |
| Q3_case_04 | 附件4样本04的三模态逐位置重要性 | 78 | [PNG](figures/Q3/Q3_case_04.png) · [PDF](figures/Q3/Q3_case_04.pdf) · [SVG](figures/Q3/Q3_case_04.svg) · [CSV](data/Q3_case_04.csv) |
| Q3_case_05 | 附件4样本05的三模态逐位置重要性 | 45 | [PNG](figures/Q3/Q3_case_05.png) · [PDF](figures/Q3/Q3_case_05.pdf) · [SVG](figures/Q3/Q3_case_05.svg) · [CSV](data/Q3_case_05.csv) |
| Q3_case_06 | 附件4样本06的三模态逐位置重要性 | 105 | [PNG](figures/Q3/Q3_case_06.png) · [PDF](figures/Q3/Q3_case_06.pdf) · [SVG](figures/Q3/Q3_case_06.svg) · [CSV](data/Q3_case_06.csv) |
| Q3_case_07 | 附件4样本07的三模态逐位置重要性 | 144 | [PNG](figures/Q3/Q3_case_07.png) · [PDF](figures/Q3/Q3_case_07.pdf) · [SVG](figures/Q3/Q3_case_07.svg) · [CSV](data/Q3_case_07.csv) |
| Q3_case_08 | 附件4样本08的三模态逐位置重要性 | 33 | [PNG](figures/Q3/Q3_case_08.png) · [PDF](figures/Q3/Q3_case_08.pdf) · [SVG](figures/Q3/Q3_case_08.svg) · [CSV](data/Q3_case_08.csv) |
| Q3_case_09 | 附件4样本09的三模态逐位置重要性 | 54 | [PNG](figures/Q3/Q3_case_09.png) · [PDF](figures/Q3/Q3_case_09.pdf) · [SVG](figures/Q3/Q3_case_09.svg) · [CSV](data/Q3_case_09.csv) |
| Q3_case_10 | 附件4样本10的三模态逐位置重要性 | 90 | [PNG](figures/Q3/Q3_case_10.png) · [PDF](figures/Q3/Q3_case_10.pdf) · [SVG](figures/Q3/Q3_case_10.svg) · [CSV](data/Q3_case_10.csv) |
| Q3_case_11 | 附件4样本11的三模态逐位置重要性 | 42 | [PNG](figures/Q3/Q3_case_11.png) · [PDF](figures/Q3/Q3_case_11.pdf) · [SVG](figures/Q3/Q3_case_11.svg) · [CSV](data/Q3_case_11.csv) |
| Q3_case_12 | 附件4样本12的三模态逐位置重要性 | 126 | [PNG](figures/Q3/Q3_case_12.png) · [PDF](figures/Q3/Q3_case_12.pdf) · [SVG](figures/Q3/Q3_case_12.svg) · [CSV](data/Q3_case_12.csv) |
| Q3_case_13 | 附件4样本13的三模态逐位置重要性 | 60 | [PNG](figures/Q3/Q3_case_13.png) · [PDF](figures/Q3/Q3_case_13.pdf) · [SVG](figures/Q3/Q3_case_13.svg) · [CSV](data/Q3_case_13.csv) |
| Q3_case_14 | 附件4样本14的三模态逐位置重要性 | 105 | [PNG](figures/Q3/Q3_case_14.png) · [PDF](figures/Q3/Q3_case_14.pdf) · [SVG](figures/Q3/Q3_case_14.svg) · [CSV](data/Q3_case_14.csv) |
| Q3_case_15 | 附件4样本15的三模态逐位置重要性 | 63 | [PNG](figures/Q3/Q3_case_15.png) · [PDF](figures/Q3/Q3_case_15.pdf) · [SVG](figures/Q3/Q3_case_15.svg) · [CSV](data/Q3_case_15.csv) |
| Q3_case_16 | 附件4样本16的三模态逐位置重要性 | 33 | [PNG](figures/Q3/Q3_case_16.png) · [PDF](figures/Q3/Q3_case_16.pdf) · [SVG](figures/Q3/Q3_case_16.svg) · [CSV](data/Q3_case_16.csv) |
| Q3_case_17 | 附件4样本17的三模态逐位置重要性 | 72 | [PNG](figures/Q3/Q3_case_17.png) · [PDF](figures/Q3/Q3_case_17.pdf) · [SVG](figures/Q3/Q3_case_17.svg) · [CSV](data/Q3_case_17.csv) |
| Q3_case_18 | 附件4样本18的三模态逐位置重要性 | 144 | [PNG](figures/Q3/Q3_case_18.png) · [PDF](figures/Q3/Q3_case_18.pdf) · [SVG](figures/Q3/Q3_case_18.svg) · [CSV](data/Q3_case_18.csv) |
| Q3_case_19 | 附件4样本19的三模态逐位置重要性 | 117 | [PNG](figures/Q3/Q3_case_19.png) · [PDF](figures/Q3/Q3_case_19.pdf) · [SVG](figures/Q3/Q3_case_19.svg) · [CSV](data/Q3_case_19.csv) |
| Q3_case_20 | 附件4样本20的三模态逐位置重要性 | 129 | [PNG](figures/Q3/Q3_case_20.png) · [PDF](figures/Q3/Q3_case_20.pdf) · [SVG](figures/Q3/Q3_case_20.svg) · [CSV](data/Q3_case_20.csv) |

## Q1_01_duration 原始样本长度与可配对状态

100条实际解码时长与词数。paired_use为当前流程可配对标志，不是人工正确率；解码时长与题面标称时长口径不同。

来源：[Q1/q1_features/runs/q1_full_20260924/reports/quality.csv](../../Q1/q1_features/runs/q1_full_20260924/reports/quality.csv)。

## Q1_02_coverage 文本定位与视觉有效覆盖率

小提琴由100条样本的实际比例估计，cut=0，保留极端值。CTC接受率不等于人工对齐精度；WER来自诊断转写，不是情感分类误差。

来源：[Q1/q1_features/runs/q1_full_20260924/reports/quality.csv](../../Q1/q1_features/runs/q1_full_20260924/reports/quality.csv)。

## Q1_03_clock 声画时钟起点与终点差异

声画媒体时钟诊断，不代表说话人身份正确或单词对齐精度。全体样本保留，零线方便区分提前与滞后。

来源：[Q1/q1_features/runs/q1_full_20260924/reports/quality.csv](../../Q1/q1_features/runs/q1_full_20260924/reports/quality.csv)。

## Q1_04_quality_rank 逐样本质量差异与视觉片段数

右图按文本接受率升序排列，视觉值使用同一排列而非独立排序。揭示低覆盖样本，不能声称所有样本均达到高质量。

来源：[Q1/q1_features/runs/q1_full_20260924/reports/quality.csv](../../Q1/q1_features/runs/q1_full_20260924/reports/quality.csv)。

## Q1_05_problem_partition 自动补查样本的问题分区

仅统计problem_partition中进入补查的样本，分母不是全部100条；身份与声文问题可以并存。该图是困难样本构成，不是消融提升。

来源：[Q1/q1_features/runs/q1_full_20260924/reports/auto_resolution/problem_partition.csv](../../Q1/q1_features/runs/q1_full_20260924/reports/auto_resolution/problem_partition.csv)。

## Q1_06_identity_similarity 人脸片段身份相似度与嵌入离散程度

所有已保存片段对的ArcFace余弦值；片段对共享样本且不独立。余弦阈值属于自动身份聚类判断，不是人工身份准确率；该图诊断质量，不证明说话人已经确认。

来源：[Q1/q1_features/runs/q1_full_20260924/reports/auto_resolution/arcface_episode_similarity.csv](../../Q1/q1_features/runs/q1_full_20260924/reports/auto_resolution/arcface_episode_similarity.csv)。

## Q1_07_sync_null 视听同步得分与错位随机参照

嘴部运动与音频能量的诊断相关统计，对照来自移位与跨样本组合。零假设排序不是经过多重比较校正的显著性，也不是说话人准确率；低于null p95的真实分数如实保留。

来源：[Q1/q1_features/runs/q1_full_20260924/reports/auto_resolution/av_sync_scores.csv](../../Q1/q1_features/runs/q1_full_20260924/reports/auto_resolution/av_sync_scores.csv)。

## Q1_08_identity_quality 身份嵌入的成功数量与稳定性

统计已检测到的人脸episode，分母不是100条原始样本。成功嵌入数量与片段内一致性分开呈现，未把没有人脸的样本当成人脸聚类正确。

来源：[Q1/q1_features/runs/q1_full_20260924/reports/auto_resolution/arcface_episode_quality.csv](../../Q1/q1_features/runs/q1_full_20260924/reports/auto_resolution/arcface_episode_quality.csv)。

## Q1_09_word_alignment 逐词CTC分数与接受区间时长

服务器1932条原始词记录（含未形成有效spoken词的条目），1793条有accepted_interval；因此不将1932替代生产spoken分母1926。左图为各状态内ECDF，排除6条无分数记录，symlog横轴完整保留尾部。分数不是校准概率，区间存在不等于人工确认；不采用陈旧pending状态字段判定接受。

来源：[doc/paper_figures/data/server_Q1_words.csv](../../doc/paper_figures/data/server_Q1_words.csv)。

## Q1_10_transcription_errors 转写误差类型分解

独立诊断转写的替换/删除/插入数按规范化参考词数归一化，三者相加等于WER。WER可超过1；它不是CTC时间对齐准确率，更不是情感分类误差。

来源：[doc/paper_figures/data/server_Q1_diagnostics.csv](../../doc/paper_figures/data/server_Q1_diagnostics.csv)。

## Q1_11_stage_coverage 正式提取与两阶段候选覆盖变化

各组按50-bin的observed_mask计数；Formal是原始保留观测，修复后是质量控制后的可用观测。全100条Vision为57.46%→70.10%→76.04%，音频99.50%→78.28%→78.28%；音频下降来自主动隔离，不是提取器退化。分组重叠，不能相加。 Stage 2/3为随新版提供的自动候选；Stage 4仅有报告，未据此伪造最终数组。

来源：[Q1/q1_features/runs/q1_alignment_repair_stage3/reports/coverage_comparison_stage3.csv](../../Q1/q1_features/runs/q1_alignment_repair_stage3/reports/coverage_comparison_stage3.csv)。

## Q1_12_paired_coverage 全部样本覆盖变化分布

同一sample_index逐样本配对，保留零变化和负变化。左图跨质量口径描述变化，不能解释为准确率增益；右图仅比较同属候选流程的Stage 2/3视觉mask。pp为百分点。 Stage 2/3为随新版提供的自动候选；Stage 4仅有报告，未据此伪造最终数组。

来源：[Q1/q1_features/runs/q1_full_20260924/features/q1_compact50.npz](../../Q1/q1_features/runs/q1_full_20260924/features/q1_compact50.npz)；[Q1/q1_features/runs/q1_alignment_repair_stage2/reports/q1_compact50_repair_candidate.npz](../../Q1/q1_features/runs/q1_alignment_repair_stage2/reports/q1_compact50_repair_candidate.npz)；[Q1/q1_features/runs/q1_alignment_repair_stage3/reports/q1_compact50_stage3_candidate.npz](../../Q1/q1_features/runs/q1_alignment_repair_stage3/reports/q1_compact50_stage3_candidate.npz)。

## Q1_13_visual_composition 视觉身份风险样本的可用与隔离成分

37条原视觉风险全部保留。直接TalkNet确认平均33.55%，ArcFace连续性新增17.89%，合计51.44%；歧义、自然缺失与未决均显示。S编号对应CSV的零基sample_index；右图按样本等权，不是按总时长加权。 Stage 2/3为随新版提供的自动候选；Stage 4仅有报告，未据此伪造最终数组。

来源：[Q1/q1_features/runs/q1_alignment_repair_stage3/reports/vision_summary_stage3.csv](../../Q1/q1_features/runs/q1_alignment_repair_stage3/reports/vision_summary_stage3.csv)；[Q1/q1_features/runs/q1_alignment_repair_stage3/reports/final_resolution_status_stage3.csv](../../Q1/q1_features/runs/q1_alignment_repair_stage3/reports/final_resolution_status_stage3.csv)。

## Q1_14_visual_transitions 视觉风险样本状态转移

Stage 2为8/14/15条full/partial/unresolved；Stage 3为16/6/5/10条full/partial/ambiguous/unresolved。状态来自自动规则，full不代表人工Gold，ambiguous不并入成功。 Stage 2/3为随新版提供的自动候选；Stage 4仅有报告，未据此伪造最终数组。

来源：[Q1/q1_features/runs/q1_alignment_repair_stage3/reports/vision_summary_stage3.csv](../../Q1/q1_features/runs/q1_alignment_repair_stage3/reports/vision_summary_stage3.csv)；[Q1/q1_features/runs/q1_alignment_repair_stage3/reports/final_resolution_status_stage3.csv](../../Q1/q1_features/runs/q1_alignment_repair_stage3/reports/final_resolution_status_stage3.csv)。

## Q1_15_gap_sensitivity 视觉身份连续间隔敏感性

0.5/1.0/1.5秒平均可用比例50.25%/51.44%/51.44%，37条状态计数相同。主配置1.0秒固定，不据此宣称最优；增加间隔并不必然增加恢复率。 Stage 2/3为随新版提供的自动候选；Stage 4仅有报告，未据此伪造最终数组。

来源：[Q1/q1_features/runs/q1_alignment_repair_stage3/reports/visual_gap_sensitivity_stage3.csv](../../Q1/q1_features/runs/q1_alignment_repair_stage3/reports/visual_gap_sensitivity_stage3.csv)。

## Q1_16_mapping_evidence 四条映射疑点的交叉证据

显示原官方与候选的文本重合得分及四指标分配一致率。只有iRB/6达到MAPPING_STRONG；ri04/2虽100%一致但绝对分数低，仍未决。所有官方mapping保留，没有自动交换clip。 Stage 2/3为随新版提供的自动候选；Stage 4仅有报告，未据此伪造最终数组。

来源：[Q1/q1_features/runs/q1_alignment_repair_stage3/reports/mapping_validation_stage3.csv](../../Q1/q1_features/runs/q1_alignment_repair_stage3/reports/mapping_validation_stage3.csv)。

## Q1_17_alignment_agreement CTC与官方文本强制对齐的一致性

100条原始记录随CSV保留；ECDF和散点只使用对应字段非空的样本，不用零填补未适用记录。两个自动对齐器的一致性不是人工时间准确率，低共同词比例也不能因小时间差被忽略。 Stage 2/3为随新版提供的自动候选；Stage 4仅有报告，未据此伪造最终数组。

来源：[Q1/q1_features/runs/q1_alignment_repair_stage2/reports/ctc_vs_official_forced_alignment.csv](../../Q1/q1_features/runs/q1_alignment_repair_stage2/reports/ctc_vs_official_forced_alignment.csv)。

## Q1_18_alignment_thresholds 时间一致性阈值敏感性

红框为固定主阈值median≤0.3秒、P90≤0.6秒。全100条71 verified、10 partial、19 not-applicable；原23条风险仅1 verified。分面色标分别以各自总数为上限，不能跨面板按深浅比较比例，也不根据最大通过数选阈值。 Stage 2/3为随新版提供的自动候选；Stage 4仅有报告，未据此伪造最终数组。

来源：[Q1/q1_features/runs/q1_alignment_repair_stage2/reports/alignment_threshold_sensitivity.csv](../../Q1/q1_features/runs/q1_alignment_repair_stage2/reports/alignment_threshold_sensitivity.csv)。

## Q1_19_resolution_axes 最终候选状态与独立语义轴

五状态计数57/19/5/4/15。交叉表保留总状态可能遮蔽的语义冲突：UNRESOLVED并不等于只有一种问题；Conflict表示安全mask方案明确，而非原内容被修好。 Stage 2/3为随新版提供的自动候选；Stage 4仅有报告，未据此伪造最终数组。

来源：[Q1/q1_features/runs/q1_alignment_repair_stage3/reports/final_resolution_status_stage3.csv](../../Q1/q1_features/runs/q1_alignment_repair_stage3/reports/final_resolution_status_stage3.csv)。

## Q1_20_semantic_masking 语义冲突与未决音频的安全屏蔽

逐样本音频50-bin观测率按最终语义状态分组；Formal与候选的质量口径不同。Conflict/Unresolved组主动屏蔽音频，表明覆盖率下降可能是安全处理结果，不能把它包装成提取失败或隐去。 Stage 2/3为随新版提供的自动候选；Stage 4仅有报告，未据此伪造最终数组。

来源：[Q1/q1_features/runs/q1_full_20260924/features/q1_compact50.npz](../../Q1/q1_features/runs/q1_full_20260924/features/q1_compact50.npz)；[Q1/q1_features/runs/q1_alignment_repair_stage2/reports/q1_compact50_repair_candidate.npz](../../Q1/q1_features/runs/q1_alignment_repair_stage2/reports/q1_compact50_repair_candidate.npz)；[Q1/q1_features/runs/q1_alignment_repair_stage3/reports/q1_compact50_stage3_candidate.npz](../../Q1/q1_features/runs/q1_alignment_repair_stage3/reports/q1_compact50_stage3_candidate.npz)；[Q1/q1_features/runs/q1_alignment_repair_stage3/reports/final_resolution_status_stage3.csv](../../Q1/q1_features/runs/q1_alignment_repair_stage3/reports/final_resolution_status_stage3.csv)。

## Q1_21_word_sources 词级候选CSR来源与覆盖分布

每个词的音频/视觉源行数量由CSR indptr相邻差计算，所有1932条均保留，包括零来源和mask词。可用词覆盖ECDF条件于mask=1。Stage 3词级NPZ自带CSR，不能与仍缺失的Formal样本级map_*.npz混为一谈；来源数量多不等于质量高。 Stage 2/3为随新版提供的自动候选；Stage 4仅有报告，未据此伪造最终数组。

来源：[Q1/q1_features/runs/q1_alignment_repair_stage3/reports/q1_word_aligned_stage3_candidate.npz](../../Q1/q1_features/runs/q1_alignment_repair_stage3/reports/q1_word_aligned_stage3_candidate.npz)。

## Q1_22_visual_timelines 受影响样本的视觉状态时间分布

全部41条受影响样本，包括37条身份风险与4条no-face。颜色表示真实时间段状态；横轴按各clip时长归一化用于比较结构，原始start/end秒数保留在CSV中。传播未跨越歧义或自然缺失，不把mask补成平滑连续数据。 Stage 2/3为随新版提供的自动候选；Stage 4仅有报告，未据此伪造最终数组。

来源：[Q1/q1_features/runs/q1_alignment_repair_stage3/reports/vision_segment_stage3.csv](../../Q1/q1_features/runs/q1_alignment_repair_stage3/reports/vision_segment_stage3.csv)；[Q1/q1_features/runs/q1_alignment_repair_stage3/reports/final_resolution_status_stage3.csv](../../Q1/q1_features/runs/q1_alignment_repair_stage3/reports/final_resolution_status_stage3.csv)。

## Q2_01_training 最终模型训练损失与验证曲线

左图是逐轮内各batch等权平均，不是逐step。第3轮总损失增加是加入受损视图损失；右图只连接实际记录的四个验证点。训练checkpoint与FP16导出包数值口径不同。

来源：[Q2/results/experiment_ledger.json](../../Q2/results/experiment_ledger.json)。

## Q2_02_seed_training 固定结构三种子训练波动

三个种子均为BERT8全层微调；部署种子预先固定1111。图呈现训练随机性，不能用单seed优势代表多seed稳定优势。

来源：[Q2/results/experiment_ledger.json](../../Q2/results/experiment_ledger.json)。

## Q2_03_schedule 学习率与延长训练的验证表现

BERT学习率为下游学习率0.1倍，左图由保存lr与最终配置计算；40轮BERT4与BERT8结构和配置不同，只能视为探索，不能把差值归于训练轮数。左图数据见Q2_01_training.csv的lr列。

来源：[Q2/results/experiment_ledger.json](../../Q2/results/experiment_ledger.json)。

## Q2_04_ablations 最终结构的三个直接消融

短线左端为72场景均值，右端为clean（按实际数值绘制）。同seed的去CLS、去人工缺失训练、均值池化实验；test已参与探索。这里使用ledger训练checkpoint评估，不与导出包最后一位小数混用。

来源：[Q2/results/experiment_ledger.json](../../Q2/results/experiment_ledger.json)。

## Q2_05_exploration 保存的无偏置探索结果全景

展示保存的test无类别偏置评价记录，点可能属于同一架构的不同配置/种子，非独立试验。两坐标使用明确的局部范围以显示接近分数，不作为独立泛化证据。

来源：[Q2/results/experiment_ledger.json](../../Q2/results/experiment_ledger.json)。

## Q2_06_grid 验证与测试72场景完整热图

F/M/R为前/中/后，20–80为名义区间比例，两个面板共享色标。每格是一个场景的全样本Macro-F1；自然缺失仍保留。全部场景均展示。

来源：[Q2/results/final/valid/metrics_per_scenario.csv](../../Q2/results/final/valid/metrics_per_scenario.csv)；[Q2/results/final/test/metrics_per_scenario.csv](../../Q2/results/final/test/metrics_per_scenario.csv)。

## Q2_07_robustness 按模态分解的缺失强度曲线

各点是前中后三个位置等权平均。连线仅引导阅读，未拟合未评估缺失率；无独立重复误差带。文本相关缺失退化较大。

来源：[Q2/results/final/valid/metrics_per_scenario.csv](../../Q2/results/final/valid/metrics_per_scenario.csv)；[Q2/results/final/test/metrics_per_scenario.csv](../../Q2/results/final/test/metrics_per_scenario.csv)。

## Q2_08_positions 缺失位置对分类和回归的影响

每条曲线均跨六种模态条件平均，valid和test分开编码。位置差异可能同时受到实际删除率和共同受损样本构成影响，应结合配对图。

来源：[Q2/results/final/valid/metrics_per_scenario.csv](../../Q2/results/final/valid/metrics_per_scenario.csv)；[Q2/results/final/test/metrics_per_scenario.csv](../../Q2/results/final/test/metrics_per_scenario.csv)。

## Q2_09_actual_deletion 名义缺失跨度与真实删除率

点为场景内样本实际删除率中位数，线为四分位范围而非置信区间。各模态横向偏移仅避免重叠，原始名义比例保留CSV；不把缺失区间长度当作实际内容删除比例。

来源：[Q2/results/final/test/deletion_rates.csv](../../Q2/results/final/test/deletion_rates.csv)。

## Q2_10_paired_conditions 共同受损样本上的配对条件比较

每个点/箱体汇集条件组的指标差值，非逐样本F1。比较在同组共同受损样本上完成；不同组仍可能具有不同分母，不能当成独立重复。

来源：[Q2/results/final/test/paired_conditions.csv](../../Q2/results/final/test/paired_conditions.csv)。

## Q2_11_stress 整模态与多模态压力场景

压力测试与主72场景分开，不纳入主均值；点图使用明确局部横坐标，不以截断柱形夸大差异。

来源：[Q2/results/final/valid/metrics_per_scenario.csv](../../Q2/results/final/valid/metrics_per_scenario.csv)；[Q2/results/final/test/metrics_per_scenario.csv](../../Q2/results/final/test/metrics_per_scenario.csv)。

## Q2_12_class_metrics 验证与测试类别瓶颈

中性类召回和F1明显低于正负类。表格记录每类support，所有色值固定0–1，避免通过各格独立缩放夸大。

来源：[Q2/results/final/valid/metrics.json](../../Q2/results/final/valid/metrics.json)；[Q2/results/final/test/metrics.json](../../Q2/results/final/test/metrics.json)。

## Q2_13_error_distribution 回归残差小提琴与绝对误差分布

727条test真实逐样本预测；不裁去离群误差。各真值类残差方向反映回归向中部收缩。累计曲线由原始误差排序计算，无平滑拟合。

来源：[Q2/results/final/figures/test_clean_predictions.csv](../../Q2/results/final/figures/test_clean_predictions.csv)。

## Q2_14_prediction_diagnostics 测试混淆矩阵与回归密度

回归图使用真实坐标聚合为六边形密度，不对真值抖动。虚线为理想预测。test已参与探索；验证集逐类诊断见Q2_12。

来源：[Q2/results/final/figures/test_clean_predictions.csv](../../Q2/results/final/figures/test_clean_predictions.csv)。

## Q2_15_special_availability 附件3全部样本的当前可观测状态

可观测位置占结构位置比例；自然零行不推断为某种具体提取失败。补偿计数均为0，最终模型没有补偿模块。专项无真值，不报告准确率。

来源：[Q2/results/final/q2_special_state.csv](../../Q2/results/final/q2_special_state.csv)。

## Q2_16_valid_diagnostics 验证集真实重载的混淆与回归密度

本地离线交付包真实推理，Accuracy/Macro-F1与Q3保存结果完全一致，MAE差约3×10^-9；验证集已用于选模，不是独立test。中性类184条中65条判对，88条误判正类。

来源：[doc/paper_figures/data/Q2_valid_reloaded_predictions.csv](../../doc/paper_figures/data/Q2_valid_reloaded_predictions.csv)；[doc/paper_figures/data/Q2_valid_reloaded_diagnostics.json](../../doc/paper_figures/data/Q2_valid_reloaded_diagnostics.json)。

## Q2_17_valid_residuals 验证集分情感类别残差与误差分布

未删除离群样本，小提琴cut=0。负向类偏高估、正向类偏低估可作为向中性回缩的描述，不能仅凭图推断训练因果机制。

来源：[doc/paper_figures/data/Q2_valid_reloaded_predictions.csv](../../doc/paper_figures/data/Q2_valid_reloaded_predictions.csv)。

## Q2_18_valid_confidence 验证集置信度与错误关系

右图固定7个等宽置信度区间，圆面积与样本量成比例，标注样本数；斜线是概率与正确率相等参照。仅做事后诊断，不进行校准拟合或阈值重新选择。分箱精确数据另见Q2_valid_confidence_bins.csv。

来源：[doc/paper_figures/data/Q2_valid_reloaded_predictions.csv](../../doc/paper_figures/data/Q2_valid_reloaded_predictions.csv)。

## Q2_19_valid_groups 验证集按有效文本长度分层

固定长度分层的描述性准确率与MAE，各组标签组成可能不同，不能声称长度造成性能变化。点图和明确局部纵轴减少空白；每组分母标出。

来源：[doc/paper_figures/data/Q2_valid_reloaded_predictions.csv](../../doc/paper_figures/data/Q2_valid_reloaded_predictions.csv)。

## Q2_20_valid_error_flows 验证集全部误分类流向与置信度

左图百分比分母为280条全部误分类，六个非对角流向完整保留，柱色按真实类别区分。右图各正确性组内ECDF；0.8仅为事后诊断参考线，不改变预测或重新选阈值。confidence≥0.8的212条中42条错误。

来源：[doc/paper_figures/data/Q2_valid_reloaded_predictions.csv](../../doc/paper_figures/data/Q2_valid_reloaded_predictions.csv)。

## Q2_21_valid_intensity_bias 验证集情感强度分层与有符号回归偏差

固定分组[-3,-1]、(-1,0)、{0}、(0,1)、[1,3]；不移除大误差。残差为预测减真值。组均值比较可检验幅度收缩现象；不据此断言损失函数、讽刺或模态冲突是已验证原因。每组样本数、MAE、bias见Q2_valid_intensity_bins.csv。

来源：[doc/paper_figures/data/Q2_valid_reloaded_predictions.csv](../../doc/paper_figures/data/Q2_valid_reloaded_predictions.csv)。

## Q2_22_valid_missing_profile 验证集局部缺失的模态、位置与跨度规律

仅使用valid的72个all_samples场景；每场景728条、相同样本复用，不是52416个独立样本。左图每格平均3个位置、色标固定0–1；右图每点平均6种模态组合，纵轴局部放大且标注。跨度是名义ρ，不等于样本实际删除比例。

来源：[Q2/results/final/valid/metrics_per_scenario.csv](../../Q2/results/final/valid/metrics_per_scenario.csv)。

## Q3_01_faithfulness 三预算忠实性增益与视频簇区间

G为关键集合D减严格匹配随机集合平均D。误差线为video_id簇bootstrap 95%区间；T/J与A/V分别缩放并显式标注，纵轴为带刻度的局部范围，不从0开始，不能据线条高度比较跨面板效应。

来源：[Q3/runs/valid_v2/summaries/faithfulness.json](../../Q3/runs/valid_v2/summaries/faithfulness.json)。

## Q3_02_gain_violin 20%预算逐样本忠实性增益分布

所有可比较样本进入小提琴，cut=0不外推密度；负增益反例全部保留。不同模态可比较分母不同，原始CSV含sample_id/video_id/n_random。

来源：[Q3/runs/valid_v2/summaries/paired_samples.csv](../../Q3/runs/valid_v2/summaries/paired_samples.csv)。

## Q3_03_gain_ecdf 忠实性增益累计分布与负增益比例

经验分布无需核带宽，显示负增益与长尾，补充小提琴的平滑显示。样本相关性不允许把24k条重复记录当成独立样本。

来源：[Q3/runs/valid_v2/summaries/paired_samples.csv](../../Q3/runs/valid_v2/summaries/paired_samples.csv)。

## Q3_04_control_coverage 严格随机对照的可比较覆盖率

灰色包含无可用模态/无匹配对照，具体拆分见CSV的not_applicable与without_controls。每条主路径计划728条；未补齐无合法随机集合的样本。

来源：[Q3/runs/valid_v2/summaries/faithfulness.json](../../Q3/runs/valid_v2/summaries/faithfulness.json)。

## Q3_05_random_counts 逐样本严格随机集合数量

每样本最多50组，少于20组的样本仍用于描述均值，但不足以支持细尾概率判断；集合数不是独立样本量。

来源：[Q3/runs/valid_v2/summaries/paired_samples.csv](../../Q3/runs/valid_v2/summaries/paired_samples.csv)。

## Q3_06_keep 保留实验的配对增益与反例

S=1−D。global保留只留下关键集合；conditional只移除当前模态域内其余内容。关键删除集合不必是最佳保留集合，负增益如实展示。

来源：[Q3/runs/valid_v2/summaries/keep.json](../../Q3/runs/valid_v2/summaries/keep.json)。

## Q3_07_granularity 整词和分块选择的粒度敏感性

历史汇总中block3与point存在成本不等配对：T/A各22条、V26条；因此不能将该图block3差值解释为纯同成本粒度因果效应。word成本差异0，区间跨0；整词可读性不等于忠实性显著提升。

来源：[Q3/runs/valid_v2/summaries/granularity.json](../../Q3/runs/valid_v2/summaries/granularity.json)。

## Q3_08_modality_shapley 全验证集整模态影响与双输出Shapley

箱体Q1–Q3，中位线与1.5IQR须；图中不显示离群点以保持主体可读，全部值在CSV且Q3_09显示全分布。带符号Shapley区分抑制与支持，不能和绝对影响D混称。

来源：[Q3/runs/valid_v2/summaries/modality.json](../../Q3/runs/valid_v2/summaries/modality.json)。

## Q3_09_modality_ecdf 整模态影响全范围分布

所有有限观测均进入经验累计分布，包含零观测模态按现有汇总定义的零贡献；可与Q3_08主体箱体互补，不隐藏长尾。

来源：[Q3/runs/valid_v2/summaries/modality.json](../../Q3/runs/valid_v2/summaries/modality.json)。

## Q3_10_special_modality 附件4全部20条模态作用与双输出贡献

各面板各自的明确色标：D非负，Shapley有符号。13号视觉无观测；14号分类与强度首位模态不一致，不能用一张平均模态图掩盖。

来源：[Q3/runs/special_v2/summaries/modality.json](../../Q3/runs/special_v2/summaries/modality.json)。

## Q3_11_direction_failures 方向重测的全部验证集失败样本

8个suppress候选联合删除后概率反而降低，1个support候选反而升高。方向只相对于固定原预测类别；局部单点符号不能保证联合集合方向。

来源：[Q3/runs/valid_v2/summaries/direction.json](../../Q3/runs/valid_v2/summaries/direction.json)。

## Q3_12_low_impact 关键集合与低影响集合对照

同样本关键集合与低影响集合的差值，区间为video_id簇bootstrap。文本与AV使用标明数值的不同局部纵轴；该对照补充严格随机对照，不取代随机集合几何匹配。

来源：[Q3/runs/valid_v2/summaries/low_comparison.json](../../Q3/runs/valid_v2/summaries/low_comparison.json)。

## Q3_13_stratification 正确错误与中性类分层忠实性

分层存在重叠，中性类可同时属于正确或错误组；并非三个互斥分组。错误预测也有正忠实性增益，说明解释可以忠实反映错误决策，不代表预测正确。

来源：[Q3/runs/valid_v2/summaries/stratified.json](../../Q3/runs/valid_v2/summaries/stratified.json)。

## Q3_14_task_after_deletion 关键删除与随机删除后的任务表现

每条路径使用自己可比较样本集合上的clean、key和random；跨路径分母不同，不能直接解释路径间性能差。随机结果先逐重复计算任务指标再取均值，未用平均概率代替。

来源：[Q3/runs/valid_v2/summaries/perturbation_metrics.json](../../Q3/runs/valid_v2/summaries/perturbation_metrics.json)。

## Q3_15_subset_predictions 20条样本八个模态保留子集的实际输出

Shapley实际使用的八子集全枚举输出，概率始终跟踪原预测类别；不是各子集最大类别概率。Empty为训练先验回退；不同样本原预测类别不同。

来源：`Q3/runs/special_v2/samples/*/shapley.json`（原始归档路径，未随图册上传）；`Q3/runs/special_v2/samples/*/original.json`（原始归档路径，未随图册上传）。

## Q3_16_budget_geometry 实际预算可达性与连续片段数

实际成本取不超过目标预算的最大合法值，不强行补齐。零观测模态可能无适用选择；对照匹配实际几何与成本。仅为20条专项的算法可行性诊断。

来源：`Q3/runs/special_v2/samples/*/selections.jsonl`（原始归档路径，未随图册上传）。

## Q3_17_local_concentration 验证集逐位置影响集中度

728条开发样本，T/A/V有观测样本728/728/713。每样本按单位置D排序，前ceil(20%n)占全部单位置D之和的比例；不是联合删除忠实性或人类情绪贡献。

来源：[doc/paper_figures/data/server_Q3_local.csv](../../doc/paper_figures/data/server_Q3_local.csv)。

## Q3_18_signed_local 验证集正负单位置影响

delta p=p原类别(x)-p原类别(删除后)。左面板以既有数值容差epsilon=6.5563e-6区分负向影响；右面板保留全部原始值，位置嵌套于样本，仅描述分布；横轴明确使用symlog，完整保留负值。

来源：[doc/paper_figures/data/server_Q3_local.csv](../../doc/paper_figures/data/server_Q3_local.csv)。

## Q3_19_equal_cost_granularity 排除成本不一致后的粒度敏感性

新增只读重汇总，不重训/重选/更改原报告。Equal cost只保留group_cost=point_cost的原配对记录；视频簇bootstrap1000次、样本加权95%区间。面板采用不同纵轴尺度，保留零线；子集过滤仍不能证明粒度因果优越。

来源：[doc/paper_figures/data/server_Q3_granularity.csv](../../doc/paper_figures/data/server_Q3_granularity.csv)。

## Q3_case_01 附件4样本01的三模态逐位置重要性

仅显示已观测位置，三角标记20% point集合；三模态独立纵轴，峰值超过中位数8倍时使用标明的symlog轴（线性阈值=2×中位数），保留峰值与小效应，禁止跨面板用柱高比较贡献。无观测V省略并在标题标出。所有位置为官方索引，不是秒数；单点D之和不等于联合D。

来源：`Q3/runs/special_v2/samples/0000/local.json`（原始归档路径，未随图册上传）；`Q3/runs/special_v2/samples/0000/mapping.json`（原始归档路径，未随图册上传）；`Q3/runs/special_v2/samples/0000/selections.jsonl`（原始归档路径，未随图册上传）。

## Q3_case_02 附件4样本02的三模态逐位置重要性

仅显示已观测位置，三角标记20% point集合；三模态独立纵轴，峰值超过中位数8倍时使用标明的symlog轴（线性阈值=2×中位数），保留峰值与小效应，禁止跨面板用柱高比较贡献。无观测V省略并在标题标出。所有位置为官方索引，不是秒数；单点D之和不等于联合D。

来源：`Q3/runs/special_v2/samples/0001/local.json`（原始归档路径，未随图册上传）；`Q3/runs/special_v2/samples/0001/mapping.json`（原始归档路径，未随图册上传）；`Q3/runs/special_v2/samples/0001/selections.jsonl`（原始归档路径，未随图册上传）。

## Q3_case_03 附件4样本03的三模态逐位置重要性

仅显示已观测位置，三角标记20% point集合；三模态独立纵轴，峰值超过中位数8倍时使用标明的symlog轴（线性阈值=2×中位数），保留峰值与小效应，禁止跨面板用柱高比较贡献。无观测V省略并在标题标出。所有位置为官方索引，不是秒数；单点D之和不等于联合D。

来源：`Q3/runs/special_v2/samples/0002/local.json`（原始归档路径，未随图册上传）；`Q3/runs/special_v2/samples/0002/mapping.json`（原始归档路径，未随图册上传）；`Q3/runs/special_v2/samples/0002/selections.jsonl`（原始归档路径，未随图册上传）。

## Q3_case_04 附件4样本04的三模态逐位置重要性

仅显示已观测位置，三角标记20% point集合；三模态独立纵轴，峰值超过中位数8倍时使用标明的symlog轴（线性阈值=2×中位数），保留峰值与小效应，禁止跨面板用柱高比较贡献。无观测V省略并在标题标出。所有位置为官方索引，不是秒数；单点D之和不等于联合D。

来源：`Q3/runs/special_v2/samples/0003/local.json`（原始归档路径，未随图册上传）；`Q3/runs/special_v2/samples/0003/mapping.json`（原始归档路径，未随图册上传）；`Q3/runs/special_v2/samples/0003/selections.jsonl`（原始归档路径，未随图册上传）。

## Q3_case_05 附件4样本05的三模态逐位置重要性

仅显示已观测位置，三角标记20% point集合；三模态独立纵轴，峰值超过中位数8倍时使用标明的symlog轴（线性阈值=2×中位数），保留峰值与小效应，禁止跨面板用柱高比较贡献。无观测V省略并在标题标出。所有位置为官方索引，不是秒数；单点D之和不等于联合D。

来源：`Q3/runs/special_v2/samples/0004/local.json`（原始归档路径，未随图册上传）；`Q3/runs/special_v2/samples/0004/mapping.json`（原始归档路径，未随图册上传）；`Q3/runs/special_v2/samples/0004/selections.jsonl`（原始归档路径，未随图册上传）。

## Q3_case_06 附件4样本06的三模态逐位置重要性

仅显示已观测位置，三角标记20% point集合；三模态独立纵轴，峰值超过中位数8倍时使用标明的symlog轴（线性阈值=2×中位数），保留峰值与小效应，禁止跨面板用柱高比较贡献。无观测V省略并在标题标出。所有位置为官方索引，不是秒数；单点D之和不等于联合D。

来源：`Q3/runs/special_v2/samples/0005/local.json`（原始归档路径，未随图册上传）；`Q3/runs/special_v2/samples/0005/mapping.json`（原始归档路径，未随图册上传）；`Q3/runs/special_v2/samples/0005/selections.jsonl`（原始归档路径，未随图册上传）。

## Q3_case_07 附件4样本07的三模态逐位置重要性

仅显示已观测位置，三角标记20% point集合；三模态独立纵轴，峰值超过中位数8倍时使用标明的symlog轴（线性阈值=2×中位数），保留峰值与小效应，禁止跨面板用柱高比较贡献。无观测V省略并在标题标出。所有位置为官方索引，不是秒数；单点D之和不等于联合D。

来源：`Q3/runs/special_v2/samples/0006/local.json`（原始归档路径，未随图册上传）；`Q3/runs/special_v2/samples/0006/mapping.json`（原始归档路径，未随图册上传）；`Q3/runs/special_v2/samples/0006/selections.jsonl`（原始归档路径，未随图册上传）。

## Q3_case_08 附件4样本08的三模态逐位置重要性

仅显示已观测位置，三角标记20% point集合；三模态独立纵轴，峰值超过中位数8倍时使用标明的symlog轴（线性阈值=2×中位数），保留峰值与小效应，禁止跨面板用柱高比较贡献。无观测V省略并在标题标出。所有位置为官方索引，不是秒数；单点D之和不等于联合D。

来源：`Q3/runs/special_v2/samples/0007/local.json`（原始归档路径，未随图册上传）；`Q3/runs/special_v2/samples/0007/mapping.json`（原始归档路径，未随图册上传）；`Q3/runs/special_v2/samples/0007/selections.jsonl`（原始归档路径，未随图册上传）。

## Q3_case_09 附件4样本09的三模态逐位置重要性

仅显示已观测位置，三角标记20% point集合；三模态独立纵轴，峰值超过中位数8倍时使用标明的symlog轴（线性阈值=2×中位数），保留峰值与小效应，禁止跨面板用柱高比较贡献。无观测V省略并在标题标出。所有位置为官方索引，不是秒数；单点D之和不等于联合D。

来源：`Q3/runs/special_v2/samples/0008/local.json`（原始归档路径，未随图册上传）；`Q3/runs/special_v2/samples/0008/mapping.json`（原始归档路径，未随图册上传）；`Q3/runs/special_v2/samples/0008/selections.jsonl`（原始归档路径，未随图册上传）。

## Q3_case_10 附件4样本10的三模态逐位置重要性

仅显示已观测位置，三角标记20% point集合；三模态独立纵轴，峰值超过中位数8倍时使用标明的symlog轴（线性阈值=2×中位数），保留峰值与小效应，禁止跨面板用柱高比较贡献。无观测V省略并在标题标出。所有位置为官方索引，不是秒数；单点D之和不等于联合D。

来源：`Q3/runs/special_v2/samples/0009/local.json`（原始归档路径，未随图册上传）；`Q3/runs/special_v2/samples/0009/mapping.json`（原始归档路径，未随图册上传）；`Q3/runs/special_v2/samples/0009/selections.jsonl`（原始归档路径，未随图册上传）。

## Q3_case_11 附件4样本11的三模态逐位置重要性

仅显示已观测位置，三角标记20% point集合；三模态独立纵轴，峰值超过中位数8倍时使用标明的symlog轴（线性阈值=2×中位数），保留峰值与小效应，禁止跨面板用柱高比较贡献。无观测V省略并在标题标出。所有位置为官方索引，不是秒数；单点D之和不等于联合D。

来源：`Q3/runs/special_v2/samples/0010/local.json`（原始归档路径，未随图册上传）；`Q3/runs/special_v2/samples/0010/mapping.json`（原始归档路径，未随图册上传）；`Q3/runs/special_v2/samples/0010/selections.jsonl`（原始归档路径，未随图册上传）。

## Q3_case_12 附件4样本12的三模态逐位置重要性

仅显示已观测位置，三角标记20% point集合；三模态独立纵轴，峰值超过中位数8倍时使用标明的symlog轴（线性阈值=2×中位数），保留峰值与小效应，禁止跨面板用柱高比较贡献。无观测V省略并在标题标出。所有位置为官方索引，不是秒数；单点D之和不等于联合D。

来源：`Q3/runs/special_v2/samples/0011/local.json`（原始归档路径，未随图册上传）；`Q3/runs/special_v2/samples/0011/mapping.json`（原始归档路径，未随图册上传）；`Q3/runs/special_v2/samples/0011/selections.jsonl`（原始归档路径，未随图册上传）。

## Q3_case_13 附件4样本13的三模态逐位置重要性

仅显示已观测位置，三角标记20% point集合；三模态独立纵轴，峰值超过中位数8倍时使用标明的symlog轴（线性阈值=2×中位数），保留峰值与小效应，禁止跨面板用柱高比较贡献。无观测V省略并在标题标出。所有位置为官方索引，不是秒数；单点D之和不等于联合D。

来源：`Q3/runs/special_v2/samples/0012/local.json`（原始归档路径，未随图册上传）；`Q3/runs/special_v2/samples/0012/mapping.json`（原始归档路径，未随图册上传）；`Q3/runs/special_v2/samples/0012/selections.jsonl`（原始归档路径，未随图册上传）。

## Q3_case_14 附件4样本14的三模态逐位置重要性

仅显示已观测位置，三角标记20% point集合；三模态独立纵轴，峰值超过中位数8倍时使用标明的symlog轴（线性阈值=2×中位数），保留峰值与小效应，禁止跨面板用柱高比较贡献。无观测V省略并在标题标出。所有位置为官方索引，不是秒数；单点D之和不等于联合D。

来源：`Q3/runs/special_v2/samples/0013/local.json`（原始归档路径，未随图册上传）；`Q3/runs/special_v2/samples/0013/mapping.json`（原始归档路径，未随图册上传）；`Q3/runs/special_v2/samples/0013/selections.jsonl`（原始归档路径，未随图册上传）。

## Q3_case_15 附件4样本15的三模态逐位置重要性

仅显示已观测位置，三角标记20% point集合；三模态独立纵轴，峰值超过中位数8倍时使用标明的symlog轴（线性阈值=2×中位数），保留峰值与小效应，禁止跨面板用柱高比较贡献。无观测V省略并在标题标出。所有位置为官方索引，不是秒数；单点D之和不等于联合D。

来源：`Q3/runs/special_v2/samples/0014/local.json`（原始归档路径，未随图册上传）；`Q3/runs/special_v2/samples/0014/mapping.json`（原始归档路径，未随图册上传）；`Q3/runs/special_v2/samples/0014/selections.jsonl`（原始归档路径，未随图册上传）。

## Q3_case_16 附件4样本16的三模态逐位置重要性

仅显示已观测位置，三角标记20% point集合；三模态独立纵轴，峰值超过中位数8倍时使用标明的symlog轴（线性阈值=2×中位数），保留峰值与小效应，禁止跨面板用柱高比较贡献。无观测V省略并在标题标出。所有位置为官方索引，不是秒数；单点D之和不等于联合D。

来源：`Q3/runs/special_v2/samples/0015/local.json`（原始归档路径，未随图册上传）；`Q3/runs/special_v2/samples/0015/mapping.json`（原始归档路径，未随图册上传）；`Q3/runs/special_v2/samples/0015/selections.jsonl`（原始归档路径，未随图册上传）。

## Q3_case_17 附件4样本17的三模态逐位置重要性

仅显示已观测位置，三角标记20% point集合；三模态独立纵轴，峰值超过中位数8倍时使用标明的symlog轴（线性阈值=2×中位数），保留峰值与小效应，禁止跨面板用柱高比较贡献。无观测V省略并在标题标出。所有位置为官方索引，不是秒数；单点D之和不等于联合D。

来源：`Q3/runs/special_v2/samples/0016/local.json`（原始归档路径，未随图册上传）；`Q3/runs/special_v2/samples/0016/mapping.json`（原始归档路径，未随图册上传）；`Q3/runs/special_v2/samples/0016/selections.jsonl`（原始归档路径，未随图册上传）。

## Q3_case_18 附件4样本18的三模态逐位置重要性

仅显示已观测位置，三角标记20% point集合；三模态独立纵轴，峰值超过中位数8倍时使用标明的symlog轴（线性阈值=2×中位数），保留峰值与小效应，禁止跨面板用柱高比较贡献。无观测V省略并在标题标出。所有位置为官方索引，不是秒数；单点D之和不等于联合D。

来源：`Q3/runs/special_v2/samples/0017/local.json`（原始归档路径，未随图册上传）；`Q3/runs/special_v2/samples/0017/mapping.json`（原始归档路径，未随图册上传）；`Q3/runs/special_v2/samples/0017/selections.jsonl`（原始归档路径，未随图册上传）。

## Q3_case_19 附件4样本19的三模态逐位置重要性

仅显示已观测位置，三角标记20% point集合；三模态独立纵轴，峰值超过中位数8倍时使用标明的symlog轴（线性阈值=2×中位数），保留峰值与小效应，禁止跨面板用柱高比较贡献。无观测V省略并在标题标出。所有位置为官方索引，不是秒数；单点D之和不等于联合D。

来源：`Q3/runs/special_v2/samples/0018/local.json`（原始归档路径，未随图册上传）；`Q3/runs/special_v2/samples/0018/mapping.json`（原始归档路径，未随图册上传）；`Q3/runs/special_v2/samples/0018/selections.jsonl`（原始归档路径，未随图册上传）。

## Q3_case_20 附件4样本20的三模态逐位置重要性

仅显示已观测位置，三角标记20% point集合；三模态独立纵轴，峰值超过中位数8倍时使用标明的symlog轴（线性阈值=2×中位数），保留峰值与小效应，禁止跨面板用柱高比较贡献。无观测V省略并在标题标出。所有位置为官方索引，不是秒数；单点D之和不等于联合D。

来源：`Q3/runs/special_v2/samples/0019/local.json`（原始归档路径，未随图册上传）；`Q3/runs/special_v2/samples/0019/mapping.json`（原始归档路径，未随图册上传）；`Q3/runs/special_v2/samples/0019/selections.jsonl`（原始归档路径，未随图册上传）。

服务器补查新增Q1_09–Q1_10与Q3_17–Q3_19；来源核对、同成本子集结论及缺失记录见[服务器补查与新增图说明](服务器补查与新增图说明.md)。

## 复算

`python scripts/build_paper_figures.py`

依赖：Python、numpy、pandas、matplotlib、seaborn、Pillow。源码相对自身定位仓库；全量重绘需要本地已有Q1/Q2/Q3运行数据，其中Q3专项samples原始归档未随图册上传，须先恢复到所列路径。图册内CSV及现成图可直接使用，服务器新增5组可单独从已提交CSV重绘。不会重新训练或更改任何实验值。
