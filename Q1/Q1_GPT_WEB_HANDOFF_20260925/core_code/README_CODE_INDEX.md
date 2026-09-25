# 核心代码索引

| 文件 | 作用 | 建议重点审查 |
|---|---|---|
| `review_annotations.py` | 多视觉片段、锚点词、分歧分析、裁决合并、金标准锁定、多片段导出 | 区间匹配、重叠策略、裁决完整性、总体结论与片段一致性 |
| `review_experiment.py` | 100 条双盲总体表、κ/α、分组 bootstrap、第三方裁决、生产配置转换 | 盲化、强制裁决条件、bootstrap 单位、状态转换 |
| `review_evaluation.py` | CTC 告警 confusion、F1、AUPRC、漏报、false release | unresolved 的排除方式、小样本 CI、AUPRC 边界情况 |
| `review_risk.py` | CTC/WhisperX 百分位风险排序 | 缺失 WhisperX 分量处理、排序与评价污染 |
| `quality.py` | 自动质量问题与 `vision_no_face` 识别 | 自然缺失与提取失败边界 |
| `runner.py` | 实际流水线阶段组织与审核写回 | `audit`、`vision --reuse-raw-csv`、状态传播 |
| `review_annotation_workflow.py` | 多片段审核 CLI | 参数门禁和失败恢复 |
| `review_workflow.py` | 总体双盲审核 CLI | 分析、裁决、导出顺序 |
| `lock_review_submissions.py` | 六份人工提交的只读副本和 SHA256 | 防覆盖性、是否需要额外 provenance |
| `evaluate_review_results.py` | 音文评价 CLI | 输入一致性和结果声明边界 |
| `build_review_risk.py` | 风险表 CLI | 是否保存完整配置和模型元数据 |

测试目录只包含与本研究问题直接相关的回归测试。当前完整项目测试为 `57 passed`，但测试通过不代表人工金标准已经建立。
