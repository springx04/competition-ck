# Q1 Stage 3：剩余异常收敛与视觉连续性修复报告

> 本阶段仅生成自动候选，不使用人工审核或情感标签，不交换 clip，不修改正式配置、正式 NPZ 或 Stage 2 产物。

## 1. 结论

Stage 3 进一步恢复了视觉静默间隔，但仍是**部分解决**而非生产写回结论。原37条视觉身份风险从 Stage 2 的8条 fully verified提升到16条；Stage 3 分布为 {'UNRESOLVED': 10, 'PARTIALLY_VERIFIED': 6, 'FULLY_VERIFIED': 16, 'AMBIGUOUS': 5}。4条映射疑点中仅1条达到 MAPPING_STRONG，仍只形成显式候选。

## 2. A：4条 clip 映射疑点

- `-iRBcNs9oI8$_$3` → `-iRBcNs9oI8$_$7`：MAPPING_WEAK；Whisper token 0.091→0.211，CTC token 0.091→0.211；Hungarian(W-token/C-token/W-char/C-char)=`-iRBcNs9oI8$_$8`/`-iRBcNs9oI8$_$8`/`-iRBcNs9oI8$_$7`/`-iRBcNs9oI8$_$7`，4指标一致率=0.50，全局rank 28→2。
- `-iRBcNs9oI8$_$6` → `-iRBcNs9oI8$_$9`：MAPPING_STRONG；Whisper token 0.000→0.500，CTC token 0.000→0.500；Hungarian(W-token/C-token/W-char/C-char)=`-iRBcNs9oI8$_$9`/`-iRBcNs9oI8$_$9`/`-iRBcNs9oI8$_$9`/`-iRBcNs9oI8$_$9`，4指标一致率=1.00，全局rank 48→1。
- `-mJ2ud6oKI8$_$1` → `-mJ2ud6oKI8$_$8`：MAPPING_UNRESOLVED；Whisper token 0.000→0.154，CTC token 0.000→0.148；Hungarian(W-token/C-token/W-char/C-char)=`-mJ2ud6oKI8$_$8`/`-mJ2ud6oKI8$_$8`/`-mJ2ud6oKI8$_$9`/`-mJ2ud6oKI8$_$6`，4指标一致率=0.50，全局rank 98→31。
- `-ri04Z7vwnc$_$2` → `-ri04Z7vwnc$_$5`：MAPPING_UNRESOLVED；Whisper token 0.000→0.143，CTC token 0.000→0.129；Hungarian(W-token/C-token/W-char/C-char)=`-ri04Z7vwnc$_$5`/`-ri04Z7vwnc$_$5`/`-ri04Z7vwnc$_$5`/`-ri04Z7vwnc$_$5`，4指标一致率=1.00，全局rank 95→2。

汇总：{'MAPPING_WEAK': 1, 'MAPPING_STRONG': 1, 'MAPPING_UNRESOLVED': 2}。没有自动修改 manifest，也没有交换音视频。

## 3. B：视觉连续性修复

Stage2 样本状态={'unresolved': 15, 'partially_verified': 14, 'fully_verified': 8}；Stage3 样本状态={'UNRESOLVED': 10, 'PARTIALLY_VERIFIED': 6, 'FULLY_VERIFIED': 16, 'AMBIGUOUS': 5}。
原37条样本平均 Stage2 可用视觉比例=0.3355；Stage3直接TalkNet=0.3355，ArcFace连续性新增=0.1789，总可用=0.5144。
绝对时间：TalkNet直接确认=100.233s，连续性恢复=54.250s，竞争身份/歧义mask=19.950s，missing=18.631s，unresolved=91.240s。
歧义=0.0525，缺失=0.0951，未决=0.3381。连续性只在同一ArcFace组件、1.0秒有界间隔且无竞争活跃身份时传播。

敏感性（最大连续间隔0.5/1.0/1.5秒）：

- 0.5s：fully=16，partial=6，ambiguous=5，unresolved=10，usable=0.5025
- 1.0s：fully=16，partial=6，ambiguous=5，unresolved=10，usable=0.5144
- 1.5s：fully=16，partial=6，ambiguous=5，unresolved=10，usable=0.5144

## 4. C：最终自动候选状态

100条分布：{'RESOLVED_VALID': 57, 'UNRESOLVED': 15, 'RESOLVED_PARTIAL': 19, 'RESOLVED_MISSING': 4, 'RESOLVED_CONFLICT': 5}。真正未决 15 条：

- `-9y-fZ3swSY$_$8`：target_visual_identity_remains_unresolved（semantic=MATCH_STRONG，visual=UNRESOLVED，mapping=NOT_SUSPECTED）
- `-HwX2H8Z4hY$_$2`：target_visual_identity_remains_unresolved（semantic=CONFLICT，visual=UNRESOLVED，mapping=NOT_SUSPECTED）
- `-HwX2H8Z4hY$_$5`：target_visual_identity_remains_unresolved（semantic=MATCH_STRONG，visual=UNRESOLVED，mapping=NOT_SUSPECTED）
- `-HwX2H8Z4hY$_$6`：target_visual_identity_remains_unresolved（semantic=MATCH_STRONG，visual=UNRESOLVED，mapping=NOT_SUSPECTED）
- `-hnBHBN8p5A$_$6`：target_visual_identity_remains_unresolved（semantic=CONFLICT，visual=UNRESOLVED，mapping=NOT_SUSPECTED）
- `-hnBHBN8p5A$_$7`：target_visual_identity_remains_unresolved（semantic=CONFLICT，visual=UNRESOLVED，mapping=NOT_SUSPECTED）
- `-iRBcNs9oI8$_$3`：clip_mapping_evidence_is_not_stable_enough（semantic=UNRESOLVED，visual=FULLY_VERIFIED，mapping=MAPPING_WEAK）
- `-iRBcNs9oI8$_$9`：target_visual_identity_remains_unresolved（semantic=CONFLICT，visual=UNRESOLVED，mapping=NOT_SUSPECTED）
- `-mJ2ud6oKI8$_$6`：semantic_pairing_remains_unresolved（semantic=UNRESOLVED，visual=FULLY_VERIFIED，mapping=NOT_SUSPECTED）
- `-mJ2ud6oKI8$_$8`：semantic_pairing_remains_unresolved（semantic=UNRESOLVED，visual=FULLY_VERIFIED，mapping=NOT_SUSPECTED）
- `-mJ2ud6oKI8$_$9`：target_visual_identity_remains_unresolved（semantic=CONFLICT，visual=UNRESOLVED，mapping=NOT_SUSPECTED）
- `-ri04Z7vwnc$_$2`：clip_mapping_evidence_is_not_stable_enough（semantic=UNRESOLVED，visual=FULLY_VERIFIED，mapping=MAPPING_UNRESOLVED）
- `-s9qJ7ATP7w$_$0`：target_visual_identity_remains_unresolved（semantic=MATCH_STRONG，visual=UNRESOLVED，mapping=NOT_SUSPECTED）
- `-s9qJ7ATP7w$_$7`：target_visual_identity_remains_unresolved（semantic=MATCH_STRONG，visual=UNRESOLVED，mapping=NOT_SUSPECTED）
- `-vxjVxOeScU$_$4`：semantic_pairing_remains_unresolved（semantic=UNRESOLVED，visual=FULLY_VERIFIED，mapping=NOT_SUSPECTED）

## 5. D：Formal / Stage2 / Stage3 覆盖率

| 分组 | 版本 | Text | Audio | Vision | Trimodal |
|---|---|---:|---:|---:|---:|
| all100 | formal | 62.56% | 99.50% | 57.46% | 39.06% |
| all100 | stage2 | 62.56% | 78.28% | 70.10% | 49.16% |
| all100 | stage3 | 62.56% | 78.28% | 76.04% | 52.14% |
| original_51_isolated | formal | 45.25% | 99.61% | 18.24% | 0.00% |
| original_51_isolated | stage2 | 45.25% | 62.51% | 43.02% | 22.75% |
| original_51_isolated | stage3 | 45.25% | 62.51% | 54.67% | 28.59% |
| original_37_visual_risk | formal | 62.38% | 99.51% | 0.00% | 0.00% |
| original_37_visual_risk | stage2 | 62.38% | 82.54% | 34.16% | 31.35% |
| original_37_visual_risk | stage3 | 62.38% | 82.54% | 50.22% | 39.41% |
| original_23_text_audio_risk | formal | 0.00% | 99.91% | 40.43% | 0.00% |
| original_23_text_audio_risk | stage2 | 0.00% | 17.65% | 41.91% | 0.00% |
| original_23_text_audio_risk | stage3 | 0.00% | 17.65% | 47.13% | 0.00% |
| semantic_conflict | formal | 0.00% | 99.83% | 24.83% | 0.00% |
| semantic_conflict | stage2 | 0.00% | 0.00% | 24.83% | 0.00% |
| semantic_conflict | stage3 | 0.00% | 0.00% | 24.83% | 0.00% |
| semantic_unresolved | formal | 12.00% | 100.00% | 76.00% | 12.00% |
| semantic_unresolved | stage2 | 12.00% | 0.00% | 76.00% | 0.00% |
| semantic_unresolved | stage3 | 12.00% | 0.00% | 76.00% | 0.00% |
| mapping_suspected | formal | 0.00% | 100.00% | 58.00% | 0.00% |
| mapping_suspected | stage2 | 0.00% | 0.00% | 58.00% | 0.00% |
| mapping_suspected | stage3 | 0.00% | 0.00% | 58.00% | 0.00% |

paired-use计数：formal=49，Stage2候选=68，Stage3候选=68。候选计数仅用于覆盖分析。
semantic conflict 组的 Stage3 audio observed=0.00%，冲突词级audio mask保持为0。

## 6. 候选特征产物

- 词级候选：`q1_word_aligned_stage3_candidate.npz`，1932词，验证 ok=true，视觉源引用2808条。
- 50-bin候选：`q1_compact50_stage3_candidate.npz`，验证 ok=true。
- 无有效时间或不安全证据保持 mask=0；没有伪造视觉或音频时间。

## 7. E：工程验收与冻结保护

- pytest：93 passed。
- pip check：No broken requirements found.。
- 正式 validation：ok=true，errors=[]。
- 冻结基线 11 项复核：ok=true，changed=0。
- 词级验证=True；50-bin验证=True；可追踪Stage3视觉源引用=2808。
- `production_writeback_proposal_stage3.csv` 已生成100行；writeback_executed 全部为0。
- candidate_only=true；production_write=false。

## 8. 停止边界

Stage 3 到此停止。所有 proposal 均未获批准、未执行写回；MAPPING_STRONG 仍需显式复核，UNRESOLVED 保持隔离。
