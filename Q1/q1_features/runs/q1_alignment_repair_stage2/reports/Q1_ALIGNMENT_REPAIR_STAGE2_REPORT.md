# Q1 ALIGNMENT REPAIR STAGE 2 实验报告

**实验性质：** 自动候选定向修复；无人工审核、无情感标签、无正式写回。
**正式基线：** `runs/q1_full_20260924`（保持冻结）
**候选目录：** `runs/q1_alignment_repair_stage2`

## 1. 总结：是否解决了对不齐问题

结论是**部分解决，尚未完全解决**。本阶段消除了旧 CTC diagnostic WER 的一票否决，完成了官方文本强制对齐，并让 TalkNet 真正参与局部 active-speaker 判定；但自动证据仍不能安全解决所有语义冲突和视觉歧义。

- 100条数据链全部通过；4条同-video clip错位候选仅标记，未交换。
- 原23条音文风险中，5条被独立语义证据支持（3条 MATCH_STRONG、2条 MATCH_WEAK）；其中1条达到严格时间 VERIFIED、4条为 PARTIAL。其余12条为 CONFLICT、6条为 UNRESOLVED，不应强制生成词级音频对应。
- 原37条视觉身份风险中，8条达到 fully_verified、14条 partially_verified、15条 unresolved；不再把局部歧义扩散为整段视觉失效。
- 原51条隔离样本的视觉观测率由 18.24% 提升至 43.02%（+24.78个百分点），三模态观测率由 0.00% 提升至 22.75%。
- 全100条三模态观测率由 39.06% 提升至 49.16%（+10.10个百分点）。
- 候选 paired-use 从正式49条增至68条（+19），但这是**候选覆盖指标，不是生产验收结论**。
- 按“semantic UNRESOLVED、temporal FAILED 或 visual unresolved”定义，仍有38条未决样本。

## 2. A：原19条 TA_CONFLICT 数据链与原因

- data-chain错误：0
- clip错位候选：4
- 两个ASR一致但均与Official冲突：8
- ASR自身不稳定：5
- 其他未决：2
- 数据链错误为0，说明主要矛盾不在文件串线；4条 mapping_suspected 仍需后续证据确认，本阶段没有自动交换 clip。

## 3. B：原23条音文风险与官方文本强制对齐

- semantic：MATCH_STRONG=3，MATCH_WEAK=2，CONFLICT=12，UNRESOLVED=6。
- temporal：VERIFIED=1，PARTIAL=4，FAILED=0，NOT_APPLICABLE=18。
- 全100条 semantic：{'MATCH_STRONG': 78, 'CONFLICT': 12, 'UNRESOLVED': 7, 'MATCH_WEAK': 3}。
- 全100条 temporal：{'VERIFIED': 71, 'PARTIAL': 10, 'NOT_APPLICABLE': 19}。
- official-text forced alignment：100/100成功，技术完成率=100.00%；失败0条。100%技术成功不等价于100%语义正确，因此CONFLICT/UNRESOLVED样本仍禁止使用生成的词时间。
- 固定主时间阈值为 common coverage≥0.80、median≤0.30s、P90≤0.60s；0.20/0.30/0.40与0.40/0.60/0.80的组合仅写入 `alignment_threshold_sensitivity.csv`，未据此调参。
- CTC diagnostic WER只保留为诊断列，不参与semantic一票否决。

重点样本：

- `-HwX2H8Z4hY$_$9`：semantic=MATCH_STRONG，temporal=VERIFIED，common-word=1.0000，midpoint median=0.2223s，P90=0.3183s。
- `-aqamKhZ1Ec$_$0`：semantic=MATCH_STRONG，temporal=PARTIAL，common-word=0.9286，midpoint median=0.3101s，P90=1.0951s。

## 4. C：原37条视觉身份风险

- ArcFace固定阈值0.60的cluster分布：1 cluster=15条，2 cluster=14条，3 cluster=3条，4 cluster=3条，5 cluster=1条，9 cluster=1条。
- Stage1（样本级证据）分布：{'V_UNCERTAIN': 27, 'V_PARTIAL': 8, 'V_STRONG': 2}。
- Stage2（TalkNet 0.5s窗口/0.25s步长局部判定）：fully_verified=8，partially_verified=14，unresolved=15。
- 时长比例：verified=0.3526，ambiguous/no-active=0.5916，missing=0.0559。
- ArcFace只连接身份，TalkNet logit≥0决定窗口 active cluster；AV-sync只增加 support/neutral/conflict 质量标记，不再整条否决。
- 4条原 no-face 样本保持 missing，未伪造视觉特征。

| Stage1证据 | Stage2局部状态 | 样本数 |
|---|---|---:|
| V_PARTIAL | fully_verified | 2 |
| V_PARTIAL | partially_verified | 2 |
| V_PARTIAL | unresolved | 4 |
| V_STRONG | fully_verified | 1 |
| V_STRONG | partially_verified | 1 |
| V_UNCERTAIN | fully_verified | 5 |
| V_UNCERTAIN | partially_verified | 11 |
| V_UNCERTAIN | unresolved | 11 |

## 5. D：全100条新旧覆盖对比

| 分组 | 版本 | Text观测 | Audio观测 | Vision观测 | 三模态观测 | Text coverage | Audio coverage | Vision coverage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| all100 | formal | 62.56% | 99.50% | 57.46% | 39.06% | 58.85% | 98.57% | 47.69% |
| all100 | repair_candidate | 62.56% | 78.28% | 70.10% | 49.16% | 58.85% | 75.69% | 57.67% |
| original_49_normal | formal | 80.57% | 99.39% | 98.29% | 79.71% | 75.35% | 98.47% | 81.15% |
| original_49_normal | repair_candidate | 80.57% | 94.69% | 98.29% | 76.65% | 75.35% | 91.33% | 81.15% |
| original_51_isolated | formal | 45.25% | 99.61% | 18.24% | 0.00% | 43.00% | 98.66% | 15.54% |
| original_51_isolated | repair_candidate | 45.25% | 62.51% | 43.02% | 22.75% | 43.00% | 60.67% | 35.10% |
| original_19_semantic_conflict | formal | 0.00% | 99.89% | 38.42% | 0.00% | 0.00% | 98.99% | 33.00% |
| original_19_semantic_conflict | repair_candidate | 0.00% | 4.32% | 39.16% | 0.00% | 0.00% | 3.65% | 33.43% |
| original_37_visual_risk | formal | 62.38% | 99.51% | 0.00% | 0.00% | 59.27% | 98.54% | 0.00% |
| original_37_visual_risk | repair_candidate | 62.38% | 82.54% | 34.16% | 31.35% | 59.27% | 80.69% | 26.96% |

候选音频覆盖下降是有意的安全结果：CONFLICT/UNRESOLVED不再沿用看似完整但语义不可信的音频时间；视觉和三模态覆盖的上升来自冻结 OpenFace 原始行上的局部 active-speaker mask，而不是补造特征。

未决原因：

- `semantic:independent_semantic_evidence_is_inconclusive`：3条
- `semantic:within_video_clip_mapping_suspected`：4条
- `visual:no_unique_TalkNet_active_identity_window`：34条

## 6. 词级与50-bin候选产物

- `q1_word_aligned_repair_candidate.npz`：1932词；text=1932、audio=1700、vision=1351个词位置可观测；验证 `ok=true`。
- `q1_compact50_repair_candidate.npz`：candidate paired-use=68；验证 `ok=true`。
- 原37条视觉风险的22维候选特征来自已冻结 `openface/features.csv`，episode按既有规则重建，并在 `raw_vision_candidate_sources.jsonl` 保存 sample/帧/face/episode/cluster/source_id 映射。正式安全隔离后的零张量未被误当作原始证据。
- semantic CONFLICT/UNRESOLVED 的词保留文本，start/end置0且audio mask=0；没有伪造词级音频对齐。

## 7. MFA

- 状态：`unavailable_optional_fallback`。No locked MFA environment, English acoustic model, and dictionary are installed; Stage 2 official-text alignment completed and MFA remains non-blocking.
- 候选fallback样本数：8。MFA未运行，不阻塞本阶段。

## 8. E：工程验证与基线保护

- pytest：78 passed in 0.29s。
- pip check：No broken requirements found.。
- 正式 validation.json：`ok=true`，errors=[]。
- 词级候选 validate：`ok=true`；50-bin候选 validate：`ok=true`。
- 正式基线610项（包括缺失态）SHA256/存在性检查：`ok=true`，changed=0。
- 候选写回：false；正式配置、正式NPZ、原数据均未覆盖。

## 9. 结论边界与下一步

自动定向修复已经证明：旧CTC高WER不能等同于音文错配；局部TalkNet判定能恢复部分视觉和三模态覆盖。但12条明确语义冲突、7条语义未决、15条原视觉风险未决仍不能被安全自动释放。后续应优先核查4条clip错位候选、对PARTIAL时间样本做MFA第三方对齐（若环境可锁定），并对剩余视觉歧义保留局部mask；在没有新独立证据前不应写回正式Q1。

本阶段到此停止：不使用情感标签、不进行人工确认、不自动交换clip、不重新训练、不覆盖正式CTC/视觉配置或`q1_compact50.npz`。
