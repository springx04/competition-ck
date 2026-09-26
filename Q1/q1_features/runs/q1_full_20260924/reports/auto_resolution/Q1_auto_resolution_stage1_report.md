# Q1 自动多证据核验阶段一报告

> 本报告仅描述候选证据，不构成人工确认，不修改正式 pairing 配置或 `q1_compact50.npz`。

## 1. 风险集合门控

- 51 条分组：`{"identity_only": 28, "identity_and_text_audio": 9, "vision_no_face_and_text_audio": 4, "text_audio_only_single_episode": 10}`。
- 分组与预注册的 28/10/9/4 完全一致后才继续执行。

## 2. 文本—音频证据

- WhisperX 完成：100/100。
- 原 23 条音文告警的证据等级：`{"TA_STRONG": 0, "TA_SEMANTIC_ONLY": 4, "TA_CONFLICT": 19}`。
- 原 49 条 paired_use=true 样本：`{"TA_STRONG": 47, "TA_SEMANTIC_ONLY": 1, "TA_CONFLICT": 1}`；新增自动风险候选 2 条：`["-I_e4mIh0yE$_$3", "-vxjVxOeScU$_$4"]`。
- 新增风险仅写入 `new_text_audio_risk_candidates.csv`，没有修改正式 paired_use。
- CTC/WhisperX 共同词 midpoint 差：n=89，median=0.1940763888888888 s，P90=0.30153878562071945 s。

## 3. 视觉身份与视听同步

- 37 条身份风险样本在 ArcFace 0.60 下的 cluster 数分布：`{"1": 15, "2": 14, "3": 3, "4": 3, "5": 1, "9": 1}`。
- 51 条候选的视觉等级：`{"V_UNCERTAIN": 27, "V_PARTIAL": 8, "V_MISSING": 4, "V_STRONG": 12}`。
- AV-sync real：n=118，P25/median/P75=0.05156818786663622/0.17854709119155487/0.31017727236535486；错位负对照：n=3041，P25/median/P75=0.04348198548556362/0.13994401633362205/0.2581033856662031；median 差=0.03860307485793282。
- 每个 cluster 相对自身错位分布的经验 percentile：median=0.5769230769230769，>=0.80 有 38，>=0.95 有 17。
- TalkNet：completed_with_explicit_short_episode_gaps；episode 165/191 可推理，逐帧 logit 19067 条；未使用不明第三方权重。

## 4. MFA

- 状态：skipped_optional_high_cost。原因：No MFA executable, English acoustic model, or pronunciation dictionary is installed. Preparing an isolated MFA environment and locked models is materially costly; the user explicitly allows MFA to be skipped when preparation cost is high.

## 5. 51 条最终候选

- trimodal：`{"uncertain": 39, "partial": 10, "verified": 2}`。
- strong candidate（trimodal verified）=2，partial=10，uncertain=39。
- 分组结果：`{"identity_and_text_audio": {"uncertain": 8, "partial": 1}, "identity_only": {"uncertain": 19, "partial": 7, "verified": 2}, "text_audio_only_single_episode": {"partial": 2, "uncertain": 8}, "vision_no_face_and_text_audio": {"uncertain": 4}}`。
- 逐条表：`auto_resolution_candidates.csv`。
- 可用/应 mask 的视觉候选区间：`visual_candidate_segments.csv`。

## 6. 词级候选包

- 验证：`ok=True`，words=1932，aligned=1793。
- 无有效 CTC 时间的词仍保留文本、mask 和失败原因，未伪造音视频时间。

## 7. 边界与限制

- 未使用情感标签，未训练或按这 100 条结果优化阈值。
- `TA_CONFLICT` 表示独立证据未能共同建立官方配对，不等同于已经证明错配。
- `V_STRONG/V_PARTIAL` 仍是自动候选，不是人工 Gold。
- 本阶段没有生产写回；下一轮应审核候选证据后再决定是否更新正式结果。
