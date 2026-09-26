# Q1 最终生产冻结与实验报告

## 1. 结论

Q1 已按固定自动策略冻结。最终产物保留100条样本，以词级对齐NPZ为主产物、50-bin NPZ为辅助表示。未使用情感标签、人工Gold或新模型，未执行clip交换。

- 最终状态：RESOLVED_VALID=57、RESOLVED_PARTIAL=19、RESOLVED_CONFLICT=5、RESOLVED_MISSING=4、UNRESOLVED=15。
- paired-use：Formal 49 → Final 68。Stage 3 相对 Stage 2 没有继续增加释放样本；主要变化是已验证身份轨迹内的视觉时间覆盖。
- 15条证据不足样本全部保留，危险模态保持mask；不再运行新的恢复算法。

## 2. 方法演进

1. Formal：完成BERT、CTC、openSMILE、OpenFace和50-bin正式基线，paired-use=49。
2. Stage 1：使用WhisperX、ArcFace、TalkNet与AV-sync生成独立自动证据，不直接写回。
3. Stage 2：拆分semantic pairing与temporal alignment，使用official-text forced alignment和局部TalkNet视觉mask。
4. Stage 3：TalkNet只建立speaker anchor；ArcFace在同身份、无竞争说话者且轨迹间隔受限时传播视觉身份。
5. Stage 4：不再优化，只按固定状态策略冻结Stage 3候选。

## 3. Coverage术语与结果

`raw_observed_coverage`表示正式基线中工具可提取/保留的观测；`verified_usable_coverage`表示经过语义、时间和身份质量控制后允许跨模态使用的观测。二者都不是accuracy。
- Stage 3 Vision verified usable=76.04%，Trimodal verified usable=52.14%。
- Audio从Formal约99.50%降至Stage 3的78.28%，原因是语义冲突/未决位置被主动mask，不是音频提取器性能下降。
- paired-use从49增至68发生在Stage 2；Stage 3主要把原37条视觉风险的平均可用视觉比例从33.55%提高到51.44%。

## 4. 视觉连续性参数

固定 `max_identity_gap_seconds=1.0`。敏感性：0.5s→50.25%，1.0s→51.44%，1.5s→51.44%。样本级状态在该范围稳定；1.0s是预先固定主配置，不宣称为最优参数。

## 5. Mapping冻结

所有100条继续使用官方sample/video/clip mapping。唯一MAPPING_STRONG `-iRBcNs9oI8$_$6 → -iRBcNs9oI8$_$9` 仅作为诊断证据保留，最终动作是RESOLVED_CONFLICT + SAFE_MASK，没有实施mapping correction。

## 6. 证据边界

没有人工Gold，因此文中的verified只表示满足冻结自动规则，不表示人工确认或真实准确率。RESOLVED_CONFLICT表示安全mask方案明确，不表示冲突被修复；RESOLVED_MISSING不插值、不借用人物或片段。

## 7. 论文附件材料

- 典型样本：`-a55Q6RWvTA$_$3`，按无标签固定规则选择，展示8个连续词及source mapping。
- 异常示例：`-NFrJFQijFE$_$1`，只展示mask和可追踪性，不宣称修复。

## 8. 工程验收

- 最终validation：ok=true，errors=[]。
- pytest：99 passed；pip check：No broken requirements found.。
- 冻结输入：772项，ok=true。
- 最小提交包：/home/jqy/shumo/q1_features/runs/q1_final_20260926/Q1_SUBMISSION_MINIMAL_20260926.zip，5947444 bytes。

## 9. 最终停止

通过Q1_FINAL_ACCEPTANCE后，Q1关闭；不再针对15条UNRESOLVED追加模型、阈值或自动恢复算法。
