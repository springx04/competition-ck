# Q2 最终实验与交付报告

文档日期：2026-09-26。本文是论文结果、复现实验和审计边界的总入口。`Q2_最终方法.md` 只讲最终模型如何计算；[最终结果表格](docs/最终结果表格.md) 保存完整逐场景表；[实验探索全记录](docs/实验探索全记录.md) 保存 73 个 trial、58 条 test evaluation 的逐条 ledger 解释。本文不把历史方案的结果写成最终模型结果。

## 1. 结论先行

最终交付为单个 `late_attn_tune` 学生：`google/bert_uncased_L-8_H-256_A-4` 的 8 个 Transformer 层全部微调，embedding 冻结，使用当前受损输入的 CLS context，三模态分别做观测掩码注意力池化，再做 387→128 晚融合，同时输出三分类和连续分数。部署 seed=1111 固定；epoch 由 valid 选择；seed=1112/1113 仅作固定配置复验。最终实际导出包重载后的结果如下。

| split | 输入条件 | Accuracy | Macro-F1 | Weighted-F1 | MAE | Pearson |
|---|---|---:|---:|---:|---:|---:|
| valid | clean | 0.6154 | 0.577807 | 0.6057 | 0.6415 | 0.6082 |
| valid | 72 个局部缺失场景等权均值 | 0.5796 | 0.546551 | 0.5742 | 0.6666 | 0.5493 |
| test | clean | 0.6740 | **0.618618** | 0.6635 | 0.6841 | 0.6081 |
| test | 72 个局部缺失场景等权均值 | 0.6267 | **0.575404** | 0.6209 | 0.7215 | 0.5498 |

因此，完整输入 test Macro-F1 超过 0.60；72 场景缺失平均为 0.575404，尚未达到 0.60。文本缺失最敏感：test 单模态 T/A/V 的 Macro-F1 为 0.5403/0.6099/0.6154，名义跨度 0.2/0.4/0.6/0.8 为 0.6017/0.5902/0.5696/0.5401，front/middle/rear 为 0.5805/0.5792/0.5665。中性类是主要瓶颈：test clean 三类 F1 为 0.6616/0.4260/0.7682，72 场景均值为 0.5969/0.3971/0.7322。

必须保留一个实验流程限制：后期架构方向曾参考 test 结果，最终选择元数据明确 `independent_holdout=false`。所以本报告把 test 数值称为“已观察 test 后的自适应探索结果”，不称为严格 valid-only 的独立泛化确认。模型没有用 test 标签训练、校准或逐样本改标签，但该事实不能消除 test 被用于方向选择的影响。

## 2. 结果、数据和文件的对应关系

| 论文或验收内容 | 直接证据 | 本报告给出的解释 |
|---|---|---|
| 数据规模、类别、ID隔离、形状 | `Q2/results/audit/data_inventory.json` | train/valid/test=3395/728/727；三划分 ID/video_id 无交集；A/V 维度=74/35。 |
| tokenizer 和输入重建 | `Q2/results/audit/tokenizer_compatibility.md`、`tokenizer_compatibility.csv` | 3395 条 train 均可由同一 WordPiece tokenizer 的 standard50 规则重建；不能由此证明物理时间对齐。 |
| 最终选择和协议偏离 | `Q2/results/final/selection.json`、`Q2/configs/final_selection.json` | 单模型、seed1111、无 class bias；epoch10由valid主缺失指标选出；architecture adoption来自已观察test。 |
| valid/test完整指标 | `Q2/results/final/{valid,test}/metrics.json`、`summary.json` | 本报告第3、4节的 clean、72场景、per-class和压力测试解释。 |
| 每个局部缺失场景 | `Q2/results/final/{valid,test}/metrics_per_scenario.csv` | 6种模态组合×3位置×4跨度的 72 行全表见 `最终结果表格.md`。 |
| 配对、实际删除率、双模态 | 同目录 `paired_conditions.csv`、`matched_deletion_rates.csv`、`dual_both_damaged.csv`、`deletion_rates.csv` | 共同受损样本和实际删除率用于避免把名义ρ误写成真实删除比例。 |
| 附件3全量预测与输入状态 | `Q2/results/final/q2_predictions_aligned.csv`、`q2_special_state.csv`、`Q2/results/audit/attachment3_availability.json` | 本报告第5节逐行给出 30 条 prediction 和当前观测计数；无标签，不计算准确率。 |
| 训练历史、消融和失败方案 | `Q2/results/experiment_ledger.json`、`实验探索全记录.md`、`Q2/results/audit/*_summary.json` | 本报告第6节按假设归组，区分观察与推断，不遗漏负结果。 |
| 推理包、重载一致性、资源 | `Q2/results/final/export.json`、`offline_verification.json`、`inference_resources.json`、`package_sizes.json` | 本报告第7节给出可复现入口、数值差异、速度和包体边界。 |

## 3. 数据、任务和最终方法摘要

Q2 只使用附件2官方 `aligned_50.pkl`。三划分类别计数为 train `[967,758,1670]`、valid `[206,184,338]`、test `[207,158,362]`；附件3 30 个文件无标签，附件4不进入 Q2 训练或选模。文本为 50 个 input/token-type/attention 位置，声学为 74 维、视觉为 35 维。自然全零 A/V 行只被视为当前不可用，不能推断是人工删除。

最终方法的输入状态、BERT 重编码、逐模态投影、masked attention、晚融合、双头输出和全空回退已完整写在 [Q2 最终方法](docs/Q2_最终方法.md)。论文需要的关键公式是：

\[
\alpha_{tm}=\frac{U_{tm}e^{a_{tm}}}{\sum_s U_{sm}e^{a_{sm}}},\quad
z_m=\sum_t\alpha_{tm}h_{tm},\quad
v=[z_T;z_A;z_V;g_T;g_A;g_V],\quad
\hat s=3\tanh(w_s^Tr+b_s).
\]

输入缺失不是先编码后置零：文本在 token 输入端换 MASK 并重新编码；A/V 行先置零再重新计算状态。最终部署不含历史 MSD、补偿、可靠性门控、教师、集成、类别偏置、GRU、z 裁剪和插补。全空时回退 train 类别先验和 score 中位数，该基线单独报告。

最终真实超参数是：BERT8 八层 Transformer 可训练、embedding 冻结；主干学习率 `1e-4`、BERT `1e-5`；AdamW、betas `(0.9,0.999)`、eps `1e-8`、weight decay `1e-4`（bias/LayerNorm一维参数为0）；20轮，前2轮 clean-only warm-up；train/eval batch=32/128；梯度裁剪1；FP32，AMP/TF32/compile关闭；hidden=128；投影 dropout=0.3、融合 dropout=0.1；损失为 train 先验平方根类权重 CE + Huber(δ=1)，回归权重1；预热后受损视图一半采自72公开场景、一半采原课程。checkpoint 选择键依次是 72 场景 Macro-F1、缺失 MAE、clean Macro-F1、clean MAE、较早 epoch。

## 4. valid 结果、错误归因和可视化证据

### 4.1 clean 与 72 场景主结果

valid clean 的 728 条样本 Accuracy=0.6154、Macro-F1=0.577807、MAE=0.6415、Pearson=0.6082；72 场景等权均值的 Accuracy=0.5796、Macro-F1=0.546551、MAE=0.6666、Pearson=0.5493。clean 与缺失均值之间 Macro-F1 差 0.0313，MAE 增加约0.0250，说明局部缺失训练带来一定保持能力，但没有消除退化。

valid clean 每类结果为：Negative Precision/Recall/F1=`0.6495/0.6117/0.6300`（support 206），Neutral=`0.4643/0.3533/0.4012`（184），Positive=`0.6523/0.7604/0.7022`（338）。72 场景均值每类 F1 为 Negative=0.5652、Neutral=0.4000、Positive=0.6744。Neutral 的 recall 和 F1最低，且正类样本最多，故不能用 Weighted-F1 替代 Macro-F1。

### 4.2 valid 导出包重载的逐样本错误诊断

最终导出包在独立本地进程中对 valid 的 728 条样本重新前向，逐样本结果保存在 [`Q2_valid_reloaded_predictions.csv`](../doc/paper_figures/data/Q2_valid_reloaded_predictions.csv)，完整诊断保存在 [`Q2_valid_reloaded_diagnostics.json`](../doc/paper_figures/data/Q2_valid_reloaded_diagnostics.json)。重载得到 Accuracy=`0.6153846`、Macro-F1=`0.5778068`、Weighted-F1=`0.6056950`、MAE=`0.6415313`、Pearson=`0.6082431`；与保存的 Q2 valid 指标分类结果完全一致，MAE 差 `3.04×10^-9`，Pearson 差约 `9.91×10^-9`。该检查使用 Python 3.9.25、torch 2.8.0+cu128 的 CUDA 进程，没有训练、改权重或重新选模。

混淆矩阵（行是真实类、列是预测类，类别顺序 Negative/Neutral/Positive）为

\[
\begin{bmatrix}126&31&49\\31&65&88\\37&44&257\end{bmatrix}.
\]

因此 Negative、Neutral、Positive 的召回率分别为 `126/206=0.6117`、`65/184=0.3533`、`257/338=0.7604`；Neutral 的 184 条样本中 31 条被判为 Negative、88 条被判为 Positive。逐样本回归残差的 MAE 按真实类为 Negative=`0.8294`、Neutral=`0.4629`、Positive=`0.6243`，预测为 Neutral 的 140 条样本的预测分数绝对值均值为 `0.2038`，低于真实 Neutral 的 `0.4629`。这支持“中性类分类边界偏向两端、回归分数向零收缩”的数据描述，但不构成因果解释。

### 4.3 valid 缺失规律

| 分组 | 条件 | 场景数 | Macro-F1 | MAE | 实际删除率 |
|---|---|---:|---:|---:|---:|
| modality | T | 12 | 0.5179 | 0.6888 | 0.5029 |
| modality | A | 12 | 0.5772 | 0.6425 | 0.5032 |
| modality | V | 12 | 0.5748 | 0.6428 | 0.5077 |
| modality | TA | 12 | 0.5196 | 0.6911 | 0.5029 |
| modality | TV | 12 | 0.5163 | 0.6893 | 0.5030 |
| modality | AV | 12 | 0.5735 | 0.6450 | 0.5032 |
| position | front/middle/rear | 24 each | 0.5531/0.5509/0.5357 | 0.6705/0.6627/0.6665 | ≈0.504 |
| rho | 0.2/0.4/0.6/0.8 | 18 each | 0.5668/0.5564/0.5406/0.5225 | 0.6533/0.6566/0.6713/0.6851 | 0.2070/0.4036/0.6038/0.8008 |

观察支持文本缺失和长跨度缺失更困难，rear 也比 front/middle 低。对实际删除率配对后，T_vs_A 在 `(0.4,0.6]` 的 Macro-F1 为 T=0.3783、A=0.4965，在 `(0.8,1.0]` 为 T=0.1105、A=0.6458；这支持“文本更敏感”的具体数据结论，但不能扩展为“文本唯一重要”。A/V 在相同实际率下接近，说明简单的名义ρ比较可能掩盖样本选择差异。

valid 双模态同时受损的 TA/TV/AV Macro-F1 为 0.5199/0.5174/0.5711；说明保留 A+V 比涉及文本的组合更稳。8 个压力场景如下，整模态删除不并入72场景均值：

| 场景 | Macro-F1 | MAE | 解释 |
|---|---:|---:|---|
| T_whole | 0.3393 | 0.8602 | 最差，文本完全缺失时仅靠 A/V 和状态标志。 |
| A_whole | 0.5661 | 0.6264 | 声学整路缺失仍高于72均值，说明文本/视觉尚能支撑。 |
| V_whole | 0.5576 | 0.6510 | 视觉整路缺失退化中等。 |
| TA_staggered_0.4 | 0.5548 | 0.6763 | 文本与声学错位缺失。 |
| TV_staggered_0.4 | 0.5450 | 0.6773 | 文本与视觉错位缺失。 |
| AV_staggered_0.4 | 0.5754 | 0.6452 | 不含文本的双模态错位较稳。 |
| TAV_middle_0.4 | 0.5491 | 0.6605 | 三路同时局部损坏。 |
| TAV_middle_0.8 | 0.4655 | 0.7180 | 更长三路缺失进一步退化。 |

valid 的基础错误归因来自 `metrics.json`：clean 中 Neutral support=184、recall=0.3533；真实 Neutral 的 score 绝对值均值为 0.4629，而预测为 Neutral 的样本均值为 0.2038（n=140），说明回归头趋向把真实中性附近的连续强度压向较小值，分类头仍大量把它们归到两端。对于局部缺失，Neutral F1 从 0.4012 降至 0.4000，Negative/Positive F1分别从0.6300/0.7022降至0.5652/0.6744；因此退化不仅是样本总数或类别比例造成的。

valid/test 的模态、位置、跨度图已有 `Q2/results/final/figures/valid_test_modal_position_rho.png` 和 `.svg`；更细训练曲线、分布、小提琴图和错误分层图的统一绘图入口暂记为 `../doc/paper_figures/README.md`。这些图应标记 valid/test、场景共享样本和 test 自适应边界，不把注意力权重解释成因果贡献。

## 5. test 结果和附件3专项输出

### 5.1 test clean、72场景和压力测试

test clean 727 条样本的混淆矩阵（行真类、列预测类）为

\[
\begin{bmatrix}131&33&43\\23&59&76\\35&27&300\end{bmatrix},
\]

Accuracy=0.6740、Macro-F1=0.618618、MAE=0.6841、Pearson=0.6081。Neutral 的 158 条样本中只有 59 条预测为 Neutral，76 条被预测为 Positive，解释了 Neutral F1=0.4260；Negative/Positive F1为0.6616/0.7682。完整混淆矩阵和回归散点图在 `Q2/results/final/figures/test_clean_confusion_matrix.png`、`test_clean_regression_scatter.png`，原始诊断在同目录 `test_clean_diagnostics.json`。

test 72 场景分组为：单 T/A/V=0.5403/0.6099/0.6154，双 TA/TV/AV=0.5369/0.5375/0.6124；front/middle/rear=0.5805/0.5792/0.5665；ρ=0.2/0.4/0.6/0.8=0.6017/0.5902/0.5696/0.5401。共同受损样本的模态比较为 T=0.5372、A=0.6111、V=0.6173、TA=0.5347、TV=0.5343、AV=0.6138；这就是报告“文本缺失更敏感”的主要证据，而不是只看不同样本的边际均值。

| 场景 | Macro-F1 | MAE | 相对 clean 的结论 |
|---|---:|---:|---|
| T_whole | 0.3581 | 0.9495 | 最严重的单压力；文本信息完全不可用。 |
| A_whole | 0.6090 | 0.6818 | 整路声学缺失仍接近 clean。 |
| V_whole | 0.6090 | 0.6862 | 整路视觉缺失仍接近 clean。 |
| TA_staggered_0.4 | 0.5612 | 0.7389 | 含文本双路缺失。 |
| TV_staggered_0.4 | 0.5634 | 0.7380 | 含文本双路缺失。 |
| AV_staggered_0.4 | 0.6075 | 0.6859 | 文本保留时较稳。 |
| TAV_middle_0.4 | 0.5710 | 0.7319 | 三路局部缺失。 |
| TAV_middle_0.8 | 0.4866 | 0.7881 | 长跨度三路缺失显著退化。 |

逐场景 72 行不能在此全部展开时，直接引用 `最终结果表格.md` 的“test 72 个主场景完整表”；该表包含 Accuracy、Macro-F1、Weighted-F1、MAE、PCC、相对 clean 变化、受损数和实际删除率。test 的实际删除率从 `deletion_rates.csv` 读取，不能用名义ρ替代。

### 5.2 全空输入

将 test 的文本、音频、视觉全部置为空后，模型回退训练先验：Accuracy=0.497937、Macro-F1=0.221610、Weighted-F1=0.331043、MAE=0.884915，Pearson 未定义（constant prediction），727 条全部预测为 Positive。该结果是信息不可用的基线，单独保存于 `Q2/results/final/test/all_empty.json`，不并入 clean 或72场景平均。

### 5.3 附件3全量30条

附件3没有标签，表中 `T/A/V` 是 `q2_special_state.csv` 的当前观测位置数，`P(*)` 是最终导出包 softmax 概率。`compensated_*` 全为0，因为最终晚融合没有补偿分支。

| sample | structural | T | A | V | pred | score | P(N) | P(Neu) | P(P) |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| 附件3_01 | 6 | 6 | 6 | 6 | Negative | -1.02227 | .69396 | .25931 | .04673 |
| 附件3_02 | 28 | 28 | 23 | 23 | Neutral | .21297 | .11833 | .45051 | .43117 |
| 附件3_03 | 32 | 32 | 27 | 23 | Positive | .81683 | .04122 | .19591 | .76287 |
| 附件3_04 | 31 | 31 | 29 | 29 | Positive | .41217 | .05781 | .28218 | .66002 |
| 附件3_05 | 12 | 12 | 12 | 12 | Negative | -1.39518 | .90576 | .06619 | .02805 |
| 附件3_06 | 25 | 25 | 22 | 22 | Positive | 1.35196 | .01041 | .07966 | .90994 |
| 附件3_07 | 21 | 21 | 18 | 18 | Positive | .16502 | .06127 | .34957 | .58916 |
| 附件3_08 | 15 | 15 | 13 | 13 | Positive | .69369 | .14709 | .19976 | .65314 |
| 附件3_09 | 14 | 14 | 12 | 12 | Positive | 1.49415 | .00893 | .05005 | .94102 |
| 附件3_10 | 6 | 6 | 6 | 6 | Neutral | .09691 | .19577 | .68867 | .11557 |
| 附件3_11 | 14 | 14 | 10 | 10 | Positive | .54437 | .02624 | .23593 | .73784 |
| 附件3_12 | 13 | 13 | 11 | 11 | Positive | .24172 | .23178 | .27530 | .49291 |
| 附件3_13 | 19 | 19 | 17 | 17 | Positive | .28867 | .05950 | .23696 | .70354 |
| 附件3_14 | 23 | 23 | 21 | 21 | Negative | -.73155 | .77443 | .13505 | .09052 |
| 附件3_15 | 27 | 27 | 24 | 24 | Positive | 1.09529 | .01022 | .08921 | .90056 |
| 附件3_16 | 34 | 34 | 30 | 30 | Positive | 1.55908 | .01307 | .07161 | .91532 |
| 附件3_17 | 22 | 22 | 14 | 14 | Neutral | .38796 | .16774 | .41842 | .41384 |
| 附件3_18 | 17 | 17 | 14 | 14 | Positive | .40568 | .15646 | .25662 | .58692 |
| 附件3_19 | 24 | 24 | 17 | 17 | Positive | .36029 | .08272 | .29064 | .62664 |
| 附件3_20 | 48 | 48 | 34 | 34 | Positive | .38622 | .07975 | .33099 | .58927 |
| 附件3_21 | 9 | 9 | 6 | 6 | Positive | .99814 | .02103 | .10309 | .87587 |
| 附件3_22 | 17 | 17 | 13 | 13 | Positive | .79317 | .06517 | .16991 | .76493 |
| 附件3_23 | 13 | 13 | 9 | 9 | Positive | .56944 | .02901 | .29378 | .67721 |
| 附件3_24 | 8 | 8 | 6 | 6 | Positive | .62681 | .13170 | .11307 | .75522 |
| 附件3_25 | 9 | 9 | 7 | 7 | Positive | .76192 | .02017 | .21266 | .76717 |
| 附件3_26 | 22 | 22 | 16 | 16 | Positive | 1.01857 | .02118 | .11261 | .86621 |
| 附件3_27 | 21 | 21 | 13 | 13 | Positive | 1.08411 | .01691 | .10268 | .88041 |
| 附件3_28 | 33 | 33 | 22 | 22 | Positive | .70366 | .01606 | .18723 | .79671 |
| 附件3_29 | 18 | 18 | 10 | 10 | Positive | .76202 | .03990 | .17599 | .78411 |
| 附件3_30 | 24 | 24 | 12 | 12 | Positive | 1.16420 | .02767 | .13188 | .84045 |

30 个样本中没有文本 MASK，29 个样本的 A/V 可用位置完全相同；A/V 零行比例按样本均值约为 20.67%/21.09%，最长连续缺口不超过4个位置。附件3_20是论文方法图可采用的真实链路：48个结构位置，T=48、A=V=34，输出 Positive、score=0.38622。附件3无标签，所以这些输出只能作为推理示例，不能作为性能证据。

## 6. 消融、失败方案和原因分析

所有历史 trial 和配置仍在 `Q2/results/experiment_ledger.json`；ledger 顶层记录 73 个 trial、58 个 test evaluation，test evaluation 均是已观察 test 后的自适应探索。下表报告最能回答“为什么采用当前方法”的对照；数值都是同一 72 场景等权口径的三分类 Macro-F1，`M/C` 分别表示缺失平均/clean。

| 方案 | test M/C | 观察和失败原因 |
|---|---:|---|
| 历史 full/MSD—补偿—可靠性，三 seed | 0.4963/0.5271 | 低于晚融合；可能的机制是旧分解、补偿和可靠性目标在零行/局部缺口上累积误差，但不能据此证明所有补偿方法无效。 |
| 历史 late_clean / late_aug，三 seed | 0.4981/0.5221；0.5113/0.5376 | 仅 clean 或早期增强不足以覆盖局部连续缺失。 |
| 原课程、reg=0.25，三 seed | 0.559022/0.593276 | 参考基线；缺失均值低于0.60。 |
| 网格混合、reg=1，三 seed | 0.566691/0.598734 | 方向上改善，但 paired bootstrap 缺失差异 CI `[-0.0117,0.0243]`，不能称稳定显著。 |
| 不做人工受损训练 | 0.550355/0.580117 | 同配置最终结构的具体失败，支持局部受损监督必要。 |
| 去 CLS context | 0.566413/0.603143 | clean低于最终单模型；CLS的增益只对该对照成立。 |
| mean pooling | 0.567688/0.606158 | 低于注意力池化；固定平均可能稀释情感显著位置，这是解释性推断。 |
| 辅助分类 `0.05/0.2`，seed1111 | 0.568961/0.597321；0.568361/0.603250 | clean可能上升，缺失未改善；`0.2` 三 seed 均值为0.566147/0.601042，仍不替换单学生。 |
| 特征教师一致性 `0.2/1.0` | 0.566903/0.592656；0.566433/0.593658 | 两个非零权重均低于零权重；教师来自完整预计算 text，缺失映射边界不足。 |
| z裁剪 `3/5` | 0.569322/0.597715；0.569163/0.596818 | 单 seed效应很小；极端值不自动等于错误，未改原始特征。 |
| BERT8 仅顶部4层微调 vs BERT8 全层微调 | `attn_bert8_v1`：0.568733/0.599249；全层候选（seed1111）：0.575404/0.618618 | 不能据此说“所有深层模型无效”；八层候选由 test 观察反馈进入后续验证，最终数值需标记自适应探索。 |
| BERT8 全层三 seed 复验 | seed1111/1112/1113 的 test M/C=`0.575469/0.618618`、`0.561138/0.592796`、`0.567513/0.597259`；最终导出包重载为 `0.575404/0.618618` | 部署 seed=1111 固定；epoch 由 valid 选择；1112/1113 仅作固定配置复验，不能按 test 逐 seed 选优。 |
| BERT8 直接消融：均值池化、去 CLS、无人工受损训练 | `0.567688/0.606158`、`0.566413/0.603143`、`0.550355/0.580117` | 三个具体对照都低于最终导出结果；它们支持注意力池化、CLS context 和受损监督在当前实现中的联合选择，不能外推为普遍定理。 |
| text lr=3e-5、GRU、gate、A/V branch | 约0.5449/0.5668、0.5531/0.5788、0.5554/0.5884、0.5574/0.5799 | 额外自由度没有在本数据上带来同步收益；这些是具体失败，非一般性定理。 |
| class weight power 0/1 | 0.551180/0.568163；0.561941/0.585832 | 均低于 train-only power=0.5；不能据test另调权重。 |
| 30轮、网格覆盖率0.75/1.0 | 30轮0.566111/0.596406；覆盖率1.0 seed1111为0.569684/0.599209 | 仍未到缺失0.60，单点改善不足以替换固定三 seed方案。 |
| 三模型 raw-logit 集成+偏置 | 0.573598/0.608641 | 探索上较高，但需要三份BERT、已观察test且非正式导出，不作为最终模型。 |

最终 BERT8all seed1111 的 test 0.575404/0.618618 高于多数历史候选，但它是被 test 观察推动进入验证的结构；该“优势”只能写成当前探索结果。辅助分类、教师蒸馏、裁剪、门控、GRU、旧 MSD/补偿和全模态压力方案的完整 valid/test 数字及资源记录，均可由 ledger 和对应 `results/audit/*_summary.json` 追溯。

## 7. 中间级数据、审计和论文图规划入口

报告不能只给最终 F1。论文可引用的中间数据包括：

- `data_inventory.json`：三划分形状、标签、自然视觉全零计数和 ID/video 隔离；它证明输入检查通过，但不证明物理时间对齐。
- `tokenizer_compatibility.*`：3395 条 train 均为 standard50 tokenizer 重建；它支持“输入 ID 语义一致”，不支持“原始视频词时间戳已知”。
- `valid/test/deletion_rates.csv`：每个模态×位置×ρ的新增删除比例；它解释为什么名义ρ不应直接写为每样本真实比例。
- `paired_conditions.csv`、`matched_deletion_rates.csv`：同一样本或同实际删除率比较；它支撑文本与 A/V 敏感度的配对结论。
- `dual_both_damaged.csv`：双模态同时新增损坏的样本数与指标；它支撑 TA/TV/AV 压力解释。
- `metrics_per_scenario.csv`：每个场景的 Accuracy、Macro-F1、Weighted-F1、MAE、PCC、类别错误、Neutral score 统计及相对 clean 变化；正文不展开时引用 `最终结果表格.md` 的对应章节。
- `experiment_ledger.json` 的 history：每轮 loss、clean/corrupt loss、学习率和 valid 检查点；这些数据用于训练曲线、收敛速度、过拟合拐点图，不能把最后一轮当作最佳轮。
- `attachment3_availability.json`、`q2_special_state.csv`：30 条专项的当前可用位置和缺口形态；它支撑专项链路图和缺口分布图，不能计算性能。

论文数据图与 CSV 已集中在 [`doc/paper_figures/README.md`](../doc/paper_figures/README.md)，图册入口为 [`index.html`](../doc/paper_figures/index.html)，当前包含 68 组图；valid 重载诊断对应 Q2_16–Q2_19。建议图至少覆盖：valid/test clean 与72场景 Macro-F1 对照；模态×位置×跨度热图；共同样本的实际删除率分层小提琴图；每类 F1/recall 和 Neutral score 误差分布；训练每轮 clean/corrupt loss、学习率和 valid 选择点；消融 M/C 点图及 seed 波动；压力测试退化瀑布图；附件3 30条观测计数与预测置信度图。每张图应保留原始坐标含义、样本数和 valid/test 标识，不能通过删掉低分区间或重标坐标制造优势。

## 8. 导出、复现和资源

最终导出包 `q2_final` 为单模型，zip=27,138,297 bytes，解压目录约30,253,681 bytes，Q2 单目录小于50 MB；全题 Q1+Q2+Q3 合包大小尚未核验。离线新进程重载通过：30 条专项逐字段一致；FP32 checkpoint 与 FP16 存储/FP32 计算包最大差异为 logits 0.002242、score 0.002018，预测类别一致。valid/test 81 个场景的 GPU 评估约25.09/24.32秒，warm batch32 中位延迟0.01291秒，约2479 samples/s（包含 CPU→GPU 复制）。环境快照在 `Q2/results/audit/environment.txt`。

配置和入口：`Q2/configs/final_train.yaml`、`Q2/configs/final_selection.json`、`Q2/src/q2/`、`Q2/scripts/evaluate_experiment.py`、`Q2/scripts/finalize_q2.py`。最终 `final_selection.json` 使用 `members` 格式，需在 Q2 根目录执行以下最终复核入口：

```bash
python scripts/finalize_q2.py prepare --selection configs/final_selection.json
python scripts/finalize_q2.py assess --selection configs/final_selection.json
python scripts/finalize_q2.py verify --selection configs/final_selection.json
python scripts/finalize_q2.py analyze --selection configs/final_selection.json
```

README 中的 `python -m q2 evaluate-test ...` 等命令仍用于旧的通用 suite/`reports/selection.json` 流程，不能替代上述 `members` selection 复核入口。

## 9. 尚未解决的边界

test 的自适应探索、单交付 seed 和 valid/test 的指标波动限制了泛化结论；72 场景是本项目预定义的评估网格，题目要求评估局部缺失及规律但未规定这 72 个具体组合，因此它不是真实缺失分布的无偏估计；附件3无标签且实际缺口分散；PKL缺少时间戳和提取映射；全空只是先验回退；Q2包体积没有替代全题包体积核验。论文中应把这些作为结果适用范围的一部分，与方法优势同时呈现。
