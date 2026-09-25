# Q2 优化实现说明 v2

2026-09-24。配合《Q2优化方案v3》使用。本文记录已实现接口和待运行实验，替代旧任务书中“所有变体均冻结BERT、固定60轮”的探索限制，其余数据使用边界不变。

## 1. 服务器与文件

本轮用户明确继续原服务器 `e-question-server`，项目 `/root/gpufree-data/shuomo_E/Q2`。本地默认配置已被迁移工作指向新服务器，不覆盖该改动。原服务器探索显式传入仍记录原路径的 `configs/quick_tune.yaml`；脚本会覆盖训练轮数等探索参数，并将最终配置保存到对应run中。

保留 `runs_baseline_20260924` 和所有已产生实验文件。新实验写入 `experiments/<name>/runs/<variant>/seed_<seed>`。不得删除旧目录后用同名重跑；新实现、课程或超参数改变时使用新名称，保留可解释的实验记录。`--resume`仅用于同配置、同方法续跑，不用于将新课程接到旧曲线上。

## 2. 关键代码约定

| 位置 | 行为 |
|---|---|
| `model/network.py` | `TUNED_VARIANTS`集中列出微调变体；`CLEAN_VARIANTS`集中列出无缺失增强变体；轻量池化候选不实例化额外Transformer |
| `text.py` | `encode_text(..., requires_grad=True)`支持BERT梯度；冻结和推理路径保持no_grad |
| `trainer.py` | 微调变体要求`text.frozen=false`；同时保存/恢复学生和`text_encoder`；不依赖冻结特征文件 |
| `evaluate.py` | `clean_cache=None`使用当前编码器。只在本次调用内复用实时clean特征；文本受损样本重新编码。禁止读写冻结磁盘缓存 |
| `__main__.py`、`export.py` | 从微调checkpoint恢复BERT；缺失文本权重时明确失败，禁止静默回退原预训练权重 |
| `masking.py` | 课程阶段依据总轮数与预热轮数缩放；60/5默认设置维持原边界 |
| `scripts/audit_data_quality.py` | 只读审计附件2处理数组的零行、NaN/Inf、范围、类别计数和重复文本，不修改数据 |
| `losses.py` | 接收回归权重、类别权重、预热和ramp设置；历史默认值保持兼容 |

优化器保存`lr_scale`；每轮使用当前余弦值乘组比例，不能根据组当前lr是否大于某阈值决定是否更新。历史checkpoint缺少该字段时按下游组和文本组角色恢复比例。RNG状态恢复到CPU ByteTensor再交给随机数API。

## 3. 可复现探索命令

以下保留早期探索命令，当前已执行；实际结果以实验汇总为准，不代表最终所选方案：

```bash
cd /root/gpufree-data/shuomo_E/Q2
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8
.venv/bin/python scripts/explore_tuning.py --config configs/quick_tune.yaml \
  --name pool_tune_lr3e5_v3 --variant late_pool_tune --text-lr 3e-5 \
  --regression-weight 0.25 --epochs 20 --seed 1111
.venv/bin/python scripts/explore_tuning.py --config configs/quick_tune.yaml \
  --name pool_tune_aug_lr3e5_v3 --variant late_pool_tune_aug --text-lr 3e-5 \
  --regression-weight 0.25 --epochs 20 --seed 1111
```

可比较参数：`--text-lr`、`--student-lr`、`--class-weight-power`、`--regression-weight`、`--epochs`。不得通过改变评估grid、删除困难样本或改变类别定义提高分数。

## 4. 已完成实验与暂停状态

`tune_lr3e5_clean_v2`已完成20轮，实测约231.6秒；最佳第5轮missing=0.53916、clean=0.56577。该时长只代表此配置，不是原full套件的同条件加速倍数。

`tune_lr3e5_aug_v2`在用户迁移暂停时运行至第12轮，保留`last.pt`。它使用尚未缩放的旧缺失课程，后续不能将它与修正课程的run混为同一个实验。原排队的轻量融合实验未启动。

原微调checkpoint已用 `scripts/reevaluate_tuned.py`恢复文本权重重评，结果见原服务器 `reports/tuned_corrected_last_20260924/summary.json`。旧微调history中的冻结缓存评估无效，保留文件以供审计，不覆写为“正确历史”。

## 5. 速度与验证

已移除“成为best时再完整评估一遍”的重复计算：首次评估保存该轮指标与预测，成为best后复制相应文件。微调评估只在一次调用中缓存实时clean文本；A/V-only场景复用，T受损场景必须重新编码。已有state向量化和关闭训练诊断注意力也可减少开销，但注意力后端变化可能改变浮点或随机路径，不能宣称训练逐位一致。

本地测试覆盖：微调梯度、checkpoint文本恢复、禁止旧缓存、改变文本权重后预测随之改变、学习率下降跨越旧阈值、RNG续跑、短课程覆盖严重双模态缺失、全无观测但有MASK槽位时回退先验。训练效果、导出FP16舍入误差、独立离线重载及最终包体积还需服务器实测。

## 6. 输出与结论

每run保存resolved_config、history、训练mask、各评估轮CSV、best/last checkpoint和resource_usage。最终汇总必须同时列完整输入、缺失平均、分模态/跨度、各类F1及种子波动，记录失败候选。新候选未验证前不更新正式selection、不生成冒充最终成绩的报告。

## 7. 2026-09-25 评估一致性与embedding对照

`evaluate-test`现在使用selection中的`checkpoint`，不再按外部配置的output_root重新拼接可能不同的路径。类别偏置统一按selection的`class_bias`优先、checkpoint配置其次恢复，适用于evaluate、predict和export；selection显式写`null`表示关闭。外部命令配置不再悄悄覆盖所选模型的偏置。旧校准实验若要复现，必须在selection明确记录偏置，并标记为看过test后的探索。

每次`evaluate-test`写到`reports/test/<UTC时间>/`，同时保存selection，避免覆盖历史CSV。评估及配对分析完成后写`reports/test/latest.json`，报告生成器通过该路径读取本次结果，避免误读根目录的旧CSV；历史无该文件时保持旧目录兼容。该命令不宣称结果是独立留出；研究流程的test观察历史需由方案文档明确记录。首次原始test为clean=0.598213、missing=0.560844；随后校准的0.602589/0.563362不具备独立确认地位，详见方案第8节。原报告模板仍含固定full方法和30run表，探索模型正式交付前须按最终结构改写，不能将旧模板直接当本轮论文结论。

`calibration.py`供评估与离线Bundle共用：偏置必须为3个有限数；仅对有实际观测内容的样本加偏置。全模态无观测时保留模型输出的train先验和得分回退，不能因仍有MASK结构槽位而加偏置。报告`empty_content`也按U全空计数，不再误用J结构槽位。测试使用同时包含空内容和有观测样本的实际Student，比较评估与Bundle logits及先验回退，并验证指定checkpoint、偏置覆盖及导出配置一致。

`text.configure_text_training`新增可选`train_embeddings=false`，历史默认保持冻结embedding。`--train-embeddings`启用后，embedding梯度进入已有BERT优化器组，与encoder使用相同低学习率；pooler保持冻结（当前BERT没有pooler）。梯度与更新测试验证仅指定层和embedding发生更新。该变化不增加参数或额外数据。

`explore_tuning.py`会移除外部配置中的`evaluation.class_bias`，探索统一使用原始logits评估。以下对照已完成，未取得明确净收益，保留命令供复现：

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
.venv/bin/python scripts/explore_tuning.py --config configs/quick_tune.yaml \
  --name attn_embeddings_v1 --variant late_attn_tune --text-lr 1e-5 \
  --student-lr 1e-4 --regression-weight 0.25 --class-weight-power 0.5 \
  --epochs 20 --seed 1111 --cls-context --train-embeddings
```

新增`train.stress_text`（默认false）与`--stress-text`，允许注意力晚融合使用此前full stress已有的采样策略；基础增强保持原样。`attn_textstress_v1`使用上面相同参数，删除`--train-embeddings`并增加`--stress-text`，只对照缺失课程。

`attn_textlr3e5_v1`沿用原注意力配置，BERT学习率单独提高到3e-5（不用上述两个新开关）。每个实验保留自身resolved_config和history。训练epoch、结构、学习率、课程的实际设置以该run文件为准，不能因variant字符串相同就认为所有实验条件相同。

测试包含embedding实际梯度与权重更新、同一模型评估/部署校准一致、空内容先验不变及报告目录恢复。同期的迁移配置、README及依赖配置不属于本轮修改，不并入checkpoint。

本轮本地和服务器均72项测试通过。服务器另用4个真实checkpoint分别重载前16条valid样本，与各run训练时保存的clean logits比较，最大绝对误差为3.95e-6、类别预测全部一致；详见`reports/inference_consistency_20260925.json`。这是checkpoint重载核查，不等同于最终离线包验收。三项新增训练均已完成，不改变正式selection；没有再次运行test评价。

## 8. 音视频表示探索实现

`model/pooling.py::consecutive_run_weights`按当前特征与观测掩码计算连续重复组倒数权重，`late_attn_runweight_tune`仅对音视频池化分数加该权重的对数。它不改源文件、模态有效位或标签。推理时复制相同音视频行不会改变池化结果；训练时逐位置dropout不同，因此不声称训练随机路径也完全不变。

`model/temporal.py::ObservedGRU`按观测顺序compact→pack→双向GRU→scatter回官方位置，排除缺失行及padding对隐状态的影响。输出为LayerNorm后的残差表示。两个GRU在公共预测头初始化之后创建，公共参数与相同种子的`late_attn_tune`初始化一致；新模块改变后续随机数流，因此不声称训练随机路径逐位一致。

新变体统一加入`TUNED_VARIANTS`，checkpoint、评估及导出均沿用微调BERT恢复机制。`late_gru_tune`使用普通注意力权重，不同时采用重复计权处理。

```bash
.venv/bin/python scripts/explore_tuning.py --config configs/quick_tune.yaml \
  --name attn_runweight_v1 --variant late_attn_runweight_tune --text-lr 1e-5 \
  --student-lr 1e-4 --regression-weight .25 --class-weight-power .5 \
  --epochs 20 --seed 1111 --cls-context
.venv/bin/python scripts/explore_tuning.py --config configs/quick_tune.yaml \
  --name gru_av_v1 --variant late_gru_tune --text-lr 1e-5 \
  --student-lr 1e-4 --regression-weight .25 --class-weight-power .5 \
  --epochs 20 --seed 1111 --cls-context
```

测试检验重复组被缺失打断、隐藏值不影响分组、复制音视频行后的预测不变、GRU对缺失位置值及梯度不敏感、原位置scatter恢复、空观测回退、公共参数同种子初始化及新变体权重加载。训练效果以各run实际history与best checkpoint为准。

`late_av_tune`新增`av_projection`，仅在`flags[text]==false`的样本上替代晚融合投影；它保持注意力池化和公共分类/回归头。该对照已完成20轮，最佳valid缺失0.557127、clean0.583849，未进入selection。当前最新代码和测试覆盖83项；服务器的seed1113 calibrated test评估目录为`reports/test/20260925T023121719340Z`，selection文件明确记录偏置和test历史污染状态。后续交付若采用该checkpoint，必须沿用相同偏置并在报告中标注探索性，不得重命名为独立最终留出。

`late_gate_tune`在三个池化向量上用`Linear(129,1)`计算可靠性分数，按当前可用模态mask做softmax，并以3倍门控向量拼接可用标记。全空样本不进入softmax后的有效内容路径，仍由Student统一回退train先验。该实现严格读取当前受损输入，不读取clean标签或测试统计；实验未改善，保留作为攻略方向的否定对照。服务器代码测试86项通过。

## 9. 缺失分布对照与攻略核验

`sample_train_descriptor`新增`grid_mix=false`。预热期不改变；预热后使用独立的确定性随机流决定50%网格分支，并在`evaluation_grid(stress=False)`的72个描述符间等概率选择。未进入网格分支的样本保留原课程随机流；返回值去掉用于报告的name字段。`trainer`、配置校验和`explore_tuning.py --grid-mix`传递同一开关，run的resolved_config记录其设置。默认行为和旧checkpoint不变。

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
.venv/bin/python scripts/explore_tuning.py --config configs/quick_tune.yaml \
  --name attn_gridmix_v1 --variant late_attn_tune --text-lr 1e-5 \
  --student-lr 1e-4 --regression-weight .25 --class-weight-power .5 \
  --epochs 20 --seed 1111 --cls-context --grid-mix
```

其他种子使用1112/1113，并分别命名`attn_gridmix_seed1112_v1`、`attn_gridmix_seed1113_v1`。选择规则仍为原72场景valid缺失Macro-F1优先，不根据clean单独挑轮。测试检查全部72种规则可采到、非网格分支保持原课程、预热不注入缺失。新增`pytest.ini`明确仅收集`tests/`并优先导入`src/`，避免本地历史审计副本中的旧同名测试及旧q2包干扰默认pytest。

`scripts/audit_special_missing.py DATA_ROOT --output REPORT.json`只读取附件3对齐版的token ID/attention及A/V特征，报告内容位置中的不可用率、缺口数量和最长缺口，不读取或推断标签、不修改数据。已在本地原始附件复核，与服务器独立审计一致；输出零行原因未知，不将它们全部标为人工缺失。该报告不输入训练流程。

网格混合三种子均已完成，选中轮次分别10/15/20；valid缺失均值0.561106、完整均值0.601508，提升幅度不足以替换现行候选。服务器汇总保存在`reports/grid_mix_summary.json`，逐轮日志与checkpoint保存在各自experiments目录。没有运行新的test评价、附件3标签评估或离线最终包验收。本轮代码完成本地88项测试；服务器同步相同文件后按同一测试目录验证。
