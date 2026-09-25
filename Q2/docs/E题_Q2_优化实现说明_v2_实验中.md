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
