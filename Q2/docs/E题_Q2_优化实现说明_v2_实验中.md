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
| `losses.py` | 接收回归权重、类别权重、预热和ramp设置；历史默认值保持兼容 |

优化器保存`lr_scale`；每轮使用当前余弦值乘组比例，不能根据组当前lr是否大于某阈值决定是否更新。历史checkpoint缺少该字段时按下游组和文本组角色恢复比例。RNG状态恢复到CPU ByteTensor再交给随机数API。

## 3. 可复现探索命令

以下是待执行的原服务器实验命令，不代表已取得结果：

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
