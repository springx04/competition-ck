# Q3：多粒度反事实解释

> **任务已结项（2026-09-26）**：本轮Q3开发与实验停止，交付以[最终方法与质量结论](reports/iteration_v2/Q3_最终方法与质量结论.md)的收尾口径为准。辅助词时听审、官方特征生成溯源和CTC独立封装不作为额外验收门槛；声画精确时间定位的证据限度仍如实保留。

本目录实现 `doc/Q3/E题_Q3_服务器具体实现说明_v1_Agent执行版.md` 中规定的 Q3 流程。Q3 只读取 Q2 最终导出包和官方数据，不修改 Q2；预测器始终是 Q2 的 `late_attn_tune` 单模型。

## 本地检查

```powershell
$env:PYTHONPATH="$PWD/Q3/src"
python -m pytest Q3/tests -q
python -m q3 --config Q3/configs/q3.yaml --help
```

本地没有服务器上的 Q2 数据/最终模型时，只运行纯算法测试。服务器上按实施说明设置 `Q3_ROOT`、`Q2_ROOT`、`DATA_ROOT` 后执行 `scripts/run_main.sh`。

## 设计边界

- 文本遮蔽只把选中 token ID 改为 103，音频/视觉先在原始输入置零；每个候选从原始输入独立构造。
- 所有差值固定跟踪原预测类别，三模态 Shapley 使用 8 个保留子集并报告效率残差。
- 连续选择使用整数预算、最多三段的 0/1 动态规划；严格随机对照按相同几何和模态成本计数。
- `q3_align` 是独立的素材定位入口，不导入情感模型；没有服务器素材或 CTC 权重时不会伪造定位结果。

## 迭代 2（2026-09-26）

本轮审查后，服务器正式计算采用 `configs/q3_v2_server.yaml`（本地 CPU 配置为 `configs/q3_v2.yaml`） 和 `runs/{valid_v2,special_v2}`，旧的 `*_main` 结果保留作审计对照。随机对照由完整连续单元构成，并同时匹配删除片段长度的多重集合、三模态删除成本和目标排除；若预算不能精确达到目标比例，选择器记录最大可达成本和原因，不用不合法集合补齐。方向实验在原预测类别上分别标记 support/suppress，整段复测后再判定方向是否保持。

valid 的忠实性区间以 `video_id` 为 bootstrap 簇，簇内样本按实际样本数加权；附件 4 无标签，不报告专项准确率。专项报告保留分类头与强度头的首位模态分歧、CTC 词时人工审核状态，以及官方 A/V 行来源的 unresolved 状态。算法完成、人工核验和官方来源核验是三个独立状态，不能相互替代。

服务器迭代 2 已完成 728/728 valid 和 20/20 专项样本。验证汇总显示 valid 有 1,011,901 条实验记录、587,281 个严格随机对照，专项有 31,805 条实验记录、18,137 个严格随机对照。保存的重建检查覆盖 23 个样本、584 个干预，最大预测概率差在数值容差内；A/V 行置换检查也通过。reports/validation_v2_final.json 的计算状态为 complete，素材状态仍为 partial：479 个候选词时均未完成听审，官方 A/V 特征行来源仍 unresolved。outputs_v2/assets/ 提供 20 个逐词播放复核页、候选音频和按实际 PTS 选出的上下文帧；这些材料不会把上下文帧冒充模型视觉证据。


## 当前结果入口与复算

- [完整实验报告](reports/iteration_v2/Q3_实验报告.md)：实际分母、区间、反例与局限。
- [本轮质量审查与最终方法修订](reports/iteration_v2/Q3_最终方法与质量结论.md)。
- [20条专项案例](outputs_v2/index.html)，机器可读结果见 `outputs_v2/`。
- 全量明细在服务器 `runs/valid_v2/samples`；本地同步了汇总、逐样本配对表和20条专项明细。交付包不含数据集、Q2权重或CTC权重。

在服务器 Q3 目录执行：

```bash
export PYTHONPATH="$PWD/src"
export Q3_PY=../Q2/.venv/bin/python
export Q3_CONFIG="$PWD/configs/q3_v2_server.yaml"
bash scripts/run_main.sh
```

`run_main.sh` 会复用已完成样本。若更改归因、抽样或预测实现，应先用新运行目录或不带 `--resume` 完整重算，再运行报告与最终核验；本轮随机顺序修正已对748条全部重算。

素材复核页面可直接打开；若浏览器限制本地音频，请在 Q3 目录运行 `python -m http.server 8765 --bind 127.0.0.1 --directory outputs_v2`，访问 `http://127.0.0.1:8765/`。播放成功只说明播放器可用，不能代替逐词听审。

## 词时定位的外部依赖

现有 `q3_align/pipeline.py` 仍导入同级 `../Q1/q1_features/src`，并从 `../Q1/q1_features/models/ctc` 加载模型；配置中的 `alignment.model_dir` 尚未被该入口使用。当前交付包能检查已保存词时和生成复核材料，**不能在缺少 Q1 源码与 CTC 权重的机器上独立重跑对齐**。本轮没有重跑或改写 CTC 对齐结果。依赖详情见 [THIRD_PARTY.md](THIRD_PARTY.md)。
