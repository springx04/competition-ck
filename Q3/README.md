# Q3：多粒度反事实解释

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
