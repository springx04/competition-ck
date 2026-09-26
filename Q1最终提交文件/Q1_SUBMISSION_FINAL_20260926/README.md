# Q1 第一问最终提交包

本包提交数学建模第一问的 100 条多模态时序特征、完整状态索引、可复现核心代码、测试和报告。主产物是词级对齐特征，50 窗口特征作为固定长度辅助表示。

## 结果快照

- 基础源码：`da6c8e5bd553d0480ca941384ae20f279929872f`
- 样本：100；词：1,932；固定窗口：每条 50
- 维度：文本 768、音频 25、视觉 22
- 状态：57 valid、19 partial、5 conflict、4 missing、15 unresolved
- 严格配对可用：68 条
- 最终验证：`ok=true`、`errors=[]`
- 工程验收：Stage 4 冻结基线 99 项通过；加入 4 项提交包回归测试后当前 103 项通过；772 个冻结输入无变化

`verified` 仅表示固定自动规则验证通过，不表示人工 Gold 或情感识别准确率。所有 unresolved、conflict 和自然缺失样本仍保留，并用 mask、status、reason 显式表达。

## 目录

```text
Q1_SUBMISSION_FINAL_20260926/
├── README.md
├── SOURCE_REVISION
├── MANIFEST.tsv
├── SHA256SUMS
├── inspect_q1_submission.py
├── features/          # 双 NPZ 与特征名
├── metadata/          # 100条索引、状态、验证和可复现性记录
├── examples/          # 典型对齐与异常掩码样例
├── code/              # src、scripts、tests、configs和依赖清单
└── documentation/     # 最终报告、映射说明和第三方声明
```

## 无网络验收

要求 Python 3.10 和 NumPy。进入解压后的包根目录：

```bash
python inspect_q1_submission.py --mode quick
python inspect_q1_submission.py --mode full
```

预期输出包含：

```text
"ok": true
"sample_count": 100
"word_count": 1932
"paired_use": 68
```

`quick` 验证逐文件 SHA256、必需文件、100条索引、双 NPZ 形状及状态计数；`full` 进一步验证有限值、二值 mask、mask=0 的零填充、coverage 范围、CSR 单调性和原生来源样本映射。

## 代码测试

```bash
cd code
python -m pip install --no-deps -e .
python -m pip check
python -m pytest tests -q
```

冻结环境的基线结果是 `99 passed`；加入提交包回归测试后，实际总数应不低于该值。若需要新建环境，原始锁定参考依赖见 `code/requirements-q1.txt`；实际验收环境见 `metadata/MODEL_TOOL_VERSIONS.json`。模型权重、OpenFace 运行时和原始视频因附件体积及数据授权限制未包含，因此代码测试不等于完整媒体重提取。

## 数据读取

```python
import numpy as np

with np.load("features/q1_word_aligned_final.npz", allow_pickle=False) as data:
    i = 0
    left, right = data["sample_indptr"][i:i + 2]
    print(data["raw_word"][left:right])
    print(data["text"][left:right].shape)    # (词数, 768)
    print(data["audio"][left:right].shape)   # (词数, 25)
    print(data["vision"][left:right].shape)  # (词数, 22)
    print(data["masks"][left:right])
```

词级 `audio_*` 和 `vision_*` CSR 字段保存聚合来源；`coverage` 是时间覆盖比例，不是准确率。50 窗口产物中的 `observed_mask[...,0/1/2]` 分别对应文本、音频和视觉。

Q1 的音频 25 维和视觉 22 维是本题第一问自生成特征，与附件二提供的 74/35 维官方对齐特征不是同一接口，不应在没有适配的情况下直接替换。

## 完整复现边界

完整重提取另需：题目附件一、锁定的 BERT/CTC 权重、FFmpeg、openSMILE、OpenFace 2.2.0 及 CUDA 环境。版本、参数和流水线见 `documentation/Q1第一问最终报告总结_20260926.md` 和 `code/configs/`。

本包不包含情感标签、原始媒体、模型权重、虚拟环境或网页端分析材料。
