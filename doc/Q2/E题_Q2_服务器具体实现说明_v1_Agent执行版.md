# E题 Q2 服务器具体实现说明（v1，Agent执行版）

> 编写日期：2026-09-24。目标环境：**Linux x86_64 + 单张 NVIDIA RTX 4090（24GB）**。
>
> 本文是交给服务器实现agent的完整任务书。服务器可以只有题目数据和本文，没有本地Windows代码、EBMC源码或先前对话。**agent必须自行下载明确列出的复用代码和预训练模型，在用户指定工作目录下单独新建大写文件夹 `Q2`，本题代码、环境、缓存、运行结果均放在其中。** 原始题目数据只读，可以位于Q2外。
>
> 本文落实已审核的Q2方法v2。网络、训练规则和默认参数在此确定，不要求实现agent重新选型。文中的 `python -m q2 ...` 命令是**agent需要实现的项目CLI，不是已经存在的上游命令**。本文件本身不代表已经完成实现或训练。

## 1. 交付目标与固定选择

### 1.1 要完成什么

创建独立Python项目，使用附件2 `aligned_50.pkl` 的固定划分完成三分类和强度回归、连续局部缺失训练、验证规律分析及模块消融；在方案选定后评价附件2 test，并对附件3对齐版30条生成预测CSV。保留能被Q3复用的同一个预测器、输入状态和中间表示接口。

必须交付：可运行源码、环境文件、复用来源说明、模型参数、全实验结果与报告、附件3的30条预测、可离线重载的推理包。不要只创建目录、训练一轮或输出一份计划就停止。

### 1.2 唯一主实现

| 项目 | 固定实现 |
| --- | --- |
| 主输入 | 官方aligned；T/A/V顺序统一为 `text,audio,vision` |
| 文本 | `google/bert_uncased_L-4_H-256_A-4`；4层、256隐藏维、4头；冻结、eval；使用末层逐词元表示，不用pooler |
| 文本权重 | 首次下载后转为FP16存储；所有训练/验证/推理从该存储版加载为FP32计算；从第一轮就使用相同舍入后的权重 |
| 文本长度 | 存储的50个位置原样保留；不重新分词替换官方输入 |
| 音/视觉 | 官方74/35维；只用train可用行估计逐维均值、标准差 |
| 共同隐藏维 | `d=128` |
| 模态时序编码 | 每模态独立2层Transformer，4头，FFN=256，dropout=0.1，GELU，pre-norm |
| MSD | 直接复用EBMC的 `MSDModule` 类；重写外层掩码池化和损失 |
| 补偿 | 一个共享的目标模态条件化跨注意力补偿器，4头，单轮补偿 |
| 可信度 | 对已知遮蔽位置监督补偿误差；原观测系数1，补偿系数 `1/(1+预测误差)` |
| 融合 | 逐位置内容得分加可信系数，经模态softmax；随后对可用位置均值池化 |
| 输出 | 线性三分类头；独立线性回归头后接 `3*tanh` |
| 教师 | 一个EMA学生副本，仅训练；EMA系数0.99，前5轮监督预热结束后初始化 |
| 主训练 | 60轮，无提前终止；batch=32；AdamW，峰值学习率3e-4；见第12节完整规则 |
| 精度 | FP32计算，关闭TF32；本轮不使用AMP、FlashAttention或torch.compile |
| 随机种子 | 1111、1112、1113；全部必做变体各3次 |
| 最终部署 | 按valid选择变体，部署该变体seed=1111的最佳学生；不挑最好的种子，不集成3个模型 |
| 本轮不实现 | 原EMC、原IMTD、Soft-MoE、未对齐版模型、音视频重提取、完整Q3归因与视频定位 |

选择小型BERT是对方法v2“兼容紧凑编码器”的具体落实。其官方配置给出4层/256维/4头/30522词表，见[模型配置](https://huggingface.co/google/bert_uncased_L-4_H-256_A-4/blob/main/config.json)。不能擅自换成bert-base、DeBERTa、DistilBERT或随机初始化模型；词表兼容性先按第6节实查。

**保持方法边界：** 真实标签和人工P不进入学生前向；原观测分支不向受损分支传入隐藏表示；附件3/4不训练、不伪标注、不用于调参，也不通过查找附件2近似样本补回内容。

## 2. 从空服务器到交付的流程

### 图A：执行总顺序

```text
本文 + 已解压题目数据 + Linux/4090
                  |
                  v
建立独立 Q2/ --> 安装专用环境 --> 下载EBMC与指定BERT
                  |
                  v
创建本文要求的源码、配置、CLI与测试
                  |
                  v
输入清点 --> 词表兼容核对 --> train统计标准化 --> 数据转换
                  |
                  v
预计算原观测BERT缓存 + 固定评估区间清单
                  |
                  v
掩码/补偿/梯度/空观测等语义测试 + 实际训练更新检查
                  |
                  v
10个变体 × 3个种子，逐个运行；每个60轮
                  |
                  v
原观测 + 72组局部缺失valid评估 --> 消融/曲线/误差估计分析
                  |
                  v
按确定规则选择变体 --> 该变体1111学生用于最终预测
                  |
         +--------+---------------------+
         v                              v
附件2 test留出评价              附件3对齐版30条预测
         +----------------+-------------+
                          v
完整推理包导出 --> 独立进程离线重载验证 --> 报告与体积统计
```

### 图B：单次训练更新

```text
train原始batch与标签（标签留在trainer）
  |
  +--> 当前状态适配 --> 原观测BERT缓存 --> 学生clean前向 --> L_task0、L_MSD
  |
  +--> 按epoch课程采样连续区间 --> 修改原始输入，得到受损副本
  |                                  |
  |                                  v
  |                    重新从受损值推断U/J/q
  |                                  |
  |                    文本被删？ 是：重新跑BERT
  |                               否：可复用该样本clean BERT
  |                                  |
  |                                  v
  |                    学生受损前向 --> L_taskP、F_hat、e_hat
  |
  +--> 第6轮起：同一clean输入 --> EMA教师eval/no_grad
                                     |
                                     v
                            F_teacher、p_teacher、y_teacher
                                     |
  已知P与U0仅在这里参与 --> L_span、L_cal、L_cons
                                     |
                                     v
  汇总损失 --> 一次backward --> 梯度裁剪 --> optimizer.step
                                     |
                                     v
                   更新EMA；记录各项损失和有效监督数
```

### 图C：学生网络内部

```text
BERT末层(B,50,256)  Audio(B,50,74)  Vision(B,50,35)
          |               |                |
          +---------------+----------------+
                          v
       各自Linear->LN + 固定位置编码 + 模态embedding
                          v
           各自2层掩码Transformer，保持(B,50,128)
                          v
             各自MSD --> C、S --> F=LN(C+S)
                          |
             +------------+------------------+
             v                               v
        U=1原观测保留              J=1且U=0的缺口查询
                                             |
                          只读取补偿前U=1的来源，单轮跨注意力
                                             |
                                      F_hat + e_hat
             +-------------------------------+
                             v
        observed / compensated / unavailable 来源分开
                             v
         内容logit + log可信系数 --> 模态softmax
                             v
                  位置融合 --> 掩码均值池化
                             v
                三分类logits + 连续情感强度
```

## 3. 必须下载的代码、模型与复用位置

### 3.1 下载清单

| 项目 | 下载地址 | 本地目标 | 使用范围 |
| --- | --- | --- | --- |
| EBMC代码 | `https://github.com/kangverse/EBMC.git` | `Q2/third_party/EBMC/` | 核对代码、复用MSD核心、保留许可证 |
| 小型BERT | `google/bert_uncased_L-4_H-256_A-4` | `Q2/models/bert_download/` | 下载config、词表、PyTorch权重及模型说明 |
| 整理后的冻结文本模型 | 从上一行加载并转换 | `Q2/models/text_encoder/` | 唯一正式文本模型；FP16文件、FP32计算 |

模型当前仓库含 `config.json`、`vocab.txt`、`pytorch_model.bin`，同时还有TensorFlow/Flax文件；只下载需要的PyTorch文件，见[官方文件列表](https://huggingface.co/google/bert_uncased_L-4_H-256_A-4/tree/main)。不需要TensorFlow、Flax、torchaudio、torchvision、OpenFace或MMSA-FET。

### 3.2 复用表：不要直接跑上游训练脚本

| 上游文件/对象 | 具体动作 |
| --- | --- |
| `EBMC/modules/msd.py::MSDModule` | 将该源文件复制到 `src/q2/vendor/ebmc_msd.py`，保留来源注释；只实例化 `MSDModule(128)`，三个模态三个实例 |
| `MSDIntegrationModule` | 不调用：其普通均值、标签处理及相似度目标不符合本文；在 `model/msd.py` 和 `losses.py` 重写外层 |
| `EBMC/modules/cce.py` | 阅读共享/特有分量增强思路，不调用原MLP反事实损失；实现本文跨注意力补偿 |
| `EBMC/ebmc.py` | 仅参考模块组织；不继承其完整类，不使用池化到长度1后再做MSD的路径，不加载其第一阶段教师 |
| `EBMC/Attention_softmoe.py` | 不复制Soft-MoE网络；用PyTorch标准注意力层实现明确掩码逻辑 |
| `EBMC/modules/emc.py`、`imtd.py` | 本轮关闭且不实现可选扩展，不把两者混进损失 |
| `dataloader_mmsa_seq.py`、`train_ebmc_seq.py` | 仅核对历史差异；重写官方aligned入口、连续遮蔽、三分类指标 |
| `LICENSE`、README引用 | 在项目 `THIRD_PARTY.md` 和 `licenses/EBMC-LICENSE` 保留；公开代码声明按上游许可证处理 |

上游[EBMC仓库](https://github.com/kangverse/EBMC)不能开箱直接完成本题。实现后在THIRD_PARTY.md记录下载日期、Git版本及上述改动，避免把新补偿器称为EBMC原有实现。使用Git版本记录即可，不新增数据/模型SHA256清单或重复校验体系。

若下载的上游文件结构与本文不一致，阅读实际代码，定位同一双MLP加共享LayerNorm的MSD核心并记录对应位置；不能无说明切换仓库或声称已经复用。网络暂时不可达时保留已完成代码，重试或使用相同官方来源的离线文件，不下载来历不明的替代权重。

## 4. 建目录与安装环境

### 4.1 只在独立Q2目录写入

用户给出工作父目录时使用它；未给出时使用实现agent启动所在工作目录作为父目录。执行：

```bash
export Q2_PARENT="${Q2_PARENT:-$PWD}"
export Q2_ROOT="$Q2_PARENT/Q2"
mkdir -p "$Q2_ROOT"
cd "$Q2_ROOT"
mkdir -p third_party models .cache data/processed configs src/q2 tests runs reports outputs delivery licenses
export HF_HOME="$Q2_ROOT/.cache/huggingface"
export PIP_CACHE_DIR="$Q2_ROOT/.cache/pip"
```

启动目录已经是本题Q2时直接把其绝对路径设为Q2_ROOT，不创建 `Q2/Q2`。如果Q2已经有本题代码，先读其README与已有记录并继续；不要覆盖已有实验。不要把代码写进Q1目录、数据目录或EBMC源码目录。

服务器需要Python 3.11及其venv支持、git和NVIDIA驱动。普通Python依赖安装进Q2的虚拟环境，不改系统Python，不为安装包自动修改驱动。

```bash
python3.11 -m venv "$Q2_ROOT/.venv"
source "$Q2_ROOT/.venv/bin/activate"
python -m pip install --upgrade pip
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

如果没有python3.11命令但服务器已有conda，可先 `conda create --prefix "$Q2_ROOT/.python311" python=3.11 -y`，再用 `"$Q2_ROOT/.python311/bin/python" -m venv "$Q2_ROOT/.venv"` 创建同一个正式虚拟环境；后续命令不变。把 `.python311/` 加入gitignore。两者均无且无权取得Python运行时时，明确报告该外部条件，继续完成不依赖运行环境的源码，不能改用不匹配的系统解释器后宣称环境已满足。

PyTorch官方提供2.6.0/cu124组合，见[安装表](https://docs.pytorch.org/get-started/previous-versions/)。主方案使用它，无需另外安装系统CUDA Toolkit。先检查 `nvidia-smi` 和实际CUDA张量运算；若服务器驱动无法运行cu124，可改装**同一torch版本的cu118轮子**，记录原因和实测结果，不改网络与数据方案，不静默转CPU完成正式训练。

### 4.2 写入requirements.txt

以下为除torch外的直接依赖；torch通过上一条专门安装：

```text
numpy==1.26.4
scipy==1.12.0
pandas==2.2.3
scikit-learn==1.5.2
transformers==4.51.3
tokenizers==0.21.4
huggingface_hub==0.34.4
safetensors==0.5.3
PyYAML==6.0.2
tqdm==4.67.1
matplotlib==3.9.2
pytest==8.3.5
```

```bash
python -m pip install -r requirements.txt
python -m pip check
```

不要安装EBMC整份requirements，它包含本项目不需要的音视频和其他依赖。生成pyproject.toml，包名 `q2-sentiment`、Python要求 `>=3.11,<3.12`，使用src布局和setuptools构建；随后 `python -m pip install -e . --no-deps`。标准库argparse足够实现CLI。

正式运行前写 `reports/environment.txt`：Python与直接依赖版本、实际torch/CUDA、GPU型号/总显存/可用显存、驱动、CPU、内存和磁盘余量；保存 `pip freeze` 供复现。建议至少16GB主存、10GB以上项目可用磁盘；实际运行占用另测，不把建议值写成实测值。

### 4.3 下载上游与文本模型

```bash
git clone --depth 1 https://github.com/kangverse/EBMC.git "$Q2_ROOT/third_party/EBMC"
git -C "$Q2_ROOT/third_party/EBMC" rev-parse HEAD
```

重复执行时若目录已是该仓库，直接记录并使用现有版本，不自动覆盖。文本下载可直接使用下列Python片段：

```python
import os
from pathlib import Path
from huggingface_hub import snapshot_download

root = Path(os.environ["Q2_ROOT"])
snapshot_download(
    repo_id="google/bert_uncased_L-4_H-256_A-4",
    local_dir=root / "models/bert_download",
    allow_patterns=["config.json", "vocab.txt", "pytorch_model.bin", "README.md"],
)
```

整理模型的程序写成 `python -m q2 prepare-model`，核心行为固定为：

```python
import os
from pathlib import Path
import torch
from transformers import BertModel, BertTokenizerFast

root = Path(os.environ["Q2_ROOT"])
download_dir = root / "models/bert_download"
text_encoder_dir = root / "models/text_encoder"
tokenizer = BertTokenizerFast(vocab_file=str(download_dir / "vocab.txt"), do_lower_case=True)
model = BertModel.from_pretrained(
    download_dir,
    add_pooling_layer=False,
    attn_implementation="eager",
    torch_dtype=torch.float32,
)
model.requires_grad_(False).eval()
model.half().save_pretrained(text_encoder_dir, safe_serialization=True)
tokenizer.save_pretrained(text_encoder_dir)
```

prepare-model在下载目录缺文件时先执行上一段的同源下载，再执行转换。之后所有组件必须重新从 `models/text_encoder` 加载 `torch_dtype=torch.float32`，`local_files_only=True`，`add_pooling_layer=False`，`attn_implementation="eager"`。不从原始FP32下载目录开始训练、到导出时才换FP16权重。

允许加载时丢弃原预训练任务头和pooler；任何embedding或encoder层权重缺失都必须定位，不能接受随机补权重继续训练。模型的字段和逐词元输出接口见[Transformers 4.51.3的BERT文档](https://huggingface.co/docs/transformers/v4.51.3/en/model_doc/bert)。

## 5. 目录、模块与CLI职责

### 5.1 必须建立的项目结构

```text
Q2/
  README.md                      # 安装、输入、完整执行命令、结果位置
  THIRD_PARTY.md
  pyproject.toml
  requirements.txt
  .gitignore                     # 排除.venv、原始数据、模型缓存和训练产物
  licenses/EBMC-LICENSE
  third_party/EBMC/               # agent下载，上游保留原状
  models/bert_download/
  models/text_encoder/           # 唯一冻结编码器，FP16存储
  configs/default.yaml
  configs/experiments.yaml
  src/q2/
    __init__.py
    __main__.py                  # argparse入口
    config.py
    data.py                      # 三种官方文件层级适配、转换、DataLoader
    state.py                     # 从当前输入推断U/J/q
    masking.py                   # 连续遮蔽、课程、固定评估区间
    text.py                      # 冻结BERT、词表核验、允许的缓存
    model/
      encoders.py
      msd.py
      compensation.py
      reliability.py
      fusion.py
      network.py
    vendor/ebmc_msd.py
    losses.py
    trainer.py
    evaluate.py
    metrics.py
    export.py
    reporting.py
  tests/                         # 第17节语义测试
  data/processed/{train,valid,test}/
  data/processed/normalizer.npz
  data/masks/                    # 固定valid/test/压力实验区间清单
  .cache/text/                   # 只缓存冻结BERT结果
  runs/<variant>/seed_<seed>/
  outputs/q2_predictions_aligned.csv
  reports/
  delivery/q2_inference/
```

所有文件是本题实现的一部分，不能只给一个调用上游脚本的包装器。采用普通Git记录代码改动；不要额外建设contract、hash、baseline或自动发布系统。

### 5.2 核心函数接口

```text
infer_state(raw_batch) -> ObservationState
make_span_mask(raw_batch, state0, descriptors) -> Perturbation
apply_span(raw_batch, perturbation) -> raw_corrupted_batch
encode_view(raw_batch, frozen_text_encoder, normalizer, text_cache=None) -> ModelInput
student.forward(model_input, return_details=False) -> ForwardOutput
compute_losses(clean_output, corrupt_output, teacher_output, labels, perturbation, state0, epoch) -> LossOutput
evaluate(model_bundle, split, grid, output_dir) -> MetricTables
predict_special(model_bundle, attachment3_dir) -> DataFrame
```

`student.forward`签名不接收label、P、U0、teacher或clean隐藏向量。`encode_view`必须从当前原值推断U/J/q，不能把训练保存的理想mask当作专项也存在的输入。

| 结构 | 字段与形状 |
| --- | --- |
| RawBatch | `sample_id`列表；`input_ids, stored_attention, token_type_ids`均 `(B,50)` int64；`audio(B,50,74),vision(B,50,35)` float32 |
| Labels | `class_id(B,)` long；`score(B,)` float32；留在trainer，不放入ModelInput |
| ObservationState | `U(B,50,3)` bool，`J(B,50)` bool，`q(B,50,3,5)` float32 |
| Perturbation | `P(B,50,3)` bool；每样本/模态 `[start,end)`；新增删除数；仅采样/损失/统计使用 |
| ModelInput | `text_features(B,50,256), audio_norm(B,50,74), vision_norm(B,50,35)`；U/J/q |
| ForwardOutput | `logits(B,3), score(B,), C/S/F/F_hat(B,50,3,128), e_hat/r/alpha(B,50,3), B_comp(B,50,3), source(B,50,3)` |

`source`固定为 `0=unavailable,1=observed,2=compensated`。普通推理可只返回logits/score；`return_details=True`才返回全部中间量。标签数组不能隐式广播成 `(B,B)`。

表中隐藏张量是full及其同骨架消融的格式。late系列不具备MSD与位置融合，details中 `C/S/F/F_hat/e_hat/r/alpha=None`，另返回H、U/J、原观测source及全False的B_comp；`fusion_type='late_concat'`。不能为满足形状伪造贡献权重。所有变体的logits/score、输入状态、ID和预测接口相同。

## 6. 数据读取、文本兼容核验与预处理

### 6.1 输入发现与三个文件层级

设置 `Q2_DATA_ROOT` 为服务器上已解压的 `E题数据` 目录。agent先在用户指定数据根目录或当前工作目录内用文件名查找；唯一命中即可填写，多个候选须依据完整附件目录选定，不能随便取第一个同名PKL。实际绝对路径写入本地配置，不把Windows路径复制到Linux。

```text
$Q2_DATA_ROOT/
  附件2-数据集特征文件/aligned_50.pkl
  附件3-模态缺失特征样本/对齐版本/附件3_01.pkl ... 附件3_30.pkl
  附件4-可解释专项视频样本与特征文件/
    附件4-可解释专项视频样本与特征文件/对齐版本/01.pkl ... 20.pkl
```

| 文件 | 读取方式 | 必需处理 |
| --- | --- | --- |
| 附件2 | `data[split][field][i]` | split为train/valid/test；不是 `data[split][i][field]` |
| 附件3 | `data['test'][field]`，已有样本维1 | 从 `(1,...)`读一条，再统一组批；ID为文件stem |
| 附件4适配接口 | `data[field]`，直接单样本 | 无test外层；输入无样本维；仅实现读取兼容，不开展Q3解释 |

已知附件2应为3395/728/727条；输入字段是text_bert、audio、vision、classification_labels、regression_labels、id、raw_text。aligned没有audio_lengths/vision_lengths，不制造这两个精确长度字段。只读 `aligned_50.pkl`，不加载未对齐版大文件。

附件2文本三路输入为int64；附件3text_bert为float32但已总结为整数值，先确认元素确为整数再转换long。三分类浮点标签也应核对其仅含0/1/2再转换long。音/视觉转float32。不要使用附件2预计算的 `text` 作为另一条训练入口。

文本轴顺序必须显式处理：附件2 `text_bert` 是 `(N,3,50)`，附件3是 `(1,3,50)`，附件4是 `(3,50)`；附件4先补一个样本维，再统一取 `input_ids=text_bert[:,0,:]`、`stored_attention=text_bert[:,1,:]`、`token_type_ids=text_bert[:,2,:]`。不能转成 `(N,50,3)` 后误把三路输入当作特征向量；RawBatch中的三个独立字段均为 `(B,50)`。

类别映射固定为 **`0=Negative`、`1=Neutral`、`2=Positive`**；logits、softmax概率、指标及CSV全部使用该顺序。读取现有classification_labels作为分类目标，regression_labels作为独立连续目标；检查其与负/零/正映射一致但不改写原标签，不用概率或预测分数重新生成分类真值。

本地已总结：train/valid/test视觉全零样本分别110/15/28条；划分之间样本ID和来源video_id不重合。检查并记录实际情况，不删样本凑数字、不重新划分。来源video_id由完整 `video_id$_$clip_id` 主键拆出；专项文件编号不是MOSEI原ID。

### 6.2 文本兼容核验：在训练之前完成

使用已下载的小型BERT词表构造uncased tokenizer。核对 `PAD=0,UNK=100,CLS=101,SEP=102,MASK=103`、词表大小30522；输出实际值。仅从train的raw_text核验以下两个已有数据常见组织方式：

1. **standard50**：`tokenizer(text, add_special_tokens=True, truncation=True, max_length=50, padding='max_length')`。
2. **prefix50**：先产生完整 `[CLS]+WordPiece(text)+[SEP]`，再取前50个ID，短序列右补PAD；注意力标记保留位置。超长序列的第49位置不强塞SEP。

比较存储的内容ID、注意力及分段ID，生成 `reports/tokenizer_compatibility.csv`：样本ID、有效长度、两种模式是否匹配、首个不匹配位置及对应token。核验时不使用情感标签。

至少一个统一模式解释全train时记录该模式；若长序列实际有两种截断来源，逐样本差异必须能**完全由这两种明确截断方式解释**，记录为混合截断。两种模式无法解释的样本逐一排查原文本规范化或词表问题，不能用“95%看起来一致”自动通过，也不能重分词覆盖官方输入。全部问题解释清楚之前不启动正式训练；其余代码和测试可继续完成。

核验的目的是证实词表和输入语义。**之后模型始终读取存储的IDs、mask和type IDs，不根据所选模式重新生成文本输入。** valid/test只进行相同结构与范围检查，不根据其效果换词表。附件3没有raw_text，依靠已证实的统一入口处理。

### 6.3 train标准化

当前音/视觉原值中 `any(row != 0)` 的行暂定为可用行。对每模态、每维按所有train可用行计算float64累计的总体均值μ和总体标准差σ（ddof=0），最后保存float32。σ小于1e-6的维度设分母为1，不除极小数；完全无可用行时μ=0、分母=1并明确记录。

`x_norm=(x-μ)/σ` 只用于可用行，不可用行最终重新置0。mask判定发生在标准化前。保存 `normalizer.npz` 的 `audio_mean/audio_std/vision_mean/vision_std` 及对应可用行计数；同一文件另存仅从train标签得到的 `class_prior(3,)` 和 `score_prior`（强度中位数），供空内容样本使用。不得在valid/test/附件3重新拟合。

### 6.4 转换与缓存

源PKL只由准备阶段读取一次，按split导出 `.npy`：input_ids、stored_attention、token_type_ids、audio、vision、class_id、score；ID/raw_text/video_id保存为UTF-8 JSON。忽略预计算text，转换后释放原大字典。Dataset通过只读mmap访问npy，禁止每个DataLoader worker重新加载原PKL。

原始数据不写回。train/valid/test必须使用独立数组与原顺序。训练只构造train DataLoader；test的模型指标直到最终选择后才计算。

冻结BERT原观测缓存为 `(N,50,256)` float32，非内容U_T位置保存0。缓存旁保存普通metadata：split、样本顺序、模型目录、词表及编码配置、生成时间；模型或预处理有变化就重建，使用普通配置和版本记录，不生成文件hash。

**受损文本缓存规则：** 如果本样本文本P全0且当前文本三路输入未变，可用其原观测缓存；否则用受损IDs和受损内容attention重新运行BERT。音频/视觉变化不要求重算未变化文本。固定valid场景可以按“split+场景名+样本序号”缓存受损BERT，训练随机文本遮蔽不做无界磁盘缓存。

## 7. 从当前输入得到U/J/q：训练与专项必须同函数

### 7.1 当前内容与结构状态

对每条样本50位置：

```python
active = stored_attention == 1
boundary = (input_ids == CLS) | (input_ids == SEP)
text_slot = active & (input_ids != PAD) & ~boundary
U_T = text_slot & (input_ids != MASK)       # 自然UNK仍是可用输入
U_A = (audio != 0).any(-1)                  # 标准化前的当前原值
U_V = (vision != 0).any(-1)
J = text_slot | U_A | U_V
K_bert = active & (input_ids != PAD) & (input_ids != MASK)
```

`K_bert`是送入BERT的attention_mask，保留CLS/SEP等编码结构，排除内容缺失槽位。`text_slot`包含MASK槽位，因而已知位置但内容删除的文本仍可申请补偿。不得把CLS、SEP的上下文输出放入内容证据池。

原始stored_attention继续保留作结构信息，不能用K_bert覆盖它再计算J。输入attention=0处的ID在BERT内部输入副本可置PAD，原文件不变。`U=[U_T,U_A,U_V]`，形状 `(B,50,3)`。由当前输入推断的全零行不宣称为真实人工P。

若某模态原本无内容，该模态U全0；其他模态继续处理。J为空时不虚构长度。所有索引0起，区间左闭右开，保留原存储位置，不删除空行压紧。

### 7.2 五维局部状态q

每条样本记 `nJ=sum(J)`，J非空时其包络长度 `Lcur=max(J_index)-min(J_index)+1`。对每模态m、每个J为1的位置t，q依次为：

1. 当前本模态可用比例：`sum(U_m)/max(nJ,1)`。
2. 当前t所属连续缺口长度：在原50位置轴上，对 `J & ~U_m` 的连续1段取长度，再除 `max(Lcur,1)`；t原可用时为0。
3. 原位置：`t/49`。
4. 距离本模态最近可用位置的距离除50；本模态无可用位置时为1。
5. 其他两模态总可用数除 `2*max(nJ,1)`。

J为0处q全0。q不读取原X0、U0、人工P、标签或“是人工删除还是天然无脸”的真因。自然缺口和人工缺口通过同一规则计算。

### 7.3 空集的统一语义

掩码均值分母0时返回零向量和 `has_observation=False`；不把零向量作为真实观测。注意力某样本没有任何有效Key时跳过该样本该操作，输出0并保持不可用；不要先做全负无穷softmax再用nan_to_num掩盖错误。

如果整条J为空，三分类概率采用train类别频率，强度采用train标签中位数，记录 `no_content=True`；保留该样本，不制造依据。先验从normalizer.npz读取，输出logits可取log(class_prior)，随后softmax恢复该分布。正常训练样本仍按各自真实标签学习，空样本常量输出不反向更新网络。

## 8. 连续遮蔽的完整定义

### 8.1 描述符与应用

遮蔽描述符包含 `modalities`（T/A/V组合）、每个模态的 `[start,end)`、名义跨度比例rho。以原观测J0包络 `[a,b)`、L=b-a定义跨度：

```text
width = min(L, max(1, floor(rho*L + 0.5)))
前段start = a
中段start = a + floor((L-width)/2)
后段start = b-width
随机start = 在整数[a,b-width]上均匀采样
end = start+width
P_m = U0_m & (start <= t < end)
```

rho=0或J0为空直接P全0。Python `round` 的银行家舍入不能替代上面的明确公式。P只表示新增删除，天然空洞不计入。

应用P时：音/视觉对应原值行置0；文本对应ID替换MASK=103，保留stored_attention、token_type_ids及50位置。随后调用 `infer_state` 从受损值重新推断U/J/q。被删除文本不重新分词，不用完整文本隐藏状态填入。

双模态同步缺失采用同一 `[start,end)`。错开缺失为两个模态独立采样start，width相同。完全没有新增删除时保留描述符并记录 `no_new_damage`，不偷偷按标签或预测正确率重新选样本。

### 8.2 60轮课程：epoch从1计

每个原train样本每轮恰好出现一次，另生成一个受损副本。原观测分支始终存在。

| epoch | 受损副本不再删除的概率 | 删除时的模态组合 | rho分布 | 起点 |
| --- | ---: | --- | --- | --- |
| 1—5 | 1.0 | 无 | 0 | 无 |
| 6—15 | 0.2 | T/A/V等概率 | Uniform[0.05,0.20] | 均匀随机 |
| 16—30 | 0.2 | 0.8概率单模态等选，0.2概率双模态等选 | Uniform[0.10,0.50] | 双模态同步 |
| 31—60 | 0.2 | 0.6概率单模态等选，0.4概率双模态等选 | Uniform[0.10,0.80] | 双模态90%同步、10%错开 |

主训练不添加整模态/三模态全段删除，它们放入压力评价。这里“模态等选”指T/A/V或TA/TV/AV内部均匀。若原模态全零，不改选成更容易的模态；照实记录此受损副本可能没有新增损伤。

用NumPy `SeedSequence([seed, epoch, original_sample_index])`创建本样本独立采样器，保证DataLoader的顺序或worker数变化不改变描述符；shuffle用独立torch Generator。保存每轮每样本的描述符到 `train_masks.jsonl`，不保存整份复制特征。

`uniform_spans`消融不随时间升难：从第6轮开始，先按权重 `10/55、15/55、30/55`选择上表后三种阶段分布，再按所选分布采样；原观测概率仍0.2。这样累计难度分布在期望上与课程接近，区别是先后顺序，而不是凭空增加严重缺失次数。

## 9. 编码器与MSD的逐层实现

### 9.1 冻结BERT是独立前端

`FrozenTextEncoder`不属于可训练学生或EMA参数副本。加载一次，始终eval、requires_grad=False；训练调用使用 `torch.no_grad()`，返回普通无梯度张量。不要在训练前端直接返回inference_mode创建的特殊张量给可训练Linear，以免其不能被保存用于反向。

输入 `input_ids/K_bert/token_type_ids`，取 `last_hidden_state(B,50,256)`，乘U_T后交下游。没有任何K_bert的样本跳过BERT并输出0；K_bert只有结构词元而无内容时仍不产生文本内容U。编码前后都不能恢复原被删词元。

### 9.2 共同空间和位置表示

三个独立输入投影分别为 `Linear(256,128)`、`Linear(74,128)`、`Linear(35,128)`，后接独立 `LayerNorm(128,eps=1e-5)`。加固定正弦位置编码p(t)及可学习模态embedding `Embedding(3,128)`。正弦定义：

\[
p_{t,2k}=\sin(t/10000^{2k/128}),\quad
p_{t,2k+1}=\cos(t/10000^{2k/128}),\quad t=0,\ldots,49.
\]

加完后按U_m置0。每个模态用两个**分别构造、独立初始化**的 `nn.TransformerEncoderLayer`：`d_model=128,nhead=4,dim_feedforward=256,dropout=0.1,activation='gelu',batch_first=True,norm_first=True,layer_norm_eps=1e-5`。不用共享层权重。

每层 `src_key_padding_mask=~U_m`（True表示屏蔽），只运行该模态至少有一个可用Key的样本；每层输出重新乘U_m。最后独立LayerNorm并再次乘U_m，得到H。对空模态整批/部分样本直接返回0，不把投影或LN偏置当作内容。

### 9.3 复用MSD

按T/A/V创建三个EBMC `MSDModule(128)`。每个类的两支均是 `Linear128→128,ReLU,Linear128→128`，按原类共用该模态的LayerNorm实例。输入H输出C/S，再把C/S乘U_m。另用独立 `LayerNorm(128)` 得到 `F=LN(C+S)`，最后乘U_m。

本轮不下载或使用EBMC训练好的预测权重，只复用代码核心并在附件2 train重新训练。`no_msd`变体直接定义 `C=H/2,S=H/2,F=LN(H)`，保持补偿器的输入意义和尺寸，关闭全部MSD辅助项。

### 9.4 MSD损失

只在clean学生输出上计算；掩码均值使用clean当前U。

- `L_shared`：对T-A、T-V、A-V三个模态对，取两模态均有观测的同一批样本I。若 `len(I)<2`，该对不参与。对池化C做L2归一化（eps=1e-8），相似度矩阵 `S=C_m @ C_n.T / 0.1`，正例为矩阵对角；计算 `0.5*(CE(S,arange)+CE(S.T,arange))`，再对有效模态对平均。
- `L_specific`：每个模态对中，两模态有观测的样本计算池化S的余弦平方，先样本均值，再有效模态对均值；仅1个样本也可计算。
- `L_uni`：每个模态的池化S输入训练用 `Linear(128,3)` 与 `Linear(128,1)`，回归输出 `3*tanh`。计算CE+Huber(delta=1)，先该模态可用样本均值，再有效模态均值。

`L_MSD=L_shared+0.1*L_specific+0.5*L_uni`。没有有效对象的项返回同设备标量0，不加入错误分母。不要复制片段标签到50位置后做逐位置有监督情感训练。

## 10. 补偿、误差估计与融合的逐层实现

### 10.1 共享补偿器

为了控制体积，三种目标模态共用Query投影、来源投影、注意力和FFN；通过目标模态embedding区分。只保留末端3个独立 `Linear(128,128)` 输出适配器。

对目标m：

1. `self_context=masked_mean(F_m,U_m)`，空集时0；计算发生在当前受损视图，不使用clean上下文。
2. `Q=Linear(384,128)(concat[p(t), embedding(m), self_context])`，生成50个查询。
3. 本模态来源是 `Linear(256,128)(concat[C_m,S_m])`；其他模态来源是共享 `Linear(128,128)(C_n)`。
4. 各来源加自己的p(u)和来源模态embedding，按固定T/A/V顺序拼成 `(B,150,128)`，来源有效掩码为同顺序的 `(B,150)` U。
5. `nn.MultiheadAttention(128,4,dropout=0.1,batch_first=True)`，Query=Q，Key=Value=上述memory，`key_padding_mask=~source_valid`；不加硬同索引限制或额外距离阈值。
6. `R=LN(Q+Dropout(attn_output))`；`R2=LN(R+Dropout(Linear256→128(GELU(Linear128→256(R)))))`；dropout=0.1、LN eps=1e-5。
7. `F_hat_m=target_output[m](R2)`，不再加原H残差。仅在 `query=J & ~U_m` 且来源非空的位置保留，其他置0。

`B_comp_m=query & any(source_valid)`。目标没有自身上下文但其他模态存在时允许补偿；所有来源为空时B_comp=0。所有m必须先从原C/S/F构造来源，再统一产生补偿，不能在循环中把已补出的值写回C/S/F供下一模态读取。

`return_details`可保存平均头注意力及来源 `(modality,index)`。注意力Key是补偿前来源；不把权重直接命名为“解释贡献”。

### 10.2 误差估计器

定义 `cross_context` 为其他两模态所有U=1位置的C均值，不是先对两个模态等权平均。空集时0。一个共享头：

```text
输入：F_hat(128) + self_context(128) + cross_context(128)
      + target_modality_embedding(128) + q(5) = 517维
整个拼接输入detach
Linear(517,64) -> GELU -> Linear(64,1) -> softplus
输出e_hat，形状(B,50,3)，只保留B_comp位置
```

误差头不使用dropout。`L_cal`只能更新这个头；模态embedding等输入也必须detach。融合时再detach e_hat，防止任务损失把误差估计器训练成任意路由器。

### 10.3 来源与内容融合

`F_tilde=where(U,F,where(B_comp,F_hat,0))`。原观测优先，U与B_comp不能同时为1。三值source的one-hot固定3维。

共享内容得分头输入：`F_tilde(128)+modality_embedding(128)+source_onehot(3)+q(5)=264维`，结构 `Linear264→64,GELU,Linear64→1`，无dropout。

```text
observed：    score = g
compensated： score = g - log1p(e_hat.detach())
unavailable： 不参加softmax
```

仅对至少一个来源可用的位置在模态维softmax；空位置alpha全0。`Z=sum_m(alpha*F_tilde)`。`z`是这些非空位置Z的简单均值，不采用attention pooling或CLS池化。最终 `Linear128→3` 输出logits；`3*tanh(Linear128→1)`输出score并只squeeze最后一维。

原观测r=1是来源参考尺度，不等于正确率100%；补偿可信系数也不是概率置信区间。保持独立分类/回归输出，不根据回归符号覆盖分类，报告符号不一致率。

## 11. 损失、梯度和EMA：不得自行补一种不确定性损失

### 11.1 任务损失与总式

分类使用无类别权重、无label smoothing的CrossEntropy；回归使用 `HuberLoss(delta=1,reduction='none')`。每个样本两者直接相加，回归权重1。

前5轮：`L = L_task(clean) + 0.05*L_MSD`。

第6—60轮，记 `ramp=min(1,(epoch-5)/10)`：

\[
L=L_{task}^{clean}+L_{task}^{corrupt}+0.05L_{MSD}
+ramp\,[0.2L_{span}+0.1L_{cal}+0.2L_{cons}].
\]

任务损失是batch样本均值，两个视图相加，不再除2。P全0的副本也保留第二次任务监督；训练dropout可不同。各消融的损失关闭规则见第14节，不能再加原CCE、trust loss或EMC。

### 11.2 span目标与误差校准

教师clean前向取**补偿之前**的F_teacher。`Omega = P & U0 & B_comp_student`。逐位置执行无仿射LayerNorm：`F.layer_norm(x.float(),(128,),weight=None,bias=None,eps=1e-5)`，得到ν(x)。

```python
e = ((nu(F_hat_student) - nu(F_teacher.detach())) ** 2).mean(-1)
L_span = masked_sample_modality_mean(e, Omega)
L_cal = masked_sample_modality_mean(
    smooth_l1_loss(e_hat, e.detach(), beta=1.0, reduction="none"), Omega
)
```

`masked_sample_modality_mean`：先对每个 `(sample,modality)` 的Omega位置取均值，再对Omega非空的样本—模态对取均值；分母不是B×50×3。全部为空时该项为0。原来无观测的视觉位置永远没有span目标，即使教师也产生了视觉补偿。

梯度必须满足：L_span更新补偿及其来源表示；L_cal只更新误差头；任务损失通过F_hat和内容得分学习补偿与融合，但不能通过r更新误差头；教师和冻结BERT均无梯度。

### 11.3 预测一致性

教师clean输出 `p_bar=softmax(logits_bar)`、`y_bar`，全部detach。每样本：

```text
w = p_bar[真实class] * exp(-abs(y_bar-真实score)/6)
KL = sum_c p_bar[c] * (log(p_bar[c]) - log_softmax(logits_corrupt)[c])
L_cons = mean_batch(w * (KL + Huber(score_corrupt, y_bar, delta=1)))
```

实际使用 `F.kl_div(log_softmax(student), teacher_prob, reduction='none').sum(-1)`，避免手工对0概率取log。温度固定1，无额外tau系数。w不进入模型，仅使用train真实标签降低错误教师影响。默认不按教师正确与否做硬阈值筛选。

### 11.4 EMA生命周期

epoch1—5没有teacher前向。完成第5轮最后一次学生更新后深拷贝学生可训练网络作为teacher，包括MSD、补偿、误差和输出头；冻结其梯度并设eval。

每次从epoch6开始的optimizer.step后，在no_grad内对同名参数执行 `teacher = 0.99*teacher + 0.01*student`；非参数buffer直接复制对应学生buffer。教师保持eval，不能随着 `student.train()`被切回train。冻结BERT在外部共享，不复制一份BERT放进teacher。

保存最佳学生时同时保存当时的teacher，仅供重现训练和valid误差分析。最后部署包不含teacher；验证选模、test和专项的预测均只能使用学生。

## 12. 固定配置、训练日程与续训

### 12.1 default.yaml至少包含以下值

```yaml
project:
  name: q2_sentiment
  data_root: /replace/with/actual/E题数据
  output_root: .
data:
  feature_version: aligned
  seq_len: 50
  modality_order: [text, audio, vision]
  dims: [256, 74, 35]
  num_workers: 4
  pin_memory: true
  train_batch_size: 32
  eval_batch_size: 128
  drop_last: false
text:
  model_dir: models/text_encoder
  model_id: google/bert_uncased_L-4_H-256_A-4
  frozen: true
  compute_dtype: float32
  stored_dtype: float16
  attention_implementation: eager
model:
  hidden_dim: 128
  encoder_layers: 2
  num_heads: 4
  ffn_dim: 256
  dropout: 0.1
  layer_norm_eps: 0.00001
  compensation_rounds: 1
  shared_compensator: true
  source_classes: 3
  q_dim: 5
  num_classes: 3
  class_names: [Negative, Neutral, Positive]
  score_min: -3.0
  score_max: 3.0
loss:
  regression_weight: 1.0
  huber_delta: 1.0
  msd_weight: 0.05
  shared_temperature: 0.1
  specific_inner_weight: 0.1
  unimodal_inner_weight: 0.5
  span_weight: 0.2
  calibration_weight: 0.1
  consistency_weight: 0.2
  ramp_epochs: 10
train:
  epochs: 60
  warmup_epochs: 5
  optimizer: adamw
  learning_rate: 0.0003
  min_learning_rate: 0.00003
  betas: [0.9, 0.999]
  adam_eps: 0.00000001
  weight_decay: 0.0001
  gradient_clip_norm: 1.0
  ema_decay: 0.99
  seeds: [1111, 1112, 1113]
  eval_epochs: [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
  amp: false
  tf32: false
  compile: false
evaluation:
  main_patterns: [T, A, V, TA, TV, AV]
  main_positions: [front, middle, rear]
  main_rhos: [0.2, 0.4, 0.6, 0.8]
  bootstrap_repeats: 1000
  bootstrap_seed: 20260924
  deployed_seed: 1111
```

把真实Linux数据路径写进本地配置后再运行。其他值本轮固定，先完成全部规定实验，不启动网格搜索或自行换一套模型。配置解析应显式识别已定义选项，拼错键不能悄悄失效；这只是正常配置读取，不另建冻结接口体系。

### 12.2 学习率与优化细节

epoch1—5的学习率为 `3e-4 * epoch/5`；epoch6—60为：

\[
lr_e=3\times10^{-5}+\frac{3\times10^{-4}-3\times10^{-5}}2
\left[1+\cos\left(\pi\frac{e-5}{55}\right)\right].
\]

在每轮开始设置一次，轮内恒定。AdamW只收requires_grad=True参数；二维及以上参数weight_decay=1e-4，一维参数及bias为0。每个batch零梯度、计算完整损失、一次backward、梯度全局范数裁剪1.0、step，然后更新EMA。

PyTorch/NumPy/Python使用相同run seed；关闭cuDNN benchmark、TF32，使用确定性算法的warn_only模式并记录警告。不承诺不同驱动/硬件逐位相同。网络使用PyTorch默认初始化，不用额外大规模预训练。

一次只运行一个GPU训练进程；默认 `CUDA_VISIBLE_DEVICES=0`、程序内部 `cuda:0`。DataLoader worker不加载CUDA模型，BERT仅在主进程运行。eval/no_grad后恢复学生train；冻结BERT和teacher仍保持eval。

若4090显存不足，先检查重复BERT、未释放计算图、全数据进GPU、其他进程占用等实际原因。eval batch可降64，缓存文本编码batch也可降32，不改变训练。正常独占24GB仍无法运行train batch32时记录实测和内存分配问题，先修复，不能无说明修改模型/精度来制造完成。

### 12.3 checkpoint和中断恢复

每个run至少保存：

```text
runs/<variant>/seed_<seed>/
  resolved_config.yaml
  history.csv
  train_masks.jsonl
  best.pt
  last.pt
  best_validation/metrics_per_scenario.csv
  best_validation/predictions.csv
  resource_usage.json
```

`last.pt`在完整epoch结束后保存学生、teacher（存在时）、optimizer、已完成epoch、当前最佳比较键、Python/NumPy/torch CPU及CUDA随机数状态，以及独立DataLoader shuffle Generator的状态。冻结BERT不重复存入checkpoint，只记录相对模型目录及配置。`best.pt`保存同一选中时刻的学生、teacher及epoch，用于重新评价。

`--resume`只接受相同变体/seed/配置的本run；从last完成的下一轮开始。中断在某轮中间时重跑这一轮，不能从一份未完整更新的history推测恢复点。每轮描述符由SeedSequence重建，完成轮记录不重复追加。与原配置不同的实验创建新的run名称，不覆盖已比较结果。

不使用早停，全部60轮保证各变体相同优化步数及课程覆盖。epoch10开始每5轮评估一次第13节完整valid网格，按选模规则保存best；训练epoch60结束不意味着必须部署最后一轮。

## 13. 评估、模型选择与统计口径

### 13.1 固定场景与清单

主网格为6种模态组合×3位置×4跨度=72个名义场景，另有clean。对每个valid样本根据自己的J0包络确定区间，保存描述符；所有种子、变体共用这些描述符。

每个描述符记录sample_id、pattern、position、rho、各模态区间、原可用数、新删数、实际删除率、是否无新增损伤。比较只使用这些记录，不按某个模型的错误样本重新抽mask。

取整后同一样本出现同一实际P时，复用同一受损输入及预测，记录 `duplicate_of`；不把它当作新的独立重复样本。每个名义场景表仍解释该条件，但不将多张表的样本数相加宣称更大样本量。若两个场景在**全split所有样本**的实际P完全一致，选模聚合只保留一个场景，另外标明等价。用数组相等比较即可，不引入hash。

额外压力网格在best模型上评估，不参与checkpoint或变体选择：

- 单模态T/A/V整个可用序列删除，共3种。
- 三模态同段删除，middle，rho=0.4与0.8，共2种。
- TA/TV/AV错开删除，rho=0.4，组合中第一个模态放front、第二个放rear，共3种。

### 13.2 全体与配对效应

每格同时输出all_samples指标和 `any(P)` 为真的damaged_only指标，另报原模态全缺、无新增损伤及空J数量。**选择模型只用all_samples主网格**，避免不同条件挑选不同样本影响排序。

题目规律分析增加共同样本比较：比较多个条件时，先取这些条件都真正删除内容的样本交集，再在同一交集计算每个条件及其clean结果。对于双模态的“两个模态均缺”分析，要求两个被指定模态均新增删除；全体表仍保留其他样本。

记录每模态实际删除率分布。比较模态敏感性时按实际删除率分到 `(0,.2],(.2,.4],(.4,.6],(.6,.8],(.8,1]`，给共同样本数；没有共同样本则标不适用，不补0、不跨组凑数。没有秒级时间映射，跨度只报官方位置数和比例。

### 13.3 指标实现

分类 `argmax(logits)`；三类顺序固定0/1/2。使用sklearn计算accuracy、macro-F1、weighted-F1及三类各自precision/recall/F1/support，F1传入 `labels=[0,1,2],zero_division=0`，额外标明组内缺少的类别。

回归使用原连续score，计算MAE与Pearson；不取整、不由类别重写score。Pearson在样本数不足2、预测或标签常量时写JSON null、CSV空值及原因。非有限预测是实现错误，不能改成0继续报告指标。

符号不一致率按 `Negative且score>=0`、`Positive且score<=0`、`Neutral且score!=0` 定义；这一严格口径会使独立连续回归头的中性样本大多不一致，报告时明确解释，不能当成网络bug强制改输出。另给Neutral预测样本的 `abs(score)` 均值和分位数，帮助审核实际偏离程度。

退化量在同一批样本上计算：clean Acc/F1减受损Acc/F1；受损MAE减clean MAE；clean PCC减受损PCC。未定义PCC的退化量仍为空，不混入平均。

### 13.4 每个run选best

对非重复主场景all_samples指标做场景等权平均，得 `missing_macro_f1`、`missing_mae`。以下顺序字典序比较，浮点并列容差1e-6：

1. missing_macro_f1更高。
2. missing_mae更低。
3. clean_macro_f1更高。
4. clean_mae更低。
5. epoch更早。

每次评估都保存指标表；改善时更新best和该轮逐样本预测。原观测表现单独展示，不能用“缺失平均更好”掩盖正常性能下降。

### 13.5 跨变体选择与固定部署种子

每个变体先取各seed自己的best，再对3个seed的四项关键指标取均值和样本标准差（ddof=1）。先剔除被另一变体在clean Macro-F1/MAE和missing Macro-F1/MAE四项均不差、至少一项更优所支配的变体，比较容差仍1e-6。

剩余候选按均值missing Macro-F1降序、missing MAE升序、clean Macro-F1降序、clean MAE升序、部署参数量升序、变体名称字典序排序，选第一名。**部署该变体seed=1111的best学生**；不临时挑三个seed中最高分者，不拼接多个权重。

若简化对照胜出，报告实际选中方法，并如实说明full未显示优势。不能隐藏对照、换指标把full硬选为第一，也不据此修改附件3结果。

选定后写 `reports/selection.json`：候选表、支配关系、排序键、最终variant、固定seed、best epoch及checkpoint相对路径。这是运行结果记录，不是新增冻结contract。

### 13.6 波动与补偿误差分析

核心比较报告3种子均值/标准差。主要差异另按来源video_id配对bootstrap1000次，seed=20260924：每次有放回抽取valid来源视频组，将该组全部片段一起带入，两个模型用相同组抽样，重新计算对应指标差。报告2.5%/97.5%分位，不把不同mask场景当独立样本扩大样本量。

对所有含误差头的变体，best对应EMA在clean valid上生成参照F，仅用于分析；与学生受损F_hat比较得到真实表示残差。保存e_hat和残差的Spearman相关、误差MAE，以及按e_hat五等分后的实际残差均值。相同值导致分位箱重复时合并并报告数量。该分析无反向传播，教师不参与受损预测。

自然全零视觉没有恢复目标，不纳入表示残差评价；单独报告该类样本的任务指标。若可信度排序不好或去可信度模型更优，如实写出，不给“校准可靠”结论。

### 13.7 test与专项

只有selection.json生成后，使用选定学生对test的clean、主网格和压力网格作一次留出报告；不比较一圈test再改selection。附件3只输出所选模型30条结果，不计算准确率。

附件3按原文件编号01—30排序，CSV至少为：

```csv
sample_id,feature_version,pred_label,pred_score,prob_negative,prob_neutral,prob_positive
```

每行sample_id为 `附件3_01` 等原stem，版本aligned，英文类别大小写固定；score输出足够精度，不四舍五入成整数。概率和为1，连续值在[-3,3]。另存状态分析文件，不把推断的零行比例写成真实人工缺失率。

## 14. 必做变体与全部实验清单

`configs/experiments.yaml`必须列出下表10个变体，每个seed=[1111,1112,1113]。训练共30个run，逐个执行，不并发抢占同一4090。全部使用同一冻结BERT、标准化、数据划分、60轮、优化器、评估清单。

该文件内容固定如下；seed列表只从default.yaml的train.seeds读取，避免两处定义不一致：

```yaml
variants:
  - full
  - late_clean
  - late_aug
  - no_msd
  - no_comp
  - no_reliability
  - no_cons
  - no_span
  - no_teacher
  - uniform_spans
```

network/trainer依据下表集中解析variant，不在多处各自猜默认开关。不存在的组件不实例化，避免将未使用参数计入模型规模或导出包。

| variant | 结构与训练差异 |
| --- | --- |
| `full` | 本文完整主方法 |
| `late_clean` | 无MSD/补偿/误差/教师；三个相同时序编码器后各自掩码池化，拼接三向量和3个可用标志得到387维，经Linear387→128、GELU、Dropout0.1，再接同样双头。前5轮一个clean任务；第6轮起两个clean前向任务相加，保持更新次数及任务项尺度 |
| `late_aug` | 同late_clean结构；第6轮起第二视图改为同一课程的受损视图；无教师和辅助损失 |
| `no_msd` | 用第9.3节C=H/2、S=H/2、F=LN(H)代替MSD，L_MSD=0，其余保留 |
| `no_comp` | B_comp全0，缺口不进融合；去补偿器、误差头、L_span/L_cal，保留MSD与预测一致性 |
| `no_reliability` | 补偿保留，所有可用来源r=1；移除误差头、L_cal，其余保留 |
| `no_cons` | 仅L_cons=0；EMA仍给span和cal目标 |
| `no_span` | 仅L_span=0；仍计算停止梯度残差供L_cal，EMA及一致性保留 |
| `no_teacher` | 无EMA、L_span/L_cal/L_cons；补偿由真实标签训练，r=1，删除误差头，保留MSD |
| `uniform_spans` | full结构与损失；使用第8.2节给出的固定混合难度采样代替课程 |

late系列的details按第5.2节用None表示不适用的隐藏结构，不编造时序融合alpha。上述联动关闭项是方法含义所要求，报告时说明，不能称为完全等参数量消融。

不把EMC、Soft-MoE、不同大模型比较再加进必做清单。先完整交付这30个run及分析；真实结果不支持方法优越性也应完成并报告，不能无边界搜索直到得到想要的结果。

## 15. 必须实现的CLI与实际执行顺序

### 15.1 CLI行为

所有命令支持 `--config configs/default.yaml`；从Q2根目录执行，相对路径以Q2根目录解释。每个命令在README说明输入、输出和重复执行方式。

| 命令 | 必须完成的动作 | 主要输出 |
| --- | --- | --- |
| `prepare-model` | 下载后的BERT转换、保存tokenizer，检查实际逐词元前向 | `models/text_encoder/`、环境与模型信息 |
| `inspect-data` | 清点附件2/3，核对层级、形状、ID、划分、标签编码及零观测；不运行test指标 | `reports/data_inventory.json/md` |
| `tokenizer-check` | 第6.2节全train词表/截断核验 | 兼容性CSV与结论MD |
| `prepare-data` | 转换npy/metadata、仅用train拟合标准化与先验 | `data/processed/` |
| `cache-text` | 为指定split保存原观测冻结文本表示；默认train/valid | `.cache/text/<split>/` |
| `make-masks` | 为valid或最终test生成主/压力场景清单；给相同P标重复 | `data/masks/<split>/` |
| `verify` | 调用第17节测试与实际模型训练更新检查 | `reports/verification.json/md` |
| `train` | 一个variant、seed的60轮训练，支持resume | 对应run的history、best/last及valid指标 |
| `train-suite` | 顺序运行experiments.yaml全部30个run；明确完成/待运行状态 | `reports/suite_progress.csv` |
| `evaluate-suite` | 重新载入每个best，在valid主/压力网格产生完整分析 | 指标、消融、误差与资源表 |
| `select-final` | 按第13节计算变体排序，选择固定seed学生 | `reports/selection.json` |
| `evaluate-test` | 只评价选定学生的test；不改selection | `reports/test/` |
| `predict-special` | 同一学生读取附件3对齐版，输出30行 | `outputs/q2_predictions_aligned.csv` |
| `export` | 导出源码、学生、完整冻结文本模型、标准化及运行入口 | `delivery/q2_inference/`及ZIP |
| `verify-export` | 在新进程中只使用交付包的代码/权重，离线重跑并比较 | `reports/export_verification.json/md` |
| `report` | 从实际结果生成技术报告、论文用结果材料、图表和交付清单 | `reports/report.md`、`reports/paper_q2_results.md` |

`train-suite`已有完整run时核对其配置后跳过；未完成run使用 `--resume` 从last续训；没有last就新建该run。不能以best.pt存在就判断60轮和评估已经全部完成。运行状态只是清单，不另外搭建任务调度平台。

### 15.2 命令模板

以下命令必须在agent实现对应CLI、创建配置并填好真实路径后执行：

```bash
cd "$Q2_ROOT"
source "$Q2_ROOT/.venv/bin/activate"
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export HF_HOME="$Q2_ROOT/.cache/huggingface"

python -m q2 prepare-model --config configs/default.yaml
python -m q2 inspect-data --config configs/default.yaml
python -m q2 tokenizer-check --config configs/default.yaml
python -m q2 prepare-data --config configs/default.yaml
python -m q2 cache-text --config configs/default.yaml --splits train valid
python -m q2 make-masks --config configs/default.yaml --split valid
python -m q2 verify --config configs/default.yaml

python -m q2 train-suite --config configs/default.yaml --experiments configs/experiments.yaml --resume
python -m q2 evaluate-suite --config configs/default.yaml --experiments configs/experiments.yaml
python -m q2 select-final --config configs/default.yaml

python -m q2 cache-text --config configs/default.yaml --splits test
python -m q2 make-masks --config configs/default.yaml --split test
python -m q2 evaluate-test --config configs/default.yaml --selection reports/selection.json
python -m q2 predict-special --config configs/default.yaml --selection reports/selection.json
python -m q2 export --config configs/default.yaml --selection reports/selection.json
python -m q2 verify-export --config configs/default.yaml
python -m q2 report --config configs/default.yaml
```

单run调试/续训必须支持：

```bash
python -m q2 train --config configs/default.yaml --variant full --seed 1111 --resume
```

不要假装以上是上游EBMC自带命令。服务器agent需要先把它们实现，再执行真实流程。正式套件日志要写文件，任务中断后可按suite_progress继续；禁止因某一轮变慢就跳过规定消融。

### 15.3 失败时执行什么

| 具体情况 | 确定的处理 |
| --- | --- |
| 找不到数据或有多个无法确定的版本 | 给出实际候选路径及缺失字段；继续独立源码实现，所需路径未确定前不冒用别的数据 |
| 网络下载失败 | 保留错误和目标官方地址，重试或使用同源离线文件；不能创建随机模型代替 |
| tokenizer不匹配 | 按样本定位差异，核实截断/文本规则；不能重分词覆盖官方输入或带错误词表训练 |
| 没有CUDA或驱动不支持主轮子 | 检查实际GPU/驱动；可改同torch版本cu118并重测；不能把CPU结果包装成4090训练完成 |
| NaN/Inf或全空注意力错误 | 定位首个出错模块和输入状态，修复掩码/损失；不能nan_to_num、丢样本或跳batch凑完成 |
| 正常输入与缺失效果都很差 | 先核对标签、mask、梯度、标准化、编码器兼容和优化日志；修实现错误后重跑受影响run |
| 实现正确但full不胜简单模型 | 完成对照、按固定规则选模型、如实报告；不伪造优越性或无限追加试验 |
| 推理包超预算 | 先移除重复checkpoint、缓存和冗余报告；不能删必需权重、标准化器或30条结果声称完整 |
| 缺失外部条件确实无法解决 | 明确缺什么、已完成什么、下一条可执行动作；不要把未完成任务写成通过 |

环境/代码错误修复属于本任务，agent应继续处理，不在每次可逆修复前请求用户确认。改变本文核心模型、输入版本或任务范围则必须明确报告理由，不能偷偷替换。

## 16. 导出、模型体积与Q3复用

### 16.1 导出包必须完整

```text
delivery/q2_inference/
  README.md
  run_inference.py
  requirements.txt
  model_config.yaml              # 只保留推理所需配置和相对路径
  student.safetensors            # 所选学生推理参数，FP32
  normalizer.npz
  assets/text_encoder/
    config.json
    model.safetensors            # 已从训练前统一的FP16文件复制
    vocab.txt
    tokenizer.json
    tokenizer_config.json
    special_tokens_map.json      # tokenizer实际产生的文件全部保留
  src/q2/                       # 自足的本项目推理代码与复用MSD源文件
  licenses/
  THIRD_PARTY.md
  q2_predictions_aligned.csv
```

student文件排除训练用单模态辅助头，推理构造时明确 `include_aux_heads=False`；只删除这个已知参数前缀，不能以宽松load掩盖其他权重遗漏。其他网络参数包括误差头（变体有时）必须保留。

冻结BERT独立存一份；不要同时把它保存在student、teacher与assets里。数据转换缓存、原下载FP32模型、优化器、30个run、teacher及原始题目数据均不放正式推理包。Q3以后直接使用同一份模型包，无需另存一套相同权重。

`run_inference.py`显式将包内src加入搜索路径，调用包内q2代码，路径相对于自身目录解析，支持：

```bash
python run_inference.py \
  --input-dir /actual/path/附件3-模态缺失特征样本/对齐版本 \
  --output ./q2_predictions_aligned.csv \
  --device cuda:0 --batch-size 128
```

不依赖Q2训练工作目录、`third_party/EBMC`、HF联网缓存或外部Windows路径。部署包在已经安装所列Python依赖的环境中可离线推理，依赖安装包本身不塞进50MB附件。

### 16.2 预期参数预算与实际核验

按指定BERT结构、去掉pooler计算11,104,768参数，FP16原始权重约22.21MB；vocab.txt、tokenizer.json及配置合计预留约1MB。按本文共享补偿器计算，full下游推理网络为1,375,046参数，FP32约5.50MB。两类权重加tokenizer按约29MB预算，均须用实际 `numel()`和文件大小核对；其他变体另算。

Q1已定设计的三组主特征原始体积约16.30MB，因此这套Q2与Q3共享参数预计仍有约5MB量级余量供Q1映射、源码与必要结果。不能据此直接宣布全题附件已小于50MB；全题限制按50,000,000字节保守计算，Q1实际包和Q3材料尚需一起汇总。

`reports/package_sizes.csv`逐文件列路径、字节数、用途；分别统计Q2完整推理目录、ZIP及和Q1/Q3合并时的已知总量。报告 `q2_package_bytes`、`remaining_before_other_parts`，未知的Q1/Q3实际体积写未知，不填估算冒充实测。

本实现包含通用BERT权重，不依赖“通用权重可以免交”的未确认豁免。不得到导出时才把学生转FP16或改成另一模型；如确需其他压缩，应另有完整valid与离线一致性验证，不能仅改文件后缀。

### 16.3 Q3接口

实现 `load_bundle(path,device)` 及 `predict_raw(raw_batch,return_details=True)`，复用本题三种字段层级适配。返回模态/位置U、J、source、B_comp、e_hat、alpha及补偿来源位置；保存预测所用官方索引和 `time_axis='official_aligned_positions'`。

不提供伪造的秒数、关键帧或词语情感标签。Q3应在原输入上删除证据并重新运行整个预测器；不能只删补偿后的视觉分支就说删掉了原视觉内容。原视觉全零的样本只能标明“没有原视觉证据”，不能把文本产生的视觉补偿当视觉贡献。

本轮只验证这套接口可用于附件4结构，不写完整Q3算法，不运行附件4来选择Q2模型。

## 17. 必须完成的语义测试与实际运行验证

测试围绕已知具体失败场景，不新增泛化的hash/gate体系。以下测试应可通过 `python -m pytest -q` 执行；涉及真实冻结BERT与CUDA的测试由 `python -m q2 verify` 调用，使用真实train输入。

### 17.1 数据与遮蔽

1. **三个文件层级**：构造附件2/3/4样式的单样本字典，核对同一内容得到同一RawBatch；特别防止附件3重复加样本维、附件4漏加维。
2. **标准化与零占位**：标准化统计只来自train可用行；修改valid值不改变统计量；带非零均值时全零不可用行在标准化后仍为0。
3. **连续区间**：用带天然洞的12位置例子验证 `[3,7)` 只新增删除原可用3/5/6，位置4不记P；原索引保留，分母正确。
4. **当前状态推断**：人工修改原值后重新infer_state得到预期U；恢复只来自原值而非传入P。自然UNK仍可用，CLS/SEP/MASK不当内容，MASK结构槽位仍在J。
5. **输入不被污染**：对Dataset取出的样本制造受损副本后，再次读取原样本仍是原值；不得因共享numpy视图永久把训练数据变成零。

### 17.2 信息隔离与注意力

6. **编码前删除测试**：同一个已选删除区间内，把原token ID或音视频原值换成另一组值，然后分别施加相同删除，重新跑完整前端和学生eval；剩余输入相同，预测应在1e-5绝对误差内一致。这验证不会读取完整缓存或隐藏原内容。
7. **文本重编码测试**：人工文本P非空时记录实际BERT调用输入，确认是MASK替换后的IDs和K_bert；不能只看最后输出零了就判通过。
8. **不可用Key隔离**：在底层注意力测试中固定U，改变被屏蔽输入的占位向量，合法输出不变；投影偏置/位置编码不能重新激活它。
9. **空观测测试**：一个模态全空、两个模态全空、局部三模态全空、整条无内容，均无NaN；来源为空不产生B_comp；整条无内容走记录明确的train先验输出。
10. **单轮来源测试**：调试输出中所有memory来源均有当前U=1，不能存在source=compensated的Key；顺序调换目标模态循环不改变eval结果。

### 17.3 梯度与损失

11. 仅反向L_cal，只有误差头有梯度；MSD、补偿器、模态embedding、教师与BERT无梯度。
12. 仅反向任务损失，误差头无梯度，但有实际参与融合的补偿器可获梯度；只用常量r实现而漏训补偿应能被发现。
13. 仅反向L_span，学生补偿及其可用来源可有梯度，teacher全部无梯度。
14. Omega只含新增且原可用的位置；在其他位置修改伪教师值不改变L_span/L_cal；没有Omega时损失合法为0。
15. 用手算两组不同长度span验证先样本—模态均值、再总均值，不被长span重复计权。
16. B=1时分类/回归形状仍是 `(1,3)/(1,)`，没有广播成矩阵；EMA更新数值等于0.99旧teacher+0.01新student。

### 17.4 小规模实际可学习性检查

从train按固定原顺序选首32条至少有内容的样本，建立独立调试模型；冻结同一BERT，关闭dropout、MSD和所有辅助损失，只优化真实双任务，学习率1e-3，最多300步。保存初始/末尾任务损失、Acc和MAE，检查损失显著降低、模型参数确实更新；这只用于发现标签/反向/数据流水线错误，不是模型性能结果，不参与正式训练或选模。

若损失基本不降或输出保持常量，继续检查真实梯度、标签广播、训练/eval状态和学习率；不能以“代码能跑完”认定实现正确。调试模型不得复用为正式run的起点，正式run按对应seed重新初始化。

对 `full` 至少完成一个真实batch的clean/受损/teacher全损失前向与反向，记录每项损失、Omega数及峰值显存。随后才启动全部实验。无CUDA时这些实际运行项目写明未执行，不能宣称verify全部通过。

### 17.5 离线导出验证

新进程设置 `HF_HUB_OFFLINE=1` 和 `TRANSFORMERS_OFFLINE=1`，从独立临时工作目录运行交付包入口；显式使用交付包src，输出实际 `q2.__file__` 路径证明没有导入训练目录的editable代码。

使用相同设备、batch大小、8条valid代表样本的clean及固定受损输入，比较训练载入版本和交付包：logits及score最大绝对差≤1e-5、最终类别一致；再重新预测附件3全部30条，数值同样比较。样本应包含文本删除、音视频缺失和原视觉全零条件。

由于文本权重从训练前已统一为同一FP16存储版，导出时不应引入新的权重舍入差异。检查包中确有冻结BERT、学生和标准化，关闭联网后仍能预测。失败时修复路径、漏权重、eval状态或配置，不放宽容差掩盖换模型。

## 18. 报告、交付与停止条件

### 18.1 技术报告report.md

首页先给实际完成状态：代码是否完成、30个run是否完成、模型选了什么、专项30条是否完成、离线验证是否通过、Q2包实际多少字节。不得把“待运行”改写为已通过。

正文必须包括：

1. 实际环境、下载模型、上游版本、代码复用位置及与本文的必要差异。
2. 数据统计、文本兼容结果、视觉全零数量、标准化分母与状态工作规则。
3. 主网络结构、参数量、每个损失、梯度隔离及测试结论。
4. 全10个变体×3个seed的best epoch、clean/缺失指标、训练时间、峰值显存。
5. 模态、位置、跨度曲线与对应有效样本数、实际删除率；原观测与受损退化量。
6. 组件消融、教师/误差估计分析、压力测试及失败案例；不只展示改善案例。
7. 最终模型选择的依据、固定seed部署说明、独立test结果。
8. 附件3全部30条预测表；没有真实专项标签，不能计算准确率。
9. 导出验证、实际文件清单与体积、Q3复用接口和未解决的数据限制。

所有方法/执行流程图使用文字框和箭头。指标曲线可用matplotlib保存PNG/PDF并同时输出原始CSV，不用网页截图代替数值结果。

### 18.2 paper_q2_results.md

形成可以继续写入论文的材料，而不是直接编造结论：方法简述、参数表、主实验表、消融表、三类缺失规律、可靠度诊断、典型/失败例、30条专项结果及结论边界。清楚区分valid开发结果、test留出结果和无真值专项应用。

主要表必须直接展开，不能只有“详见CSV”的占位句。图表列出来源文件和指标定义，PCC未定义处保留原因。图中“时长”注明官方位置跨度，不虚构秒数。

### 18.3 完成条件

只有同时满足以下事实才可报告本次Q2实现完成：

- `Q2/`为独立可运行项目，源码、依赖和README齐全，EBMC和模型来源可追踪。
- 指定模型实际下载并证实输入词表兼容；没有靠维度相同混接官方text。
- 掩码、补偿、梯度与实际运行验证完成，无已知未修复的数据泄漏或空注意力错误。
- 必做30个run均有60轮记录、best和完整valid结果；未完成部分明确列出，不拿一个run代替套件。
- 依规则确定最终学生，test仅在此后评价，专项30行唯一且完整。
- 自足推理包真实离线重载通过；模型/标准化齐全，实际体积已统计。
- 技术报告与论文结果材料引用的是当前代码和当前产物，没有旧报告配新权重。

“完整方法没有超过简单对照”不妨碍如实完成实现，但必须限制论文的效果结论；“模型能导入或loss是有限值”不足以代表全部工作完成。全题50MB合包状态与Q2自身完成状态分别报告，尚无Q1/Q3实际包时不能替它们宣布通过。

## 19. 可直接复制给服务器agent的任务文字

> 请按本文完整实现并运行E题Q2。服务器为Linux x86_64 + NVIDIA RTX 4090，本机没有现成Q2代码。先在我的工作父目录下**单独新建大写Q2文件夹**；本题源码、虚拟环境、下载模型、上游代码、缓存、实验和交付包都放在Q2中，题目原始数据只读引用。
>
> 自行下载 `https://github.com/kangverse/EBMC.git` 和 `google/bert_uncased_L-4_H-256_A-4`，按本文只复用MSDModule核心，不运行原EBMC训练入口。严格实现输入适配、编码前连续遮蔽、当前状态推断、共享/特有分解、单轮补偿、监督误差系数融合、EMA与双任务；不要自行增加EMC/IMTD/Soft-MoE或更换aligned输入。
>
> 本文已确定依赖、模型层次、损失、60轮训练、10变体×3种子、评估和选择规则。请真正写出CLI并执行，不把示例命令当现成工具，不只交脚手架或计划。先完成文本兼容核验与关键语义测试，再顺序训练全部run、汇总valid、选择固定seed学生、运行test及附件3全部30条，最后导出完整离线推理包并验证。
>
> 遇到代码/环境/数值问题继续定位修复并重跑受影响部分；遇到确实缺失的数据路径、无法取得的指定模型或无法解释的词表差异，说明精确阻碍，继续独立可完成的实现，不能用随机权重、删除样本或虚构结果凑完成。不要无边界调参追逐预期结论。
>
> 输出README、THIRD_PARTY、测试结果、全实验表、技术report.md、paper_q2_results.md、30条预测CSV、实际模型包及体积清单。流程图用文字；Q3只预留和验证预测器/来源接口，不开始完整Q3算法。完成时明确每项的真实状态、最终选中模型、复现命令和仍未解决事项。

## 20. 来源与本文件的验证范围

- 方法依据：已审核《E题_Q2_模态局部缺失鲁棒情感预测方法_v2_审核稿.md》；本任务书已展开服务器实现所需内容，不要求服务器有该原稿。
- 数据事实：已有《E题数据细节.md》中的实际字段、划分、标签与零观测总结；本轮未重新读取大规模源PKL。
- 复用代码：[EBMC官方仓库](https://github.com/kangverse/EBMC)。本地副本中已核对MSD、CCE、EMC、IMTD、序列前向、读取及评估；远端下载时记录实际版本。
- 文本模型：[Google小型BERT配置](https://huggingface.co/google/bert_uncased_L-4_H-256_A-4/blob/main/config.json)、[文件列表](https://huggingface.co/google/bert_uncased_L-4_H-256_A-4/tree/main)及[Transformers 4.51.3 BERT接口](https://huggingface.co/docs/transformers/v4.51.3/en/model_doc/bert)。
- 环境：[PyTorch官方历史安装表](https://docs.pytorch.org/get-started/previous-versions/)。查阅日期2026-09-24；这些是选定的可复现版本，不宣称为最新版本。

本文已进行文档层面的结构、配置、张量尺寸、参数预算和执行依赖检查；未在用户4090服务器实际安装或训练。因此运行成功、模型效果与最终包体积必须由服务器agent执行后报告，不能把本说明当作运行证明。
