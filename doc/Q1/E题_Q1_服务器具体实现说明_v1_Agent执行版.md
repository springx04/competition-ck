# E题 Q1 服务器具体实现说明（v1，Agent执行版）

> 编写日期：2026-09-23。适用环境：**Linux x86_64 + NVIDIA GPU**。本文件是已审核的Q1方法的实现任务书，不是已经运行完成的结果报告。服务器上可以只有题目数据和本文；实现 agent 须自行下载下述代码、依赖及预训练权重，再创建本文指定的项目。
>
> 方法来源：`E题_Q1_多模态特征构建方法_v3_审核稿.md`。本文已包含实现必需的信息，不要求服务器存在本地Windows项目或先前对话。此次工作仅交付实施文档；实际实现、下载大模型和100条提取由下一阶段在服务器完成。

## 1. 实现范围与不可自行改变的选择

实现一个命令行项目 `q1_features`：读取附件1的100条MP4及已有英文转写，输出每条样本的三模态原生特征、50窗特征、可用状态、覆盖率和可回到原素材的来源表。**不训练情感分类器，不重新ASR替换转写，不读取情感标签调参，不实现Q2/Q3预测网络。**

| 项目 | 本次唯一主实现 |
| --- | --- |
| 文本编码器 | `google-bert/bert-base-uncased`，`BertModel`，末层768维；冻结、eval、float32 |
| 音频提取器 | `opensmile==2.5.1`，`eGeMAPSv02`，`LowLevelDescriptors`，25维 |
| 视觉提取器 | OpenFace `OpenFace_2.2.0`，`FaceLandmarkVidMulti`，静态AU模式；17个AU强度+3个头部旋转角+2个视线角，共22维 |
| 词级对齐 | ESPnet Conformer的CTC输出 + `ctc-segmentation==1.7.4`；使用给定转写 |
| 对齐模型 | `espnet/kamo-naoyuki_librispeech_asr_train_asr_conformer5_raw_bpe5000_schedule-truncated-c8e5f9` |
| 输入音轨 | 第一条音频流，16,000 Hz，单声道float32；不做响度归一化、降噪或VAD裁剪 |
| 输入视频 | 第一条非封面视频流；按原始PTS抽取目标25 Hz图像，低帧率不复制源帧 |
| 输出时间轴 | 覆盖完整片段的50个等宽时间窗；不是50个词或官方50个序列位置 |
| 聚合 | 有效区间与目标窗的交叠时长加权；角度用圆均值；覆盖率用区间并集 |
| 缺失 | 保留时间位置；零占位并记录状态；不插值、前向填充或删除缺失位置 |
| 标签 | 另存标签表，提取模块的函数签名不接收label/annotation |
| 并发 | 一张可用GPU、一个GPU工作进程、模型分阶段加载；OpenFace默认1个进程 |
| 随机数 | Python/NumPy/PyTorch种子2026；不承诺跨硬件逐位一致 |

CPU退路只用于GPU架构不支持、驱动不可用或单样本显存不足，使用相同模型与float32计算，记录实际设备。不能因此换成小模型、Whisper、wav2vec2、MediaPipe、88维声学特征或整句CLS。

### 1.1 服务器必须获得的输入

1. 本文。
2. 已解压的题目数据，至少包含附件1。原始目录名可以保留中文。
3. 一个用户有写权限的项目目录。代码、模型与中间结果均写这里。

已知附件1目录结构：

```text
E题数据/
└── 附件1-数据集原始多模态样本/
    └── MOSEI数据集部分原始视频-100条/
        ├── label-100.xlsx
        ├── <video_id>/<clip_id>.mp4
        └── ...
```

主表列为 `video_id, clip_id, text, label, annotation`。共37个来源、100个片段，已知时长约2.648–34.567秒。样本主键必须是 `video_id + "$_$" + clip_id`，不能只用clip_id。上述数字用于核对输入版本；以实际文件为处理对象，发现不一致写明差异，不能删行凑成100条。

### 1.2 阅读顺序

审核者优先看第2节流程图、第3节下载清单、第8—13节算法和第18节完成条件。实现 agent 按第4节建环境、第5节下载、第6节创建代码结构，再按第16节逐阶段运行。文中 `python -m q1_features ...` 是**需要 agent 实现的CLI**，不是已经存在的上游命令。

## 2. 详细流程图

图内R表示复用计算核心，A表示改造封装，N表示新增题目组织逻辑。所有分支都回填同一份样本索引；局部失败不删除样本。

### 图A：从空服务器到全量输出

```text
[附件1 + 本文]
       │
       ▼
[N 建项目与索引：100个完整主键，标签另存]
       │
       ▼
[下载指定代码/模型 → 安装依赖 → 真实模型前向与设备核验]
       │
       ▼
[N 解析源PTS、各流偏移、共同起点t0及片段时长D]
       │
       ├── 原文 ──→ [N 字符/词映射] ──→ [R+A 冻结BERT] ────┐
       │                    │                               │
       │                    └─→ 大写英文BPE ──┐             │
       │                                      ▼             ▼
       ├── 音轨 ──→ [A 16k波形] ──→ [R+A CTC词对齐] → [按word_id连接]
       │                 │                                  │
       │                 └─→ [R+A eGeMAPSv02 LLD，25维] ──┐ │
       │                                                  │ │
       └── 视频 ──→ [N 25Hz选源帧]                         │ │
                         │                                │ │
                         ▼                                │ │
                  [R+A OpenFace：身份+质量+22维] ──┐        │ │
                                                 ▼        ▼ ▼
                                    [N 50窗聚合，详见图E]
                                                 │
                                                 ▼
                           [100条特征 + V/O/P + coverage + 来源映射]
                                                 │
                                                 ▼
                         [全量验证 → 完整/局部成功/失败报告 → 体积统计]
```

### 图B：时钟与源帧，不能用抽帧序号替代原始时间

```text
[ffprobe流信息 + PyAV逐帧PTS]
                 │
                 ▼
[t0=min(所选音视频首个有效PTS)；公共时间=源时间-t0]
                 │
        ┌────────┴─────────────────────────────┐
        ▼                                      ▼
[音轨首样本偏移a0]                    [源帧v_j及展示区间]
        │                                      │
        ▼                                      ▼
[源音频帧是否连续、可解码？]           [q=r/25时选择正在展示的源帧]
        │                                      │
        ├─是→ [x[n]对应a0+n/16000]              ▼
        │                             [同一源帧只导出一次]
        └─否→ [A及词对齐失败；其他继续]           │
                                               ▼
                             [保留源展示区间，不填满舍弃帧的间隔]
                                               │
                                               ▼
                             [image_index → 源帧编号/PTS/支持区间]

上述两支统一用片段内秒数；D取所有可信展示末端最大值。
不能把音轨、视频分别从0重新计时后直接拼接。
```

### 图C：文本同源映射与长文本处理

```text
[原始text：字符不改动]
           │
           ▼
[非空白词段 → word_id + 原文半开字符区间]
           │
    ┌──────┴──────────────────────────────────┐
    ▼                                         ▼
[BERT完整分词一次]                  [CTC专用规范化：大写/数字展开]
    │                                         │
    ▼                                         ▼
[全局子词ID/word_id/字符offset]     [SentencePiece piece → ESPnet词表ID]
    │                                         │
    ▼                                         ▼
[510子词一块，步长382]              [整段音频CTC分段，每词一个候选区间]
    │                                         │
    ▼                                         ▼
[BERT末层 → 重叠块内同一子词平均]   [得分/区间/UNK检查]
    │                                    │             │
    ▼                                    │通过         │失败
[词内所有内容子词平均，768维]              ▼             ▼
    └─────────────────────→ [word_id连接]       [原词和词向量保留]
                                  │             [时间null + 失败原因]
                                  ▼
                     [按交叠时长聚合进50窗]
                     [子词多的词不增加时间权重]
```

### 图D：视觉身份和观测状态

```text
[图像序列 + 源帧映射]
           │
           ▼
[FaceLandmarkVidMulti：所有脸候选与失败记录]
           │
           ▼
[整片只有一条可信连续轨迹，且无其他成功脸候选？]
           │是                              │否
           ▼                                ▼
[采用单一可见人物假设]             [有人工目标区间表？]
[assumed_single_visible]                    │
           │                          ┌─────┴─────┐
           │                          │有         │无
           │                          ▼           ▼
           │                  [指定face_id]  [身份未知，视觉缺失]
           │                          │      [全部候选原样保留]
           └────────────┬─────────────┘
                        ▼
             [选定目标唯一 + success=1 + confidence≥0.8 + 22项有限？]
                        │
                  ┌─────┴──────┐
                  │是          │否
                  ▼            ▼
         [有效22维，含合法零AU] [无脸/低质量/身份未知/坏数值]
                  │            │
                  └─────┬──────┘
                        ▼
       [按源展示区间聚合；失败不移动后续帧的时间]
```

人工表明确选择目标后，多脸同帧可以保留**被指定的唯一目标**；图D的“只有一个候选”指按身份规则筛选后的候选数。无人工表时，不用最大脸或靠近中心等规则冒充说话人识别。

### 图E：一个目标窗内的计算与缺失区别

```text
[目标窗 I_k=[kD/50,(k+1)D/50)]
                 │
                 ▼
[读取相交原生记录及eligible、失败原因]
                 │
                 ▼
[仅eligible=1：w_i=max(0,min(e_i,窗末)-max(s_i,窗首))]
                 │
                 ▼
             [Σw_i > 0？]
                 │
         ┌───────┴─────────────────────────┐
         │是                               │否
         ▼                                 ▼
[普通维度：Σ(w_i*x_i)/Σw_i]       [特征=0；observed=0；coverage=0]
[角度：atan2(Σw_i*sin,Σw_i*cos)]  [记录无词/无脸/失败等原因]
         │                                 │
         ▼                                 │
[observed=1；coverage=有效区间并集/窗宽]     │
         │                                 │
         └──────────────┬──────────────────┘
                        ▼
       [time_map：source_id、交叠秒数、归一化权重]
                        │
                        ▼
       [valid=真实时间结构；Q1正常全1]
       [perturb=人工删除；Q1全0]
       [usable = valid * observed * (1-perturb)]
```

## 3. 下载清单和真正复用的位置

### 3.1 仓库版本及模型定位

| 对象 | 下载位置 / 版本 | 服务器落点 | 用途 |
| --- | --- | --- | --- |
| MMSA-FET | [thuiar/MMSA-FET](https://github.com/thuiar/MMSA-FET)，提交 `f8fbd2d88d4f77580ea1ded0b3469073488c5c19` | `third_party/MMSA-FET/` | 本文核对的提取与CTC代码参考 |
| OpenFace | [TadasBaltrusaitis/OpenFace](https://github.com/TadasBaltrusaitis/OpenFace)，标签 `OpenFace_2.2.0` | `third_party/OpenFace/` | 编译C++提取程序与下载脸模型 |
| ESPnet | PyPI `espnet==202402`；[对应源码](https://github.com/espnet/espnet/tree/v.202402) | Python环境 | ASRTask、Conformer及CTC头 |
| CTC segmentation | PyPI `ctc-segmentation==1.7.4`；[官方实现](https://github.com/lumaku/ctc-segmentation) | Python环境 | 给定文字与CTC帧的动态规划 |
| BERT | [google-bert/bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased) | `models/bert/` | 文本语义特征 |
| 英文Conformer | [正式模型仓库](https://huggingface.co/espnet/kamo-naoyuki_librispeech_asr_train_asr_conformer5_raw_bpe5000_schedule-truncated-c8e5f9) | `models/ctc/` | 词对齐，非情感模型 |

MMSA-FET本地副本的包版本写为0.4.2，但不能仅凭这个数字选择历史tag：本次核对的 `v_0.4.2` 路径没有现行 `aligner/default.py`。因此下载上表的**普通Git提交版本**。不要建立额外文件哈希系统。

原代码中很长的 `kamo-naoyuki/librispeech_...valid.acc.ave` 是旧模型标识。本文给出了现有ESPnet组织下的迁移地址；其 `meta.yaml` 明确列出相同Conformer5训练目录及 `valid.acc.ave_10best.pth`。按新地址下载，不能让agent继续搜索或猜测替代模型。[模型文件目录与元数据](https://huggingface.co/espnet/kamo-naoyuki_librispeech_asr_train_asr_conformer5_raw_bpe5000_schedule-truncated-c8e5f9/blob/main/meta.yaml)

### 3.2 复用矩阵：下载后读这些文件，按右列改造

| 上游文件（相对MMSA-FET仓库） | 复用内容 | 本题必须做的改造 / 禁止直接沿用的行为 |
| --- | --- | --- |
| `src/MSA_FET/aligner/default.py` | ASRTask加载、CTC log_softmax、prepare_token_list、ctc_segmentation、determine_utterance_segments | 改模型路径；修正大写词表；eval；保留原始log分数；建立word_id；不静默丢UNK；补流偏移；不调用ASR识别 |
| `src/MSA_FET/extractors/text/bert.py` | BertTokenizerFast、BertModel末层提取 | 一次完整分词贯穿映射与编码；增加长文本块；排除特殊词元；保留字符offset |
| `src/MSA_FET/extractors/audio/opensmile.py` | Smile初始化、process_signal、DataFrame输出 | 强制LLD，保留DataFrame起止时间及列名，不能只留中点 |
| `src/MSA_FET/extractors/video/openface.py` | OpenFace外部进程和CSV读取的组织思路 | 用subprocess参数列表；换多脸程序；按列名取22维；保存质量和脸ID；使用真实源PTS；不调用其虚构FPS时间轴 |
| `src/MSA_FET/single.py` | 单样本分阶段组织、词与子词连接思路 | 新建自己的runner，不用原来的词区间平均+子词复制作为50时间窗算法 |
| `src/MSA_FET/utils.py` | FFmpeg调用思路 | 新增解码PTS表和偏移；不能只输出BMP/WAV路径 |
| `src/MSA_FET/dataset.py` | 仅阅读以理解旧接口 | 不作为入口；它要求附件1没有的split与标签字段，异常会漏样本，部分Aligner接口不一致，还会截取序列前缀 |
| `src/MSA_FET/example_configs/aligned.json` | 对照配置结构 | 只作参考，视觉开关、FPS和聚合规则全部使用本文配置 |

**复用实施方式：** 下载整个仓库供核对，随后把需要的短计算核心适配进 `src/q1_features/extractors/` 与 `alignment.py`。不要 `pip install MMSA-FET`，不要 `from MSA_FET import FeatureExtractionTool`，否则会引入本题不使用的MediaPipe/ASD等依赖和旧批处理假设。保留复制代码的版权头，在 `THIRD_PARTY_NOTICES.md` 写来源、提交、复制文件和改动摘要；随复用代码保留其GPL许可证。OpenFace及预训练模型各自保留原许可说明。

## 4. 环境、依赖和安装次序

### 4.1 目录变量与机器前提

下面是Bash命令，不能在Windows PowerShell照搬。实现 agent 将 `Q1_ROOT` 指向当前任务的可写目录，将 `Q1_DATA` 指向已解压的 `E题数据` 目录；这两项是部署路径，不是模型设计选择。若任务已有工作目录，使用该目录下的 `q1_features`，不要写死 `/root` 或某个Windows盘符。

```bash
export Q1_ROOT="$PWD/q1_features"
export Q1_DATA="/绝对路径/E题数据"
mkdir -p "$Q1_ROOT"/{third_party,models,env,configs,src/q1_features,tests,scripts,runs}
cd "$Q1_ROOT"
uname -m
nvidia-smi
```

要求 `uname -m` 为x86_64。若是aarch64，不把本文x86_64二进制装上去；保留已经生成的源码和数据索引，报告架构差异。磁盘先按额外30GB、内存16GB、GPU显存8GB作为资源准备参考，不作为已测量峰值；首批实际记录内存与显存。源数据不复制进项目。

环境使用已有conda/mamba创建独立Python环境。若均不存在，在用户目录安装micromamba后使用相同conda-forge创建语句；安装程序来源使用[官方micromamba安装说明](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html)，不得改系统Python。下面命令以 `micromamba` 为例，已装conda可将 `micromamba create/run -p` 等价替换成 `conda create/run -p`。

无环境管理器时，先运行下列引导命令。这里latest只用于环境管理器自身；项目模型和提取依赖仍按指定版本安装。

```bash
mkdir -p "$Q1_ROOT/tools"
curl -fL --retry 3 https://micro.mamba.pm/api/micromamba/linux-64/latest \
  -o "$Q1_ROOT/tools/micromamba.tar.bz2"
tar -xjf "$Q1_ROOT/tools/micromamba.tar.bz2" \
  -C "$Q1_ROOT/tools" bin/micromamba
export PATH="$Q1_ROOT/tools/bin:$PATH"
export MAMBA_ROOT_PREFIX="$Q1_ROOT/env/mamba-root"
```

```bash
micromamba create -y -p "$Q1_ROOT/env/python" -c conda-forge \
  python=3.10.14 pip=24.0 ffmpeg=6.1.1 libsndfile=1.2.2 \
  c-compiler cxx-compiler make pkg-config
export Q1_PY="$Q1_ROOT/env/python/bin/python"
export PATH="$Q1_ROOT/env/python/bin:$PATH"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export HF_HOME="$Q1_ROOT/models/hf_cache"
```

编译Python扩展时应在环境激活后执行，或使用 `micromamba run -p "$Q1_ROOT/env/python" ...`，使conda编译器的环境变量生效。`Q1_PY` 用于后续统一调用，不依赖交互shell切换。

### 4.2 Python依赖文件的确切内容

创建 `env/requirements-q1.txt` 如下。PyTorch先单独安装，后续此文件同时限制它不被依赖解析升级。锁定会影响模型/数值和已知兼容问题的直接依赖；普通传递依赖由pip解析后保存 `pip freeze`，不引入额外依赖管理框架。

```text
torch==2.1.2
torchaudio==2.1.2
numpy==1.23.5
scipy==1.10.1
pandas==1.5.3
numba==0.57.1
llvmlite==0.40.1
scikit-learn==1.3.2
librosa==0.9.2
soundfile==0.12.1
resampy==0.4.3
transformers==4.38.2
tokenizers==0.15.2
huggingface-hub==0.23.5
safetensors==0.4.3
espnet==202402
typeguard==2.13.3
sentencepiece==0.1.97
ctc-segmentation==1.7.4
importlib-metadata==4.13.0
opensmile==2.5.1
audinterface==1.2.3
audformat==1.1.2
audresample==1.3.3
audiofile==1.3.2
av==12.3.0
openpyxl==3.1.5
PyYAML==6.0.2
num2words==0.5.13
matplotlib==3.7.5
Pillow==10.4.0
tqdm==4.66.5
pytest==8.3.3
```

依据：ESPnet202402明确要求 `numpy<1.24`、`librosa==0.9.2`、`sentencepiece==0.1.97`、`typeguard==2.13.3` 和 `importlib-metadata<5`。因此不能套用“最新版NumPy+最新版ESPnet”的安装指令。[ESPnet版本依赖](https://pypi.org/pypi/espnet/202402/json)

安装顺序：

```bash
"$Q1_PY" -m pip install setuptools==69.5.1 wheel==0.43.0 Cython==0.29.37 numpy==1.23.5
"$Q1_PY" -m pip install torch==2.1.2 torchaudio==2.1.2 \
  --index-url https://download.pytorch.org/whl/cu118
"$Q1_PY" -m pip install --no-build-isolation -r env/requirements-q1.txt
"$Q1_PY" -m pip check
"$Q1_PY" -m pip freeze > env/installed-python.txt
ffmpeg -version > env/ffmpeg-version.txt
```

先安装NumPy和Cython是为了CTC segmentation等源代码扩展的编译。这里不安装 `espnet[all]`、CUDA Toolkit、Kaldi训练工具、ASR数据集或语言模型；ESPnet标准依赖中带来的辅助包由pip正常安装。`espnet-model-zoo` 不作为运行依赖，模型直接由HF下载并按本地路径加载，避免旧下载器的地址转换和路径修补行为。[PyTorch官方2.1.2 CUDA11.8安装组合](https://pytorch.org/get-started/previous-versions/)

若pip提示依赖冲突，先检查是否确实在新环境、是否被其他requirements升级，再恢复上表。若确有当日传递依赖不兼容，仅把报错依赖调整到满足上游声明的版本，写入 `env/compatibility-notes.md` 并重新 `pip check`；不能忽略冲突或改变模型/特征定义。本文做过上游声明核对，**没有在目标Linux服务器执行整套安装**，不能将这份依赖表描述成已经服务器实测通过。

### 4.3 NVIDIA设备选择

默认 `cuda:0`，`batch_size=1`。先执行真实张量运算，再在第5节下载后执行一次BERT和一次CTC前向。仅 `torch.cuda.is_available()` 为True不够。

```python
import torch
if torch.cuda.is_available():
    x = torch.ones((64, 64), dtype=torch.float32, device="cuda:0")
    y = x @ x
    torch.cuda.synchronize()
    print(torch.cuda.get_device_name(0), float(y[0, 0]))
else:
    print("CUDA_UNAVAILABLE: use the same models on CPU")
```

决策顺序固定：CUDA运算成功→使用GPU；显存不足→释放其他提取器，仅保留当前模型再试一次；仍不足→该阶段切CPU；架构不支持/no kernel image/驱动不兼容→该阶段切CPU并记录。特别新的GPU可能不支持本旧版wheel，不自动升级整套ESPnet依赖或安装驱动。CPU也失败才记该阶段失败。BERT和CTC分两阶段运行，避免同时占显存；OpenFace、openSMILE均在CPU运行。

### 4.4 OpenFace原生依赖与编译

OpenFace不通过Python pip安装。为避免要求服务器sudo，创建独立C++依赖环境；它的Python包与Q1的Python环境互不混装。

```bash
micromamba create -y -p "$Q1_ROOT/env/openface" -c conda-forge \
  cmake=3.28 ninja=1.11 gcc_linux-64=12 gxx_linux-64=12 \
  opencv=4.8.1 dlib=19.24 libopenblas=0.3.26 boost-cpp=1.82
```

OpenFace2.2.0声明使用C++17、OpenCV4、dlib和OpenBLAS。[版本说明](https://github.com/TadasBaltrusaitis/OpenFace/releases/tag/OpenFace_2.2.0) 若服务器已有Ubuntu22.04系统开发依赖，也继续使用上述独立环境以减少路径混用。若conda源无法解析该组合，使用Ubuntu22.04容器的 `build-essential cmake libopencv-dev libdlib-dev libopenblas-dev libboost-all-dev` 编译同一tag；需要容器服务已可用，不在任务中配置宿主机Docker或改驱动。不得悄悄换成浮动的 `algebr/openface:latest`。

第5.3节模型文件下载后再运行CMake；OpenFace在配置时会复制模型，顺序反过来会导致build/bin缺模型。

```bash
micromamba run -p "$Q1_ROOT/env/openface" cmake \
  -S "$Q1_ROOT/third_party/OpenFace" \
  -B "$Q1_ROOT/third_party/OpenFace/build" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$Q1_ROOT/env/openface" \
  -DCMAKE_INSTALL_RPATH="$Q1_ROOT/env/openface/lib" \
  -DOpenBLAS_INCLUDE_DIR="$Q1_ROOT/third_party/OpenFace/lib/3rdParty/OpenBLAS/include" \
  -DOpenBLAS_LIB="$Q1_ROOT/env/openface/lib/libopenblas.so" \
  -DDLIB_USE_CUDA=OFF
micromamba run -p "$Q1_ROOT/env/openface" cmake \
  --build "$Q1_ROOT/third_party/OpenFace/build" \
  --target FaceLandmarkVidMulti FeatureExtraction --parallel 2
```

上述 `OpenBLAS_INCLUDE_DIR` 和 `OpenBLAS_LIB` 是该tag实际使用的CMake项；头文件使用上游自带版本，避免conda库包缺 `f77blas.h`。编译进程内存不足时将并行度降至1。运行OpenFace也用 `micromamba run -p env/openface`，工作目录设置为 `third_party/OpenFace/build/bin`；不要把此环境的动态库路径永久写到系统配置中。保存 `micromamba list -p "$Q1_ROOT/env/openface" --json` 的实际解析版本到 `env/installed-openface.json`。

## 5. 代码和模型下载：不依赖本地预置代码

### 5.1 下载源码

```bash
cd "$Q1_ROOT"
git clone https://github.com/thuiar/MMSA-FET.git third_party/MMSA-FET
git -C third_party/MMSA-FET checkout f8fbd2d88d4f77580ea1ded0b3469073488c5c19
git clone --branch OpenFace_2.2.0 --depth 1 \
  https://github.com/TadasBaltrusaitis/OpenFace.git third_party/OpenFace
git -C third_party/MMSA-FET rev-parse HEAD > env/mmsa-revision.txt
git -C third_party/OpenFace rev-parse HEAD > env/openface-revision.txt
```

目录已存在时先检查其remote和当前版本；匹配则复用，不覆盖未知目录。下载失败是环境问题，保留日志并重试同一地址，不改用来源不明的打包文件。

### 5.2 下载BERT与CTC所需文件

在项目根目录执行以下Python代码，可保存为 `scripts/download_models.py`。`snapshot_download`下载实体权重而非Git LFS文本指针。模型下载后，提取阶段使用本地路径并关闭自动远程加载。

```python
import json
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download

root = Path.cwd()
jobs = [
    ("google-bert/bert-base-uncased", "bert", [
        "config.json", "model.safetensors", "tokenizer.json",
        "tokenizer_config.json", "special_tokens_map.json", "vocab.txt", "README.md"
    ]),
    ("espnet/kamo-naoyuki_librispeech_asr_train_asr_conformer5_raw_bpe5000_schedule-truncated-c8e5f9",
     "ctc", ["meta.yaml", "README.md", "data/token_list/**",
             "exp/asr_train_*/config.yaml", "exp/asr_train_*/*.pth",
             "exp/asr_stats_*/train/feats_stats.npz"]),
]
records = []
api = HfApi()
for repo, name, patterns in jobs:
    revision = api.model_info(repo).sha
    dst = root / "models" / name
    snapshot_download(repo_id=repo, revision=revision, local_dir=dst,
                      local_dir_use_symlinks=False, allow_patterns=patterns)
    records.append({"repo_id": repo, "revision": revision,
                    "local_dir": str(dst.relative_to(root))})
(root / "models" / "download-record.json").write_text(
    json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
```

这里记录的是Hub自带的版本标识，不对数据建立额外SHA256机制。模型名与架构固定；记录实际下载版本后，续跑使用同一批本地文件。无需下载CTC仓库里的 `exp/lm_train_*`，本题不用解码语言模型。

CTC必须包含以下文件（`ASR_DIR` 为完整训练目录名）：

```text
ASR_DIR = exp/asr_train_asr_conformer5_raw_bpe5000_scheduler_confwarmup_steps25000_batch_bins140000000_optim_conflr0.0015_initnone_accum_grad2_sp
models/ctc/<ASR_DIR>/config.yaml
models/ctc/<ASR_DIR>/valid.acc.ave_10best.pth
models/ctc/data/token_list/bpe_unigram5000/bpe.model
models/ctc/exp/asr_stats_raw_bpe5000_sp/train/feats_stats.npz
```

编写 `prepare_ctc_config()`：读取下载的ASR YAML，把 `bpemodel` 和 `normalize_conf.stats_file` 的相对路径解析到 `models/ctc`，写为绝对路径，另存 `models/ctc/runtime-config.yaml`。其他模型参数保持不变。`token_list` 在此模型中是直接内嵌的列表；如果读取结果不是列表，明确报版本不匹配。原配置的 `normalize=global_mvn` 使用模型自带训练统计，这是预训练模型组成部分，不能删除，也不能用附件1重新估计。原始YAML不覆盖。

**已核对的模型结构：** 16k前端、Conformer编码器12块、输出512维、8个注意力头、卷积核31、BPE5000、英文大写词表。CTC头只为时间定位服务，不作为25维音频情感特征。[模型配置](https://huggingface.co/espnet/kamo-naoyuki_librispeech_asr_train_asr_conformer5_raw_bpe5000_schedule-truncated-c8e5f9/blob/main/exp/asr_train_asr_conformer5_raw_bpe5000_scheduler_confwarmup_steps25000_batch_bins140000000_optim_conflr0.0015_initnone_accum_grad2_sp/config.yaml)

### 5.3 OpenFace额外模型文件

源码Git仓库中的AU预测器、PDM和眼部模型保留。还须下载官方 `download_models.sh` 指定的四个CEN patch专家文件到：

`third_party/OpenFace/lib/local/LandmarkDetector/model/patch_experts/`

```bash
OF_PATCH="$Q1_ROOT/third_party/OpenFace/lib/local/LandmarkDetector/model/patch_experts"
mkdir -p "$OF_PATCH"
curl -fL --retry 3 'https://www.dropbox.com/s/7na5qsjzz8yfoer/cen_patches_0.25_of.dat?dl=1' -o "$OF_PATCH/cen_patches_0.25_of.dat"
curl -fL --retry 3 'https://www.dropbox.com/s/k7bj804cyiu474t/cen_patches_0.35_of.dat?dl=1' -o "$OF_PATCH/cen_patches_0.35_of.dat"
curl -fL --retry 3 'https://www.dropbox.com/s/ixt4vkbmxgab1iu/cen_patches_0.50_of.dat?dl=1' -o "$OF_PATCH/cen_patches_0.50_of.dat"
curl -fL --retry 3 'https://www.dropbox.com/s/2t5t1sdpshzfhpj/cen_patches_1.00_of.dat?dl=1' -o "$OF_PATCH/cen_patches_1.00_of.dat"
```

这些是项目官方脚本中的公开模型链接，不涉及用户Dropbox账号。[原脚本](https://github.com/TadasBaltrusaitis/OpenFace/blob/OpenFace_2.2.0/download_models.sh) 下载后核实文件不是HTML错误页，运行程序加载真实模型，并核对 `build/bin/model/main_ceclm_general.txt`、patch文件和AU预测器存在。链接若失效，仅使用OpenFace官方安装页提供的同名模型备用地址；找不到时记录视觉环境未就绪，不换检测器后继续声称使用OpenFace2.2.0。

## 6. 需要创建的项目结构、接口与配置

### 6.1 文件职责

```text
q1_features/
├── pyproject.toml                   # src布局；安装包名q1-features；不重复声明另一套版本
├── README.md                        # 从数据路径到运行命令的使用说明
├── THIRD_PARTY_NOTICES.md            # 复用文件、版本、许可证及改造内容
├── env/                             # 环境、requirements、实际版本记录
├── models/                          # BERT、CTC权重；不提交到Git
├── third_party/                     # 下载的两套源码；不复制进竞赛代码包
├── configs/q1.yaml                  # 下面的完整参数
├── configs/target_face_segments.csv # 初始只有表头；身份审核结果
├── configs/alignment_review.csv     # 初始只有表头；错配复核结果
├── scripts/download_models.py
├── src/q1_features/
│   ├── __init__.py
│   ├── __main__.py                  # argparse子命令
│   ├── config.py                    # YAML加载、相对路径解析
│   ├── manifest.py                  # Excel、主键、路径、标签分离
│   ├── media.py                     # ffprobe、PyAV、时间轴、WAV与图像
│   ├── text_map.py                  # 原文词段、BERT子词、CTC规范化
│   ├── alignment.py                 # 适配MMSA-FET的CTC核心
│   ├── extractors/
│   │   ├── bert.py
│   │   ├── audio.py
│   │   └── vision.py
│   ├── pooling.py                  # 交叠权重、并集覆盖、圆均值
│   ├── quality.py                  # 模态错配排查、人工复核合并
│   ├── storage.py                  # NPZ/JSONL/CSV读写
│   ├── runner.py                   # 按阶段执行、保留失败样本、续跑
│   └── report.py                   # 统计表、文本时间线、来源示例、体积
└── tests/                          # 第18节的实质性测试
```

不新增数据库、Web服务、任务队列、自动调参框架或分布式训练。`argparse + dataclasses + NumPy` 足够。流程说明图、审核时间线均输出**文本框、箭头和表格**，不使用浏览器绘图、Mermaid或网页截图。原始视频截帧仅作为待检查素材，不承担流程图表达。

`pyproject.toml`使用以下最小内容；依赖已由requirements安装，不在这里再求解另一套依赖：

```toml
[build-system]
requires = ["setuptools==69.5.1", "wheel==0.43.0"]
build-backend = "setuptools.build_meta"

[project]
name = "q1-features"
version = "0.1.0"
requires-python = ">=3.10,<3.11"

[tool.setuptools.packages.find]
where = ["src"]
```

### 6.2 关键函数输入输出

以下是建议的明确接口，agent可拆分内部辅助函数，但不能改字段含义。`NativeSeries` 为简单dataclass：`values float32[n,d]`、`intervals float64[n,2]`、`eligible bool[n]`、`source_ids int64[n]`；更丰富的信息留在旁边的JSONL。无数据时n=0，维度仍为d。

| 函数 | 输入 | 输出与副作用 |
| --- | --- | --- |
| `build_manifest(data_root, out_dir)` | 题目数据根目录 | manifest.jsonl、labels.csv、输入差异说明 |
| `probe_and_decode(sample, cfg, work_dir)` | 不含标签的样本记录 | MediaInfo、WAV、帧图像、audio_frames.jsonl、video_frames.jsonl |
| `build_words(raw_text)` | 原始字符串 | 按原文顺序的Word列表；word_id从0开始 |
| `encode_words(words, bert_dir, device)` | Word列表 | word_features、子词表、块表、每词语义状态 |
| `align_words(words, wav, media, ctc_model)` | 相同Word列表与波形 | 逐词候选时间、log得分、CTC中间状态、诊断识别文字 |
| `extract_audio(wav, media, smile)` | 波形及公共偏移 | 25维NativeSeries、列名与源帧区间表 |
| `extract_vision(frames, media, cfg, reviews)` | 源帧表、身份人工表 | 22维NativeSeries、全部候选CSV、身份及质量说明 |
| `audit_pairing(sample, media, words, native, reviews)` | 不含label的数据与诊断 | 样本错配状态、逐问题记录、是否可作为已对齐组合使用 |
| `pool50(series, duration, kind)` | 已确定eligible的单模态序列 | 50窗特征、observed、coverage、CSR来源映射 |
| `write_sample(...)` | 三支输出与质量结果 | 固定命名的样本文件；失败也写status |
| `collect_run(run_dir, manifest)` | 全部样本结果 | 100行紧凑数组、全量报告；不自动排除失败行 |

每个外部命令用 `subprocess.run(list_of_args, check=True, capture_output=True, text=True)`，路径作为单独参数，不用shell拼接。单样本外部命令最长30分钟；超时记录原命令、返回信息和样本ID。Python异常保留 traceback；只在样本/阶段边界捕获，不能在特征循环里把任意异常换成0而标成功。

### 6.3 `configs/q1.yaml`

```yaml
schema_version: 1
seed: 2026
paths:
  data_root: null       # CLI --data-root必填一次，随后写入run配置
  bert_dir: models/bert
  ctc_dir: models/ctc
  openface_bin: third_party/OpenFace/build/bin/FaceLandmarkVidMulti
  openface_env: env/openface
  face_review: configs/target_face_segments.csv
  alignment_review: configs/alignment_review.csv
media:
  sample_rate: 16000
  target_video_hz: 25.0
  audio_gap_tolerance_samples: 2
  audio_duration_tolerance_s: 0.02
text:
  model_id: google-bert/bert-base-uncased
  hidden_layer: last
  content_tokens_per_chunk: 510
  chunk_stride: 382
  word_pool: mean
alignment:
  model_id: espnet/kamo-naoyuki_librispeech_asr_train_asr_conformer5_raw_bpe5000_schedule-truncated-c8e5f9
  ctc_blank_id: 0
  min_log_score: -5.0
  score_min_mean_over_L: 30
  min_window_size: 8000
  max_window_size: 100000
  transcript_case: upper
audio:
  feature_set: eGeMAPSv02
  feature_level: LowLevelDescriptors
vision:
  confidence_min: 0.8
  au_mode: static
  target_policy: single_episode_else_review
  trajectory_gap_s: 0.20
  bbox_iou_split: 0.10
  bbox_center_jump_fraction: 0.25
  scene_hist_l1_split: 0.65
pool:
  bins: 50
  coverage: union
  angle_resultant_min: 0.000001
quality:
  diagnostic_wer_warn: 0.65
  aligned_word_fraction_warn: 0.80
  crossmodal_review_required: true
runtime:
  device: cuda:0
  gpu_workers: 1
  openface_workers: 1
  batch_size: 1
  dtype: float32
  allow_same_model_cpu_fallback: true
  external_command_timeout_s: 1800
```

这些阈值是预先指定的工程规则，未经本数据集质量标注优化；报告应这样说明。不能看情感label调整它们。模块中不再保留另一个相互矛盾的默认值。所有相对路径以项目根目录解析；写入run目录的 `config.yaml` 保存最终路径与参数。首次prepare的 `--data-root` 覆盖YAML的null；后续命令未传该参数时使用run里保存的data_root，不能重新把它变回null。比较续跑配置前，先按这个规则解析成实际生效参数。

## 7. 样本索引与输入读取

1. 在 `--data-root` 下递归定位名为 `label-100.xlsx` 的文件。恰有一个则使用；有多个时按预期附件1目录优先精确匹配，仍有多个则要求CLI `--labels-xlsx` 指定，不任意取第一个。
2. 使用openpyxl读取工作簿，寻找表头同时含五个规定字段的sheet。只把字段名去两端空格；保留text原文。唯一匹配即使用；多个匹配输出各sheet名称、行数，让数据路径参数确定具体表，不合并重复sheet。
3. `video_id` 保留字符串；数值型 `clip_id` 必须为整数，转十进制无小数的字符串。原本字符串如 `01` 不擅自去前导零；先按原值匹配文件，只有数值Excel单元格才执行整数转换。空ID、重复主键记输入错误，不能无记录丢弃。
4. MP4路径严格为Excel所在目录的 `<video_id>/<clip_id>.mp4`。如某个数值ID的实际文件名有前导零，列出同目录候选；唯一数字等价值可建立显式 `path_resolution=integer_equivalent`，否则记ambiguous_path，不靠全文搜索另一附件补文件。
5. 排序使用 `(video_id, clip_id)` 的Unicode字符串字典序，固定 `sample_index=0..N-1`。同时保留Excel原行号，不能用不同阶段各自的文件遍历顺序拼接数组。
6. manifest只含 `sample_index, sample_id, video_id, clip_id, video_relpath, raw_text, excel_sheet, excel_row, input_status`。labels.csv单独含 `sample_id,label,annotation`。缺文件的样本仍在manifest中。
7. 工作目录用 `samples/000000/` 等数值索引，避免样本ID中的符号造成shell路径问题；目录与完整主键映射写入manifest。
8. 验证37个来源/100条，与已知统计不一致就报告。无需为了Q1加载附件2巨型PKL；本阶段只保留18条与附件2可能重合的已知风险说明。

空text保留为空字符串，文本原生数组长度0、50窗文本全缺失；音视频继续。Excel缺失值不能转成字面字符串 `nan`。读取过程既不改变数据，也不把标签喂给任何模型。

## 8. 媒体解码、共同时间轴与抽帧

### 8.1 使用真实PTS建立时间轴

先调用：

```bash
ffprobe -v error -show_format -show_streams -of json /绝对路径/sample.mp4
```

命令由Python列表传参，保存原始JSON。选第一条 `codec_type=video` 且非 `attached_pic` 的流和第一条音频流；存在其他流记录到日志。PyAV解码这些流，以解码输出顺序记录 `source_frame_index`，从0开始。保存整数PTS和有理数time_base，公共秒数用float64计算。

- 帧时间优先 `frame.pts * frame.time_base`；缺PTS时检查ffprobe相同展示帧的 `best_effort_timestamp_time`，明确记录 `timestamp_basis=best_effort`。两者都不存在时该记录不可用于时间定位。
- 视频按展示PTS排序；同一PTS的重复帧保留源记录，但仅首个解码成功帧作为该时刻候选。明显逆序、负duration和坏帧记录原因，不能以循环下标/平均FPS重建成“真值”。
- 共同起点 `t0 = min(第一有效视频展示PTS, 第一有效音频帧PTS)`，只对存在的流取最小。公共时间均减t0。元数据同时保留容器/流声明的start_time供比较。
- 视频源展示末端来自帧duration；不可用时用下一不同PTS的间隔，保守限制不超过本片正相邻PTS差的中位数，并标 `duration_basis=estimated`；末帧也用该中位数。只有一个有效帧时可用流声明的平均帧率推定一帧，标估计。无法得出正时长则媒体结构失败。
- 音频帧末端为 `PTS + nb_samples / sample_rate`。D取所选音视频可信展示末端的最大值再减t0。容器format.duration只用于核对，不覆盖实际解码时间。
- 输出的 `bin_edges = np.linspace(0.0,D,51,dtype=np.float64)`。若D无效，全样本status=failed，valid全0、时间占位0；不虚构1秒的片段。

这种D约定覆盖有音无画、先画后音和片尾静音。由样本文件内部时间定义的偏移，并不能证明内容属于同一片段；第14节另查语义/素材错配。

### 8.2 音频连续性与16k波形

先检查原始已解码音频帧相邻边界。容差为原采样率的2个采样点：`abs(next_start-current_end) <= 2/source_sr`。超过容差、内部样本率/通道布局突然改变、解码出非有限PCM，都列出具体时段。

**本版遇到内部缺段/重叠不自动拼接或填静音**：保存源帧表和异常，A提取及CTC词时间阶段标 `audio_discontinuous`，视觉和BERT语义继续。音轨整体晚于视频的正常起点偏移不算内部缺段。该保守分支避免把解码故障伪装成说话人的停顿；需要修复的样本进入明确问题清单。

连续音轨使用FFmpeg：

```bash
ffmpeg -nostdin -v error -i /绝对路径/sample.mp4 \
  -map 0:a:0 -vn -ac 1 -ar 16000 -c:a pcm_f32le /本次样本目录/audio_16k.wav
```

既不加 `-ss/-t` 截头尾，也不加 `loudnorm`、`dynaudnorm` 或 `silenceremove`。不做VAD删静音。FFmpeg执行前目标文件应不存在；显式重跑时仅替换本样本派生WAV，不覆盖原视频。

`a0=首个有效解码音频PTS-t0`。重采样波形x的第n个采样对应 `a0+n/16000`。读取WAV确认16000Hz、单声道、float32，检查 `abs(len(x)/16000 - 源连续音轨解码时长) <= 0.02s`；超限进入 `audio_duration_mismatch` 复核，不通过拉伸波形匹配视频时长。

记录重采样前采样率/布局、首PTS、解码总样本数、派生样本数及偏移。派生采样区间回溯到源音频帧时按时间交叠记录，不声称重采样后一个采样点与原PCM点严格一一对应。

### 8.3 视频采样的明确规则

1. 建立每个源视频帧的公共展示区间 `[v_j,e_j)`。
2. 生成 `q_r=r/25`，`r=0..ceil(25D)-1`，仅处理 `q_r<D`。
3. 对每个q选择区间包含q的源帧。相同源帧被多个q命中时只导出一次，保存 `requested_grid_indices`。q落在视频起点前、片尾后或可确认缺段时记录无候选，不从最近帧复制补齐。
4. 导出图像按公共PTS升序命名 `000001.png` 等。`image_index` 从1开始，与OpenFace CSV的 `frame` 对应；`source_frame_index` 仍是原解码索引。
5. 图像保持原宽高；应用视频显示旋转元数据（90/180/270度无插值旋转），保存原尺寸、显示尺寸与旋转值。无需缩放、裁脸或预平均；其他旋转值列为需检查，不静默忽略。
6. 特征支持区间只采用**被选中的源帧展示区间**，不扩到相邻选中帧的中点。30/60fps降采样后的部分未观测间隔因此会降低视觉coverage；这不是人脸检测失败。报告同时给“采样覆盖”和“在已采样时段内的人脸有效率”。
7. PNG源帧映射写成表：`image_index,filename,source_frame_index,source_pts,time_base_num,time_base_den,start,end,duration_basis,requested_grid_indices,decode_status`。

这样10fps输入不会被复制成25个独立观测，而每帧真实展示的0.1秒仍可被其区间覆盖。遇到可变帧率使用实际PTS，不以声明FPS作为时钟。选帧为空时视觉阶段失败但样本保留。

## 9. 文本768维：词、字符与子词全部可追溯

### 9.1 原始词段

使用Python标准库 `re.finditer(r"\S+", raw_text)`，每个匹配作为一个父词段。编号按原文顺序，不因大小写、标点或数字规范化改变。字段：

```text
word_id: 0起的整数
raw_word: 原文完整非空白词段，如 "don't," 或 "2020"
char_start, char_end: 原始Python字符串字符索引，半开区间
bert_piece_indices: 全局内容子词索引列表
ctc_text: 仅对齐用的规范化文本
semantic_status: ok / empty / model_failed
alignment_status: 第10节定义
```

字符索引按Python Unicode code point，不是UTF-8字节或JavaScript UTF-16下标。必须满足 `raw_text[char_start:char_end] == raw_word`。标点不从BERT输入删掉，原文不强制转小写；uncased tokenizer自行处理。

独立纯标点词段记录 `attached_to_word_id`：优先附着左侧最近的非纯标点词，没有左词则附着右词；全篇都是标点则为null。附着用于来源阅读，不为标点创建独立发音区间或额外时间权重；标点语义仍通过BERT上下文影响对应词。原word_id与原字符区间不删除、不重排。

### 9.2 BERT一次分词，完整分块

```python
from transformers import BertTokenizerFast, BertModel
tokenizer = BertTokenizerFast.from_pretrained(bert_dir, local_files_only=True)
model = BertModel.from_pretrained(bert_dir, local_files_only=True)
model.eval()
model.requires_grad_(False)
model.to(device=device, dtype=torch.float32)
encoded = tokenizer(
    [w.raw_word for w in words],
    is_split_into_words=True,
    add_special_tokens=False,
    truncation=False,
    return_offsets_mapping=True,
    return_attention_mask=False,
    return_token_type_ids=False,
)
word_ids = encoded.word_ids()
```

此段运行模块需另行 `import torch`，`bert_dir/device/words` 由函数参数提供。`offset_mapping` 在预分词模式中是**父词内部**的索引，加该词char_start才成为原文索引。记录token_id、token文本、word_id、父词内offset、全局字符区间。不要第二次对字符串重新分词生成另一套映射。

全局内容子词数为L。若L=0，输出空词语义序列。否则块起点从0开始，每次增加382，截取最多510个内容子词；一块已到达L后停止，不再创建完全被已有末块覆盖的多余块。每块两端加 `[CLS]` 和 `[SEP]`，位置ID在块内从0重置，attention全1，token_type_ids全0；单块batch1无需padding。

每块输入长度不超过512。使用 `torch.inference_mode()` 得到 `last_hidden_state[0,1:-1,:]`，不用pooler_output、MLM logits或多层拼接。记录块覆盖的全局子词区间。一个全局子词在重叠块出现多次时，其向量为所有出现的等权平均；再对每个父词的内容子词取均值，形成 `(W,768)`。所有求和先float64累加、最终float32；模型本身float32。

`[UNK]` 是内容子词，保留其语义向量并记unknown标志；CLS/SEP/PAD不进入父词特征，也不分配视频时间。少见的单个父词超过510子词时允许跨块，最后仍汇成一个词。完整保留词序列尾部，绝不 `[:50]` 截断。

词向量能包含其所在编码块的上下文信息；它的词时间只是定位锚点。后续证据说明必须保留这个限制，不能把时间窗向量说成仅由窗内词内容产生。

## 10. 词级强制对齐：模型加载、输入和质量处理

### 10.1 CTC文本规范化与词表映射

仍以第9节的父word_id为单位；为每个父词构造一条 `ctc_text`：

1. Unicode NFKD分解并去组合音标；把弯引号转 `'`。先识别数字表达，再清理标点，避免把小数点/百分号过早删掉。
2. 数字串采用 `num2words(...,lang="en",to="cardinal")` 展开。数字之间的千位逗号先删除；小数按整数部分+`POINT`+逐数字英文展开；百分号加 `PERCENT`；数字序数后缀 `st/nd/rd/th` 用ordinal模式。以上变换仍属于原父词，可产生多个空格分开的发音词。含数字均标 `normalization_assumed=true`，实际读法可能不同，列入质量抽查。
3. 英文字母转大写；连字符/斜杠等分隔符转空格，去其余纯标点并合并连续空格；保留词内撇号。文本仅由标点组成时结果为空，记 `no_spoken_content`，不加入CTC序列但原词和BERT向量保留。
4. 其他无法规范化的有意义字符不直接吞掉；记 `unsupported_text`。完整保留原词、规范化结果、变换类型，不自动润色、补词或改写题目转写。
5. 用下载的SentencePiece模型 `encode(ctc_text,out_type=str)` 得到piece字符串。通过 `asr_model.token_list` 建立 `piece -> ESPnet ID` 字典；**不直接采用SentencePiece整数ID**，二者编号体系可能不同。

对齐token_list使用模型完整词表，空白ID=0，模型中的 `<sos/eos>` 不作为文本目标。CTC log概率列数必须等于词表长度；不能照旧代码盲目 `token_list[:-1]` 后仍传未裁列的概率矩阵。目标piece若不在词表或是 `<unk>`，使用模型UNK ID作为定位占位并把父词标 `contains_unk`，该词最终不可进入时间窗文本。这样不会静默丢掉中间一个词造成后续word_id移位。整个父词无可编码内容时不向prepare_token_list传空数组。

### 10.2 加载与前向

```python
from espnet2.tasks.asr import ASRTask
model, train_args = ASRTask.build_model_from_file(
    config_file=str(runtime_yaml),
    model_file=str(asr_weight),
    device=device,
)
model.eval()
model.requires_grad_(False)
with torch.inference_mode():
    speech = torch.from_numpy(wav_float32).unsqueeze(0).to(device)
    lengths = torch.tensor([speech.shape[1]], dtype=torch.long, device=device)
    encoder_out, encoder_lengths = model.encode(speech=speech, speech_lengths=lengths)
    frame_count = int(encoder_lengths[0].item())
    lpz = model.ctc.log_softmax(encoder_out)[0, :frame_count].cpu().numpy()
```

`torch`在模块导入；WAV已是16000Hz单声道，不再次 `librosa.load` 隐式重采样。模型eval保证不启用训练时SpecAugment/dropout。输入整段音频，最多约35秒；不随意切成固定10秒后重复同一全文对齐。显存不足用相同模型CPU退路。

`lpz` 为 `(F,vocab_size)` 的对数概率。`index_duration=(len(wav)/16000)/F`，保存该值和F，表示CTC帧时间近似；不宣称达到采样点精度。模型自带global_mvn必须生效。

### 10.3 CTC分段与接受条件

保持MMSA-FET核心调用顺序：

```python
import numpy as np
from ctc_segmentation import (
    CtcSegmentationParameters, prepare_token_list,
    ctc_segmentation, determine_utterance_segments,
)
params = CtcSegmentationParameters(
    char_list=token_list,
    blank=0,
    index_duration=index_duration,
    min_window_size=8000,
    max_window_size=100000,
    score_min_mean_over_L=30,
)
ground_truth, begin = prepare_token_list(params, token_arrays)
timings, char_probs, state_list = ctc_segmentation(params, lpz, ground_truth)
segments = determine_utterance_segments(params, begin, char_probs, timings, ctc_texts)
```

`token_arrays`与 `ctc_texts` 长度一致，通过显式 `ctc_row_to_word_id` 回到原词，不能与全量words直接zip。默认保留库的转移规则，不调整blank跳过代价来强行提升成功率。过长文本/过短音频、回溯异常、模型加载问题分别标原因。

对每个返回 `(s_local,e_local,score)`：

- score原样作为 `ctc_log_score` 保存，阈值 `score >= -5.0`。可另存 `exp(score)`，但列名为score_exp，不能称为校准置信概率。
- s/e必须有限且 `e>s`；允许不超过一个CTC帧宽的边界溢出，裁到 `[0,audio_duration]` 并记录 `boundary_clipped`，超过则该词不可定位。
- 原词有UNK或unsupported_text、得分不足、区间无效，时间只保存为candidate，最终accepted时间为null。
- 连续返回词的start/end应各自非递减；若倒序，两词均记 `nonmonotonic_alignment`。相邻词重叠超过一个CTC帧宽时两词均标 `overlap_alignment`，不通过排序词条掩盖问题。
- 初步通过者的公共区间是 `[a0+s_local,a0+e_local)`，不是直接用local时间。裁至公共D仅用于浮点末端误差；明显出界是时钟问题。
- 最后还必须经过第14节的整段错配排查。整段转写与音频可疑/确认错配时，先保留候选结果，**所有时间窗文本eligible设为0**，不能因有CTC路径就称为对齐成功。

词状态至少包含：`ok,no_spoken_content,contains_unk,unsupported_text,low_ctc_score,invalid_interval,nonmonotonic_alignment,overlap_alignment,audio_missing,audio_discontinuous,alignment_failed,pairing_suspected,pairing_mismatch`。多个问题保存为reason列表，主状态按“素材错配→音轨问题→区间问题→分数问题→ok”的顺序决定。

### 10.4 仅用于检查的CTC贪心识别

从同一lpz沿类别轴argmax；先压缩相邻重复ID，再删除blank和模型特殊符号；按模型piece恢复可读文本。得到 `diagnostic_transcript`，与规范化题目文字计算第14节的诊断WER。此输出**只用于发现题目转写与音轨不相符**，不回填raw_text、不替代BERT输入、不作为新的训练标签。无须另下载语言模型或运行大模型ASR。

## 11. 音频25维的具体提取

```python
import opensmile
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
    sampling_rate=16000,
    resample=False,
    num_workers=1,
)
df = smile.process_signal(wav_float32, 16000)
```

必须拿到25列。程序把 `list(df.columns)` 原顺序保存为 `audio_feature_names.json`，全run每样本都检查同一顺序；不字母排序、不挑数值型额外列、不截前25列掩盖配置错误。LLD含F0、能量、谱及共振峰等描述量；精确列名由指定版本API给出，不能把88维Functionals列名套上去。[官方Python接口和特征维度](https://audeering.github.io/opensmile-python/)

索引的 `start/end` 为Timedelta，用 `total_seconds()` 读取，不能把纳秒整数直接当秒。第i行原生区间为 `[a0+start_i,a0+end_i)`，保留转换前索引文本。仅保留与音轨真实范围相交且有正宽度的支持区间；尾端若索引end缺失，使用实际WAV末端，并记该处理。无法解释的内部缺失end不能用下一行中点糊补。

每行25项全有限且区间合法时eligible=1。若某项NaN/Inf，该行整个A向量不可用，原生值和异常维度记录于本地诊断；主原生数组可用零占位，但eligible=0。F0未发声取0、静音能量、合法的零值均不是缺失证据。不要写 `if np.all(values==0): missing`。

记录 `audio_row_id,start,end,local_start,local_end,wav_sample_start,wav_sample_end,eligible,reasons`。采样范围用 `floor(local_start*16000)` 到 `ceil(local_end*16000)`，裁到 `[0,len(wav)]`，是该输出时间区间对应的派生采样范围，**不把它说成openSMILE内部所有滤波器/平滑操作的精确感受野**。

不对100条一起fit scaler，不按单片减均值/除方差。保留工具原尺度；如果将来训练自己的Q1预测器，标准化参数只能从对应训练集的可用位置估计，这不属于本阶段。

## 12. 视觉22维、跟踪编号与目标身份

### 12.1 调用OpenFace及字段顺序

使用 `FaceLandmarkVidMulti`，静态AU策略对所有样本一致。它能保留多脸候选，但AU没有单人视频模式的人物专属校准优势；这个选择是为明确处理多脸/切镜头而作的实现取舍，不宣称优于单人动态AU。[OpenFace AU模式说明](https://github.com/TadasBaltrusaitis/OpenFace/wiki/Action-Units)

程序由Python构造参数，等价命令如下：

```bash
micromamba run -p "$Q1_ROOT/env/openface" \
  "$Q1_ROOT/third_party/OpenFace/build/bin/FaceLandmarkVidMulti" \
  -fdir /绝对路径/本样本/frames \
  -out_dir /绝对路径/本样本/openface \
  -of features.csv -au_static -aus -pose -gaze -2Dfp \
  -fx FX -fy FY -cx CX -cy CY
```

由代码把FX等替换成数值：按显示宽高W/H取 `FX=500*W/640`、`FY=500*H/480`、`CX=W/2`、`CY=H/2`，记录 `camera_intrinsics=estimated`。没有真实相机标定，角度是工具估计值，不是测量学真值。无需开启tracked视频、HOG、对齐脸图片或显示窗口。`-2Dfp`只为身份连续性和诊断计算bbox，不能加入最终22维。[命令行与输出字段](https://github.com/TadasBaltrusaitis/OpenFace/wiki/Command-line-arguments)

CSV列名先strip两端空格，按下面顺序显式提取：

```text
0  AU01_r       1  AU02_r       2  AU04_r       3  AU05_r
4  AU06_r       5  AU07_r       6  AU09_r       7  AU10_r
8  AU12_r       9  AU14_r      10  AU15_r      11  AU17_r
12 AU20_r      13  AU23_r      14  AU25_r      15  AU26_r
16 AU45_r      17  pose_Rx     18  pose_Ry     19  pose_Rz
20 gaze_angle_x              21  gaze_angle_y
```

17个AU为强度 `_r`，不采用发生 `_c`，没有AU28_r。后5维单位为弧度，AU原范围通常0–5。若原输出超出预期范围，记录并复核，不进行无说明裁剪。[输出含义](https://github.com/TadasBaltrusaitis/OpenFace/wiki/Output-Format)

必须另保留 `frame,face_id,timestamp,success,confidence` 及2D关键点bbox。CSV某帧无任何行时，在原source表中补“no_face”，不压缩序列。这个tag的 `SequenceCapture` 是读帧后递增frame计数，因此第一张图对应CSV frame=1；用 `image_index` 精确连接。CSV自带timestamp来自图像序列的内部FPS，**禁止用于公共时间**。

### 12.2 `face_id`不是人的身份证

已核对C++实现：`FaceLandmarkVidMulti.cpp` 将 `model`跟踪槽位编号写入face_id，失败后槽位会重用，且默认约每8帧才执行一次新脸检测。因此不能凭“face_id始终为0”认定始终同一个人。保留以下保守流程，不添加新的说话人模型：

1. 从成功行的2D关键点计算bbox。`success=1`的候选参与人数/身份检查；最终特征另要求confidence≥0.8。
2. 对同一face_id构造连续episode：相邻成功源时刻间隔>0.20秒，或bbox IoU<0.10且中心距离>显示图像对角线的0.25，或出现下述明显切镜头信号，则开始新episode。
3. 明显切镜头信号：对相邻所选图像缩到64×64 RGB，各通道计算归一化16格直方图；三个通道L1差值的均值>0.65。只作为断开身份假设的信号，不宣称是准确镜头检测器。
4. 只有整片恰有一个成功episode，且没有其他同时或先后成功脸候选，才自动选它，记录 `identity_basis=assumed_single_visible`。不是 `speaker_verified`；单一可见人物仍可能是配音/插入画面，须经第14节排查。
5. 多episode、多脸或无法确认时，默认视觉主特征全不可用，但原始候选全部保留。不要选择面积最大的脸、最长轨迹或第0号脸来自动解决身份。
6. 人工/具备视听检查能力的agent可填写目标表，指定原视频中已经确认的人物片段。每行字段：`sample_id,start,end,face_id,episode_id,reviewer,evidence_note`。start/end为公共秒；同一时刻不得指定两个目标。未覆盖时间保持identity_unknown。
7. 候选选定后仍须 `success==1 && confidence>=0.8 && finite(all_22)`；目标表不能把检测失败帧改成成功。合法零AU保持有效。

这套规则宁可输出明确的部分视觉覆盖，也不偷偷将两个不同人物的AU混成一个说话人。无需在本文提前列出具体多脸样本；实现后在报告中列出其实际ID、时间和原图像索引。

### 12.3 视觉原生记录

选中的每个图像位置都保留一条记录：`vision_row_id,image_index,source_frame_index,start,end,face_id,episode_id,identity_basis,success,confidence,eligible,reasons`。无脸或未知目标的face_id/episode_id为null，向量零占位。另保存未经筛选的原始CSV，以便审核选择是否正确。22维全部一起定义一个视觉观测；本版不做某个AU缺失时的逐维插值。

## 13. 50窗聚合、掩码、覆盖率和来源记录

### 13.1 时间定义与计算精度

模态顺序固定为 `['text','audio','vision']`。公共D>0时，`I_k=[kD/50,(k+1)D/50)`，k从0开始；最后右边界等于D。所有时间、交叠时长和累加计算使用float64，最终特征/coverage转float32。

对一个模态的第i条原生记录 `[s_i,e_i)`，设 `eligible_i` 为第8—14节规则后的可用性，计算：

```text
w_ki = eligible_i * max(0, min(e_i, I_k.end) - max(s_i, I_k.start))
z_k  = Σ_i w_ki

z_k > 0:  X_k = Σ_i w_ki * x_i / z_k；observed_k = 1
z_k = 0:  X_k = 全零；observed_k = 0

coverage_k = length(union(全部有效交叠区间)) / (D/50)
```

coverages不能用z_k直接除窗宽，因为声学区间/候选词可能重叠。并集算法：先按起点、终点排序，遍历合并当前终点≥下一起点的区间，累计合并长度。覆盖率仅因浮点误差轻微越界时clip到[0,1]；明显越界应修复代码。

主序列中词向量先完成“子词→词”平均，然后一个父词只对应一个区间。不要把它复制给多个子词后再按时间累计。无词的停顿窗文本O=0，而不是沿用上一词。

### 13.2 五个角度维度

视觉第17–21维的常规结果用交叠权重的圆均值：

```text
S = Σ_i alpha_i * sin(theta_i)
C = Σ_i alpha_i * cos(theta_i)
theta_mean = atan2(S,C)       # 输出[-pi,pi]
R = sqrt(S*S+C*C)
alpha_i = w_ki / z_k
```

AU仍为普通加权均值。若任一角度R<1e-6，则该窗角度方向没有确定圆均值：本版把**该窗整个视觉观测**设为不可用，22维0、O=0、coverage=0，记录 `circular_mean_undefined`；另存 `candidate_coverage` 和原贡献项供复核。不要返回 `atan2(0,0)=0` 并称为可信正面朝向。这是图E常规分支后的极端退化处理；其他模态不受影响。

### 13.3 V/O/P的固定语义

| 名称 | 单样本形状 | dtype | 含义 |
| --- | --- | --- | --- |
| `valid_mask`（V） | `(50,)` | uint8 | 是否有真实时间结构；D有效时全1，媒体结构失败时全0 |
| `observed_mask`（O） | `(50,3)` | uint8 | 按上述质量规则，该窗是否有可聚合观测 |
| `perturb_mask`（P） | `(50,3)` | uint8 | 1代表后来人工删除；Q1全部0 |
| `coverage` | `(50,3)` | float32 | 已接受的实际时间支持覆盖率 |
| `valid_length` | 标量 | int64 | `sum(V)`，正常为50；不是三模态共同非零窗数 |

`usable = V[:,None] * O * (1-P)`。数值是否为0不反推掩码。覆盖率0.02但O=1表示只有很少时间有观测；不要把O=1说成该窗完整。已经确认不同步的样本，还受第14节 `paired_use` 控制，不能只看O决定是否可以作同步多模态输入。

### 13.4 来源映射采用CSR数组

每模态保存一个轻量NPZ，避免在几万条JSON中重复768维向量：

```text
indptr:        int64[51]       # 第k窗贡献项是[indptr[k],indptr[k+1])
source_ids:    int64[E]        # text为word_id；audio/vision为各自row_id
overlap_s:     float64[E]      # 该原生记录与目标窗交叠秒数
weights:       float64[E]      # overlap_s / 当前窗总有效交叠时长
```

每个O=1窗口的weights之和应在1e-6内等于1。贡献项按source_id升序存储，所有权重>0。无观测窗口没有接受贡献；退化角度的候选贡献另存诊断JSON，不混进已接受CSR。源区间可能跨几个窗，所以同一source_id可出现多次，但同一窗内不重复列同一源行。

**可手算例子：** D=10秒，第5窗（k=5）为[1.0,1.2)。词a=[0.9,1.1)，词b=[1.1,1.3)，向量标量简化为2和6。交叠各0.1秒，权重各0.5，结果4，coverage=1。如果b不可用，结果2、O=1、coverage=0.5，而不是结果1或O=0。若只有一个有效静音声学帧且25维都是合法0，则A结果0、O仍为1。

## 14. 必须执行的数据错配检查：产出具体问题样本

**用户已明确提示数据存在多种模态对齐问题，包括音频可能完全对不上。实现 agent 必须把排查作为实际工作的一部分，不能仅写“假设已对齐”或仅检查数组长度。本文不预先猜测哪些样本有问题；实现后必须给出具体sample_id、文件路径、时段、问题类型、证据和处理结果。**

### 14.1 检查层次与处理原则

```text
[附件1全100条，不按标签抽样]
              │
              ▼
[第一层：文件配对、主键、流起点、音视频时长、PTS连续性]
              │
              ▼
[第二层：原转写 ↔ 音频内容]
[CTC诊断转写差异 + 词对齐覆盖/得分 + 全段/首尾问题]
              │
              ▼
[第三层：音频 ↔ 画面/说话人物]
[音画明显不同步、配音/插画、换人、局部缺段、整段错配]
              │
              ▼
[逐样本问题表：没有发现 / 可疑 / 确认错配 / 无法核验]
              │
       ┌──────┴──────────────────────────────┐
       │没有发现或复核通过                   │可疑、确认错配、无法核验
       ▼                                     ▼
[按已接受区间生成配对特征]         [原生单模态照实保留]
                                  [paired_use=false]
                                  [受影响词时间不进入主输出]
                                  [具体ID、时段、证据进入问题清单]

“有CTC输出”“能解码”“长度相近”均不能单独证明内容对齐。
```

需要检查的类型至少包括：错文件/错clip、整段音频与转写不符、音频与画面不是同一段、固定音画偏移、局部漂移、开头/结尾截断、内部缺段或重复、转写缺词/多词、画面人物更换、配音或画外音、人脸来源不确定。只要求实现时识别和记录，不在本文编造具体异常数量或修复结果。

### 14.2 全量自动检查的最低实现

每条样本必须填写一行 `quality.csv`：

1. 文件主键匹配情况，所选流、解码时长、首PTS差、末端差、内部缺段、坏帧和重采样时长差。
2. 可发音父词总数、初步CTC通过数及比例、低分词比例、无时间词列表、首末接受词与音轨边界距离。
3. CTC贪心诊断文本和规范化题目文本的词级编辑距离；`diagnostic_wer=(S+D+I)/max(1,reference_word_count)`，不截断到1。使用确定性动态规划/已安装editdistance；完整保存两份对比文字。WER>0.65或初步可对齐词比例<0.80，标 `suspected_text_audio_mismatch`，进入复核。
4. 视觉有效帧比例、成功候选数、episode数、切镜头信号、画面可见人物与单人假设状态。只有声学数据或只有图像时不能宣称音画检查通过。
5. 为所有样本给出检查完成状态；对被自动触发的异常列出证据时间。没有自动报警只能写 `no_issue_detected`，不是人工验证正确。

诊断识别也会出错，尤其噪声、口音和专有词；阈值触发的是可疑状态，不是自动确定错文件。人工确认音频确实匹配时，复核可以解除“整段隔离”，但低CTC得分的单词仍不能自动变成有效时间。

### 14.3 原素材核验与可执行边界

实现 agent 必须实际检查原素材，尤其**全部报警样本**以及按时长分层的正常样本；检查完整片段或覆盖其首/中/尾，不能只看一张截图。可使用其环境具备的视频播放、音频识别/听取能力，记录使用了哪一种。至少核验最短、中位、最长样本，并额外核验多脸、低分、长停顿等案例；重复样本只核验一次。

音画内容是否属于同一情境、是否画外音/配音，不能仅由CTC或流PTS判断。**若服务器agent没有可靠视听检查能力，必须明确写“未完成视听核验”，输出需要核验的具体ID和时间、文本证据与对应帧路径，保持unverifiable/needs_review；不能生成虚假的“人工确认正常”。** 同样不要求它凭肉眼精确估计几十毫秒口型偏移；没有可量化参考时标可疑并说明依据。

`alignment_review.csv`字段固定：

```text
sample_id,issue_type,start,end,review_status,reviewer,evidence,action

review_status = confirmed_match / confirmed_mismatch / unresolved
action        = release_pairing / quarantine_pairing / keep_unresolved
```

首版不做自动“修复”：不搜索附件2替换音轨，不从相似样本借文字，不平移直到CTC得分变好，不直接覆盖原素材。若复核证明确有固定偏移，记录测量依据与偏移候选，另输出建议修正表；实际修正应作为可复查的数据修正，保存修正前后关系。没有依据时隔离而非猜测。

### 14.4 错配状态如何进入输出

样本元数据 `pairing_status`取：`no_issue_detected,reviewed_match,suspected,confirmed_mismatch,unverifiable`；另存 `review_scope` 和 `paired_use`。

- `no_issue_detected`：全量自动检查完成且无报警；paired_use=true，但报告写清仅自动检查，不能称全面人工通过。
- `reviewed_match`：所需问题经过复核且没有未解决项；paired_use=true。
- `suspected/confirmed_mismatch/unverifiable`：paired_use=false。其中“unverifiable”用于已经触发的必要核验无法完成，不把每一条未人工逐帧检查的正常样本都伪称异常。
- 转写—音频错配涉及全段时，50窗文本O=0，768维全0；原文及BERT词向量在native保留，CTC时间仅作为candidate。局部已明确问题可按问题区间禁用相交词；没有可证明的局部边界则按全段处理。
- 音画/目标人物错配：A和V独立提取结果保留原有O；样本paired_use=false，不能作为同步三模态组合使用。这区分“确实观测到某段声音”和“它与此画面对应”。
- 不把paired_use直接乘到所有原生特征里销毁数据；读取器默认返回它，使用者可明确筛选，Q1仍交付全100条。

最终至少输出 `alignment_issues.csv` 和 `alignment_audit.md`，按具体主键列问题，不只写汇总百分比。无异常也要记录实际检查范围与证据，不能预设数据一定存在某种数量的问题。

## 15. 输出文件、dtype、读取方式

### 15.1 单样本与全量目录

```text
runs/<run_name>/
├── config.yaml
├── manifest.jsonl
├── labels.csv
├── environment.json
├── sample_index.csv
├── samples/000000/
│   ├── status.json
│   ├── media.json
│   ├── words.jsonl
│   ├── bert_tokens.jsonl
│   ├── bert_chunks.jsonl
│   ├── audio_frames.jsonl
│   ├── video_frames.jsonl
│   ├── audio_rows.jsonl
│   ├── vision_rows.jsonl
│   ├── bert_words.npz
│   ├── audio_native.npz
│   ├── vision_native.npz
│   ├── native.npz
│   ├── compact50.npz
│   ├── map_text.npz
│   ├── map_audio.npz
│   ├── map_vision.npz
│   ├── bin_quality.jsonl
│   ├── diagnostic.json
│   ├── audio_16k.wav
│   ├── frames/000001.png
│   └── openface/features.csv
├── features/q1_compact50.npz
├── features/feature_names.json
├── reports/summary.md
├── reports/quality.csv
├── reports/alignment_issues.csv
├── reports/alignment_audit.md
├── reports/review_samples.txt
├── reports/timelines/<sample_index>.md
├── reports/package_size.csv
└── logs/events.jsonl
```

`run_name`显式指定如 `q1_full_20260923`，不能隐式复用旧输出目录。工作用WAV和PNG保留到审核结束；不把它们全部打入竞赛提交包。每个步骤先写本样本临时文件，再 `os.replace` 到目标，避免中断留下半个NPZ被当作完成；这是普通文件写入流程，不需要额外哈希或数据库。

### 15.2 数组规格

全量 `q1_compact50.npz`：

| 字段 | 形状 | dtype | 说明 |
| --- | --- | --- | --- |
| `text` | `(N,50,768)` | float32 | 窗内已定位父词向量 |
| `audio` | `(N,50,25)` | float32 | 声学LLD窗表示 |
| `vision` | `(N,50,22)` | float32 | AU与角度窗表示 |
| `valid_mask` | `(N,50)` | uint8 | V |
| `observed_mask` | `(N,50,3)` | uint8 | O |
| `perturb_mask` | `(N,50,3)` | uint8 | P，全0 |
| `coverage` | `(N,50,3)` | float32 | 接受支持覆盖率 |
| `time_intervals` | `(N,50,2)` | float64 | 公共秒区间 |
| `duration` | `(N,)` | float64 | 结构失败行占位0并由状态说明 |
| `valid_length` | `(N,)` | int64 | 真实时间窗数 |
| `sample_index` | `(N,)` | int64 | 指向索引表 |
| `paired_use` | `(N,)` | uint8 | 是否可按当前证据作为配对输入使用 |

N正常为100，即使有失败也保持N不变。NPZ不保存Python object数组或pickle对象；字符串放JSONL/CSV。使用 `np.savez_compressed`；读取使用 `allow_pickle=False`。单样本compact文件是同字段去掉N轴，duration/valid_length/sample_index/paired_use为0维数组。

`native.npz`至少包含 `word_features(W,768)`、`audio_features(A,25)`、`vision_features(V,22)`，三者的 `*_intervals(*,2)`、`*_eligible(*)`。未定位词intervals使用(0,0)占位，eligible=0，真实缺失原因和null时间在words.jsonl中；不让NaN污染后续聚合。原始坏数值所在CSV/诊断文件另留。source_id对应各自表中固定row_id，不随删除/筛选重编号。

分阶段写入避免覆盖其他模态：text只写 `bert_words.npz` 的word_features与word_ids；audio写 `audio_native.npz` 的values/intervals/eligible/source_ids；vision写同结构的 `vision_native.npz`；align写words.jsonl中的candidate字段与诊断，保留原词/语义映射字段。audit每次从这些原始提取状态和当前review表重建最终eligible，不能在上次已经置0的eligible上继续累乘，否则解除误报警也恢复不了数据。pool汇总生成最终 `native.npz` 与compact文件。单阶段结果不就地删除原候选区间。

`status.json`包含：sample_id、每阶段状态、总体processing_status、pairing_status、paired_use、实际CPU/GPU设备、耗时、异常摘要、复核状态。环境信息保存Python包版本、ffmpeg版本、源码Git版本、模型来源版本及GPU名称。不要在日志写账号token。

### 15.3 逐窗质量表

`bin_quality.jsonl`每窗一行，字段至少：`bin_index,start,end,text/audio/vision`。各模态子对象含 `observed,coverage,source_count,reasons`；空窗用以下原因或其组合：

```text
text:   no_word / word_unaligned / pairing_suspected / pairing_mismatch / extractor_failed
audio:  outside_audio / audio_missing / audio_discontinuous / invalid_output / extractor_failed
vision: unsampled_interval / outside_video / no_face / low_confidence / identity_unknown /
        invalid_output / circular_mean_undefined / extractor_failed
```

区别 `no_word` 和 `word_unaligned`：只有CTC整体运行有效且附近没有不可定位内容时，才可认为该窗确实没有已定位词；不知道未定位词应该在哪个窗时，相关样本的空文本窗同时带 `unlocated_words_present` 提示，不把未定位词强行分配到某个空窗。

### 15.4 消费端示例

```python
import csv
import numpy as np
from pathlib import Path

run = Path("runs/q1_full_20260923")
with np.load(run / "features/q1_compact50.npz", allow_pickle=False) as f:
    i = 0
    T, A, V = f["text"][i], f["audio"][i], f["vision"][i]
    usable = (f["valid_mask"][i, :, None]
              * f["observed_mask"][i]
              * (1 - f["perturb_mask"][i]))
    paired_use = bool(f["paired_use"][i])
    seconds = f["time_intervals"][i]
with (run / "sample_index.csv").open(encoding="utf-8", newline="") as fp:
    index_rows = list(csv.DictReader(fp))
print(index_rows[i]["sample_id"], T.shape, A.shape, V.shape, paired_use)
```

读取器不能认为text[:,0]就是第一个词，也不能把这份25/22维特征直接传给要求74/35维官方输入的Q2模型。

## 16. 实现顺序、CLI和全量运行

### 16.1 开发顺序

```text
[1 下载/读复用代码，安装环境]
          ↓
[2 实现manifest与media，先把源时间表做正确]
          ↓
[3 实现text_map、BERT与CTC；确认词ID未错位]
          ↓
[4 实现openSMILE和OpenFace；保留质量与真实时间]
          ↓
[5 实现错配检查；受影响样本/词有明确状态]
          ↓
[6 实现pool50、CSR来源记录和全量存储]
          ↓
[7 运行数学/状态测试 + 短中长真实样本验证]
          ↓
[8 全100条按阶段提取 → 问题清单 → 素材复核]
          ↓
[9 应用复核表，仅重算受影响聚合与报告]
          ↓
[10 验证全量输出，统计实际体积，提交代码和结果说明]
```

先完成能够正确处理空分支和局部失败的单样本，再批量。不要先用一个假的随机数组让全量形状验证通过，然后把真实提取留成TODO。

### 16.2 CLI子命令及行为

全部命令均有 `--config`、`--run-dir`，stage命令另有 `--ids-file` 和 `--resume`。ids文件每行完整sample_id，UTF-8，读取后仍按manifest顺序执行。

| 子命令 | 必须实现的行为 |
| --- | --- |
| `prepare` | 解析路径、创建run、构建manifest/标签、记录配置与环境；不进行模型推理 |
| `check-models` | 用本地模型执行BERT/CTC实际前向；运行OpenFace解析一个真实选帧序列；验证25/22维列；记录设备 |
| `media` | 解码音视频、建立时钟、导出WAV/PNG与原始映射 |
| `text` | 建words、一次BERT分词、分块编码，输出词特征 |
| `align` | 加载一次CTC模型，对选择样本逐条强制对齐及生成诊断转写 |
| `audio` | 加载一次Smile，对选择样本输出LLD |
| `vision` | 逐样本调用OpenFace，做身份与质量筛选；保留全部候选 |
| `audit` | 合并自动错配诊断及人工review表，生成问题清单和最终eligible |
| `pool` | 根据当前审核状态重算50窗、CSR和单样本状态 |
| `collect` | 按manifest顺序汇总全量NPZ，不跳过失败行 |
| `validate` | 第18节的数据形状、时间、映射与状态一致性；输出逐项结果 |
| `report` | 汇总质量、错配ID、文本时间线、手算示例、实际资源与体积 |
| `run` | 按media→text→align→audio→vision→audit→pool→collect→validate→report执行 |

`run`遇某样本阶段失败时保留状态并继续处理独立分支/其他样本；全局模型缺失或加载普遍失败时停止该阶段并报告环境错误，不重复100次同样失败。CLI正常完成返回0；输入/环境导致无法执行返回2；结果已生成但验证发现实现错误返回3。自然缺失、已明确隔离的数据错配写在结果状态中，不通过退出码伪装成程序崩溃。

### 16.3 服务器运行命令

agent完成代码后创建最小pyproject，包依赖使用requirements文件管理，再：

```bash
cd "$Q1_ROOT"
"$Q1_PY" -m pip install --no-deps -e .
"$Q1_PY" scripts/download_models.py
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

"$Q1_PY" -m q1_features prepare \
  --config configs/q1.yaml --data-root "$Q1_DATA" \
  --run-dir runs/q1_full_20260923

"$Q1_PY" -m pytest tests -q

"$Q1_PY" -m q1_features media \
  --config configs/q1.yaml --run-dir runs/q1_full_20260923
```

完成media后，按duration升序选择第0、`floor((N_valid-1)/2)`、最后一条有效样本，重复去掉，写到 `reports/review_samples.txt`。首批至少有这3条，不按情感标签选择。若有效片段不足3条，说明实际输入问题，不能任意制造样本。

```bash
"$Q1_PY" -m q1_features check-models \
  --config configs/q1.yaml --run-dir runs/q1_full_20260923 \
  --ids-file runs/q1_full_20260923/reports/review_samples.txt

"$Q1_PY" -m q1_features run \
  --config configs/q1.yaml --run-dir runs/q1_full_20260923 \
  --ids-file runs/q1_full_20260923/reports/review_samples.txt --resume
```

ids子集运行时，collect仍按全manifest写N行，但未处理样本明确标 `not_processed`，对应V/O为0；validate只对显式选择的子集判定阶段完成，同时报告全量待处理数。该文件只能称为首批结果，不能称100条完成。检查首批实际源时间、词/BPE、列名与三模态内容，修复实现问题后再运行全量：

```bash
"$Q1_PY" -m q1_features run \
  --config configs/q1.yaml --run-dir runs/q1_full_20260923 --resume
```

读取具体问题清单、复核报警样本、填写身份与错配review表后：

```bash
"$Q1_PY" -m q1_features vision \
  --config configs/q1.yaml --run-dir runs/q1_full_20260923 --reuse-raw-csv
"$Q1_PY" -m q1_features audit \
  --config configs/q1.yaml --run-dir runs/q1_full_20260923
"$Q1_PY" -m q1_features pool \
  --config configs/q1.yaml --run-dir runs/q1_full_20260923
"$Q1_PY" -m q1_features collect \
  --config configs/q1.yaml --run-dir runs/q1_full_20260923
"$Q1_PY" -m q1_features validate \
  --config configs/q1.yaml --run-dir runs/q1_full_20260923
"$Q1_PY" -m q1_features report \
  --config configs/q1.yaml --run-dir runs/q1_full_20260923
```

`--reuse-raw-csv`只重做目标身份筛选，不调用OpenFace重新计算；要求源选帧表、CSV和提取配置未改变。没有身份表变化则无需运行该条vision命令。

### 16.4 续跑与修改规则

不建设内容哈希缓存。`--resume`只对**同一run、同一配置与同一代码版本**使用：读取保存的config和当前config做普通字段比较，检查本阶段status为ok且所需输出可读。发现配置或提取代码变化，新建run，或用显式 `--redo-stage stage --ids-file ...` 重算受影响样本；不能根据文件存在就默认正确。

依赖关系：media变化→所有依赖分支和聚合；BERT变化→词特征、T聚合；CTC变化→词时间、audit、T聚合；OpenFace/目标表变化→视觉筛选、audit、V聚合；质量review变化→audit、受影响eligible和pool；bins变化→pool。只改报告排版无需重新推理。resume跳过已完成纯提取阶段时，audit仍须合并当前review；pool、collect、validate、report重新执行。

每个阶段记录样本ID、开始/结束、耗时、设备、原生行数、成功或具体错误。单GPU下BERT整阶段结束后释放模型并清理引用，再加载CTC；不用多进程复制GPU模型。OpenFace默认串行，确认内存充足后最多2个样本并行，各自独立目录，禁止共用临时CSV。

## 17. Q1与Q2/Q3的衔接，以及提交体积

### 17.1 本轮必须预留的复用接口

```text
[Q1：真实媒体 → 时钟/词/源帧 → 状态与time_map]
                       │ 方法与代码复用
                       ▼
[Q3：附件4原文/视频的辅助定位]
                       │
                       ▼
[是否证明“官方第k行”对应这个词/这段原素材？]
             │已证明                    │未证明
             ▼                          ▼
[给出来源区间和映射依据]       [保留官方位置贡献；秒级映射未知]

[Q2：附件2官方特征训练] → [同一预测器] → [Q3归因及缺失干预]
          ↑
          └─ 继承Q1的状态/来源思想；不直接读取Q1的25/22维替换官方74/35维
```

Q1本轮只开发通用的manifest、原文字符映射、媒体时钟、CTC对齐、来源表示与缺失状态工具，不开发Q2网络。以下事实放进README的边界说明，避免下一个agent误接：

- 附件2官方对齐版train/valid/test分别3395/728/727；A/V维度74/35，50位置含义不能由Q1等时间窗定义推断。
- 附件3对齐版30条只有text_bert/audio/vision，没有预计算text；Q2与Q3应统一官方text_bert编码入口，并先在附件2 train/valid核对词表及attention mask。维度768不等于词表兼容。
- 附件4对齐版20条中13号视觉全零，不能把新提取视频里的动作说成原预测器实际使用的视觉证据。
- Q1的100条与附件2重合18条，其中7条属于附件2 test，故不能将Q1全部加入训练或调参。
- 本次发现的数据错配清单须交给后续Q2/Q3工作；不要将“已知可疑的原素材映射”作为解释真值。官方特征来源规则未知时，新的CTC词时间不能证明官方音频/视觉每行的真实秒数。

Q1输出已保留模态坐标类型 `relative_time_bins`、源ID和paired_use；Q2/Q3输入应另用 `official_sequence_positions`。本次不写官方输入的猜测性转换器，也不因二者长度都是50而共享已经训练好的输入层权重。

### 17.2 本地计算结果与竞赛提交文件分开

三模态核心float32数组的未压缩体积：

`100 × 50 × (768+25+22) × 4 = 16,300,000 bytes`，约16.30 MB（十进制），不含映射/报告/代码/Q2Q3权重。

模型和原生结果用于服务器计算，不默认装进小型成果包。拟选成果文件为全部100条compact特征、索引/标签、列名、压缩来源映射、质量/错配报告、代码与环境说明；原生高频向量、全部PNG/WAV、下载源码、环境和大型权重作为服务器计算材料保留。

但题目要求**总提交材料50 MB**，目前没有证据证明预训练权重可以豁免。BERT和Conformer权重显然超过此限；仅给下载脚本是否符合提交规定，需要赛事口径确认。实现agent必须：

1. 用实际文件大小生成 `package_size.csv`，明确字节数，按50,000,000 bytes做保守预算。
2. 分列“Q1紧凑成果包”“运行所需外部预训练权重”“其他Q2/Q3材料预留”，不能把仅特征包小于50MB写成全题合规。
3. 权重无需纳入最终包的规则没有确认前，报告 `submission_weight_policy=unresolved`，不得宣布正式提交条件全部满足。
4. 不为压到50MB擅自换BERT、删除异常样本、只交若干示例或改成float16；若确有体积问题另列实际占用和可审查的压缩方案。

这项外部提交口径不阻止完成Q1代码与全部特征提取，但阻止无依据的“最终提交包已满足要求”结论。

## 18. 验证、报告与完成判据

### 18.1 必须写的测试：只检查实质性错误

使用pytest构造小型人工输入，不下载测试数据集。至少覆盖以下场景；不写仅仅检查函数存在或重复实现代码的测试。

| 测试 | 输入与正确结果 |
| --- | --- |
| 字符映射 | 原文有大小写、撇号、标点、连续空白；每词/子词回到正确原文，特殊词元无时间 |
| 长文本尾部 | 超过510内容子词；每个全局子词均有向量，尾词不丢，重复块子词只在词内统计一次 |
| 子词权重 | 一个词含3子词，另一个含1子词但时长相同；先词内平均后两个词的时间权重相同 |
| 流起点 | 音轨比画面晚0.4秒；所有音频与CTC公共时间统一+0.4秒，不分别置0 |
| 非连续音轨 | 中间有真实PTS缺段；A/CTC标异常，不能把缺段压没或当合法静音 |
| 低帧率 | 10fps源帧按25Hz选取，source_frame_index不重复导出；源展示支持仍正确 |
| 交叠与并集 | 重叠的两个有效音频区间，平均权重按交叠，coverage按并集且不超过1 |
| 静音与零AU | 合法全零行仍O=1；真正无观测才O=0 |
| 中间失败帧 | 第3帧失败，后续源帧PTS和窗口位置不前移 |
| 角度环绕 | +179°与−179°等权，结果接近±180°；正好相反角导致R不足时按不可用规则处理 |
| 人为遮蔽 | P=1使U=0，但O原始事实不变；Q1输出P全0 |
| 整段错配 | 有有限CTC区间但诊断报警未复核，paired_use=false、时间窗T缺失、native BERT仍保留 |
| 完整索引 | 3条输入含1条媒体失败，汇总仍为3行，对应顺序正确，失败行V=0 |

外部工具必须另用真实样本验证：实际调用BERT、ESPnet、openSMILE、OpenFace，不能只用mock通过后称环境可用。先短/中/长3条，若检查通过且代码未变，无需反复重复同一安装验证。

### 18.2 全100条结果检查

- manifest与sample_index与全量数组逐行一致，缺文件/错误/未处理状态不能被跳过。
- 正常输出形状与dtype完全符合第15节；数值均有限；labels不混入特征。
- D>0样本50个时间窗覆盖[0,D]；D失败与not_processed行明确为0占位；V/O/P只取0或1。
- 每条接受原生区间有正宽度并在D内；所有应参与聚合的区间都有可核对CSR贡献，尾部不丢失。
- coverage在[0,1]；O=0时对应主特征全0且coverage=0；O=1的普通窗至少一个贡献项、权重和1。零特征不推出O=0。
- 随机数种子固定选10个有观测窗口，根据native与CSR重算，普通维度 `rtol=1e-5,atol=1e-6`；角度比较圆周差，不直接比较+pi和−pi。
- 被标错配/待复核样本不能出现在paired_use=true行中；所有异常都有具体ID与原因。原始CTC候选不得冒充已接受时间。
- 各阶段success数量、特征可用数量、配对可用数量分别统计，不能只输出“100/100成功”。

### 18.3 状态、文本时间线与报告格式

`processing_status`：

- `ok`：所需阶段均执行完成且三模态至少各有一个接受观测；仍可能paired_use=false，二者分开报告。
- `partial`：D可建立，但某模态不可用或某阶段失败；其他成果仍有价值。
- `failed`：无法建立可靠公共时间结构或所有特征分支失败。
- `not_processed`：仅用于首批/中断过程，不允许在“全量处理完成”结论中隐藏。

时间线用Markdown表格和等宽文本，至少显示原文词、接受/候选CTC、A/V支持、窗边界、缺失和错配区域。示意：

```text
sample_id: <真实完整ID>       D=<实测秒数>       pairing=<实际状态>

公共秒轴    0.0 ────── 0.4 ────── 0.8 ────── 1.2 ────── ... ── D
音轨        [无音轨]  [实际波形支持─────────────────────────────]
文本词      [word0候选?]   [word1接受]       [未定位词：单列说明]
源画面      [f0][f1] ... [无脸] ... [目标身份未知] ... [f_last]
50窗        [k0][k1][k2][k3] ...                           [k49]

窗口k | 起止秒 | T/A/V的O | coverage(T,A,V) | source_id及权重 | 原因
------|--------|----------|-----------------|------------------|-----
实际值由report程序填写；不手工伪造示例结果。
```

报告至少包括：数据总数、成功/局部/失败数、每模态有效比例、词覆盖、采样与人脸覆盖、各类错配具体ID、复核范围、未解决事项、实际资源、运行环境、输出读取例子。主报告选1个跨窗词做权重手算，并列短/中/长样本及全部问题样本的时间线文件。

若没有人工词边界，报告自动覆盖和检查结论，不报告“对齐准确率”。若实际标注人工边界，另存标注表、标注者与定义，再计算 `(|s-s*|+|e-e*|)/2` 的中位数、90分位数和样本量。不要用CTC得分代替边界误差。

50窗最长片段窗宽约0.69秒，应报告实测最大窗宽、每窗词数量和短事件被平均的情况。只有确有压缩问题再建议更多窗比较；本次主输出仍为50窗，不擅自把比较实验当作最终方案。

### 18.4 实现阶段完成条件

代码完成与数据结论分开陈述。实现agent最终必须交付：

1. 可从空项目安装、下载、运行的源码与README，清楚标注复用代码。
2. 实际执行过的环境/真实模型验证和上述必要测试结果。
3. 附件1所有100条的输出索引、固定形状特征、状态、coverage与来源记录；异常条目保留，不承诺每条所有模态都成功。
4. **实际数据错配检查结果**，包含具体ID/时段/证据及隔离结果，不能只交本文的计划性检查列表。
5. 自动检查、实际视听复核和未完成核验的范围分别说明；需要人工确认的具体条目列出，不声称未做过的审核已通过。
6. 实测提交体积与预训练权重提交口径的真实状态。

核心代码/依赖未能运行、还有not_processed样本、来源映射重算不一致，均不能称实现完成。全量处理完成但自然缺失/确认错配仍存在，可以交付完整的部分可用成果，并明确其限制；无法完成必要视听核验则该项保持未完成。

## 19. 可直接交给服务器agent的执行指令

> 请在当前Linux + NVIDIA GPU服务器按本文实现Q1项目。服务器没有本地MMSA-FET代码，请先下载本文指定的MMSA-FET Git版本、OpenFace2.2.0及两套预训练模型，阅读指定复用文件，按复用矩阵改造。本文模型、维度、采样、阈值、词映射、时间定义、缺失规则和输出结构均作为本次实现选择，不重新进行模型选型。
>
> 先检查任务目录和题目数据位置，建立独立环境；不要修改系统驱动，不要使用情感标签调参。实现本文的CLI、状态和来源记录，先通过必要数学测试及短/中/长真实样本验证，再处理全部100条。流程图与审核时间线使用文本框/箭头/表格，不用网页绘图。
>
> 数据可能存在多种模态错配，甚至整段音频完全对不上。必须执行全量自动排查并实际核验报警素材，输出具体样本ID、时段、错配类型、证据和处理结果。CTC有输出不等于内容配对正确。可疑/确认错配样本先保留与隔离，不猜测修复，不把诊断ASR替换题目转写。如果缺少可靠视听检查能力，明确列出未核验的具体数据，不编造人工确认。
>
> 保存所有样本，包括失败条目；生成全部特征、掩码、覆盖率、原生来源映射、质量与错配报告。结尾说明实际完成内容、验证结果、剩余具体问题和体积。只实现Q1及通用时间/状态工具，不训练Q2/Q3、不宣称官方50行与Q1时间窗兼容、不把大型权重提交豁免当作既定事实。

---

本文的上游代码、模型地址、依赖声明与关键字段已经静态核对；Linux安装、提取速度、100条错配样本清单、检测覆盖与最终体积必须由服务器运行产生。本文没有把尚未执行的验证写成已完成结果。
