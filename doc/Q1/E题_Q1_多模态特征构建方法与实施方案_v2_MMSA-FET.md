# E题 Q1 多模态特征构建方法与实施方案

## 1. 任务目标与数据边界

### 1.1 Q1目标

Q1面向附件1中的100条原始多模态视频样本，完成可复现、可核验的文本、音频、视觉三模态时序特征构建。输出应同时保留样本标识、有效性掩码、模态观测状态、时间区间及原素材映射信息，为后续Q2的局部缺失鲁棒预测和Q3的可解释分析提供统一的数据组织规则。

Q1输出不是100个情感预测值，而是100条样本的三模态时序表示及配套元数据。

### 1.2 实际数据

附件1每条样本由以下内容组成：

- 原始MP4视频；
- 英文转写文本 `text`；
- 连续情感强度标签 `label`；
- 离散情感极性标签 `annotation`；
- 主键 `(video_id, clip_id)`。

当前数据没有提供：

- 独立音频文件；
- 逐词时间戳；
- 抽帧时间表；
- 预先生成的Q1三模态特征。

因此，音频需从MP4中分离，文本与音频/视频之间的时间对应关系需自行建立。

### 1.3 与Q2、Q3的衔接原则

Q1不强行复制附件2的原始特征空间，而是统一以下规则：

1. 样本标识方式；
2. 时序位置定义；
3. 文本、音频、视觉的时间对应关系；
4. `valid_mask`、`observed_mask` 等掩码语义；
5. 特征位置到原始素材的追溯方式；
6. 预处理与提取配置的记录方式。

Q1建议采用固定50个标准时序位置作为最终统一表示，以便与附件2/3/4的对齐版数据组织形式衔接。该长度是方案设计参数，不是题目强制要求。原始高分辨率特征在进入50步表示前保留变长形式，避免直接粗暴截断原始信息。

---

## 2. 可复用项目与总体技术路线

### 2.1 主要复用仓库：MMSA-FET

已跑通并作为Q1主要代码基座的项目：

- MMSA-Feature Extraction Toolkit（MMSA-FET）：https://github.com/thuiar/MMSA-FET

MMSA-FET对应M-SENA工作中的多模态特征提取工具链，Q1优先复用其已有的视频读取、模态预处理、特征提取和批处理实现，而不是重新搭建一套独立的特征提取框架。

可直接复用或在其基础上改造的部分主要包括：

- 原始视频读取与数据集批处理框架；
- 文本、音频、视觉预处理流程；
- BERT/RoBERTa类文本特征提取接口；
- Librosa/OpenSMILE/wav2vec2等音频特征提取接口；
- OpenFace/MediaPipe等视觉特征提取接口；
- 特征保存、配置管理和批量提取逻辑。

M-SENA论文及平台用于总体方法结构参考：

- M-SENA：https://github.com/thuiar/M-SENA

其“数据管理—特征提取—模型训练—结果分析”的模块化设计用于组织比赛方案，但Q1实际代码实现以MMSA-FET为主。

Q1需要在MMSA-FET基础上自行新增或针对比赛数据改造的关键部分：

- 附件1专用数据读取与唯一样本索引；
- 已有英文转写与音频之间的强制时间对齐；
- 三模态统一时间坐标与特征时间锚定；
- 原始变长序列到固定50步的自适应时序聚合；
- `valid_mask`、`observed_mask` 的显式生成；
- 为Q2预留 `perturb_mask` 的统一语义；
- 特征位置到文本、音频和视频原素材的追溯映射；
- 异常样本、局部提取失败与质量信息记录。

---

## 3. Q1总体方法流程

流程中的实现标签统一采用：

- `【MMSA-FET直接复用】`：现有代码可直接调用，仅需修改配置或数据路径；
- `【基于MMSA-FET改造】`：保留现有提取器或批处理框架，增加比赛数据适配逻辑；
- `【比赛方案新增】`：MMSA-FET中没有现成实现，需要单独新增。


```text
附件1：100条原始视频 + transcript + label
                    │
                    ▼
┌───────────────────────────────────────────┐
│ M1 数据索引与完整性检查                    │
│                                           │
│ 输入：video_id, clip_id, mp4, text, label │
│ 输出：唯一sample_id、路径、标签、元信息     │
│                                           │
│ 作用：                                     │
│ ① 防止clip_id跨视频来源重复                │
│ ② 保证100条视频与标签一一对应              │
│ ③ 为后续批处理建立稳定主键                 │
│                                           │
│ 【比赛方案新增】                                │
└─────────────────────┬─────────────────────┘
                      ▼
┌───────────────────────────────────────────┐
│ M2 原始多模态信号解析                      │
│                                           │
│ Video ──► 视频帧 + 原始时间戳              │
│ Audio ──► 从MP4中分离音轨                  │
│ Text  ──► 使用题目已有英文转写             │
│                                           │
│ 作用：                                     │
│ 建立后续所有模态共享的原始时间坐标          │
│                                           │
│ 【基于MMSA-FET改造：音视频读取与预处理】    │
└─────────────────────┬─────────────────────┘
                      ▼
┌───────────────────────────────────────────┐
│ M3 文本—音频时间对齐                       │
│                                           │
│ 已有transcript + 原始audio                 │
│              ↓                            │
│      word/subword timestamps              │
│                                           │
│ 作用：                                     │
│ ① 为文本赋予真实时间信息                   │
│ ② 建立文本 ↔ 音频 ↔ 视频的统一时间锚点     │
│ ③ 为Q3关键片段回溯提供基础                 │
│                                           │
│ 【比赛方案新增：核心模块】                      │
└─────────────────────┬─────────────────────┘
                      ▼
          ┌───────────┼───────────┐
          │           │           │
          ▼           ▼           ▼
┌────────────────┐ ┌────────────────┐ ┌────────────────┐
│ M4-T 文本特征  │ │ M4-A 音频特征  │ │ M4-V 视觉特征  │
│                │ │                │ │                │
│ BERT类编码器   │ │ OpenSMILE /    │ │ OpenFace       │
│                │ │ eGeMAPS        │ │                │
│ 输出：         │ │ 输出：         │ │ 输出：         │
│ token/subword  │ │ 帧级声学序列   │ │ 帧级视觉序列   │
│ 时序表示       │ │                │ │                │
│                │ │                │ │                │
│【MMSA-FET直接复用】│ │【MMSA-FET直接复用】│ │【MMSA-FET直接复用】│
└───────┬────────┘ └───────┬────────┘ └───────┬────────┘
        │                  │                  │
        └──────────────────┼──────────────────┘
                           ▼
┌───────────────────────────────────────────┐
│ M5 自适应时序对齐与聚合                    │
│                                           │
│ 三模态原始变长序列 + 各自时间区间          │
│                    ↓                      │
│       映射到统一的50个标准时间位置         │
│                    ↓                      │
│ 每个时间位置分别对T/A/V局部特征进行聚合    │
│                                           │
│ 作用：                                     │
│ ① 统一Q1三模态时序结构                     │
│ ② 与Q2/Q3对齐版的50步组织形式衔接          │
│ ③ 保留模态间真实时间对应                   │
│                                           │
│ 【比赛方案新增：核心模块】                      │
└─────────────────────┬─────────────────────┘
                      ▼
┌───────────────────────────────────────────┐
│ M6 有效性与观测状态建模                    │
│                                           │
│ valid_mask      ：是否属于真实有效位置      │
│ observed_mask   ：该位置该模态是否成功观测  │
│                                           │
│ 区分：                                     │
│ padding ≠ 原始提取失败 ≠ Q2人工缺失        │
│                                           │
│ 作用：                                     │
│ 为Q2缺失建模和Q3解释提供一致的掩码语义      │
│                                           │
│ 【比赛方案新增】                                │
└─────────────────────┬─────────────────────┘
                      ▼
┌───────────────────────────────────────────┐
│ M7 原素材追溯映射                          │
│                                           │
│ 对每个标准位置k保存：                       │
│ ① [start_time, end_time]                  │
│ ② 对应word/subword                        │
│ ③ 对应audio sample/frame范围              │
│ ④ 对应source video frame indices          │
│                                           │
│ 作用：                                     │
│ 将Q3中的“位置17-20重要”重新映射回原始视频   │
│                                           │
│ 【比赛方案新增：Q3预留接口】                    │
└─────────────────────┬─────────────────────┘
                      ▼
┌───────────────────────────────────────────┐
│ M8 质量控制与异常记录                      │
│                                           │
│ 检查：                                     │
│ 人脸检测失败 / 静音 / 文本对齐失败         │
│ 空时间窗 / 视频解码异常 / 模态局部不可用   │
│                                           │
│ 原则：                                     │
│ 不静默删除失败样本，保留状态与处理日志      │
│                                           │
│ 【基于MMSA-FET改造：批处理与状态记录】         │
└─────────────────────┬─────────────────────┘
                      ▼
┌───────────────────────────────────────────┐
│ M9 标准化打包输出                          │
│                                           │
│ text_features                             │
│ audio_features                            │
│ vision_features                           │
│ time_intervals                            │
│ valid_mask                                │
│ observed_mask                             │
│ token_or_word_spans                       │
│ source_frame_indices                      │
│ metadata                                  │
│                                           │
│ 【基于MMSA-FET改造：特征保存与接口封装】      │
└───────────────────────────────────────────┘
```

---

## 4. 各方法模块设计

## 4.1 M1 数据索引与完整性检查

### 输入

- `video_id`
- `clip_id`
- `text`
- `label`
- `annotation`
- 对应MP4路径

### 主键

使用：

```text
sample_id = video_id + "$_$" + clip_id
```

不单独使用 `clip_id`，因为不同视频来源文件夹可能出现相同的片段编号。

### 检查项

- 100条标签记录是否全部存在对应MP4；
- 是否存在重复主键；
- MP4是否可正常读取；
- 视频时长、FPS、分辨率；
- 音频采样率、声道数；
- 文本是否为空；
- 标签是否完整。

### 输出

```text
sample_index.csv
```

建议字段：

```text
sample_id
video_id
clip_id
video_path
text
label
annotation
duration
fps
audio_sample_rate
status
```

---

## 4.2 M2 原始多模态信号解析

### 文本

直接使用附件1已有英文转写，不重新将完整ASR作为主流程。

如需ASR，仅作为：

- 对齐失败辅助检查；
- 原始音频质量核验；
- 转写明显异常时的备选信息。

### 音频

从MP4中提取音频轨道，并统一为固定采样率后进入音频特征提取器。

需要保留：

- 原始采样率；
- 目标采样率；
- 重采样规则；
- 音频起止时间。

### 视频

逐视频读取真实流参数，不将示意图中的FPS或分辨率直接当作全部样本参数。

需要保留：

- 原始FPS；
- 总帧数；
- 每帧时间戳或可恢复时间戳的信息；
- 抽帧规则；
- 缩放规则。

---

## 4.3 M3 文本—音频强制时间对齐

### 目的

附件1已有完整转写，但没有逐词时间戳。Q1需要补建文本到原始时间轴的映射。

### 输入

```text
transcript
audio waveform
```

### 输出

建议至少获得词级记录：

```text
word
char_start
char_end
start_time
end_time
confidence/status
```

若最终使用BERT子词，则继续建立：

```text
word → subword/token → timestamp
```

### 对齐结果用途

1. 将文本特征映射到统一时间位置；
2. 建立文本与音频/视频同步关系；
3. Q3解释时恢复关键英文片段；
4. 对齐失败时提供可追踪错误记录。

### 异常处理

出现以下情况时不直接删除样本：

- 某些词无法对齐；
- 对齐区间重叠；
- 对齐超出视频时长；
- 长停顿；
- 文本和音频存在明显不一致。

应保存：

```text
alignment_status
alignment_coverage
unaligned_words
```

---

## 4.4 M4-T 文本特征

### 推荐方案

```text
Transcript
   ↓
Tokenizer
   ↓
BERT类文本编码器
   ↓
token/subword hidden states
   ↓
保留token时间映射
```

### 选择理由

- 与M-SENA现有文本特征管线兼容；
- 便于复用当前已跑通项目；
- BERT类表示与附件2中768维文本表示的形式接近；
- 后续Q2/附件3可以采用统一的 `text_bert → text encoder` 路线。

### Q1输出

不要求必须证明与附件2的 `(50,768)` 是完全相同的特征空间。若采用768维BERT表示，可以在形状上与后续对齐版保持一致，但需要明确编码器版本、Tokenizer和所取层。

### 必须记录

```text
model_name
tokenizer_name
max_length
selected_hidden_layer
pooling_rule
```

---

## 4.5 M4-A 音频特征

### 主方案

优先采用：

```text
OpenSMILE / eGeMAPS
```

提取帧级声学特征，再依据真实时间戳进入时序聚合。

### 主要原因

相比只使用深层wav2vec2 embedding，OpenSMILE/eGeMAPS特征：

- 更容易追踪到真实时间片；
- 声学含义更明确；
- Q3中更容易解释声音变化；
- MMSA-FET已有相关提取流程可复用；
- 存储与计算成本较低。

### 可选增强

若后续实验时间充足，可将wav2vec2作为第二套音频表示做消融或对比，但不作为Q1第一版必需项。

### 不做的事情

不为了与附件2形式一致而随意裁剪或投影成74维并宣称与附件2音频特征相同。

若最终主动选择74维，必须有明确的特征定义或可解释的降维方案，并将其写成自定义Q1表示，而不是官方特征复现。

---

## 4.6 M4-V 视觉特征

### 主方案

优先采用OpenFace，提取与情感分析直接相关的行为特征：

```text
Facial Action Units
Head Pose
Eye Gaze
Facial Landmarks / 其低维统计表示
```

### 选择理由

- 与MMSA-FET已有视觉提取工具链兼容；
- 适合视频情感场景；
- 对Q3的原始片段解释更友好；
- 能明确处理“没有检测到有效人脸”的情况。

### 观测失败处理

若某时间区间没有检测到有效人脸：

```text
vision_observed_mask[k] = 0
```

该位置不能只依赖“特征是否为全零”进行判断。

---

## 4.7 M5 自适应时序对齐与聚合

### 4.7.1 原始序列

设三模态原始时序特征为：

\[
X_T\in\mathbb{R}^{L_T\times D_T},
\]

\[
X_A\in\mathbb{R}^{L_A\times D_A},
\]

\[
X_V\in\mathbb{R}^{L_V\times D_V}.
\]

每个原始特征位置同时拥有对应时间区间：

\[
\tau_T,\quad \tau_A,\quad \tau_V.
\]

### 4.7.2 标准时间位置

对当前视频总时长 \(D\)，定义50个标准时间区间：

\[
I_k=
\left[
\frac{k-1}{50}D,\,
\frac{k}{50}D
\right),
\qquad k=1,\ldots,50.
\]

### 4.7.3 局部特征聚合

对模态 \(m\in\{T,A,V\}\)，第 \(k\) 个标准位置聚合所有与 \(I_k\) 有时间交集的原始特征：

\[
z_k^{(m)}
=
\operatorname{Pool}
\left\{
x_i^{(m)}
\mid
\tau_i^{(m)}\cap I_k\neq \varnothing
\right\}.
\]

得到：

\[
Z_T\in\mathbb{R}^{50\times D_T},
\]

\[
Z_A\in\mathbb{R}^{50\times D_A},
\]

\[
Z_V\in\mathbb{R}^{50\times D_V}.
\]

### 4.7.4 Pool策略

第一版建议使用简单稳定的：

```text
时间加权Mean Pooling
```

若后续验证发现短时变化损失明显，再比较：

```text
Mean Pooling
Max Pooling
Attention Pooling
```

不建议Q1第一版直接引入较复杂的可训练聚合器，避免让Q1特征提取与Q2/Q3预测模型耦合过深。

### 4.7.5 为什么不是直接截断或填充原始序列

视频时长差异较大，直接将原始帧/音频帧截断到固定长度会改变时间意义。先建立真实时间坐标，再进行时间区间聚合，可以保证三个模态的第 \(k\) 个位置对应当前视频中的同一相对时间范围。

---

## 4.8 M6 掩码设计

Q1开始统一定义三类状态，避免后续Q2把不同类型的零值混为一类。

### 4.8.1 `valid_mask`

表示标准位置是否属于有效输入范围。

```text
valid_mask[k] = 1
```

表示该位置属于真实序列。

```text
valid_mask[k] = 0
```

表示该位置仅用于批处理填充。

如果Q1最终所有样本均完整映射到50个标准时间位置，则其50个位置通常均为有效位置；该字段仍保留，以便与Q2/Q3统一接口。

### 4.8.2 `observed_mask`

形状：

```text
(50, 3)
```

分别对应：

```text
Text / Audio / Vision
```

表示有效位置上的该模态是否真正提取到可用观测。

例如：

```text
valid_mask[k] = 1
vision_observed_mask[k] = 0
```

表示该时间段是真实视频区间，但视觉特征未成功提取。

### 4.8.3 `perturb_mask`

Q1不主动生成缺失扰动，但接口预留给Q2。

Q2中人工遮蔽某个原本存在的局部位置时：

```text
valid_mask = 1
observed_mask = 1
perturb_mask = 0
```

从而明确区分：

```text
padding
原始模态不可用
人工局部缺失
```

---

## 4.9 M7 原素材追溯映射

每个标准位置 \(k\) 至少保存：

```json
{
  "position": 17,
  "start_time": 8.20,
  "end_time": 8.68,
  "text": {
    "words": [],
    "token_indices": [],
    "char_spans": []
  },
  "audio": {
    "sample_start": 0,
    "sample_end": 0,
    "frame_indices": []
  },
  "vision": {
    "frame_indices": []
  }
}
```

该映射用于Q3将特征空间结果恢复到原始素材。

例如：

```text
Q3发现 Vision position 17~20 贡献最大
                  │
                  ▼
            查询time_map
                  │
                  ▼
       恢复原视频时间 8.2~10.1 s
                  │
                  ├─ 对应视频帧
                  ├─ 对应音频区间
                  └─ 对应英文文本
```

注意：附件2/3本身没有完整原素材映射，因此该映射规则是Q1建立的方法接口；Q3针对附件4的视频需要按同样方法重新建立并核验，而不是直接假设附件4的50个位置天然等距对应视频时间。

---

## 4.10 M8 质量控制

每条样本需要保留处理状态，失败样本不静默删除。

建议记录：

```text
video_decode_ok
audio_extract_ok
text_alignment_ok
alignment_coverage
face_detect_ratio
silent_ratio
text_observed_ratio
audio_observed_ratio
vision_observed_ratio
empty_window_count
processing_status
error_message
```

### 重点异常

#### 文本

- 强制对齐失败；
- 文本过短；
- Tokenizer截断；
- 某些词无法定位。

#### 音频

- 无音轨；
- 长静音段；
- 解码异常；
- 特征提取失败。

#### 视觉

- 无有效人脸；
- 主说话人检测失败；
- 连续若干时间窗无视觉观测；
- 极低帧率或视频读取异常。

---

## 5. Q1输出设计

建议目录：

```text
q1_features/
├─ features/
│  ├─ <sample_id>.npz
│  └─ ...
├─ metadata/
│  ├─ <sample_id>.json
│  └─ ...
├─ sample_summary.csv
├─ extraction_config.json
└─ README.md
```

### 5.1 单样本特征

建议NPZ保存：

```text
text_features
audio_features
vision_features
time_intervals
valid_mask
observed_mask
```

其中：

```text
text_features   : (50, D_t)
audio_features  : (50, D_a)
vision_features : (50, D_v)
time_intervals  : (50, 2)
valid_mask      : (50,)
observed_mask   : (50, 3)
```

### 5.2 单样本metadata

建议JSON保存：

```text
sample_id
video_id
clip_id
video_path
duration
fps
audio_sample_rate
text
label
annotation

text_encoder
audio_extractor
vision_extractor

token_or_word_spans
source_audio_ranges
source_frame_indices

alignment_status
quality_info
processing_status
```

### 5.3 汇总表

`sample_summary.csv` 至少包括：

```text
sample_id
duration
text_dim
audio_dim
vision_dim
sequence_length
alignment_coverage
text_observed_ratio
audio_observed_ratio
vision_observed_ratio
processing_status
```

---

## 6. 与Q2、Q3的接口预留

## 6.1 Q1 → Q2

Q2训练仍以附件2的train/valid为主，不把附件1的100条样本直接并入Q2训练。

Q1向Q2继承的是：

```text
统一的位置编号
统一的mask语义
统一的输入组织
统一的模态投影接口
统一的时间追踪原则
```

Q2可以将官方三模态特征分别投影到共同隐藏维度：

\[
H_m = P_m(X_m),\qquad m\in\{T,A,V\},
\]

其中 \(P_m\) 为模态投影层。

Q1的自提取特征同样可以采用对应投影接口，但若原始特征语义不同，不直接共用Q2已经训练好的输入投影权重。

### Q2人工缺失接口

Q2在附件2上构造局部连续缺失时，沿用：

```text
valid_mask
observed_mask
perturb_mask
```

因此缺失实验可以明确区分：

```text
原始有效信息
原始不可观测信息
人为遮蔽信息
```

---

## 6.2 Q1 → Q3

Q3继承Q1的：

```text
time_map
位置编号
模态观测状态
原素材回溯方法
```

Q3模型得到局部重要性后：

```text
feature position
      ↓
time_map
      ↓
原始时间区间
      ↓
文字 / 音频 / 视频帧
```

从而输出可核验的关键片段，而不是只给出抽象特征索引。

---

## 7. 推荐实现选择

| 环节 | 主方案 | 备选 | 当前选择 |
|---|---|---|---|
| 数据管理 | MMSA-FET批处理结构 + 附件1适配器 | 自定义全新框架 | 基于MMSA-FET改造 |
| 文本来源 | 题目已有transcript | 完整ASR重转写 | 已有transcript |
| 文本时间定位 | transcript-audio强制对齐 | 视频等距映射 | 强制对齐 |
| 文本特征 | BERT类表示 | RoBERTa | BERT优先 |
| 音频特征 | OpenSMILE/eGeMAPS | wav2vec2 | OpenSMILE优先 |
| 视觉特征 | OpenFace | MediaPipe/深层视觉embedding | OpenFace优先 |
| 原始特征组织 | 保留变长序列 | 直接50步提取 | 变长优先 |
| 最终时序长度 | 自适应聚合到50 | 保持完全变长 | 50步 |
| 时间聚合 | 时间加权Mean Pooling | Max/Attention Pooling | Mean优先 |
| 掩码 | valid + observed | 仅零值判断 | 双掩码 |
| Q2预留 | 增加perturb_mask | 人工置零不记录 | 显式扰动掩码 |
| Q3预留 | time_map回溯 | 仅保存位置索引 | time_map |
| 音频维度 | 保持自提特征定义 | 强行74维 | 不强行74 |
| 视觉维度 | 保持自提特征定义 | 强行35维 | 不强行35 |

---

## 8. 实施顺序

```text
Step 1
读取label-100.xlsx并建立100条唯一sample_id
        │
        ▼
Step 2
逐视频读取真实duration / FPS / audio sample rate
        │
        ▼
Step 3
从MP4分离音频，保留原始时间信息
        │
        ▼
Step 4
使用已有transcript完成文本—音频强制对齐
        │
        ▼
Step 5
运行MMSA-FET可复用特征提取模块：
BERT / OpenSMILE / OpenFace
        │
        ▼
Step 6
生成三模态原始变长时序特征及时间戳
        │
        ▼
Step 7
通过Adaptive Temporal Alignment聚合到50步
        │
        ▼
Step 8
生成valid_mask与observed_mask
        │
        ▼
Step 9
建立每个位置的time_map
        │
        ▼
Step 10
保存100条feature + metadata + summary
        │
        ▼
Step 11
选择典型样本核验：
文本 ↔ 音频时段 ↔ 视频帧 ↔ 特征位置
```

---

## 9. 最低完成标准

Q1完成时必须满足：

1. 100条附件1样本全部有处理记录；
2. 每条样本均有唯一 `sample_id`；
3. 三模态特征均有清晰提取配置；
4. 最终三模态具有统一时序位置；
5. 不使用零值直接判断所有缺失；
6. 每条样本保存 `valid_mask` 与 `observed_mask`；
7. 保存位置到原素材的时间映射；
8. 提取失败或局部不可用的样本不被静默删除；
9. 可以从至少一个典型样本中展示：
   - 原始英文文本；
   - 对应音频区间；
   - 对应视频帧；
   - 对应时序特征位置；
10. 输出全部100条特征、汇总表、提取配置与读取说明；
11. 整体输出规模需要与比赛50 MB提交限制同时规划。

---

## 10. 论文中建议展示的Q1结果

### 10.1 方法主图

直接采用第3节流程图的压缩版：

```text
原始视频 + Transcript
          │
          ▼
  多模态信号解析
          │
          ▼
  文本—音频时间锚定
          │
          ▼
┌─────────┼─────────┐
│         │         │
▼         ▼         ▼
BERT   OpenSMILE  OpenFace
│         │         │
└─────────┼─────────┘
          ▼
 Adaptive Temporal Alignment
          │
          ▼
 Feature + Mask + Time Map
          │
          ▼
      Q1标准输出
```

### 10.2 典型样本对齐图

建议展示一条视频：

```text
原视频时间轴
0s ---------------------------------------------------- D s

位置        1        2        3            ...       50
            │        │        │                       │
Text      words    words    words                   words
Audio     frames   frames   frames                  frames
Vision    frames   frames   frames                  frames
            │        │        │                       │
            └──────统一time_map───────────────────────┘
```

同时显示：

```text
valid_mask
observed_mask
```

以证明模型输入位置与原始素材之间存在可核验对应关系。

---

## 11. 实现注意事项

1. 不把附件1理解成附件2的100条训练子集，Q1样本不直接整体加入Q2/Q3训练。
2. 不将 `50` 写成题目强制长度；它是为三问统一接口选定的设计参数。
3. 不将 `768/74/35` 写成Q1强制特征维度。
4. 不因为输出维度相同就声称Q1特征与附件2官方特征完全一致。
5. 不重新做完整ASR替代已有transcript；优先利用已有文本做强制对齐。
6. 不通过删除中间零行压紧序列，否则会破坏时间关系。
7. 不将全零特征自动视为人工缺失。
8. 标准化统计量只能在相应训练数据的有效观测位置上估计；Q1的特征提取和Q2训练统计量分开管理。
9. 任何无法精确恢复的时间对应关系必须记录为近似或未知，不能伪造精确时间戳。
10. 若最终使用大型预训练模型，需同步考虑模型参数与比赛50 MB附件限制。
