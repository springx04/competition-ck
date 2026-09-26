# E题Q3服务器具体实现说明 v1｜Agent执行版

> **最终结项**：Q3迭代已于2026-09-26结束。题目要求与交付范围按[最终收尾口径](../../Q3/reports/iteration_v2/Q3_最终方法与质量结论.md)解释；下文历史计划不再产生新的待办。保留输入空间归因与原文定位结论，不夸大声画时间映射证据。

> **当前执行状态与方法修订｜2026-09-26**：728条valid和20条专项的迭代2已完成；本文件下文“尚未启动”“服务器没有Q3”“先编写入口”等措辞属于2026-09-25初版历史，不再表示当前状态。当前实际方法、核验结果与诚实限制，以[最终方法与质量结论](../../Q3/reports/iteration_v2/Q3_最终方法与质量结论.md)和[完整实验报告](../../Q3/reports/iteration_v2/Q3_实验报告.md)为准。
>
> 最终方法修订：完整单元严格随机匹配（片段长度多重集合、模态成本、目标排除），Floyd无放回抽样后独立随机排列，避免跨模态按位置排序引入联合保留相关性；预算取不超过目标的最大可达成本，整词不切断；整组真实联合遮蔽并设置同成本point比较。block3实际是固定不重叠官方索引块，只能作粒度敏感性，不能称为已核实的复制组或真实时间片。valid按239个video_id簇抽样，簇内按样本数加权，报告真实可比分母和失败候选。
>
> 服务器配置为 `Q3/configs/q3_v2_server.yaml`，GPU推理环境为 `Q2/.venv/bin/python`；具体复算命令见[Q3 README](../../Q3/README.md)。初版 `q3.yaml` 与本地CPU配置不能替代当前服务器配置。CTC入口仍依赖同级Q1源码和权重，尚不满足初版“无Q1项目也能独立运行”的便携性目标；本轮复用了既有词时，不把机械检查称为人工听审。官方A/V来源仍 unresolved。

> 日期：2026-09-25。依据已审核的《Q3多粒度反事实解释方法v9：Q1/Q2闭环与代码核查版》。本文件是实施任务书，执行Agent应按本文件完成开发、验证、全部valid评价和附件4的20条预测解释，不再自行选择另一套方法。
>
> **迭代 2 执行补充（2026-09-26）**：服务器上的第二轮结果使用 `configs/q3_v2_server.yaml`、`runs/valid_v2` 和 `runs/special_v2`，旧运行目录不得覆盖。随机对照必须由完整连续单元严格枚举/抽样，匹配片段长度多重集合、模态成本和目标排除；预算不可达时保存最大可达成本。方向实验必须保存整段重测后的 `direction_retained`，失败候选保留并标记。valid 区间按 `video_id` 簇 bootstrap，并按簇内样本数加权。报告分别保留分类/强度首位模态、tokenizer/CTC人工审核状态和官方 A/V 来源状态。
>
> **必须在服务器单独新建 `Q3` 文件夹，Q3新增代码、配置、环境、下载内容和运行产物全部放在该文件夹内。服务器目前没有Q3代码；下文所有 `python -m q3 ...`、`scripts/...` 均是要求Agent先编写、再执行的新入口，不是声称已经存在。服务器已有Q2全部内容，优先复用其最终交付包。**
>
> 本次写作核对了本地Q2最终包及源码、Q1可复用代码和环境记录。本次SSH连接被拒绝，以下服务器路径和设备信息依据Q2已归档记录，执行时先检查；不能把它们称为2026-09-25本轮在线确认。本文没有执行Q3开发或服务器安装。

## 0. 给执行Agent的总指令

1. 默认项目父目录 `/root/gpufree-data/shuomo_E`，新建 `/root/gpufree-data/shuomo_E/Q3`。只向Q3写文件；Q2代码、权重、标准化及数据只读。不得把Q3实现塞入Q2，也不得运行会覆盖Q2结果的训练、选模或finalize流程。
2. 预测器固定为Q2最终 `late_attn_tune` 单模型，八层BERT、seed1111、CLS上下文开启、无类别偏置。不得下载一个原始BERT代替已经微调的权重；不得重训Q2/Q3预测器。
3. 全量valid为728条；专项为附件4对齐版01—20共20条。test默认不读、不新增评估；已有test结果只能带“自适应探索”身份引用。专项无标签，不调参数或报告准确率。
4. 完成三路及联合位置扫描、8组合Shapley、连续选段、方向卡、随机对照、保留实验、预算和分组粒度分析，再输出可复算明细、CSV、图和解释卡。不得只做几张案例图后宣布完成。
5. 需要的外部源码和词时对齐模型按第5节自行下载。没有Q1项目也能按本文完成；不要求用户另行上传Q1代码。无需下载MMSA-FET整套运行环境、OpenFace、WhisperX、EBMC或新的情感模型。
6. 原文定位、词发音时间、官方音视频特征来源是三种不同关系。无法证实的音视频行来源保留`unresolved`，不得伪造秒数。数据缺乏来源证据造成的未完成项须明确报告，同时完成不依赖该证据的全部计算。
7. 不新增SHA256流程、冻结contract、baseline文件或多重gate。使用Git版本、普通配置、主键、实际模型加载与必要测试记录即可。不得移除Q2已有检查。
8. 本任务没有发布、删数据或推送授权。所有模型下载和实验产物留在Q3；最终只提交源代码、配置、说明和适量结果材料，遵守项目已有Git规则。

## 1. 先确认目录与已有资源

### 1.1 默认路径和检查

服务器Agent从服务器终端开始，不需要再次SSH到自己。以下为实际可执行的只读检查与目录建立：

```bash
set -euo pipefail
export PROJECT_ROOT=/root/gpufree-data/shuomo_E
export Q2_ROOT="$PROJECT_ROOT/Q2"
export Q3_ROOT="$PROJECT_ROOT/Q3"
export DATA_ROOT="$PROJECT_ROOT/data"
export Q2_BUNDLE="$Q2_ROOT/delivery/q2_final"
test -d "$Q2_ROOT"
test -d "$DATA_ROOT"
mkdir -p "$Q3_ROOT"
export PIP_CACHE_DIR="$Q3_ROOT/.cache/pip"
export HF_HOME="$Q3_ROOT/models/.hf_cache"
export TORCH_HOME="$Q3_ROOT/models/.torch_cache"
export XDG_CACHE_HOME="$Q3_ROOT/.cache"
export TMPDIR="$Q3_ROOT/.tmp"
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$PIP_CACHE_DIR" "$HF_HOME" "$TORCH_HOME" "$TMPDIR"
cd "$Q3_ROOT"
ls -ld "$Q2_ROOT" "$DATA_ROOT"
if test -d "$Q2_BUNDLE"; then ls -ld "$Q2_BUNDLE"; fi
df -h "$Q3_ROOT"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
```

若父目录不同，先查看Q2的 `results/final/export.json` 和配置中的真实路径；只能把找到且核对过的实际路径写入Q3配置，不能创建一个空Q2目录冒充已有项目。默认路径存在就不要遍历整台服务器。

Q2归档机器为RTX4090、约24GB显存，Python3.11.16、torch2.6.0+cu124。内存、空闲磁盘以执行时读数为准。不要删除历史实验或数据释放空间；空间不足时报告实际所需内容和缺口。

### 1.2 必需已有文件

| 资源 | 默认路径/内容 | 用途 |
|---|---|---|
| 最终包 | `Q2/delivery/q2_final/ensemble_config.json` | 实际加载入口 |
| Q2推理源码 | 最终包的`src/q2/` | 使用与交付模型一致的实现 |
| 标准化 | 最终包的`normalizer.npz` | 已由train拟合；禁止重新fit |
| 学生参数 | 最终包的`students/student_0.safetensors` | 下游预测网络 |
| 文本参数 | 最终包的`assets/text_encoder/shared_embeddings.safetensors`及`member_0.safetensors` | 微调后的完整BERT参数拆分 |
| 词表与架构 | 同目录`config.json`、`vocab.txt`及已有tokenizer文件 | 构造相同编码器与字符映射 |
| valid原始输入 | `Q2/data/processed/valid/*.npy`与`metadata.json` | 优先复用；不加载约1GB PKL反复重读 |
| 附件4 | `DATA_ROOT/附件4-可解释专项视频样本与特征文件/附件4-可解释专项视频样本与特征文件/对齐版本/` | `01.pkl`至`20.pkl`，视频在`videos/` |
| 既有结果 | `Q2/results/final/valid/metrics.json`、`selection.json`、`offline_verification.json` | 原样结果和来源核对 |

**最终包缺失时的确定处理顺序：**

1. 若 `Q2/delivery/q2_final.zip` 存在，先列出ZIP成员，核对无绝对路径或`..`，解压到 `Q3/models/q2_predictor/`，自动定位唯一包含`ensemble_config.json`的根目录，写入配置。
2. 若只有训练checkpoint，使用Q2源码的 `q2.ensemble.export_ensemble(root, selection, dest)`：`root=Q2_ROOT`，`selection`读取`Q2/configs/final_selection.json`，`dest=Q3/models/q2_predictor`。先检查其唯一成员、variant、seed及class_bias；该函数可写目的目录，不能调用`finalize_q2.py`。需要的原始BERT目录和checkpoint必须确实存在。已有dest不覆盖，核对后复用。
3. 两者均不可恢复则记录“缺少最终微调权重”，继续完成算法单元测试、源码和非预测文档，停止依赖模型的真实实验。GitHub不含这些权重，下载官方原始BERT不能恢复本项目模型。

### 1.3 最终包内容应满足的事实

`ensemble_config.json`的`members`长度为1，成员`variant=late_attn_tune`、`seed=1111`、`text.cls_context=true`、`text.clip_z=null`；顶层及evaluation的`class_bias`为null。BERT config为8层、hidden_size256、4个头、vocab_size30522。加载函数名含ensemble只是包装器名称，当前模型不是多模型集成。

这些是模型身份核对；只检查本文直接依赖的字段。不从Q2通用配置里残留的MSD、补偿字段推断实际架构。

## 2. Q3目录与模块分工

Agent应创建以下目录。无需提前建立空的结果数据文件；需要时写入真实结果。

```text
Q3/
  README.md
  THIRD_PARTY.md
  pyproject.toml
  requirements-main.txt
  requirements-align.txt
  configs/q3.yaml
  src/q3/
    __init__.py
    __main__.py             # CLI，只负责分派
    config.py
    data.py                # valid/附件4适配、元数据与manifest
    predictor.py           # 唯一Q2导入和模型调用入口
    interventions.py       # 150位布尔集合、遮蔽、保留
    attribution.py         # D、方向、TV、模态/Shapley/局部扫描
    grouping.py            # 原文词与复制组、3位置窗口
    selection.py           # 连续区间DP、主/方向/低影响选择
    controls.py            # 可行对照计数、无放回均匀抽样
    experiments.py         # 预算、删除、保留与粒度实验
    statistics.py          # 视频簇bootstrap及性能指标
    text_mapping.py        # tokenizer核对及Unicode字符坐标
    mapping.py             # 字符/词时/音视频来源三层连接
    results.py             # 文件读写、稳定ID和断点续算
    plots.py
    report.py
    export.py
  src/q3_align/
    __init__.py             # 不导入q3预测器
    __main__.py             # 独立环境运行的对齐CLI
    pipeline.py
    vendor/                # 下述Q1模块的限定复用副本
      __init__.py
      alignment.py
      text_map.py
      media.py
      storage.py           # media.py的本地JSON写出依赖
  scripts/
    run_main.sh
    run_align.sh
    fetch_alignment_assets.py
    collect_environment.py
  tests/
  third_party/q1_reference/ # 下载的只读参考代码，或稀疏Git工作树
  models/ctc/              # 仅词时对齐模型；按需另有恢复的q2_predictor
  data/                    # 小型清单及专项解析结果，不重复官方原数据
  runs/valid_main/
  runs/special_main/
  reports/
  outputs/
  delivery/
  .venv/                  # 仅当Q2环境不可复用时建立
  .venv-align/            # 独立对齐环境
```

职责约束：预测器只接收原始模型字段，不接收标签；归因器不读取真实标签；真实标签仅进入statistics/report。对齐进程只消费原文与媒体，不修改用于预测的文本或特征。`q3/__init__.py`与`q3_align/__init__.py`保持轻量，避免对齐命令误导入另一个环境的依赖。

## 3. 情感推理模型与主环境

### 3.1 模型选择已确定

- 使用Q2最终微调模型，原始预训练来源为 `google/bert_uncased_L-8_H-256_A-4`，**正常流程不重新下载原始模型**。
- 文本参数使用包内shared embeddings和member0共同恢复；不拿Q1的bert-base-uncased替换。
- 三路输入为token ID、74维音频、35维视觉；文本隐藏维256。Q1的768/25/22维特征不进入Q2模型。
- 推理全程FP32计算，文本磁盘存储FP16由Q2加载器转float；eval、no_grad，禁用AMP、TF32和torch.compile，使用Q2已设定的eager attention。
- 分类为logits softmax，顺序0负向、1中性、2正向；强度直接取score，不按类别修正。

### 3.2 优先只读使用Q2已有Python环境

```bash
export Q3_PY="$Q2_ROOT/.venv/bin/python"
test -x "$Q3_PY"
"$Q3_PY" -c 'import sys,torch,numpy,transformers,tokenizers,safetensors,yaml,pandas,scipy,sklearn,matplotlib,pytest; print(sys.version); print(torch.__version__); print(numpy.__version__); print(transformers.__version__)'
"$Q3_PY" -m pip check
```

若实际Q2解释器在别处，从Q2 README/执行脚本查到路径后填写`configs/q3.yaml`及`run_main.sh`。只读使用该解释器和包，不向Q2环境执行pip install/upgrade。通过 `PYTHONPATH="$Q3_ROOT/src"` 运行Q3即可，不需要在Q2环境editable安装Q3。

不能通过“用Q2解释器创建`--system-site-packages`虚拟环境”假设继承了Q2虚拟环境全部依赖；虚拟环境嵌套不保证这种行为。

若Q2环境不存在或不兼容，在Q3创建独立环境：

```bash
python3.11 -m venv "$Q3_ROOT/.venv"
export Q3_PY="$Q3_ROOT/.venv/bin/python"
"$Q3_PY" -m pip install 'setuptools>=68,<81' wheel
"$Q3_PY" -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
"$Q3_PY" -m pip install -r "$Q3_ROOT/requirements-main.txt"
"$Q3_PY" -m pip check
```

若PATH无python3.11，使用已确认的Q2解释器执行`-m venv`。不得为得到不同Python版本覆盖系统Python。

`requirements-main.txt`逐项采用Q2实际记录：

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

torch通过上面的官方CUDA索引独立安装。主环境不需要torchaudio、ESPnet、Captum、SHAP包或LLM API。表格CSV和绘图用已有pandas/matplotlib，解释卡用静态HTML与原生JavaScript，无Node构建依赖、无在线服务。

## 4. 唯一模型适配器的具体写法

### 4.1 导入与加载

`q3.predictor`在任何`import q2`前将最终包的`src`插入`sys.path[0]`，再调用：

```python
from q2.ensemble import load_ensemble
bundle = load_ensemble(bundle_dir, device="cuda:0")
```

检查 `Path(q2.__file__).resolve()` 在所选包的`src/q2`内。不要同时import服务器另一份`Q2/src/q2`，也不要把最终包源代码复制改写成新的模型。缺失包需要恢复时才在独立恢复进程使用Q2源码，恢复结束退出，正式进程重新从包加载。

将导入路径准备单独实现为ensure_q2_import_root(bundle_dir)，不加载模型。CLI在导入data/state/metrics等Q2辅助模块之前先调用它；模块顶层不得提前import另一份q2。prepare只需包内源码和tokenizer，不应为了读数据加载BERT权重。

固定：

```python
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
```

加载后验证单成员、模型eval状态、BERT层数及前述设置。按实际命令记录device；CUDA不可用可以用同一模型CPU完成接口测试，真实全量计算改用CPU时必须明确报告，不能悄悄换轻模型。

### 4.2 RawBatch与返回结构

只送入以下五个Tensor，所有批均有样本轴：

| 字段 | dtype | shape |
|---|---|---|
| input_ids | torch.int64 | `(B,50)` |
| stored_attention | torch.int64 | `(B,50)` |
| token_type_ids | torch.int64 | `(B,50)` |
| audio | torch.float32 | `(B,50,74)` |
| vision | torch.float32 | `(B,50,35)` |

`sample_id/raw_text/video_id/label`作为Q3侧元数据，不送入BERT。调用 `bundle.predict_raw(raw_batch, return_details=True)`，返回对象不是dict；用`.logits/.score/.U/.J`取值。批量归因只把这四项转CPU保存，必要时才请求H，避免存海量中间张量。

Q3统一返回 `Prediction(logits[N,3], probs[N,3], score[N], U[N,50,3], J[N,50])`。softmax在FP32计算后转CPU；差值和统计用float64。记录原始logits供复核，不保留计算图。

最终late分支的`alpha/attention/e_hat`是None、`B_comp`为零。不得把它们当真实可靠性或补偿信息。注意力排序对照本版不执行，避免为了一个非必需对照修改Q2接口。

### 4.3 当前观测与遮蔽

优先复用 `q2.state.infer_state`。文本观测：stored_attention=1且ID不为0/101/102/103；100即UNK仍是观测。音视频一行任意维非零即观测。干预集合永远从原`U0`选取。

采用Q2 `Perturbation(P=..., descriptors=...)` + `apply_span`，或在Q3内写下列严格等价薄函数并与Q2比较：

```python
changed = {key: value.clone() for key, value in raw.items()}
changed["input_ids"][P[..., 0]] = 103
changed["audio"][P[..., 1]] = 0
changed["vision"][P[..., 2]] = 0
```

`P.shape=(B,50,3)`，bool，且`P & ~U0`恒为空。**stored_attention不改、位置不压缩、不删行、不先标准化。** MASK通过Q2状态函数自动从BERT attention与池化排除。每次完整`predict_raw`重新编码受损文本及CLS；首版不实现跨不同文本输入的隐藏层缓存。

一个样本一次加载原输入。多干预通过clone/expand后clone组成batch，每批128个视图；GPU OOM时仅减为64、32、16并重试同一批，仍失败才记录异常，不能跳样本。一次只运行一个GPU工作进程。

### 4.4 数值一致性与方向容差

先选valid顺序中的首8条，以及附件4全部20条，比较单条和batch128前向、同批重复前向；保存最大logits/prob/score绝对差及类别一致性。方向数值容差定义为：

- `eps_prob=max(1e-7, 10*max_observed_probability_difference)`；
- `eps_score=max(1e-6, 10*max_observed_score_difference)`（原[-3,3]尺度）；
- `eps_D=0.5*eps_prob+0.5*eps_score/6`。

这里max_observed_difference取“同批重复”与“同输入单条/批量比较”两类差异的最大值，不能只取重复同批误差而漏掉batch形状的影响；文件分别保存两类原始最大差。若某专项输入失败，记录实际probe数，不因缺少必测样本仍宣称完整核对通过。

这只是舍入噪声边界，不是解释质量阈值。若相同输入出现类别不一致，或概率差>1e-4、score差>1e-3，先检查eval、输入串扰、TF32及导入来源，不靠扩大容差掩盖问题。后续使用同一个`numeric_tolerance.json`，不逐样本调容差。

原样Q3结果与同一调用的Q2结果应一致；与FP32训练checkpoint不是逐bit相等目标。Q2已报告导出存储差异，不能把checkpoint的p0与导出包的干预p混算。

## 5. 需要自行下载的代码、模型和对齐依赖

### 5.1 外部资源清单：必须区分用途

| 资源 | 确定来源 | 本题用途 | 是否默认下载 |
|---|---|---|---|
| Q2最终情感模型 | 服务器已有最终包 | 全部预测和归因 | 不下载其他情感模型 |
| Q1薄适配源码 | `springx04/competition-ck`提交`139aa00ba794b21355aeeb6c55629a2d1d32c3e1` | 文本分词记录、CTC调用和媒体时钟 | 无本地同版本源码时下载 |
| ESPnet CTC模型 | 下表固定Hub模型与revision | 仅附件4词时对齐 | 无完整本地快照时下载 |
| ESPnet与ctc-segmentation | 固定PyPI版本 | 运行既定CTC模型和强制对齐 | 安装在独立对齐环境 |
| Captum/SHAP | 无 | 三模态8组合自行精确求和 | 不安装 |
| WhisperX/OpenFace/MMSA-FET/EBMC | 无新增运行需求 | 不是本Q3算法的运行依赖 | 不下载、不编译 |

Q1源码不是全套特征重提取任务；不要运行其runner、审核流程或`download_models.py`（后者还会下载本题不需要的BERT）。只参考并复用下面明确列出的模块。

### 5.2 Q1代码下载与复制范围

```bash
mkdir -p "$Q3_ROOT/third_party"
git clone --filter=blob:none --no-checkout \
  https://github.com/springx04/competition-ck.git \
  "$Q3_ROOT/third_party/q1_reference"
git -C "$Q3_ROOT/third_party/q1_reference" sparse-checkout init --cone
git -C "$Q3_ROOT/third_party/q1_reference" sparse-checkout set \
  Q1/q1_features/src/q1_features Q1/q1_features/env \
  Q1/q1_features/configs Q1/q1_features/scripts
git -C "$Q3_ROOT/third_party/q1_reference" checkout --detach \
  139aa00ba794b21355aeeb6c55629a2d1d32c3e1
```

路径已存在则检查其提交，不重复clone或覆盖。Git传输不可用时从同一仓库固定提交归档下载：

```text
https://codeload.github.com/springx04/competition-ck/zip/139aa00ba794b21355aeeb6c55629a2d1d32c3e1
```

解压到Q3/third_party内保留来源记录。不要用main的未来变化替换固定参考版本。

复制到 `src/q3_align/vendor/` 的文件仅为：

- `alignment.py`：复用`find_ctc_files`、`prepare_ctc_config`、`load_ctc_model`、`ctc_forward`、`align_token_arrays_detailed`。
- `text_map.py`：复用`build_words`、`normalize_ctc_text`、`prepare_ctc_rows`及其同文件依赖。Q3文本预测映射仍按第8节从原字符串分词，不能调用Q1的BERT分块流程替代官方50位置。
- `media.py`：复用`probe_and_decode`与其时钟、连续性、ffprobe/ffmpeg辅助逻辑；不使用25Hz选帧结果充当官方行映射。
- `storage.py`：保留`media.py`依赖的`write_json/write_jsonl`及其同文件依赖；输出目录由Q3传入。不得漏掉其相对导入。

保持原文件版权/说明，`THIRD_PARTY.md`注明复制来源、提交、用途及Q3修改。若原仓库无明确开源LICENSE，记录“项目内部代码复用，未发现独立许可声明”，不要自拟MIT许可。第三方库与CTC模型的许可证另行保留。

### 5.3 对齐环境固定版本

ESPnet202402的发布依赖要求NumPy<1.24，与Q2的NumPy1.26.4冲突，因此必须使用单独进程和环境。Q1实际运行曾使用torch2.5.1；本说明选同一torch/torchaudio组合，CUDA12.1轮子。Python使用可用的3.11；这套独立环境必须由Agent安装并实际验证，不把Q1在Python3.10上的运行成功冒充新环境验证。

```bash
"$Q3_PY" -m venv "$Q3_ROOT/.venv-align"
export ALIGN_PY="$Q3_ROOT/.venv-align/bin/python"
"$ALIGN_PY" -m pip install 'setuptools>=68,<81' wheel 'Cython<3'
"$ALIGN_PY" -m pip install torch==2.5.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu121
"$ALIGN_PY" -m pip install -r "$Q3_ROOT/requirements-align.txt"
"$ALIGN_PY" -m pip check
```

`requirements-align.txt`：

```text
numpy==1.23.5
scipy==1.10.1
numba==0.57.1
llvmlite==0.40.1
librosa==0.9.2
soundfile==0.12.1
resampy==0.4.3
espnet==202402
typeguard==2.13.3
sentencepiece==0.1.97
ctc-segmentation==1.7.4
importlib-metadata==4.13.0
transformers==4.38.2
tokenizers==0.15.2
huggingface-hub==0.23.5
safetensors==0.4.3
PyYAML==6.0.2
num2words==0.5.13
av==12.3.0
Pillow==10.4.0
```

pip会安装ESPnet声明的间接依赖，实际清单写入`reports/environment_align.txt`。遇到依赖冲突先根据错误修复该独立环境；不得升级主环境NumPy或更换CTC模型。缺少C/C++编译器、Python头文件、FFmpeg或libsndfile时，服务器为Debian/Ubuntu且有权限才安装所需系统包`build-essential python3-dev ffmpeg libsndfile1`，不为本题安装完整CUDA toolkit。FFmpeg/ffprobe至少支持当前视频解码，并记录实际版本；Q1归档为6.1.1，不要求通过重编译系统强行凑同一patch版本。

在对齐环境检查`import espnet, sentencepiece, ctc_segmentation, av, soundfile`以及`from espnet2.tasks.asr import ASRTask`；最终有效性以实际加载指定模型并对一条专项音轨前向为准，不以import成功替代模型可运行。

### 5.4 CTC模型下载：来源、文件和加载

固定模型：

```text
repo_id = espnet/kamo-naoyuki_librispeech_asr_train_asr_conformer5_raw_bpe5000_schedule-truncated-c8e5f9
revision = e6a0f274799b5a4c157d1d5bc20de4c569e25f7e
source = https://huggingface.co/espnet/kamo-naoyuki_librispeech_asr_train_asr_conformer5_raw_bpe5000_schedule-truncated-c8e5f9
```

Q1来源记录注明模型卡为CC-BY-4.0。Agent下载后保存README/meta/许可信息及来源记录。本轮无法重新访问Hub模型页，固定revision来自已有Q1记录；实际下载成功及文件齐全由Agent确认。

新建 `scripts/fetch_alignment_assets.py`，只下载以下内容：

```python
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="espnet/kamo-naoyuki_librispeech_asr_train_asr_conformer5_raw_bpe5000_schedule-truncated-c8e5f9",
    revision="e6a0f274799b5a4c157d1d5bc20de4c569e25f7e",
    local_dir=ctc_dir,
    local_dir_use_symlinks=False,
    allow_patterns=[
        "README.md", "meta.yaml", "LICENSE*",
        "data/token_list/**",
        "exp/asr_train_*/config.yaml",
        "exp/asr_train_*/valid.acc.ave_10best.pth",
        "exp/**/feats_stats.npz",
    ],
)
```

下载后读取ASR YAML `normalize_conf.stats_file`：若其实际相对路径没有被通配下载覆盖，使用同一个repo/revision的`hf_hub_download`补该**确切相对文件**。不把所有模型历史checkpoint都下载下来。必须有唯一ASR config、`valid.acc.ave_10best.pth`、`data/token_list/bpe_unigram5000/bpe.model`和配置指定的global-MVN统计量。

使用Q1的`prepare_ctc_config`把BPE及统计量路径转为绝对路径，输出Q3模型目录内`runtime-config.yaml`，不覆盖原config。调用`ASRTask.build_model_from_file`经Q1函数加载，eval、FP32；模型输出仅供CTC，不参与情感预测。

若服务器已有完整同revision CTC目录，可只读复用模型大文件，但runtime YAML写到Q3。对Q1原`prepare_ctc_config`做这个单一薄改造；不得把runtime配置写进只读Q1目录。没有完整来源记录时按固定revision自行下载。

网络暂时失败可重试同一来源。无法获得权重时完成全部非对齐模块并明确词时定位未完成；不能静默换WhisperX、均匀时间映射或ASR新转写。

## 6. 配置文件：Agent必须显式写出的选项

`configs/q3.yaml`至少包含下列值，路径按第1节实际检查填写：

```yaml
project:
  root: /root/gpufree-data/shuomo_E/Q3
  q2_root: /root/gpufree-data/shuomo_E/Q2
  bundle_dir: /root/gpufree-data/shuomo_E/Q2/delivery/q2_final
  data_root: /root/gpufree-data/shuomo_E/data
  valid_dir: /root/gpufree-data/shuomo_E/Q2/data/processed/valid
runtime:
  python: /root/gpufree-data/shuomo_E/Q2/.venv/bin/python
  device: cuda:0
  view_batch_size: 128
  gpu_workers: 1
  num_workers: 0
  seed: 20260925
  amp: false
  tf32: false
  text_cache: false
method:
  modality_order: [T, A, V]
  class_names: [Negative, Neutral, Positive]
  cls_weight: 0.5
  score_weight: 0.5
  score_scale: 6.0
  budgets: [0.10, 0.20, 0.30]
  main_budget: 0.20
  max_intervals: 3
  direction_budgets: [0.20]
  keep_budgets: [0.20]
  grouped_budgets: [0.20]
  controls_per_set: 50
  min_controls_for_tail_description: 20
  bootstrap_repeats: 1000
  replication_groups_file: data/review/official_replication_groups.jsonl
alignment:
  python: /root/gpufree-data/shuomo_E/Q3/.venv-align/bin/python
  model_dir: /root/gpufree-data/shuomo_E/Q3/models/ctc
  device: cuda:0
  sample_rate: 16000
  min_window_size: 8000
  max_window_size: 100000
  score_min_mean_over_L: 30
  min_log_score: -5.0
  batch_size: 1
  allow_cpu_on_cuda_oom: true
  review_file: data/review/word_times.csv
  av_provenance_file: data/review/official_av_mapping.csv
  external_command_timeout_s: 120
  audio_duration_tolerance_s: 0.02
  playback_context_s: 0.20
output:
  valid_run: runs/valid_main
  special_run: runs/special_main
  float_csv_digits: 8
  save_all_view_outputs: true
  include_source_videos_in_delivery: false
```

`min_log_score=-5.0`沿用Q1配置，为词时对齐的算法候选筛选；不是官方声画时间正确性的阈值。bootstrap等实验随机数采用第12节的命名空间种子，不依赖Python`hash()`。

主环境与对齐环境分别记录Python路径、版本、pip freeze、设备、FFmpeg版本。记录配置的普通JSON/YAML副本和Git版本即可，不添加文件摘要系统。

## 7. 数据适配、输入清单与输出数值身份

### 7.1 valid数据

复用 `q2.data.AlignedDataset` 或直接以mmap读取已有 `.npy`，两者必须保持同一顺序。字段为`input_ids/stored_attention/token_type_ids/audio/vision/class_id/score`，元数据从`metadata.json`读`id/raw_text/video_id`。ID用于展示，索引0—727用于稳定计算种子与文件夹。

若processed/valid不存在，用Q2 `load_pickle`读取一次附件2 aligned_50.pkl，取`source['valid']`，再用`unpack_record(...,'attachment2')`，结果只写Q3/data/valid。**不调用Q2 prepare_data，不fit normalizer，不处理train/test缓存。** `video_id`优先元数据；缺失时沿用Q2 `sample_id.split('$_$')[0]`，记录解析方式。

校验样本数728、ID唯一、三类顺序、shape和有限值。发现与归档数量不同先查路径，不调整预期数量绕过。

### 7.2 附件4数据

显式构建01—20文件名列表，不以字典序或通配结果猜编号。每个PKL经Q2 `unpack_record(data,'attachment4')`读取；顶层是记录，不是`data['test']`。原text_bert为`(3,50)`，适配后为`(1,3,50)`拆出三行；音视频加样本轴后为`(1,50,74)/(1,50,35)`。

`sample_id`用于输出一律是文件编号`01`…`20`；内部原ID另存`source_id`，不让字符串化的内部对象覆盖文件编号。raw_text保持原Unicode字符串，不stripping、不小写化原文。核对`videos/01.mp4`…`20.mp4`与PKL成对；视频不可用不阻止特征预测，但标记素材定位缺口。

字段缺失、NaN/Inf或非法ID不能静默填补。逐样本写错误原因，其余样本继续；最终导出20行时该样本预测列为缺失且状态为failed，不能伪造预测，并明确没有实现20条成功推理。已总结的13号视觉全零应作为状态检查，不强行把它修成有视觉。

### 7.3 原观测记录

每条保存：官方50位置的`U0[50,3]`和`J0[50]`、各路n_observed、原始预测logits/probs/score、`c_star`。解释目标`c_star`在该样本运行中固定。

所有待干预集合P按 `(t,m)`映射为整数`j=3*t+m`（m顺序T/A/V），用递增tuple作为缓存主键。保留干预也转换为**实际删除集合**再调用预测器。同一样本相同删除集合只前向一次；不同样本不能共用该缓存。普通tuple即可，不引入hash指纹文件。

score差保存原[-3,3]尺度和归一化/6两个值；full-precision JSON/NPZ用于计算，CSV最后展示时保留8位小数。JSON中缺失值用null，禁止输出NaN/Infinity或把未算量写0。

## 8. 文本映射、分组与粒度：确定实现规则

### 8.1 从实际50位置回到原文

使用包内assets/text_encoder的BertTokenizerFast，local_files_only=True；直接在**原始字符串**上调用，不先按空格重拼、不改标点。要求return_offsets_mapping=True和return_special_tokens_mask=True，分别生成两种候选：

1. standard50：add_special_tokens=True、truncation=True、max_length=50、padding="max_length"。
2. prefix50：先add_special_tokens=True、truncation=False、padding=False取得全序列，再截取前50位置；不足补PAD0及offset(0,0)。attention按Q2 tokenizer_check定义为token_id!=0，token_type全0。

同时逐位置核对IDs、stored_attention和token_type，不能只比较非PAD词元。两者都匹配时优先standard50，并记录两个布尔结果；均不匹配时token_mapping_status=unexplained，不做模糊词匹配、字符串搜索或手工补齐。该样本的官方位置归因照常运行，字符定位和整词归因记不适用。

offset采用Python Unicode code point的左闭右开[char_start,char_end)，不是UTF-8字节或JavaScript UTF-16索引。保留原文、token_id、token字符串、offset、官方t、is_observed和is_unknown。UNK100如有有效offset可以定位字符，但记录未知词元；不能据此声称模型识别了具体词义。CLS/SEP/PAD/MASK不产生文本归因原子。

前端高亮使用Array.from(raw_text)按code point切片，或在Python预生成转义HTML；不能直接用JS slice处理这些offset。HTML通过html.escape写入，不把原文当HTML执行。

### 8.2 整词组的唯一划分

沿用Q1 build_words，以正则非空白串产生完整词表，word_id从0递增，纯标点词仍有自己的ID。**此处“词”是原文空白分隔段，不是语言学词汇分析，也不是BERT内部word_ids。** 对每个观测token，用其offset与原词区间的正长度重叠匹配；必须恰好归入一个词。零长度、跨两个原词或无法定位的token视为组映射失败，不能取第一个猜测。

同一原词的全部已存储观测token构成一个组。用未截断tokenization的offset确定该词是否还有未存储piece：有则truncated_word=true、group_label=stored_word_part，否则group_label=whole_word。这一判断按完整offset和piece次序，不靠字符串中是否有##。同一个词在原文重复出现也保持不同word_id。

每条文本组应为O_T的互不相交、完整、按位置排序的划分；组内可能含标点piece。只要某个观测token无法确定归属，该样本不进入整词选择比较，仍可输出已成功词的扫描值作局部诊断，并运行第8.4节索引块补充。纯标点词不自动合并到相邻词；CTC可有另一个发音锚点，两种语义不得混用。

### 8.3 音视频复制组：证据文件决定是否启用

可选输入data/review/official_replication_groups.jsonl，每行含split、sample_id、modality、group_id、positions、source_kind、source_ref、source_locator、explanation；positions为官方0—49原观测位置。有效依据必须给出该样本的词—行关联及同词展开/复制关系，如可复核的生成代码、官方对齐表或已完成且记录前提的Q2审计。**仅特征数值相等、与文本行号相同、肉眼感觉同时发生，均不足以启用语义复制组。**

先从Q2已有MD总结和审计文件找依据，不重新扫描全量原视频。没有足够依据即replication_group_status=unavailable，默认不把全数据集相邻同值行合组。若部分复制组成立，其余行以singleton组补齐，但明确partial_replication_coverage及有来源组覆盖率。组不得重叠或交错；同组首尾之间若有其他观测行，应归属同组，否则该分组不能进入连续组选择。

每个组以原始输入联合遮蔽后计算D；不将组分数均分给各行。文本字符关系成立，也不自动使音视频复制关系或时间来源成立。

### 8.4 语义分组不可用时的固定索引补充

对该模态每个观测中心t扫描[max(0,t-1),min(50,t+2))内的全部原观测行，空洞不补内容；相同实际集合只计算一次，但保留对应中心列表。这是center3扫描，无词义或真实时间含义。

为实现预算选择，另使用**不重叠**的block3分组：[0,3)、[3,6)、…、[48,50)，各块与O_m相交，去掉空组。不能把有重叠的center3窗口直接送入可加组DP。无语义组时采用block3选择并进行同粒度随机对照；报告“索引粒度补充”，不填入整词/复制组实验成功分母。

## 9. 归因计算：公式、数值和空状态

### 9.1 基础视图与变化量

每条先计算p0、s0、c_star、U0、J0。c_star=argmax(p0)，完全相等按0、1、2较小索引，之后保持固定。原观测集合O={(t,m):U0[t,m]}。对实际删除集合E调用唯一预测器，保存：

```python
delta_class = p0[c_star] - pE[c_star]
delta_score_raw = s0 - sE
delta_score = delta_score_raw / 6.0
D = 0.5 * abs(delta_class) + 0.5 * abs(delta_score)
TV = 0.5 * sum(abs(p0[c] - pE[c]) for c in range(3))
class_changed = int(argmax(pE) != c_star)
```

数值方向：delta_class>eps_prob支持原类，<-eps_prob抑制原类，其余indistinguishable；delta_score_raw>eps_score推向积极，<-eps_score推向消极，其余数值上不可区分。保存原差值，不把微小原差值覆写成0。TV只作诊断，不参与主证据排序。

### 9.2 模态消融与主要参考模态

对T/A/V分别取E=O_m。原模态无观测时删除集合为空，D_m=0、局部解释状态not_applicable。三路总量D_sum若≤eps_D，weights=null、primary_modalities=[]、状态undistinguished；否则w_m=D_m/D_sum，主要模态包含所有max(D)-D_m≤eps_D的**有观测**模态，按T/A/V排序保留并列。

分类排序以abs(delta_class_m)，强度排序以abs(delta_score_raw_m)，分别按eps_prob、eps_score判并列；全部未超过其容差时首位集合为空。两个非空首位集合不相同记head_top_disagreement=true；任一为空时该项为null并给原因。权重展示同时列原D及原观测数，不能只展示百分比。

三路全空时只保存预测器真实返回的先验输出、八子集和空参与者结果；内容归因/选段/保留比较均not_applicable，不生成虚假零热力图。不得用人工构造均匀概率替代Q2全空输出。

### 9.3 精确三模态Shapley

子集bitmask固定T=1、A=2、V=4，遍历0—7；输入为只保留子集中的原观测，其余删除。v_c(S)=p_S[c_star]，v_s(S)=s_S/6。对m循环其余两模态的四个子集，权重为大小0的1/3、大小1的1/6、大小2的1/3。

```python
# subset_probs.shape == (8,3); subset_scores.shape == (8,)
# 第0维严格按保留bitmask 0,1,...,7排序。
subset_probs = np.asarray(subset_probs, dtype=np.float64)
subset_scores = np.asarray(subset_scores, dtype=np.float64)
values = np.column_stack((subset_probs[:, c_star], subset_scores / 6.0))
phi = np.zeros((3, 2), dtype=np.float64)
for m in range(3):
    bit = 1 << m
    for subset in range(8):
        if subset & bit:
            continue
        weight = {0: 1/3, 1: 1/6, 2: 1/3}[subset.bit_count()]
        phi[m] += weight * (values[subset | bit] - values[subset])
residual = phi.sum(axis=0) - (values[7] - values[0])
```

以缓存的八个预测float64求和；数学效率残差应接近浮点误差，普通测试abs(residual)≤1e-12。若不同路径重新前向比较，另用第4.4节预测容差。空参与者phi应为0（同一删除集合共用输出）。保存phi_class[3]、phi_score_scaled[3]、phi_score_raw=6*phi_score_scaled、空集预测及残差。不得对D做同一效率分解，也不把abs(phi)归一化替代w。

### 9.4 全位置、联合位置和组扫描

- 单模态：每个(t,m)∈O恰好有一次扫描请求，E={(t,m)}。逻辑矩阵为[50,3]，图上转置为3×50；缺失格用valid mask显示N/A，不写作真实0。
- 联合位置：每个非空O_t={(t,m)∈O}计算E=O_t，cost_t=|O_t|为1/2/3；D必须重测，不能加三路D。
- 组：第8节每个有效word/replication/block3组计算联合删除D，记录actual_cost、span和完整位置；center3单列。

用缓存合并重复请求，不合并不同角色记录。基础扫描名义上最多208个不同视图（8子集+150单点+50联合）；组和随机实验另计。每条记录requested_view_count、unique_forward_count、cache_hits、实际耗时与最大批量，不将208当全实验成本。

## 10. 连续证据选择的精确算法

### 10.1 预算、区间和可行集合的统一表示

比例用整数百分数10、20、30保存到选择器，避免浮点四舍五入差异：B=min(n,max(1,(pct*n+50)//100))；n=0直接not_applicable，**不调用Python round**。区间均为官方索引左闭右开[a,b)，a≥0、b≤50。

选择器输入有序不可分割单元Unit(unit_id,atoms,lo,hi,cost,proxy)，atoms为递增j列表；主单模态每行一单元、cost1；联合为同一位置全部原观测、cost1—3；组为完整word/replication/block3、cost为实际原子数。单元须互不相交并覆盖该路径原观测域，顺序不得交错。

将选中单元中连续的run映射为从首单元lo到末单元hi的官方区间。两个选中单元之间只有原本空位置时可合成一段，**区间包含这些空洞，其跨度必须计入**；中间有未选观测单元时必须断段，不能把未删位置画成已删。不同run之间至少隔一个未选单元。此为同一实际删除集合的规范区间表示，删除无效的边缘空位置，避免同一mask由多个窗口重复计数。

例：模态观测位置为[1,2,5,6]，选[1,2,5]的规范区间为[1,6)，删除3行、跨度5、含2个自然空位；选[1,5]则为[1,2)与[5,6)，不能跨过仍保留的观测2合并。此规则没有把官方坐标压缩成新时间轴。

### 10.2 DP状态、转移和平局顺序

在最多50个单元上做0/1选择。状态(i,b,k,z)：已处理前i个单元、实际成本b、已开启k段、上一单元是否选择z。初态(0,0,0,0)，每个状态保存最优代理和回溯。

- 不选当前单元：到(i+1,b,k,0)。
- 选择：cost增加该单元cost；若z=0则k+1、开启[lo_i,hi_i)，若z=1则延长末段到hi_i。仅保留b≤B、k≤3。
- 代理增加该单元proxy；跨度开启时增加hi_i-lo_i，延长时增加hi_i-hi_(i-1)，因此自然空洞计入跨度。

float64分数按单元递增顺序累加，选择器用确定的精确数值比较，不在DP内部用不传递的“相差eps就相等”。eps仅用于数值可辨识和方向标签。相同目标值依次比较：段数更少、官方总跨度更短、区间端点序列((a1,b1),…)字典序更小、实际原子j序列字典序更小。最后两项解决同起点等情况。每个状态的同分保留也用相同次序。

| 路径/selection_kind | 成本要求 | 代理 | 终态排序 |
|---|---|---|---|
| key单模态 | 必须b=B | 单点D之和最大 | 目标后按上述平局规则 |
| low单模态 | 必须b=B | 单点D之和最小 | 目标后同上 |
| joint_key | 先找b≤B的最大正可达b | 联合位置D之和最大 | 先成本、再代理、再平局 |
| group_key | 先找b≤B的最大正可达b | 组D之和最大 | 先成本、再代理、再平局 |
| support_candidate | 0<b≤B | sum(max(delta_class,0))最大 | 代理后先成本更少、再段数/跨度/字典序 |
| suppress_candidate | 0<b≤B | sum(max(-delta_class,0))最大 | 同支持候选 |

方向候选用原始单点差值计算代理，若最佳代理≤eps_prob则不产生候选；允许零代理单元作为连接两处方向信息的桥，但它占成本。不能简单删除全部零代理位置后再选段。所有主单点D≤eps_D时仍完成预算实验，但记local_effect_status=indistinguishable，卡片称“预算选择片段”，不得称已发现显著证据。

联合和组路径连一个单元都放不下时budget_unreachable；不能拆同位置模态、拆词或突破预算。若b<B，保存target_budget和actual_cost，随机对照以actual_cost为准。分组约束不适用于主单点路径。

### 10.3 集合重测与展示层级

每个候选从原X0构造E，计算真实D、delta_class、delta_score_raw、TV和类别变化，代理和重测值分开字段。主证据中的每个独立区间也单独重测，另存segment_view_id；多段总分只属于**整段并集**，不能给每段贴同一个总分或相加单段分数冒充联合结果。

支持候选重测delta_class>eps_prob才direction_retained=true；抑制候选重测delta_class<-eps_prob才为true；其他为false并保存reversed/indistinguishable。不反复换候选直到得到想要的符号，也不重新挑预算。候选失败仍进入方向保持率分母及原方向统计。

同一样本、模态、实际集合相同的证据合成一条证据实体，roles可含key/support/suppress等多个角色；实验请求仍分别保留，以明确预算/方法。不同角色不重复增加映射覆盖数。

联合证据保留跨模态整体view_id和整体D。做素材覆盖统计时，将其原子按模态展开，与单模态证据按相同实际原子子集去重；这只是素材工作量统计，不能把联合D分摊成三路D。所有请求按split样本顺序、表中实验顺序、预算10/20/30、模态T/A/V、官方位置递增生成，保证view注册及断点恢复不依赖并发完成顺序。

## 11. 严格随机对照：数清可行集合，再等概率抽样

### 11.1 每个目标集合的匹配条件

目标为一个已选证据E，不依模型输出再筛随机集合。严格匹配以下完整签名：干预路径、每路实际删除数(c_T,c_A,c_V)、规范区间数k、**区间官方跨度的多重集合**sorted(b-a)、组不可拆约束。单模态路径其余两路成本为0；联合路径每个位置原有几路就同时删几路，严格匹配三路成本向量，不能只匹配总数。

从第10节相同有序单元中选完整的run，所得实际mask不得等于E，可以与E部分重叠。段的长度可换次序，不能把“相同长度多重集合”误做“第1段必须相同长度”。相同mask只对应一套规范区间，因此不需要按窗口抽样后用偏置去重的做法。方向候选随机对照不按单点方向或重测方向筛选。

默认只做严格匹配；本实现不自动生成放宽跨度的补充对照。若以后新增，应另设experiment_kind，不能混入当前主表。

### 11.2 有限计数DP与rank/unrank

采用最多3段的精确整数计数，不使用反复随机起点加拒绝的无限循环。先枚举可作为单段的单元范围[i,j]：实际集合为这些单元全部atoms，lo=unit_i.lo、hi=unit_j.hi；预计算官方跨度、三路成本向量。仅保留长度属于目标长度多重集合且成本逐分量不超目标的候选，按(lo,hi)排序。

递归状态C(next_i, remaining_lengths, remaining_costs)，其中remaining_lengths为“长度→剩余个数”的有序tuple。遍历起点i≥next_i的所有合法候选段[i,j]，消耗一个对应长度及其成本，递归进入next_i=j+2。这样两段之间至少有一个未选单元，避免本应合并的相邻run被重复计数。若段已耗尽且三个剩余成本都是0，返回1，否则返回0。记忆化缓存用普通tuple，计数用Python int，不转float或int64。

同长度只从multiset消耗一次，不给相同长度段添加人为身份。每个分支按(lo,hi)排序；完整集合的rank为此前分支完成数之和，再加后缀rank。unrank按累计完成数找到分支并递归。将目标E也放入计数域，得总数N_all和它的唯一rank q；可用对照数N=N_all-1。

目标不在自身匹配域内是实现错误，不能当“无随机对照”。每次unrank后重新核对mask、规范区间、成本与签名；这针对去重/边界错误，不新增额外校验框架。

### 11.3 无放回抽50个或枚举全部

R=min(50,N)。N=0时状态no_alternative、对照均值和G为null；1≤N<50则取全部，不能复制集合凑50。N≥50时用Floyd抽样在整数[0,N)抽R个不同rank，支持任意大Python整数：

~~~python
chosen = set()
for j in range(N - R, N):
    t = rng.randrange(j + 1)
    chosen.add(j if t in chosen else t)
ranks = sorted(chosen)
rng.shuffle(ranks)
for r in ranks:
    full_rank = r if r < q else r + 1  # 跳过目标本身
    control = unrank(full_rank)
~~~

rng为第12.4节确定种子的random.Random。N<50枚举后也shuffle得到可复现的随机顺序，供任务指标按control_index配成重复实验。记录N_all、N为**十进制字符串**（防止JSON/浏览器整数精度溢出），R为普通整数，并保留每个full_rank和control_index。

每个对照按与目标相同的原样X0重测。R<20仍报告差值、样本级均值和可行数；不作该样本尾部比较。R≥20可报告random_D_q05/q50/q95与描述性尾部比例(1+#(D_random≥D_target))/(R+1)，字段名tail_fraction，不称p_value，不赋予stronger/显著/通过标签，不做解释准确率。

### 11.4 三路并集的保留对照

三路key_20各自的完整可行域包括本身；自然空模态只有唯一空集合。全输入只保留三路证据并集时，严格对照域是三路完整可行域的笛卡尔积，再排除目标三元组。总数为N_T_all×N_A_all×N_V_all-1，使用混合进位rank/unrank和相同Floyd抽样。

不得简单拼“各模态已有50个非自身对照”，该做法排除了部分路恰好等于目标的合法组合，会改变分布。当仅一路存在内容，该定义自然退化为该路域；三路并集本身为空则不开展内容充分性实验。

## 12. 实验矩阵、统计和确定的随机数

### 12.1 必须完成的实验表

每个split各用一个run；valid全728条，special固定01—20。下表全部执行，不因某样本解释差而跳过；数据或映射确实不可用时按状态记缺口。低影响作为本实施版固定诊断，无须另行做参数选择。

| experiment_kind | 范围与预算 | 选择/输入 | 重测与对照 |
|---|---|---|---|
| original | 全部 | 原观测 | 预测、U0/J0、valid任务指标 |
| modality/shapley | 全部 | 8保留组合 | 三路D、权重、双目标phi、全空先验 |
| local/joint_local | 全部观测单元 | 单点及联合位置 | 完整扫描，不只Top-k |
| key_delete | T/A/V各10/20/30% | 精确成本DP | 目标、各段、最多50严格随机删除 |
| low_delete | T/A/V各10/20/30% | 同可行域最小代理 | 目标重测；不额外生成50组低影响随机对照 |
| direction_delete | T/A/V各20% | support/suppress候选 | 无正目标则N/A；有候选均重测和匹配随机 |
| joint_delete | 全输入20% | joint_key、最大可达成本 | 目标、各段、三路成本匹配随机 |
| group_scan/group_delete | 各模态20% | word/replication，缺失时block3补充 | 组扫描、最大可达成本、同粒度严格随机 |
| center3_scan | 无语义分组的模态 | 中心3位置 | 仅索引粒度诊断 |
| conditional_keep | 各模态20%主key | 仅该模态压缩，其他模态全留 | key/low及key对应随机保留 |
| global_keep_union | 三路20%主key并集 | 其余所有原观测删除 | 第11.4节联合随机保留 |
| global_keep_joint | 联合20%主证据 | 仅保留joint_key | 复用joint_delete的随机集合，但算保留预测 |
| mapping/report | special全部20条 | 20%主/联合/方向证据 | 原文、词时、AV来源、全量卡片 |

本版方向、joint、group及keep只做20%；主单模态删除做三个预算。未来新增分析使用新run名，并明确与当前结果分开，不以未来扩展为当前完成条件。可选的缺失后重归因、注意力、IG和新生成器不实现。

### 12.2 删除和保留各算什么

主删除每样本：G_D=D_key-mean(D_random)，G_class_abs=abs(delta_class_key)-mean(abs(delta_class_random))，G_score_abs=abs(delta_score_raw_key)/6-mean(abs(delta_score_raw_random)/6)。低影响与key共用同一预算可行域，但最终段数/跨度可能不同；报告几何差异，不能声称二者实际几何完全相同。

方向比较：support用delta_class，suppress用-delta_class作有向目标；G_direction=target_value-mean(control_value)。对所有生成过的候选报告，不以direction_retained筛掉失败。方向保持率分母为该方向全部已重测候选数，另列无候选数。

条件保留实际删除集合为O_m减去E_m；全输入保留实际删除集合为O减去E（均为集合差，不是乘积）。所有保留计算：

~~~python
tv_keep = 0.5 * np.abs(p0 - p_keep).sum()
S = 0.5 * tv_keep + 0.5 * abs(s0 - s_keep) / 6
keep_class_agreement = int(argmax(p_keep) == c_star)
keep_delta_class = p0[c_star] - p_keep[c_star]
G_keep = mean(S_random) - S_key  # 正值表示关键保留更接近原输出
~~~

此处TV=0.5*sum(abs(p0-p_keep))，不是D中的固定类概率差。另列全空先验的S、类别保持和原类概率；不能以先验已保持类别证明空证据充分。conditional_keep、global_keep_union、global_keep_joint分表，不混平均。区间并集的实际总保留数及每路保留数须一起输出。

### 12.3 粒度比较和任务指标

group实际成本若比20%目标小，必须另在**同样本、同模态、该确切成本**重新选择一次单点key，作为group对照（selection_kind=point_cost_matched）；不能拿未减预算的单点key比较。保存两者删除集合的Jaccard（空并集记N/A），两者D及各自严格随机增益。单点与组的几何约束/不可拆约束仍不同，如实说明；不把D增大一律解释为分组方法更好。

valid原样及确定性视图复用q2.metrics.compute_metrics(y_true_class, logits, y_true_score, predicted_score)；额外输出固定labels=[0,1,2]、zero_division=0的每类precision/recall/F1/support和混淆矩阵。标签不进入任何归因、选择或随机抽样。

随机任务指标不能先平均logits/probabilities再算F1。对某path/budget先建立共同样本支持集A={目标成功且R_i≥1}，R_common=min(50,min_i R_i)。按随机control_index r=0…R_common-1，在A上各算一遍任务指标，再对这些标量求均值/标准差；目标、原样和low均在同一A上重算供配对比较。报告A大小、独立video数、R_common及R_i分布；空A或只有1次随机重复时分别输出N/A或std=null。全728原样指标另列，不能偷换分母。R_i很少会限制R_common，这是精确几何约束的真实限制，不补抽重复集合。

附件4无真值，不调用任务指标函数；只列预测、扰动效应、保留情况和映射结果。分类回归头分歧按Negative且score>eps_score、Positive且score<-eps_score标记；Neutral伴随非零score单独描述其连续值，不发明一致性阈值把它判错。

### 12.4 随机数不随循环次序或断点续算变化

统一使用numpy.random.SeedSequence。整数命名表：split(valid=1,special=2)，用途(controls=101,bootstrap=201)，path(T=0,A=1,V=2,joint=3,union=4)，粒度(point=0,word=1,replication=2,block3=3)，角色(key=0,support=1,suppress=2,group=3)。所有编号在代码中为固定常量，不使用Python hash或按运行时间播种。

随机对照熵列表为[20260925,101,split_id,sample_index,pct,path_id,granularity_id,role_id,len(E),*sorted(E)]；E为原子j整数列表，这是实际主键内容，不另造hash文件。SeedSequence生成4个uint32，按低位到高位拼成Python int，传给random.Random。目标与匹配签名完全相同时允许复用对照；必须使用首次已记录的同一seed，不因新角色重复抽有利结果。

bootstrap熵列表为[20260925,201,split_id,analysis_id,path_id,pct,granularity_id,role_id,stratum_id]。analysis_id固定：G_D=1、G_class_abs=2、G_score_abs=3、G_direction=4、G_keep=5、direction_retained_rate=6。stratum_id固定：总体0，真实类10/11/12，原预测正确20/错误21，概率间隔30/31/32，单路观测数量40/41/42/43。角色另补conditional_keep=4、global_keep_union=5、global_keep_joint=6、point_cost_matched=7。枚举在run配置明文记录；由SeedSequence构造numpy.random.default_rng。按原始样本索引排序后抽样，不取系统随机状态。恢复运行不得改变已完成样本的control列表。

### 12.5 配对视频簇bootstrap与分母

主汇总单位是样本，不是窗口或随机对照。分别对每个path/预算计算G的样本平均、中位数、四分位数、有效数及G>0比例。95%区间使用1000次video_id簇bootstrap：在有效支持集的G个唯一video_id中有放回抽G个；每抽中一次带入该视频全部有效样本，重复抽中就重复计权，最终仍取抽中样本的平均。目标与其随机均值作为一对整体抽，不分别重采样。

至少两个来源视频才能给该统计的区间；更少时CI=null并说明样本不足。区间为bootstrap均值分布的2.5/97.5百分位。方向保持率可同样按簇重采样；special仅全20条描述统计，不以20例做泛化结论。任务指标的随机重复标准差与bootstrap区间是不同量，分列，不把50次对照作为50倍样本量。

固定报告分层：真实类0/1/2；原预测正确/错误；原top1-top2概率间隔[0,.2)、[.2,.5)、[.5,1]；单路观测数0、1—10、11—30、31—50。观测数分组按对应模态分别统计，联合图给三路数量向量。分层只用于描述，不参与参数选择。主表列全样本数、输入失败数、自然空模态数、预算不可达数、无随机替代数、有效对照数；不把N/A转0参加平均。

对分类/强度差以及各预算均保留结果，不挑CI不跨0的分析作唯一结论。valid已参与Q2选模；Q2历史test还影响过探索，论文应承认评价身份，不能称独立盲测。

## 13. 原素材回溯：独立环境、三种关系分别输出

### 13.1 q3_align与Q1媒体模块的薄改造

只对附件4的20个视频运行。主环境先导出data/special_manifest.jsonl，每行有sample_id、video_path、raw_text和词表/字符映射路径；对齐环境读取这些JSON文件，**不加载Q2、pickle或情感模型**。词表使用完整原文，包括未进入50位置的尾部词；全转写帮助CTC对齐，但不能把尾部词列为预测器输入证据。

在vendor/media.py上新增Q3薄入口probe_q3_media(video_path, work_dir, timeout_s=120)，复用原probe_and_decode的选流、音视频帧记录、build_media_clock、连续性检查、声道混合/重采样和时钟输出。具体改动必须限定为：

1. sample.video_relpath改为传入的确切Path，sample_id显式传入；不要为调用旧runner构造完整Q1任务。
2. 复用的内部cfg为runtime.external_command_timeout_s=120，pool.bins=50，media.sample_rate=16000，media.target_video_hz=25，media.audio_duration_tolerance_s=0.02。bins和25Hz仅为兼容旧辅助函数，不代表官方50位置的时间坐标。
3. 关闭原函数的25Hz PNG批量导出、重复帧选择和所有Q1特征提取；保留原始video_source_frames.jsonl及audio_frames.jsonl。改造后不访问selection.selected_frames等被关闭分支的局部变量，media.json中的selected_video_frames初始为0。
4. 保留storage.py的相对导入，保留PyAV、Pillow、soundfile依赖。原视频不修改；只在Q3/runs/special_main/media/{sample_id}/写ffprobe.json、media.json、帧表和audio_16k.wav。
5. 选第一个非封面视频流和第一个音频流，保存流索引，其余流数单列；没有音轨时词时对齐not_available，没有视频轨时图片提取not_available，不能影响已有特征预测。

帧时间优先pts×time_base，缺PTS按原模块明示的best_effort_timestamp_time降级。共同零点t0是所选有效音/视频流首个有效时间的最小值；common_seconds=container_seconds-t0。音频WAV零点对应audio_offset，CTC回到共同时间必须加audio_offset。记录container、common和wav三类字段，不把播放器从0开始等同于原始stream PTS从0开始。

音频只在原模块判连续、来源与重采样时长差≤0.02秒时送入CTC；有间断时保留帧表和失败原因，不把片段拼紧后继续按连续音轨解释，也不人工填充静音伪装来源。重采样为16k、单声道、float32；不得使用语音增强、时间拉伸或ASR新转写替换给定原文。视频旋转、source/display尺寸和帧失败记录沿用原模块。

这里float32指soundfile.read(..., dtype="float32", always_2d=False)交给模型的波形数组；WAV编码沿用Q1的FFmpeg提取结果，不因该措辞再做一次有损转换。Q1原函数在时长差过大时只是写issues，Q3薄入口必须读取该issues并停止该条CTC，不能把“已调用原函数”等同于已阻止错误时轴继续传播。

### 13.2 CTC加载、输入和词身份

一个进程只加载一次第5节指定的Conformer模型和SentencePiece模型，逐条运行。prepare_ctc_config增加runtime_output参数，runtime YAML永远写Q3；若原stats_file为旧机器绝对路径，应在下载快照内按明确同名相对路径核对唯一文件后重写，不能仍指向不存在的训练机目录。

调用build_words(raw_text)和prepare_ctc_rows(words, sentencepiece_model, model.token_list)。关键约束：

- SentencePiece的**piece字符串**映射到ESPnet的token_list，不能将SentencePiece整型ID直接当CTC类别ID；unk_id使用token_list中实际的未知标记索引，不硬编码BERT的100。
- 使用返回的ctc_row_to_word_id连接输出；纯标点/无发音内容可能没有CTC行，不能将输出与全词表直接zip。保留normalizations、contains_unk_word_ids、unsupported_word_ids、skipped_word_ids。
- prepare_ctc_rows可能为含UNK的词仍生成行。按其原设计对完整可生成序列进行对齐，相关词结果标无效；不得删除失败词后重编号并把下一词时间错配过来。
- CTC token_arrays转np.int64一维数组列表；ctc_texts、row_to_word和arrays长度严格相等。零可用行则no_spoken_content，不调用空对齐。

ctc_forward输入整段16k WAV，得到lpz[T_ctc,V]及index_duration=(len(wav)/16000)/T_ctc。它是该CTC模型的帧时间尺度，**不是50个官方特征行的时间尺度**。使用blank=0、min_window_size=8000、max_window_size=100000、score_min_mean_over_L=30调用align_token_arrays_detailed。贪心转写只允许作为诊断附录，不替换raw_text。

CUDA OOM时先释放该条中间量再以**同模型FP32 CPU**重算一次，记录实际设备及原因；不得把音频切成无来源的短段、截短原文或静默换模型。主Q2进程与对齐进程顺序执行，避免抢占同一张卡。其他异常逐条记录，不将技术失败改写为低置信词。

### 13.3 算法接受与人工审核分开

每个有CTC行的词保存原始start_wav/end_wav/log_score及row_id。算法accepted要求：时间和score有限、start<end、无UNK/unsupported、log_score≥-5、位于WAV时长内且次序合理。边界仅在越界≤一个index_duration时允许截回合法边界，并记录boundary_adjusted及原值；更大越界拒绝。相邻有发音行若出现非单调起点或超过一帧的重叠，两词均标时间冲突，不能排序结果掩盖错位。

加audio_offset得到start_common/end_common。若原文规范化涉及数字展开等假设，保留normalization_assumed；词即使算法accepted也仍review_pending。纯标点可保存attached_to_word_id供上下文播放，但自己的词时字段为null、time_role=punctuation_anchor，不能冒充独立发音。截断词的存储piece可以借用整词发音区间作词级定位，time_resolution=word，不伪造单WordPiece发音时间。

data/review/word_times.csv为**人工填写文件**，首次生成后后续命令不覆盖。字段：sample_id、word_id、raw_word、auto_start_common、auto_end_common、auto_log_score、algorithm_status、review_status(pending/accepted/corrected/rejected)、reviewer、reviewed_at、review_start_common、review_end_common、notes。审核者修正必须同时提供两个边界、署名和实际时间；原算法结果永久保留作对照。不能由Agent批量填写“人工通过”。

未人工审核时，自动播放区间标“CTC候选词时”；accepted/corrected且有合法审核记录后才称“已审核词时”。仅纠正被展示证据的词不代表全转写审核完毕。统计同时报告算法接受数、审核完成数、选中证据覆盖，不把它们合为一个成功率。

### 13.4 官方A/V来源文件与无法定位的处理

可选人工/资料整理文件data/review/official_av_mapping.csv，字段固定为sample_id、modality(A/V)、official_t、source_kind、source_ref、source_locator、clock(container/common)、start_s、end_s、mapping_status(verified/bounded/unresolved)、reviewer、notes。一个官方行可有多个来源区间，另加source_interval_index从0递增；没有来源用一行unresolved且时间null。

verified需要官方行到生成窗口/帧的可复核生成关系，以及正确媒体坐标；bounded需要独立依据给出有限候选时间范围，并写明为何无法精确；只有行号、猜测词对齐、均匀时间或肉眼回看不能创建官方生成来源。人看过某段情绪明显，也不能把未知行来源升级verified。默认无法找到依据时50行中有观测的行均unresolved；不影响模态D排序，不把主要模态改成方便展示的文本。

对已选A/V证据展开为原行的来源区间并取并集，仅全部选中行都满足某种状态时才给整条相应状态；部分成功记录partial和已映射/应映射原子数，不能用一个已知行替整段背书。特征维度没有说明时不写“第x维即某AU/微笑强度”。仅数值重复的组也不提供真实时间证据。

### 13.5 音频片段、代表帧与时间呈现

文本的已接受CTC区间只标“文本证据的语音定位”，不称音频分支证据。A/V有verified或bounded来源时才生成相应素材入口，并在标题保留状态。为方便听辨，播放可加前后0.20秒上下文、裁到媒体合法范围；core_start/core_end保留未扩展区间，播放上下文不计证据时间、不进入归因成本。

每个视觉来源并集区间[a,b)，在video_source_frames中筛选eligible且展示区间与[a,b)有正长度相交的帧。选展示区间中点最接近(a+b)/2者；并列选较早PTS，再较小source_frame_index。按该源帧精确解码并应用已记录旋转，保存PNG及source_frame_index、PTS、time_base、展示起止、与目标区间关系。无有效帧即frame_unavailable，不取全视频最夸张画面替代。贡献属于官方行集合，图片标题必须写“区间代表帧”，不能称单帧获得整段D。

播放时间统一使用common_seconds。短音频片段由audio_16k.wav按(common_time-audio_offset)裁切；没有WAV或区间不在音轨范围就不导出。若展示视频clip，由选定媒体流按共同时间转换回其实际时间坐标后裁切，并记录导出片段的时间原点；不得直接把common_seconds当container PTS传入并假设正确。首版必须交音频片段和代表帧；视频clip为可选便利输出，不是必需依赖。

### 13.6 映射覆盖的具体分母

附件4分别汇总主20%证据与方向证据；证据实体按(sample_id,modality,actual_atoms)去重。报告：文本token字符定位数量/应定位token数，文本证据完整字符映射条数/应定位条数；申请词时的有发音词算法接受率与人工审核率；A/V被选原子中verified/bounded/unresolved比例；完整/部分素材映射证据数；20条中主要参考模态至少有一条verified素材证据的样本数/20。

文本的verified字符定位可构成文本素材证据，词时审核状态另列；A/V须有生成依据才能计verified，纯背景截图不计。并列主要模态另给“至少一路有证据”和“所有并列主要模态都有证据”两个比例。三路全空仍在20总分母，另列其数。未映射时间不记为0秒；已有时间只按去重并集计长度，并注明覆盖比例。没有独立边界标注就不报告毫秒定位误差或IoU准确度。

## 14. 文件格式、字段和接口

### 14.1 核心Python接口

这些是要求新实现的Q3接口，Q2函数以实际源码为准。除predictor外不得直接import情感模型；除statistics外不得消费label。配置对象在主入口解析一次，传入函数，不在库函数读取CLI全局变量。

~~~text
load_samples(config, split) -> list[Sample]
Predictor(bundle_dir, device).predict(raw_batch) -> Prediction
apply_delete(raw_batch, delete_mask_bool, original_U) -> dict[str, Tensor]
evaluate_sets(sample, requested_sets, predictor, cache) -> list[ViewResult]
compute_attribution(sample, evaluator) -> AttributionResult
build_units(sample, path, granularity, attribution, mapping) -> list[Unit]
select_units(units, budget, max_intervals, objective, cost_mode) -> Selection
enumerate_control_space(units, target_selection) -> ControlSpace
ControlSpace.count() -> int
ControlSpace.rank(selection) -> int
ControlSpace.unrank(rank) -> Selection
sample_controls(control_space, target_rank, n, rng) -> list[Selection]
evaluate_experiments(sample, attribution, selections, evaluator) -> ExperimentResult
summarize_run(run_dir, labels, config) -> Summary
build_case_report(sample_result, mappings, config) -> CaseReport
~~~

Sample包含split、sample_index、sample_id、source_id、video_id、raw_text、raw五字段及独立labels；RawBatch严格用第4节五字段。为了实现“归因不消费标签”，向attribution/evaluator传Sample的无标签视图，不把label复制到model raw。分组和媒体记录以sample_id关联，绝不靠输出文件行号拼表。

### 14.2 每个run的布局与完成状态

~~~text
runs/{valid_main|special_main}/
  run.json                   # 日期、Git版本、实际路径、配置、环境、状态
  manifest.jsonl             # 每个预期样本一行，含失败样本
  numeric_tolerance.json     # 全运行共用，special引用同一份数值
  samples/{sample_index:04d}/
    sample.json
    original.json
    views.jsonl
    requests.jsonl
    modality.json
    shapley.json
    local.npz
    tokens.jsonl
    groups.jsonl
    selections.jsonl
    controls.jsonl
    experiments.jsonl
    evidence.jsonl
    status.json
  summaries/
    prediction_metrics.json
    perturbation_metrics.csv
    faithfulness.csv
    granularity.csv
    strata.csv
    failures.csv
  media/{01..20}/            # 仅special，对齐环境写入
  alignment/{01..20}/        # words.jsonl、diagnostics.json
~~~

路径用整数sample_index，valid 0000—0727；special 0000—0019对应01—20。sample_id始终另存字符串，不能把0000当专项01的外部ID。特殊数据失败时manifest、sample.json和status.json仍有一行/文件。

状态枚举按阶段记录pending/running/complete/partial/failed/not_applicable和reason；完整样本预测成功不代表素材映射complete。文件写入用同目录临时文件完成后rename，避免中断留下半个JSON；无需引入数据库或hash系统。单样本视图注册表按delete_j tuple去重，v000000固定为原样，后续视图ID按首次确定请求顺序递增并持久化；恢复时加载注册表，不重新编号已引用视图。

--resume仅在相同配置、数据路径/shape/ID和模型版本记录下复用已完成阶段；模型、数据或选择规则改变使用新的--run-name，不覆盖旧结果。外部输入在同次run内按只读处理；人工review文件变更只重做mapping/report阶段，不重跑预测。记录普通文件大小/mtime辅助提示即可，不增加SHA流程。

### 14.3 关键记录字段

所有记录默认带schema_version=1、split、sample_index、sample_id；schema_version只是文件版本号，不设额外冻结流程。j=3*t+m；t零基；模态用T/A/V字符串。JSON数组使用数值类型，所有NA为null并有status/reason。下表为必需字段，允许增添明确诊断字段但不得改含义。

| 文件/记录 | 必需字段及类型 |
|---|---|
| manifest.jsonl | source_id:string或null，video_id:string，input_path:string，video_path:string或null，raw_text:string或null，input_status:string，reason:string或null |
| original.json | logits:float[3]，probs:float[3]，score:float，c_star:int，U0:bool[50][3]，J0:bool[50]，n_observed:int[3]，prediction_status |
| views.jsonl | view_id:string，delete_j:int[]，delete_counts:int[3]，logits/probs/score，predicted_class:int，delta_class、delta_score_raw、delta_score、D、TV:float，class_changed:bool，status/reason |
| requests.jsonl | request_id:string，view_id:string，experiment_kind:string，path:string，budget_pct:int或null，granularity:string，role:string，source_selection_id:string或null；不同逻辑角色可指同一view |
| modality.json | 各路n_observed、view_id、D、delta_class、delta_score_raw、weight；D_sum、primary_modalities:string[]、class_top_modalities、score_top_modalities、head_top_disagreement、status |
| shapley.json | subsets共8项含bitmask/view_id/v_class/v_score；phi_class[3]、phi_score_scaled[3]、phi_score_raw[3]、residual_class/score、empty_view_id |
| local.npz | observed:bool[50,3]；D/delta_class/delta_score_raw/TV:float64[50,3]；joint_observed:bool[50]和对应joint_*数组；无观测数值可存NaN但必须有mask，转JSON/CSV时改null/空白 |
| tokens.jsonl | official_t、token_id、token_string、stored_attention、token_type、is_observed、is_unknown、char_start/end、word_id、token_mapping_status、tokenization_mode |
| groups.jsonl | group_id、modality、granularity、word_id或null、positions:int[]、atoms:int[]、cost:int、lo/hi:int、truncated_word、source_ref、view_id、D、status |
| selections.jsonl | selection_id、experiment_kind、path、granularity、budget_pct、target_budget、actual_cost、counts_by_modality[3]、intervals:[[a,b],...]、delete_j、proxy_value、proxy_kind、joint_view_id、segment_view_ids、local_effect_status、direction_retained、status/reason |
| controls.jsonl | selection_id、control_index:int、full_rank:string、N_all/N_alternatives:string、seed_entropy:int[]、delete_j、intervals、counts_by_modality、delete_view_id、keep_view_ids、status |
| experiments.jsonl | experiment_id、selection_id、view_id、operation(delete/conditional_keep/global_keep)、control_role(target/random/low/cost_matched)、control_index或null、D/TV/S及有符号变化、cost/keep_counts、status |
| evidence.jsonl | evidence_id、roles:string[]、selection_ids:string[]、modality/path、atoms、intervals、union_view_id、segments及view_id、tokens/words引用、character_status、word_time_status、av_mapping_status、core/playback_intervals、media_refs、映射覆盖数、status/reason |
| faithfulness.csv | split/path/budget_pct/granularity/role、n_total/n_eligible/n_videos、R_min/median/max、mean_G/median_G/q25_G/q75_G/positive_fraction、ci_low/high、n_no_alternative、n_unreachable |

selection_id使用确定标签，如key.T.p20.point；source_id与文件ID不可混用。evidence_id按同一实际原子集合排序后编号e0001等，并保留selection_ids。view文件按需只存预测和集合，不存每个视图的AV特征、BERT hidden、完整模型或重复视频。

### 14.4 专项主表与可读卡片

outputs/Q3_attachment4_predictions.csv严格按01—20一行一个样本，列至少为：sample_id、source_id、status、predicted_class_id、predicted_class_name、sentiment_score、p_negative/p_neutral/p_positive、n_T/n_A/n_V、D_T/D_A/D_V/D_sum、w_T/w_A/w_V、primary_modalities、head_top_disagreement、evidence_T/evidence_A/evidence_V、mapping_status、case_json、case_html。多值格使用合法JSON字符串，CSV通过标准库或pandas正确引号转义；不以逗号拼列表破坏列数。失败预测行保留ID和reason、数值为空。

outputs/cases/01.json和01.html至20.json/20.html内容一致。静态HTML将所需JSON以内联脚本或安全HTML直接嵌入，禁止使用file://页面上fetch旁边JSON导致浏览器跨源失败；小图和音频用相对路径资源，复制到outputs/assets的仅是必要片段。图表用本地PNG/SVG，无CDN。源视频不放进提交包，卡片写明原视频路径及是否有可播放副本；缺素材时显示明确状态，不留看似可用的断链。

每张卡固定顺序：原预测及三类概率 → 原模态观测和整路D/权重 → 双头Shapley与空先验 → 三路3×50热力图 → 20%主选段及整段重测 → 方向候选成功/失败 → 严格随机均值/数量/增益 → 条件/全输入保留 → 原文和可追溯素材 → 缺口。热图mask格灰色斜线或N/A，观测零为真实色阶0；方向图使用以0为中心对称色阶；不把单点和整段D混同一图例。

自然语言采用固定模板，明确“移除这些已观测输入后，原类别概率由…变为…，强度由…变为…”。不调用LLM API润色、编情绪原因或生成素材解释。原文上下文最多各一原词，仅作为灰色展示上下文；高亮范围严格对应实际被删piece。

## 15. Agent的开发顺序与全部命令

### 15.1 先编写入口，不能对空目录直接执行后面的命令

按以下顺序完成源码，每阶段使用实际输入或相应的必要单元测试确认后再推进：

1. 目录、pyproject、两套requirements、配置、轻量CLI、结果写出；所有新文件位于Q3。
2. data/predictor/interventions以及原样/干预数值测试；先证明与现有Q2包完全同一路径。
3. text_mapping/grouping/attribution；完成原文匹配、八子集、三路全位置、联合位置和组扫描。
4. selection/controls；用第16节小域穷举验证DP与严格随机计数，再接真实样本。
5. experiments/statistics；接入三个预算、方向、组、随机、保留和视频簇统计。
6. q3_align四个vendor文件及薄入口；安装独立依赖、下载CTC模型、完成媒体和词时记录。
7. mapping/report/plots/export；生成完整20条及valid报告，汇总素材未解问题。

pyproject使用setuptools构建，requires-python为>=3.11,<3.12，package discovery指向src并包含q3*与q3_align*；依赖清单以两个requirements文件为安装入口，不通过一个project.dependencies同时安装两套冲突环境。主环境无需editable安装，使用PYTHONPATH。pytest注册integration与alignment标记；纯算法测试不得在模块导入时加载真实模型或ESPnet。

CLI采用argparse。所有主命令统一为：python -m q3 --config PATH SUBCOMMAND OPTIONS。q3_align同样约定，配置路径可为绝对路径；配置内相对路径均相对project.root解析，不相对当前工作目录。每个命令支持--help，报错指出具体文件/样本/字段。

| 子命令 | 必需参数/默认 | 实际动作与产物 |
|---|---|---|
| prepare | --split valid或special | 解析输入、生成manifest/原观测字段和token映射；不做预测 |
| check-predictor | 两个manifest已存在 | 按valid首8和专项20核对模型、数值和原样接口，生成共享容差文件 |
| attribute | --split；--resume可选 | 原样、八子集、点/联合/组扫描，写每样本视图与归因 |
| select | --split；--resume可选 | 按全部实验表生成key/low/方向/joint/group选择及同成本point；不启动新训练 |
| controls | --split；--resume可选 | 精确计数、种子和最多50个严格对照，写实际集合；此阶段不按模型输出挑选 |
| evaluate | --split；--resume可选 | 批量重测选段、各段、随机、删除/保留，写experiments和evidence实体 |
| summarize | --split | 全量状态、忠实性、粒度和统计；valid另算任务指标 |
| map | --split special | 合并字符、词时、人工文件和官方AV来源，写素材与覆盖，不修改预测 |
| report | --split | valid图表/报告；special全量20个JSON/HTML及主CSV |
| export | 无split | 组装delivery材料，引用现有Q2权重，生成体积报告 |
| validate | --stage inputs或final | 只读检查数据/输出完整性与必要一致性，写reports/validation_{stage}.json |

除check-predictor、export、validate外，支持--run-name NAME覆盖配置run目录的末级名称，用于明确的新实验；默认valid_main/special_main。正式主实验不提供看分数后任意挑样本的--ids快捷入口。开发测试直接使用测试夹具，不将开发子集报告替代全量运行。

批处理命令按样本捕获并记录输入/运行失败，继续其余样本，末尾清楚打印成功/失败/不适用数；不能用裸except或全局“忽略异常”。配置不存在、模型身份不符、状态前置文件缺失等全局错误立即非零退出。最终validate对技术性缺失返回非零；AV来源unresolved或人工pending属于已披露研究缺口，输出material_status=partial，不伪装完成，也不妨碍已完成计算的报告导出。

### 15.2 主环境初始化和测试命令

完成第1—6节文件编写及安装后，在同一个shell执行。下列默认使用Q2已有环境；若第3节确实新建了Q3/.venv，将Q3_PY改为该唯一确定路径，记录到run_main.sh。禁止尝试一串解释器直到某个恰好import成功。

~~~bash
set -euo pipefail
export PROJECT_ROOT=/root/gpufree-data/shuomo_E
export Q2_ROOT="$PROJECT_ROOT/Q2"
export Q3_ROOT="$PROJECT_ROOT/Q3"
export DATA_ROOT="$PROJECT_ROOT/data"
export Q3_PY="$Q2_ROOT/.venv/bin/python"
export ALIGN_PY="$Q3_ROOT/.venv-align/bin/python"
export PYTHONPATH="$Q3_ROOT/src"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PIP_CACHE_DIR="$Q3_ROOT/.cache/pip"
export TORCH_HOME="$Q3_ROOT/models/.torch_cache"
export MPLBACKEND=Agg
export TOKENIZERS_PARALLELISM=false
export HF_HOME="$Q3_ROOT/models/.hf_cache"
export XDG_CACHE_HOME="$Q3_ROOT/.cache"
export TMPDIR="$Q3_ROOT/.tmp"
mkdir -p "$HF_HOME" "$XDG_CACHE_HOME" "$TMPDIR" "$Q3_ROOT/reports"
mkdir -p "$PIP_CACHE_DIR" "$TORCH_HOME"
cd "$Q3_ROOT"
"$Q3_PY" scripts/collect_environment.py --output reports/environment_main.txt
"$Q3_PY" -m pytest tests -m 'not integration and not alignment' -q
"$Q3_PY" -m q3 --config configs/q3.yaml prepare --split valid
"$Q3_PY" -m q3 --config configs/q3.yaml prepare --split special
"$Q3_PY" -m q3 --config configs/q3.yaml validate --stage inputs
"$Q3_PY" -m q3 --config configs/q3.yaml check-predictor
"$Q3_PY" -m pytest tests -m integration -q
~~~

collect_environment.py只调用当前sys.executable、pip freeze/pip check及可用的nvidia-smi/ffmpeg --version，记录未安装项；不自行安装、升级或遍历凭据。主check-predictor将reports/numeric_tolerance.json复制到两个run并记录共同来源。integration测试读取configs/q3.yaml指定包和输入，不写入Q2或数据目录。

### 15.3 全量归因与实验命令

~~~bash
for SPLIT in valid special; do
  "$Q3_PY" -m q3 --config configs/q3.yaml attribute --split "$SPLIT" --resume
  "$Q3_PY" -m q3 --config configs/q3.yaml select --split "$SPLIT" --resume
  "$Q3_PY" -m q3 --config configs/q3.yaml controls --split "$SPLIT" --resume
  "$Q3_PY" -m q3 --config configs/q3.yaml evaluate --split "$SPLIT" --resume
  "$Q3_PY" -m q3 --config configs/q3.yaml summarize --split "$SPLIT"
done
~~~

scripts/run_main.sh封装第15.2节的数据/模型检查及以上正式顺序，使用set -euo pipefail，给每阶段写开始/结束日志；已完成且同配置时--resume跳过对应计算。--resume不是跳过错误，failed阶段修复后仍须重算。每个样本一行进度，含完成数、unique_views、耗时与失败数，不打印海量Tensor。

记录前8个实际完成样本各阶段耗时与实际view数，给出剩余运行的粗略估计；这8条保留为全量结果的一部分，不创建另一份挑选集。ETA不是承诺，不从Q2普通批推理速度推算CTC或随机计数耗时。每次仅载一条原样和其视图批，写完释放中间量，不把728条所有受损BERT hidden留在内存。

### 15.4 独立对齐命令

先按第5节下载Q1参考代码并完成限定vendor适配，再建立对齐环境。对齐时停止主情感推理进程，释放GPU，不要求重启服务器。

~~~bash
"$ALIGN_PY" scripts/collect_environment.py --output reports/environment_align.txt
"$ALIGN_PY" scripts/fetch_alignment_assets.py --config configs/q3.yaml
"$ALIGN_PY" -m q3_align --config configs/q3.yaml media --resume
"$ALIGN_PY" -m q3_align --config configs/q3.yaml align --resume
"$ALIGN_PY" -m q3_align --config configs/q3.yaml make-review
~~~

fetch_alignment_assets.py只依赖argparse/pathlib、PyYAML、huggingface_hub；读取alignment.model_dir和第5节常量，不import q3。media执行第13.1节所有20条媒体解析；align先实际加载指定CTC并处理顺序第一条可用音轨，成功后同进程继续其他样本，此次结果直接保留，不另外重复整条试跑。make-review合并新增word_id，保留已填写人工审核列；另生成official_av_mapping.csv的unresolved待补条目，不自动断言来源。

scripts/run_align.sh封装这四步，使用同一Q3_ROOT、PYTHONPATH和缓存环境变量。CTC下载或媒体/人工工作未完成，主推理与valid统计仍应交付。Agent应明确列出需要人听看的词和缺少生成来源的AV行；不能等待人审时隐去其余已完成产物。

### 15.5 报告、导出和最终验证

~~~bash
"$Q3_PY" -m q3 --config configs/q3.yaml map --split special
"$Q3_PY" -m q3 --config configs/q3.yaml report --split valid
"$Q3_PY" -m q3 --config configs/q3.yaml report --split special
"$Q3_PY" -m q3 --config configs/q3.yaml export
"$Q3_PY" -m q3 --config configs/q3.yaml validate --stage final
~~~

后续有人填写review CSV，仅重复map、special report、export和final validate；若修改的是复制组而非词时/素材审核，分组输入已影响选择，必须用新run重做相应归因/实验，不能仅改展示。正式输出保留原始算法时间和审核时间两套值。

## 16. 必要测试与可检查的验收

### 16.1 纯算法测试：无需外部权重

这些测试验证真实失效场景，不写重复赋值或只检查函数存在的测试；不新增大型检查框架。

| 测试文件 | 必须覆盖的情形与正确性依据 |
|---|---|
| test_interventions.py | 文本只变ID为103，stored_attention/token_type/自然零行不变；A/V先原始置零；每个候选从原样clone，不互相污染；keep严格等于对应原观测补集删除 |
| test_attribution.py | 手工八子集数值验证两目标Shapley和效率式；空参与者贡献为0；固定c_star不随干预argmax变；D与S中TV定义不同 |
| test_selection.py | 对≤8单元穷举全部2^n选择，与DP比较最优值、实际成本、最多3段和全部平局次序；覆盖自然洞、成本1/2/3、组预算不可达、方向含零桥与最小成本规则 |
| test_controls.py | 小域穷举构建严格匹配集合，与count/rank/unrank逐一对照；rank唯一且往返一致；长度多重集合含重复值；联合总成本相同但模态向量不同必须拒绝；目标排除、非自身全枚举、N=0与N<50 |
| test_text_mapping.py | standard/prefix截断差异、重复词、UNK、纯标点、空白、非BMP字符、已存储部分词；匹配失败不猜offset；组不重叠且覆盖所有观测token |
| test_statistics.py | 两个视频含不同片段数，重采样按视频整簇带入并保持目标/随机配对；F1先逐随机重复计算，不能用平均概率；N/A不按0聚合；标签不影响归因选择 |
| test_results.py | 视图相同delete_j去重、不同样本不串缓存；partial样本仍在20行；JSON无NaN/Infinity；中断恢复的控制集合与未中断一致 |
| test_media_mapping.py | 合成音视频不同起点的共同时钟、audio_offset加法、连续性失败、CTC少于一帧越界修正/更大越界拒绝、词ID跳过标点不移位；选代表帧以PTS展示区间而非帧号/25Hz |

随机抽样不以“小量频次恰好均匀”作易波动的通过条件；有限域双射和Floyd算法是无放回均匀性的可验证依据。Floyd测试用固定seed验证数量/唯一性/范围，同时直接验证全部rank都能对应合法集合。纯media测试使用供应的帧记录，不为了测试下载真实视频。

### 16.2 真实模型集成测试

使用valid首8和专项20中输入成功的样本，至少覆盖一条有三模态及13号视觉全空；若实际13号不是全空，先核对附件版本，不修改数据使测试通过。

1. Q3原样与直接调用同一bundle.predict_raw相比logits/probs/score在第4.4节容差内；单条/批量/重复一致，模型处于eval。单次推理与BERT编码均不参与训练。
2. T删除前后检查MASK不进入U和BERT attention，CLS/SEP按Q2规则保留；比较Q3薄mask与Q2 apply_span输出，避免误改attention或先归一化置零。
3. A/V自然全空没有候选、weight为0（D_sum有效时）、phi为0；全空输入返回Q2已有prior，不报NaN。
4. 同一A或V模态将完整原始行顺序置换，再重新推断掩码，最终late_attn_tune输出应在数值容差内不变；该验证对应模型无时序编码的已核查结构，不能据此推断原素材无时间结构。
5. 八子集、至少一条点/联合/整词、一个多段集合与其保留均能完成；所有删除原子属于原U0，实际数量与预算匹配。真实模型测试不要求关键D一定大于随机，也不要求删关键一定降低Accuracy。
6. 执行真正CTC模型加载和至少一条真实音轨全流程；在独立对齐环境保存该结果，模型/数据无法获得时明确未通过此项，不将纯函数测试冒充模型运行成功。

### 16.3 最终输出验收

- valid manifest有728条，special有20条且外部ID恰为01—20。所有成功输入的每个原观测单元均完成扫描；未计算量不得填0。输入失败列表和影响范围清晰。
- 每样本有8个Shapley逻辑组合；重复实际输入可以共用view。权重和phi有明确NA规则，三路/联合原观测域不混用。
- 所有应运行实验都有完成或具体not_applicable原因；默认预算、实际成本、区间及随机签名相符。每个随机集合唯一、合法、非目标且可按保存seed复现。
- 三预算独立选择，允许非嵌套和非单调；方向候选失败不消失；group不可达不超预算；low的实际删除量与key一致。
- 主CSV严格20行，JSON/HTML20份，包括失败或未定位卡片；无标签专项没有Accuracy/F1等列。归因值可通过view_id回到原预测和真实干预输出。
- 词时、字符、官方AV来源分层展示；没有t/50×时长伪造定位；13号不出现虚构视觉证据。review_pending/unresolved出现在主报告和卡片中。
- 报告列有效样本和来源video数，随机任务指标计算口径正确；不能把解释分数变大写成预测性能提升。
- 环境、复用代码来源、模型revision、实际运行配置和命令齐全；无新代码写入Q2，无Q2权重/数据改写，无test新增选参。

validate --stage final不以D正值、G均值、CI跨不跨0或定位率达到某任意百分比作为验收阈值。它检查计算与文件是否完整、结论是否忠实；方法效果不好是要报告的实验结果。

## 17. 报告、交付体积与停止条件

### 17.1 必交材料

~~~text
Q3/README.md
Q3/THIRD_PARTY.md
Q3/pyproject.toml
Q3/requirements-main.txt
Q3/requirements-align.txt
Q3/configs/q3.yaml
Q3/src/、Q3/scripts/、Q3/tests/
Q3/reports/environment_main.txt、environment_align.txt
Q3/reports/numeric_tolerance.json
Q3/reports/validation_inputs.json、validation_final.json
Q3/reports/Q3_实验报告.md
Q3/reports/Q3_素材定位缺口.md
Q3/outputs/Q3_attachment4_predictions.csv
Q3/outputs/cases/01.json ... 20.json
Q3/outputs/cases/01.html ... 20.html
Q3/outputs/assets/                 # 必要短音频/代表帧/图，不是全部视频
Q3/runs/valid_main/、special_main/ # 全明细留服务器，可按记录复算
Q3/delivery/q3_submission.zip
Q3/delivery/size_report.json
~~~

Q3实验报告按“原样valid表现—三路删除/Shapley—三预算及随机/low—两类保留—组粒度—中性/错误案例—20条专项—映射覆盖与限制”组织。至少输出三预算G分布/均值区间图、模态D/相对权重与phi图、保留S对照图、粒度成本匹配表、真实中性与错误分组表。所有图旁列实际n，不只放最好看的样本。

报告附Q1—Q2—Q3闭环表：Q1提供的字符/媒体追溯方法及其未完成验证，Q2固定预测器与缺失算子，Q3同预算关键/随机/low扰动及中性错误分析。可直接引用服务器Q2既有valid clean/missing72汇总作为背景，但保留原场景定义；Q3定向删除的预算和几何不等于Q2 missing72，不能比较两个不同协议的数值就声称提高了缺失鲁棒性。无需重新跑train/test或返工Q1。

20条全量卡片已交时，报告中的典型案例按确定规则选择：各主要模态中D_m最大的一个、分类/强度首位分歧一个、方向未保持一个、13号、素材unresolved一个；重复样本去重，某类不存在写“未观察到”，不得造一个符合叙事的例子。典型样本选择规则仅用于展示，不改变统计集。

### 17.2 压缩包范围和全题50MB限制

q3_submission.zip包括源码/配置/依赖/README/来源说明、专项主CSV和20条轻量JSON/HTML、必要汇总图表和报告。默认**不放CTC权重、虚拟环境、pip/HF缓存、原数据、完整视频、第三方Git工作树、全量views.jsonl、Q2重复权重**。CTC是生成词时所需工具模型，按说明可重新下载；完整运行记录保留服务器。

必要素材附件按size_report逐项统计后纳入；若体积不允许则报告中用路径/来源记录并明确包内素材不可播放，不留下伪装已嵌入的链接。源代码打包时不跟随软链接把Q2数据或大模型拉入。外部源码保留准确来源、原作者说明、许可，不能因压缩包小省略第三方声明。

全题≤50MB应统计Q1+共享Q2预测权重+Q3合并后的真实ZIP。已有Q2约27.14MB只是背景；Q1已有大包不在本任务擅自删减。Agent若只具备Q2与Q3，size_report标明full_competition_total_bytes=null、原因“缺少Q1最终交付物”，分别报告Q3包和可见共享资源体积，不宣称整题已满足50MB。

### 17.3 Agent何时可以宣布完成

计算与源码部分只有在真实模型可用、规定实验全部完成且技术检查通过后才computational_status=complete；缺权重、输入失败或未跑全valid即partial/failed，列出精确范围。素材部分只有证据链实际完成才相应verified；算法已运行但待人审或官方AV来源未知，material_status=partial，并保留可复核数值结果。

外部网络失败、原始数据缺失、人工审核未完成、无法取得官方行生成来源是四种不同阻塞，分别说明，不用统一“已完成”掩盖。本任务不要求返工Q1未完成的多模态对齐；Q3中的独立专项映射也不能替Q1宣布验证成功。

最终向用户报告：Q3目录、实际运行环境与模型、完成样本/实验数量、验证结果、20条CSV/报告位置、真实素材缺口及包体积。不得报告预估指标为实验值，不启动新的情感模型训练或自行切换设计。本说明授权实施Agent在新Q3目录完成上述开发和实验，不授权改写Q2结论或用专项/test重新调参。

## 18. 来源索引与本文件验证边界

本文件可单独交给服务器Agent使用，已将方法v9实施所需公式、默认值与边界写入正文；Agent无需拥有本机旧Q3代码。项目内相对路径以competition-ck根为起点，服务器找不到的Q1参考文件按第5节下载固定提交。

| 依据 | 本文件采用的内容 |
|---|---|
| doc/Q3/Q3_多粒度反事实解释方法_v9_Q1Q2闭环与代码核查审核版.md | 主D、双输出方向、八子集、预算、连续选择、随机/保留、素材关系和评价身份 |
| Q2最终交付包的src/q2、ensemble_config.json、文本config及normalizer | 实际推理入口、单成员模型身份和输入遮蔽语义 |
| Q2/src/q2/data.py、state.py、masking.py、text.py、ensemble.py、metrics.py | 数据适配、tokenizer匹配、状态、包装返回对象和指标接口；部署时以包内版本为准 |
| Q2/configs/final_selection.json、final_train.yaml以及既有最终报告 | 最终选模、归档环境、已有结果和test探索限制 |
| Q1/q1_features/src/q1_features/{alignment,text_map,media,storage}.py | 实际可复制模块、CTC调用、原文词身份、媒体共同时间与输出依赖 |
| Q1已归档环境与模型来源记录 | CTC模型固定revision与依赖组合的来源 |
| ESPnet202402 PyPI发布元数据 | 核对其NumPy版本约束，不把主/对齐依赖强塞同一环境 |

外部下载与核查地址如下，均为实现所需来源，不是Q3实验结果：

~~~text
https://github.com/springx04/competition-ck/tree/139aa00ba794b21355aeeb6c55629a2d1d32c3e1
https://pypi.org/pypi/espnet/202402/json
https://pypi.org/project/ctc-segmentation/1.7.4/
https://huggingface.co/espnet/kamo-naoyuki_librispeech_asr_train_asr_conformer5_raw_bpe5000_schedule-truncated-c8e5f9/tree/e6a0f274799b5a4c157d1d5bc20de4c569e25f7e
https://download.pytorch.org/whl/cu124
https://download.pytorch.org/whl/cu121
~~~

本文件初版未执行服务器实验；迭代2已在服务器完成环境核验、Q3计算和技术验收。CTC候选词时已生成并完成结构检查，但人工听审尚未完成；官方A/V行来源仍未提供，因此素材状态保持partial。

> 本轮服务器核验已完成 728 条 valid、20 条专项；最终技术核验为 complete。479 条词时候选仍是待人工听审，官方 A/V 行来源仍 unresolved。逐词复核页、候选音频和 PTS 上下文帧只提供人工核验入口，不自动升级证据等级。
