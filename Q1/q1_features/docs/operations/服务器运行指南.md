# q1_features

这是附件1 Q1 的静态工程入口。实现目标是读取 100 条 MP4 与已有英文转写，保留真实媒体时钟、词/子词映射、CTC 候选时间、25 维音频、22 维视觉、50 窗特征、可用状态、覆盖率和可回到原素材的来源表。Q1 不训练情感分类器、不重新 ASR 替换转写、不读取情感 label 调参，也不实现 Q2/Q3 预测网络。

仓库已包含执行版要求的 `src/q1_features/` 命令行实现与无模型实质测试；`models/`、`third_party/` 和 `runs/` 由服务器安装与运行步骤产生。本 README 不把源码或本地纯函数测试描述成已经完成服务器真实模型验证或100条全量提取。

通过 `git archive` 发布时，`SOURCE_REVISION` 会被替换成对应的40位提交号；因此无 `.git` 目录的服务器归档仍能记录代码版本，并允许同一干净归档内按规范续跑。工作树直接复制且该占位符未展开时，`--resume` 会保守拒绝。

## 固定选择

- Python 3.10.14；pip 24.0；ffmpeg 6.1.1；libsndfile 1.2.2；依赖精确版本见 env/requirements-q1.txt。
- PyTorch 2.1.2 与 torchaudio 2.1.2 使用 CUDA 11.8 wheel；默认 cuda:0、batch_size=1，真实张量运算成功后才使用 GPU。BERT 与 CTC 分阶段运行；OpenFace 和 openSMILE 在 CPU 运行。
- 文本模型固定为 google-bert/bert-base-uncased，BertModel 末层 768 维、冻结、eval、float32。
- 词级强制对齐固定为 espnet/kamo-naoyuki_librispeech_asr_train_asr_conformer5_raw_bpe5000_schedule-truncated-c8e5f9；它只用于时间定位，不是情感模型。CTC 使用模型自带的 global-MVN 训练统计，不用附件1重新估计。
- 视觉固定为 OpenFace 2.2.0 的 FaceLandmarkVidMulti、静态 AU 模式：17 个 AU 强度 + 3 个头部旋转角 + 2 个视线角，共 22 维。
- 音频固定为 openSMILE eGeMAPSv02 的 LowLevelDescriptors，25 维；聚合固定 50 个相对时间窗，coverage 使用有效交叠区间并集。
- target_face_segments.csv 与 alignment_review.csv 初始只有表头；不以最大脸、最长轨迹或第 0 张脸自动解决身份，不以相似样本、平移音轨或提高 CTC 得分猜测修复。

`scripts/download_models.py` 在下载时通过 Hub API 解析两个模型仓库的实际不可变 revision，并写入 `models/download-record.json`。以下是编写本说明时核对到的 revision，仅作核对值；运行记录以下载脚本实际写入值为准：

~~~text
google-bert/bert-base-uncased
  86b5e0934494bd15c9632b12f734a8a67f723594

espnet/kamo-naoyuki_librispeech_asr_train_asr_conformer5_raw_bpe5000_schedule-truncated-c8e5f9
  e6a0f274799b5a4c157d1d5bc20de4c569e25f7e
~~~

## 环境与安装

以下是服务器 Bash 命令；不能在 Windows PowerShell 照搬。Q1_ROOT 指向当前任务可写目录，Q1_DATA 指向已解压的 E题数据 目录，不把源数据复制到项目。要求 uname -m 为 x86_64；aarch64 不安装本文 x86_64 二进制。磁盘按额外 30 GB、内存 16 GB、GPU 显存 8 GB 作为资源准备参考，不当作已测量峰值。

~~~bash
export Q1_ROOT="$PWD/q1_features"
export Q1_DATA="/绝对路径/E题数据"
mkdir -p "$Q1_ROOT"/{third_party,models,env,configs,src/q1_features,tests,scripts,runs}
cd "$Q1_ROOT"
uname -m
nvidia-smi
~~~

若没有 conda/mamba，按执行版指定的官方 micromamba 安装说明安装到项目工具目录；项目模型和提取依赖仍按固定版本安装：

~~~bash
mkdir -p "$Q1_ROOT/tools"
curl -fL --retry 3 https://micro.mamba.pm/api/micromamba/linux-64/latest \
  -o "$Q1_ROOT/tools/micromamba.tar.bz2"
tar -xjf "$Q1_ROOT/tools/micromamba.tar.bz2" \
  -C "$Q1_ROOT/tools" bin/micromamba
export PATH="$Q1_ROOT/tools/bin:$PATH"
export MAMBA_ROOT_PREFIX="$Q1_ROOT/env/mamba-root"
~~~

~~~bash
micromamba create -y -p "$Q1_ROOT/env/python" -c conda-forge \
  python=3.10.14 pip=24.0 ffmpeg=6.1.1 libsndfile=1.2.2 \
  c-compiler cxx-compiler make pkg-config
export Q1_PY="$Q1_ROOT/env/python/bin/python"
export PATH="$Q1_ROOT/env/python/bin:$PATH"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export HF_HOME="$Q1_ROOT/models/hf_cache"
~~~

安装次序固定为：

~~~bash
"$Q1_PY" -m pip install setuptools==69.5.1 wheel==0.43.0 Cython==0.29.37 numpy==1.23.5
"$Q1_PY" -m pip install torch==2.1.2 torchaudio==2.1.2 \
  --index-url https://download.pytorch.org/whl/cu118
"$Q1_PY" -m pip install --no-build-isolation -r env/requirements-q1.txt
"$Q1_PY" -m pip check
"$Q1_PY" -m pip freeze > env/installed-python.txt
ffmpeg -version > env/ffmpeg-version.txt
~~~

不能安装 espnet[all]、CUDA Toolkit、Kaldi 训练工具、ASR 数据集或语言模型；espnet-model-zoo 不作为运行依赖。若确有当日传递依赖不兼容，只能记录满足上游声明的兼容性调整并重新 pip check，不能改变模型或特征定义。

## 源码与 OpenFace

~~~bash
cd "$Q1_ROOT"
git clone https://github.com/thuiar/MMSA-FET.git third_party/MMSA-FET
git -C third_party/MMSA-FET checkout f8fbd2d88d4f77580ea1ded0b3469073488c5c19
git clone --branch OpenFace_2.2.0 --depth 1 \
  https://github.com/TadasBaltrusaitis/OpenFace.git third_party/OpenFace
git -C third_party/MMSA-FET rev-parse HEAD > env/mmsa-revision.txt
git -C third_party/OpenFace rev-parse HEAD > env/openface-revision.txt
~~~

OpenFace 使用独立 C++ 环境，不与 Q1 Python 环境混装：

~~~bash
micromamba create -y -p "$Q1_ROOT/env/openface" -c conda-forge \
  cmake=3.28 ninja=1.11 gcc_linux-64=12 gxx_linux-64=12 \
  opencv=4.8.1 dlib=19.24 libopenblas=0.3.26 boost-cpp=1.82
~~~

四个 CEN patch 专家文件下载到 third_party/OpenFace/lib/local/LandmarkDetector/model/patch_experts/，再配置和编译；第 5.3 节的四条官方 URL 必须原样使用。模型文件未就绪时，不换检测器后继续声称使用 OpenFace 2.2.0。

~~~bash
OF_PATCH="$Q1_ROOT/third_party/OpenFace/lib/local/LandmarkDetector/model/patch_experts"
mkdir -p "$OF_PATCH"
curl -fL --retry 3 'https://www.dropbox.com/s/7na5qsjzz8yfoer/cen_patches_0.25_of.dat?dl=1' -o "$OF_PATCH/cen_patches_0.25_of.dat"
curl -fL --retry 3 'https://www.dropbox.com/s/k7bj804cyiu474t/cen_patches_0.35_of.dat?dl=1' -o "$OF_PATCH/cen_patches_0.35_of.dat"
curl -fL --retry 3 'https://www.dropbox.com/s/ixt4vkbmxgab1iu/cen_patches_0.50_of.dat?dl=1' -o "$OF_PATCH/cen_patches_0.50_of.dat"
curl -fL --retry 3 'https://www.dropbox.com/s/2t5t1sdpshzfhpj/cen_patches_1.00_of.dat?dl=1' -o "$OF_PATCH/cen_patches_1.00_of.dat"
~~~

第 5.3 节要求在模型下载后再运行 CMake：

~~~bash
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
~~~

## 模型下载与 CTC runtime config

从项目根目录执行：

~~~bash
"$Q1_PY" -m pip install --no-deps -e .
"$Q1_PY" scripts/download_models.py
~~~

下载器使用固定 Hub revision 和 snapshot_download 实体文件，不使用 Git LFS 文本指针。CTC 必须包含：

~~~text
models/ctc/<ASR_DIR>/config.yaml
models/ctc/<ASR_DIR>/valid.acc.ave_10best.pth
models/ctc/data/token_list/bpe_unigram5000/bpe.model
models/ctc/exp/asr_stats_raw_bpe5000_sp/train/feats_stats.npz
~~~

其中 <ASR_DIR> 为：

~~~text
asr_train_asr_conformer5_raw_bpe5000_scheduler_confwarmup_steps25000_batch_bins140000000_optim_conflr0.0015_initnone_accum_grad2_sp
~~~

download_models.py 的 prepare_ctc_config(root) 是显式入口。它读取原始 ASR YAML，验证 token_list 是直接内嵌列表，仅将 bpemodel 和 normalize_conf.stats_file 解析为 models/ctc 下的绝对路径，输出 models/ctc/runtime-config.yaml，不覆盖原始 YAML，不猜测或替换模型。

## 配置与复核表

所有相对路径相对项目根目录解析。configs/q1.yaml 中的阈值是预先指定的工程规则，未经数据集质量标注优化，不能看情感 label 调整。首次 prepare 的 --data-root 覆盖 YAML 的 null 并保存到 run 配置；后续命令不传时使用 run 中保存的路径。

身份复核表 configs/target_face_segments.csv 的列为：

~~~text
sample_id,start,end,face_id,episode_id,reviewer,evidence_note
~~~

错配复核表 configs/alignment_review.csv 的列为：

~~~text
sample_id,issue_type,start,end,review_status,reviewer,evidence,action
~~~

其中 review_status 只能为 confirmed_match / confirmed_mismatch / unresolved；action 只能为 release_pairing / quarantine_pairing / keep_unresolved。没有可靠视听核验能力时必须写明“未完成视听核验”，保持 unverifiable/needs_review，不能伪造人工确认正常。

## 服务器运行命令

实现代码完成后按执行版顺序运行：

~~~bash
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
~~~

完成 media 后，按 duration 升序选择第 0、floor((N_valid-1)/2)、最后一条有效样本，去重后写入 reports/review_samples.txt；首批至少 3 条，不按情感标签选择。

~~~bash
"$Q1_PY" -m q1_features check-models \
  --config configs/q1.yaml --run-dir runs/q1_full_20260923 \
  --ids-file runs/q1_full_20260923/reports/review_samples.txt

"$Q1_PY" -m q1_features run \
  --config configs/q1.yaml --run-dir runs/q1_full_20260923 \
  --ids-file runs/q1_full_20260923/reports/review_samples.txt --resume
~~~

首批结果不能称为 100 条完成。检查源时间、词/BPE、列名和三模态内容并修复后，再执行：

~~~bash
"$Q1_PY" -m q1_features run \
  --config configs/q1.yaml --run-dir runs/q1_full_20260923 --resume
~~~

完成复核表后：

~~~bash
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
~~~

reuse-raw-csv 只重做目标身份筛选，不重新调用 OpenFace；要求源选帧表、CSV 和提取配置未改变。run 遇单样本阶段失败时保留状态并继续独立分支；全局模型缺失或普遍加载失败才停止该阶段。resume 只用于同一 run、同一配置与同一代码版本，不能因文件存在就默认正确。

## Q1 与 Q2/Q3 的边界

Q1 只开发通用的 manifest、原文字符映射、媒体时钟、CTC 对齐、来源表示和缺失状态工具，不开发 Q2 网络。

- 附件2官方对齐版 train/valid/test 分别为 3395/728/727；音频/视觉维度为 74/35，50 个位置的含义不能由 Q1 时间窗定义推断。
- 附件3对齐版 30 条只有 text_bert/audio/vision，没有预计算 text；Q2 与 Q3 统一官方 text_bert 编码入口，并先在附件2 train/valid 核对词表和 attention mask。维度 768 不等于词表兼容。
- 附件4对齐版 20 条中 13 号视觉全零；不能把新提取视频中的动作说成原预测器实际使用的视觉证据。
- Q1 的 100 条与附件2重合 18 条，其中 7 条属于附件2 test；不能将 Q1 全部加入训练或调参。
- 数据错配清单必须交给后续 Q2/Q3；已知可疑的原素材映射不能作为解释真值。新的 CTC 词时间不能证明官方音频/视觉每行的真实秒数。
- Q1 输出保留模态坐标类型 relative_time_bins、源 ID 和 paired_use；Q2/Q3 输入另用 official_sequence_positions。不写官方输入的猜测性转换器，也不因长度都为 50 而共享已训练输入层权重；不把 Q1 的 25/22 维直接传给要求官方 74/35 维的 Q2 模型。

## 提交体积边界

三模态核心 float32 数组未压缩体积为：

~~~text
100 × 50 × (768+25+22) × 4 = 16,300,000 bytes
~~~

实际运行后必须生成 package_size.csv，按 50,000,000 bytes 做保守预算，并分列“Q1紧凑成果包”“运行所需外部预训练权重”“其他Q2/Q3材料预留”。在权重是否可以豁免的规则确认前，必须报告 submission_weight_policy=unresolved，不能宣布正式提交条件全部满足。不得为压缩体积擅自换 BERT、删除异常样本、只交示例或改成 float16。
