# Third-party notices

本目录的提取实现必须按 doc/E题_Q1_服务器具体实现说明_v1_Agent执行版.md 的固定版本执行。本文件记录下载源码、预训练模型及适配边界；它不替代上游源码随附的许可证、版权头、模型卡或数据集许可证。下载的 models/、third_party/ 和编译环境不提交到 Git。

## 复用源码

| 组件 | 固定来源 | 用途 | 许可证与保留要求 |
| --- | --- | --- | --- |
| MMSA-FET | https://github.com/thuiar/MMSA-FET，commit f8fbd2d88d4f77580ea1ded0b3469073488c5c19 | 参考并适配 CTC、BERT、openSMILE、OpenFace 的短计算核心 | 上游 GNU GPL v3（GPL-3.0）；复制的文件保留版权头和上游 LICENSE，改造版本须标明已修改 |
| OpenFace | https://github.com/TadasBaltrusaitis/OpenFace，tag OpenFace_2.2.0 | 编译 FaceLandmarkVidMulti，静态 AU、头部旋转和视线角 | 遵守 OpenFace 原始 Copyright.txt、许可证/商业授权说明，以及 dlib、OpenBLAS、OpenCV 和训练数据集的许可证 |
| TalkNet | https://github.com/TaoRuijie/TalkNet_ASD，源码归档 SHA256 `7ba19f3dd877c8839445d5278130cabccd839df9f6a18f83b157219bdfbe029a` | 仅用于主动说话候选排序，不写入确认结论 | MIT；保留上游 LICENSE；预训练权重当前未下载，不得声称已运行 |
| InsightFace/ArcFace | https://github.com/deepinsight/insightface，计划使用 `buffalo_l` | 仅用于单样本内跨 episode 身份连接 | 代码与权重许可分离；公开预训练权重仅限非商业研究；当前未安装或下载 |
| WhisperX | https://github.com/m-bain/whisperX，计划固定 v3.8.6 | 仅用于独立音文诊断 | 当前未安装；模型及对齐权重须另行锁定和记录许可 |

MMSA-FET 下载后仅作为核对和适配来源，不执行 pip install MMSA-FET，不导入 MSA_FET.FeatureExtractionTool。计划适配的上游文件是 src/MSA_FET/aligner/default.py、extractors/text/bert.py、extractors/audio/opensmile.py、extractors/video/openface.py、single.py 和 utils.py；dataset.py 只用于理解旧接口，不作为附件1入口。适配必须保留真实 PTS、原生时间区间、质量状态和来源映射，不沿用旧批处理的标签/split 假设、FPS 虚构时间轴、词区间均值或前缀截断行为。

## 预训练模型

| 模型 | 编写时核对的 Hugging Face revision | 本地目录 | 用途 | 上游许可 |
| --- | --- | --- | --- | --- |
| google-bert/bert-base-uncased | 86b5e0934494bd15c9632b12f734a8a67f723594 | models/bert/ | 冻结、eval、float32 的末层 768 维文本语义特征 | Hub model card 标注 Apache-2.0 |
| espnet/kamo-naoyuki_librispeech_asr_train_asr_conformer5_raw_bpe5000_schedule-truncated-c8e5f9 | e6a0f274799b5a4c157d1d5bc20de4c569e25f7e | models/ctc/ | 16 kHz Conformer CTC 词级强制对齐；不是情感模型 | Hub model card 标注 CC-BY-4.0 |

scripts/download_models.py 在运行时通过 Hub API 取得实际 commit revision，再把该不可变 revision 显式传给 snapshot_download 并写入 models/download-record.json；上表 revision 仅用于核对。CTC 下载完成后，prepare_ctc_config(root) 只将原 ASR YAML 中的 bpemodel 和 normalize_conf.stats_file 相对 models/ctc 解析为绝对路径，输出 models/ctc/runtime-config.yaml；其他模型参数不变，原始 YAML 不覆盖。若 token_list 不是内嵌列表，脚本明确报模型版本不匹配，不猜测词表或替换模型。

OpenFace 的四个 CEN patch 专家文件必须由 OpenFace_2.2.0/download_models.sh 指定的官方公开链接下载到 third_party/OpenFace/lib/local/LandmarkDetector/model/patch_experts/，并核对不是 HTML 错误页。链接失效时只使用 OpenFace 官方安装页同名模型备用地址；找不到时记录视觉环境未就绪，不更换检测器后继续声称使用 OpenFace 2.2.0。

## Python 与原生依赖

Python 版本与包版本以 env/requirements-q1.txt、执行版第 4 节为准；PyTorch 2.1.2/torchaudio 2.1.2 使用官方 CUDA 11.8 index 单独安装。OpenFace 不通过 pip 安装，而是在独立 env/openface 环境中编译。各 PyPI/conda 依赖仍受各自上游许可证约束，安装时保留其许可证信息。

本工程的 Python 实现是按上述上游接口与算法调用顺序重新封装的适配代码，没有逐文件复制 MMSA-FET 包源码，也不导入其高层工具。若后续把上游代码段直接复制进本工程，必须同步保留对应版权头、GPL 许可证与改动摘要。
