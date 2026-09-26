# Q3 第三方来源

- Q2 最终推理包：本仓库 `../Q2/delivery/q2_final`（只读复用，按 Q2 交付许可）。
- Q1 对齐模块：按 Q3 实施说明固定提交 `139aa00ba794b21355aeeb6c55629a2d1d32c3e1` 的限定复制；当前运行入口实际依赖同级 Q1 项目的源码和模型目录，并非已将该固定提交的全部必要代码独立打包。固定提交是原方案约定，不应理解为本轮重新核验过服务器 Q1 的版本。
- ESPnet CTC 模型：`espnet/kamo-naoyuki_librispeech_asr_train_asr_conformer5_raw_bpe5000_schedule-truncated-c8e5f9`，revision `e6a0f274799b5a4c157d1d5bc20de4c569e25f7e`，仅用于词时定位。

Q3 不新增 Captum/SHAP/WhisperX/OpenFace/MMSA-FET/EBMC 运行依赖。

当前词时入口 `src/q3_align/pipeline.py::align` 使用 `../Q1/q1_features/src` 和 `../Q1/q1_features/models/ctc`。交付包不附带这些源码、CTC权重、Q2权重或官方数据。本轮只复用既有对齐候选并核验边界、状态与映射，479个候选词时的人工听审确认数仍为0。
