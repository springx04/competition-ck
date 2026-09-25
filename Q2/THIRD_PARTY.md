# Third-party sources

本项目严格按 `doc/Q2/E题_Q2_服务器具体实现说明_v1_Agent执行版.md` 复用外部来源。不把密码、令牌或本地私钥写入项目。

## EBMC

- 来源：<https://github.com/kangverse/EBMC.git>
- 下载日期：2026-09-24；main 提交：`18ee1f2e1993d87748c0033b496eeba983c2ce81`。
- 服务器连续两次 `git clone --depth 1` 遇到 TLS 中断，故从同一官方仓库的 <https://codeload.github.com/kangverse/EBMC/zip/refs/heads/main> 下载归档；提交号由官方 GitHub API 的 `commits/main` 返回。归档中的 `EBMC/modules/msd.py` 和 `LICENSE` 已分别与本项目 `src/q2/vendor/ebmc_msd.py`、`licenses/EBMC-LICENSE` 逐字节比较一致。
- 目标目录：`third_party/EBMC/`
- 用途：只复用 `modules/msd.py` 中的 `MSDModule` 核心，并保留上游许可证；不直接运行上游训练脚本，不使用其 `MSDIntegrationModule`、EMC、IMTD 或 Soft-MoE 网络。
- 本项目改造：Q2 自己实现输入状态、外层掩码池化、跨注意力补偿、可信度融合和损失；新补偿器不称为 EBMC 原有实现。
- 许可证：上游许可证原文已保存到 `licenses/EBMC-LICENSE`。

## BERT

- 来源：Hugging Face `google/bert_uncased_L-4_H-256_A-4`
- 来源页：<https://huggingface.co/google/bert_uncased_L-4_H-256_A-4>
- 下载目录：`models/bert_download/`
- 允许下载的正式文件：`config.json`、`vocab.txt`、`pytorch_model.bin`、`README.md`。
- 整理目录：`models/text_encoder/`。首次下载后转换为 FP16 存储；训练、验证和推理重新加载为 FP32 计算，使用末层逐词元输出，不使用 pooler。历史基线冻结整个 BERT，后续微调的层数与 embedding 策略由每个实验的配置记录。
- 词表、模型配置和实际文件可用性必须由 `prepare-model` 与 `tokenizer-check` 在服务器记录；不使用来历不明的替代权重。

## 八层 BERT 对照及最终候选

- 官方来源：[google/bert_uncased_L-8_H-256_A-4](https://huggingface.co/google/bert_uncased_L-8_H-256_A-4)。2026-09-25 下载官方 `config.json`、`vocab.txt`、`pytorch_model.bin`、`README.md`，不增加数据集；模型页注明 Apache-2.0。
- 下载目录 `models/bert8_download/`；整理目录 `models/text_encoder_8/`。原始权重 57,739,749 字节，整理后的 FP16 权重 28,542,304 字节，FP32 计算。
- 实测八层、隐藏维256、四注意力头；词表全部30,522条与四层BERT逐项同序相等，不只是大小一致。因此继承已核验的附件2 WordPiece ID 映射。
- `attn_bert8_v1` 只解冻顶部四层；`attn_bert8_all_v1` 及1112/1113复验解冻全部八个 Transformer 层。三者均冻结 embedding。不能将前一实验写成“全八层微调失败”。
- `prepare-model` 根据配置中的 model_id/model_dir 整理对应模型，不需要下载其它模型。网络受限时可把上述官方原文件复制到下载目录后离线整理。
- 最终采用情况及实测推理包体积以 [`doc/Q2/report.md`](../doc/Q2/report.md) 为准。

除上述来源外，不安装 EBMC 的完整依赖，也不把音频、视频、OpenFace、MMSA-FET 等非本方案依赖带入 Q2 环境。
