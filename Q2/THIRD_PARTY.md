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
- 整理目录：`models/text_encoder/`。首次下载后转换为 FP16 存储；训练、验证和推理重新加载为 FP32 计算，冻结并使用末层逐词元输出，不使用 pooler。
- 词表、模型配置和实际文件可用性必须由 `prepare-model` 与 `tokenizer-check` 在服务器记录；不使用来历不明的替代权重。

除上述来源外，不安装 EBMC 的完整依赖，也不把音频、视频、OpenFace、MMSA-FET 等非本方案依赖带入 Q2 环境。
