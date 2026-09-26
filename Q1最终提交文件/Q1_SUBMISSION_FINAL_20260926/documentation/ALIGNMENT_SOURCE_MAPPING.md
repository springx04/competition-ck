# Q1词级对齐与来源映射

主产物以 `sample_indptr` 划分100条样本，以 `audio_indptr`/`vision_indptr` 构成词到原生来源的CSR映射。`*_source_sample_index` 必须等于该词的 `sample_index`；`*_overlap_s` 非负；mask=0时对应特征为零。词级 start/end 来自已冻结时间对齐，语义冲突或未决样本不会伪造音频时间。视觉仅保留 VERIFIED_ACTIVE 或 VERIFIED_CONTINUITY 来源；no-face视觉始终mask。
