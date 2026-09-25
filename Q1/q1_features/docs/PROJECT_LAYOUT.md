# 项目布局与维护规则

## 1. 布局原则

本项目采用“源码与配置稳定、运行资产原位保留、说明文档分类归档”的布局。`configs/`、`models/`、`runs/`、`third_party/`、`env/` 和 `tools/` 的相对路径已被配置、运行记录和构建证据引用，不应为了目录美观而移动。

## 2. 目录职责

| 路径 | 内容 | 维护规则 |
|---|---|---|
| `configs/` | 特征配置、身份审核表、对齐审核表 | 可通过规定流程更新审核表；阈值和路径变更必须重新验证 |
| `src/q1_features/` | Python 实现 | 仅做可审计的最小修改，修改后运行全部测试 |
| `scripts/` | 模型准备脚本 | 模型 revision 和落盘结构不得临时替换 |
| `tests/` | 单元测试 | 不得为通过测试而删除、跳过或弱化测试 |
| `models/` | 固定模型权重、缓存和下载记录 | 作为运行资产保留，不手工修改 |
| `runs/` | manifest、native/pooled 特征、状态、报告、审核证据 | 由 CLI 生成，不手工编辑生成物 |
| `third_party/` | OpenFace 等第三方源码、模型和构建目录 | 版本及构建证据必须可追溯 |
| `env/` | 依赖清单、日志、版本和真实运行证据 | 日志与证据保留；运行环境不得随意替换 |
| `tools/` | 本地工具和构建辅助组件 | 仅在重建环境时更新 |
| `docs/requirements/` | 脱敏的执行与验收要求 | 不记录连接凭据或内部网络地址 |
| `docs/operations/` | 安装、运行、恢复与审核操作手册 | 命令变化时同步维护 |
| `docs/reports/` | 实验结果和最终验收记录 | 作为结论性文档保留 |
| `docs/delivery/` | 历史归档内容、文件清单和校验说明 | 历史快照不可伪装成当前动态清单 |

## 3. 标准数据流

```text
prepare → media → check-models → 抽样 run → 完整 run
                                         │
                                         ▼
人工审核 → vision --reuse-raw-csv → audit → pool → collect → validate → report
```

- `prepare` 建立 manifest、标签和样本索引。
- `media` 固化媒体信息、音频和源时间轴。
- `check-models` 在正式运行前验证固定模型。
- `run --resume` 负责各样本可恢复的模态提取。
- 人工审核只能依据真实音视频、时间轴或人脸证据填写。
- 最终阶段必须按规定顺序执行，不得跳过 `audit`、`collect` 或 `validate`。

## 4. 可编辑与不可编辑边界

允许通过正常开发或审核流程编辑：

- `src/`、`scripts/`、`tests/` 中的项目文件；
- `configs/q1.yaml`，但变更后需使用匹配的新运行记录；
- 两个人工审核 CSV，但结论必须有逐项证据；
- `docs/` 下的说明与报告。

不得手工编辑或覆盖：

- 原始数据集；
- 下载后的模型权重和 `models/download-record.json`；
- OpenFace 模型、二进制及构建证据；
- `runs/` 中的 `.npz`、native 特征、CSR、时间轴、状态和验证结果；
- 已发布归档、校验文件和历史内容清单。

## 5. 缓存与清理

项目根目录中的 `.pytest_cache/`、源码和测试目录的 `__pycache__/`、`.pyc` 以及 editable install 产生的 `src/q1_features.egg-info/` 可以安全重建。环境、模型、第三方构建、工具链和运行目录内即使存在同名缓存，也不纳入常规清理范围。

## 6. 版本与交付

- 代码来源以根目录 `SOURCE_REVISION` 为准。
- 模型来源以 `models/download-record.json` 为准。
- 运行配置以相应 run 目录中保存的配置为准。
- 每次正式交付必须拥有独立文件名、内容清单、文件列表和 SHA256。
- 2026-09-24 归档信息见 [`delivery/交付归档说明.md`](delivery/交付归档说明.md)。

