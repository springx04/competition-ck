# Q2：模态局部缺失鲁棒情感预测

本目录是 Q2 的独立 Python 项目。实现先在本地完成代码和语义测试，再上传到 Linux/RTX 4090 服务器完成环境安装、数据核对、验证、30 个训练 run、评估、专项预测和推理包导出。实现边界、固定超参数、输入语义、变体及结果判定以 `doc/Q2/E题_Q2_服务器具体实现说明_v1_Agent执行版.md` 为准；方法解释参考 `doc/Q2/E题_Q2_模态局部缺失鲁棒情感预测方法_v2_审核稿.md`。

## 目录与磁盘规划

服务器系统盘只有约 30 GB、数据盘约 50 GB 时，把本项目根目录 `Q2_ROOT` 放在数据盘挂载点，并把已解压的只读 `E题数据` 也放在数据盘。项目代码、`.venv`、Hugging Face/Pip 缓存、预处理数据、训练 run、报告和交付包都置于 `Q2_ROOT`；系统盘只保留操作系统和必要的 SSH/Python 工具。训练完成后可清理缓存，但不要删除尚未归档的 run、best checkpoint 或报告。

当前服务器路径：

```bash
export Q2_ROOT=/root/gpufree-data/shuomo_E/Q2
export Q2_DATA_ROOT=/root/gpufree-data/shuomo_E/data
mkdir -p "$Q2_ROOT"
cd "$Q2_ROOT"
```

`configs/default.yaml` 的 `project.data_root` 已填写上述服务器数据目录。不要把本地 Windows 路径写入配置，也不要改动原始题目数据。

## 本地阶段

在本地只完成源码实现、配置检查和不依赖 GPU/题目大数据的语义测试。项目采用 `src` 布局；安装 torch 时按服务器的 CUDA 组合执行，torch 不在 `requirements.txt` 中重复安装。代码完成后检查：

```bash
cd Q2
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
python -m pytest -q
```

本地没有 Linux/4090 或题目数据时，不把本地测试结果写成正式训练、GPU 或数据核验结果。

## 服务器环境

服务器默认 SSH 别名是 `e-question-server`。在 Q2 根目录创建专用环境，不修改系统 Python：

```bash
ssh e-question-server
cd "$Q2_ROOT"
.python311/bin/python -m venv --prompt Q2 .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
python -m pip check
nvidia-smi
```

若没有 `python3.11` 但已有 conda，按方案在 Q2 数据盘目录建立 `.python311`，再用它创建同一个 `.venv`。若驱动不能运行 cu124，只改用同一 torch 版本的 cu118 轮子并在 `reports/environment.txt` 记录原因和实测 CUDA；不要静默改为 CPU 完成正式训练。正式运行前记录 Python、直接依赖、torch/CUDA、GPU 显存、驱动、CPU、内存和磁盘余量，并保存 `pip freeze`。

下载 EBMC 和指定 BERT 的目录、文件范围及许可证记录见 `THIRD_PARTY.md`。模型下载/整理由 `prepare-model` 完成，正式文本模型是 `models/text_encoder/` 中 FP16 存储、FP32 计算的冻结小型 BERT。

## 完整执行顺序

所有命令从 Q2 根目录执行，均支持 `--config configs/default.yaml`。先设置运行环境变量：

```bash
cd "$Q2_ROOT"
source "$Q2_ROOT/.venv/bin/activate"
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export HF_HOME="$Q2_ROOT/.cache/huggingface"
```

按以下顺序执行。每个命令由 Q2 自己实现，不是 EBMC 上游命令：

```bash
python -m q2 prepare-model --config configs/default.yaml
python -m q2 inspect-data --config configs/default.yaml
python -m q2 tokenizer-check --config configs/default.yaml
python -m q2 prepare-data --config configs/default.yaml
python -m q2 cache-text --config configs/default.yaml --splits train valid
python -m q2 make-masks --config configs/default.yaml --split valid
python -m q2 verify --config configs/default.yaml

python -m q2 train-suite --config configs/default.yaml --experiments configs/experiments.yaml --resume
python -m q2 evaluate-suite --config configs/default.yaml --experiments configs/experiments.yaml
python -m q2 select-final --config configs/default.yaml

python -m q2 cache-text --config configs/default.yaml --splits test
python -m q2 make-masks --config configs/default.yaml --split test
python -m q2 evaluate-test --config configs/default.yaml --selection reports/selection.json
python -m q2 predict-special --config configs/default.yaml --selection reports/selection.json
python -m q2 export --config configs/default.yaml --selection reports/selection.json
python -m q2 verify-export --config configs/default.yaml
python -m q2 report --config configs/default.yaml
```

单个 run 的调试或续训：

```bash
python -m q2 train --config configs/default.yaml --variant full --seed 1111 --resume
```

`train-suite` 按 `configs/experiments.yaml` 的 10 个变体和 `default.yaml` 的 3 个 seed 顺序运行，共 30 个 run，不在同一张 4090 上并发。已完整的 run 经配置核对后跳过；未完成的 run 使用 `--resume` 从 `last` 续训。不得只因已有 `best.pt` 就判定 60 轮和评估已完成。

当前执行门控：先只运行 ull 的 1111、1112、1113 三个 seed，等待 valid 指标和稳定性审核通过后，才允许启动 late_clean、late_aug 及其他消融。服务器使用 scripts/train_full_only.sh，已完成的 full run 会自动跳过，未完成的从 last.pt 续训。

服务器长任务可先启动 `train-suite`，再运行 `bash scripts/finish_suite.sh`；后者等待 `reports/train_suite.pid` 对应进程结束，按上述顺序续训、分析、选模、test、专项、导出及报告，并把标准输出写入 `reports/finish_suite.log`。若任一步失败，脚本立即停止，修正原因后从该步继续。

## 固定变体

`experiments.yaml` 固定列出：`full`、`late_clean`、`late_aug`、`no_msd`、`no_comp`、`no_reliability`、`no_cons`、`no_span`、`no_teacher`、`uniform_spans`。网络和 trainer 集中解析这些名称；关闭的组件不实例化，避免把未使用参数计入模型规模或导出包。

## 主要结果位置

- `reports/data_inventory.json/md`：附件层级、形状、ID、标签和零观测清点。
- `data/processed/`、`data/masks/`：预处理数据、仅由 train 拟合的标准化和固定扰动清单。
- `.cache/text/<split>/`：冻结 BERT 原观测文本缓存。
- `runs/<variant>/seed_<seed>/`：每个 run 的 resolved config、history、mask 记录、best/last checkpoint 和 valid 指标。
- `reports/selection.json`：按方案规则选出的变体以及固定部署 seed=1111 的学生。
- `reports/test/`：选定学生的附件 2 test 留出评价。
- `outputs/q2_predictions_aligned.csv`：附件 3 对齐版 30 条预测。
- `outputs/q2_special_state.csv`：附件 3 每条当前可观测位置和补偿位置计数；全零当前视觉不等于人工缺失真相。
- `delivery/q2_inference/`：可离线重载的推理代码、学生权重、标准化、冻结文本模型和入口。
- `reports/report.md`、`reports/paper_q2_results.md`：技术报告和论文结果材料。

原始数据、模型缓存、训练产物和交付文件默认由 `.gitignore` 排除；需要提交代码时只提交源码、配置和来源说明。正式结果必须以实际服务器运行日志和文件为准。
