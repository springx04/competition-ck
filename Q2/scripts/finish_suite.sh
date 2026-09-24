#!/usr/bin/env bash
# Continue the documented Q2 workflow after the formal training process exits.
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export HF_HOME="$PWD/.cache/huggingface"
source .venv/bin/activate

if [[ -f reports/train_suite.pid ]]; then
  train_pid="$(cat reports/train_suite.pid)"
  while kill -0 "$train_pid" 2>/dev/null; do sleep 30; done
fi

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
