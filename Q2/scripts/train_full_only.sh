#!/usr/bin/env bash
# Run only the full Q2 model until its valid metrics are reviewed.
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export HF_HOME="$PWD/.cache/huggingface"
source .venv/bin/activate
for seed in 1111 1112 1113; do
  if [[ -f "runs/full/seed_${seed}/last.pt" ]]; then
    completed="$($PWD/.venv/bin/python -c 'import sys,torch; print(torch.load(sys.argv[1], map_location="cpu", weights_only=False)["epoch"])' "runs/full/seed_${seed}/last.pt")"
    if [[ "$completed" -ge 60 ]]; then
      echo "skip full seed=${seed}: already complete"
      continue
    fi
  fi
  echo "start/resume full seed=${seed}"
  python -u -m q2 train --config configs/default.yaml --variant full --seed "$seed" --resume
done