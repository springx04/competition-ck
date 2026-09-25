#!/usr/bin/env bash
set -euo pipefail
Q3_ROOT="${Q3_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
export PYTHONPATH="$Q3_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
PYTHON_BIN="${Q3_PY:-python}"
"$PYTHON_BIN" "$Q3_ROOT/scripts/collect_environment.py" --output "$Q3_ROOT/reports/environment_main.txt"
"$PYTHON_BIN" -m q3 --config "$Q3_ROOT/configs/q3.yaml" prepare --split valid
"$PYTHON_BIN" -m q3 --config "$Q3_ROOT/configs/q3.yaml" prepare --split special
"$PYTHON_BIN" -m q3 --config "$Q3_ROOT/configs/q3.yaml" validate --stage inputs
"$PYTHON_BIN" -m q3 --config "$Q3_ROOT/configs/q3.yaml" check-predictor
"$PYTHON_BIN" -m q3 --config "$Q3_ROOT/configs/q3.yaml" attribute --split valid --resume
"$PYTHON_BIN" -m q3 --config "$Q3_ROOT/configs/q3.yaml" attribute --split special --resume
