#!/usr/bin/env bash
set -euo pipefail
Q3_ROOT="${Q3_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
export PYTHONPATH="$Q3_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
PYTHON_BIN="${Q3_PY:-python}"
CONFIG="${Q3_CONFIG:-$Q3_ROOT/configs/q3_v2_server.yaml}"
mkdir -p "$Q3_ROOT/reports"
"$PYTHON_BIN" "$Q3_ROOT/scripts/collect_environment.py" --output "$Q3_ROOT/reports/environment_main_v2.txt"
"$PYTHON_BIN" -m pytest "$Q3_ROOT/tests" -q
"$PYTHON_BIN" -m q3 --config "$CONFIG" validate --stage inputs
"$PYTHON_BIN" -m q3 --config "$CONFIG" check-predictor
for SPLIT in valid special; do
  "$PYTHON_BIN" -m q3 --config "$CONFIG" run --split "$SPLIT" --resume
done
"$PYTHON_BIN" -m q3 --config "$CONFIG" map --split special
for SPLIT in valid special; do
  "$PYTHON_BIN" -m q3 --config "$CONFIG" report --split "$SPLIT"
done
"$PYTHON_BIN" "$Q3_ROOT/scripts/verify_saved_predictor.py" --config "$CONFIG"
"$PYTHON_BIN" "$Q3_ROOT/scripts/build_media_review.py" --config "$CONFIG"
"$PYTHON_BIN" -m q3 --config "$CONFIG" validate --stage final
"$PYTHON_BIN" -m q3 --config "$CONFIG" export
