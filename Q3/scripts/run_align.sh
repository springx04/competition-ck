#!/usr/bin/env bash
set -euo pipefail
Q3_ROOT="${Q3_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
export PYTHONPATH="$Q3_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
PYTHON_BIN="${ALIGN_PY:-python}"
"$PYTHON_BIN" "$Q3_ROOT/scripts/collect_environment.py" --output "$Q3_ROOT/reports/environment_align.txt"
"$PYTHON_BIN" -m q3_align --config "$Q3_ROOT/configs/q3.yaml" media --resume
"$PYTHON_BIN" -m q3_align --config "$Q3_ROOT/configs/q3.yaml" align --resume
"$PYTHON_BIN" -m q3_align --config "$Q3_ROOT/configs/q3.yaml" make-review
