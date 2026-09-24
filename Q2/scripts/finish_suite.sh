#!/usr/bin/env bash
# Full-only gate: run the full model first; wait for metric review before ablations.
set -euo pipefail
cd "$(dirname "$0")/.."
exec bash scripts/train_full_only.sh