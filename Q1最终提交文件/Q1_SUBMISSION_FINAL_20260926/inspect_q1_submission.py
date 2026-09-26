from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "code" / "src"))
from q1_features.submission import inspect_submission

parser = argparse.ArgumentParser(description="Inspect the Q1 final submission without network access")
parser.add_argument("--mode", choices=("quick", "full"), default="quick")
args = parser.parse_args()
print(json.dumps(inspect_submission(ROOT, mode=args.mode), ensure_ascii=False, indent=2))
