from __future__ import annotations

import argparse
import json
from pathlib import Path

from q1_features.word_aligned_candidate import build_word_aligned_candidate


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a candidate-only word-aligned multimodal package from frozen native features"
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--validation-json", required=True)
    args = parser.parse_args()

    result = build_word_aligned_candidate(
        Path(args.run_dir).resolve(),
        Path(args.output).resolve(),
        Path(args.validation_json).resolve(),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
