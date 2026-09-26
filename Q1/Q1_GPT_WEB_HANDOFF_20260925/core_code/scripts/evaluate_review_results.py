from __future__ import annotations

import argparse
import json

from q1_features.review_evaluation import evaluate_audio_review


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate Q1 adjudicated audio/text decisions")
    parser.add_argument("--final-decisions", required=True)
    parser.add_argument("--risk-csv", required=True)
    parser.add_argument("--quality-csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--iterations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    result = evaluate_audio_review(
        args.final_decisions,
        args.risk_csv,
        args.quality_csv,
        args.output,
        iterations=args.iterations,
        seed=args.seed,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

