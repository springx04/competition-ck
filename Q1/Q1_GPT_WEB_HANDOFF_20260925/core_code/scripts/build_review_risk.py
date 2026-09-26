from __future__ import annotations

import argparse
import json

from q1_features.review_risk import build_audio_risk_ranking


def main() -> int:
    parser = argparse.ArgumentParser(description="Build evidence-only Q1 audio/text review risk ranking")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--whisperx-csv")
    args = parser.parse_args()
    result = build_audio_risk_ranking(
        args.run_dir,
        args.output,
        whisperx_csv=args.whisperx_csv,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

