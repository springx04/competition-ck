from __future__ import annotations

import argparse
import json

from q1_features.review_experiment import (
    analyze_agreement,
    export_proposed_configs,
    finalize_decisions,
    prepare_workspace,
)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Q1 blinded double-review workflow")
    sub = root.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare")
    prepare.add_argument("--run-dir", required=True)
    prepare.add_argument("--output-dir", required=True)
    prepare.add_argument("--seed", type=int, default=2026)
    prepare.add_argument("--force", action="store_true")

    analyze = sub.add_parser("analyze")
    analyze.add_argument("--reviewer-a", required=True)
    analyze.add_argument("--reviewer-b", required=True)
    analyze.add_argument("--run-dir", required=True)
    analyze.add_argument("--output-dir", required=True)

    finalize = sub.add_parser("finalize")
    finalize.add_argument("--reviewer-a", required=True)
    finalize.add_argument("--reviewer-b", required=True)
    finalize.add_argument("--adjudication", required=True)
    finalize.add_argument("--run-dir", required=True)
    finalize.add_argument("--output", required=True)

    export = sub.add_parser("export")
    export.add_argument("--final-decisions", required=True)
    export.add_argument("--baseline-alignment", required=True)
    export.add_argument("--output-dir", required=True)
    return root


def main() -> int:
    args = parser().parse_args()
    if args.command == "prepare":
        result = prepare_workspace(args.run_dir, args.output_dir, seed=args.seed, force=args.force)
    elif args.command == "analyze":
        result = analyze_agreement(args.reviewer_a, args.reviewer_b, args.run_dir, args.output_dir)
    elif args.command == "finalize":
        result = finalize_decisions(
            args.reviewer_a, args.reviewer_b, args.adjudication, args.run_dir, args.output,
        )
    else:
        result = export_proposed_configs(
            args.final_decisions, args.baseline_alignment, args.output_dir,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
