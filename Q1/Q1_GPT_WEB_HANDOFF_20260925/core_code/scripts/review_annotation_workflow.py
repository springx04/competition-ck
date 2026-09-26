from __future__ import annotations

import argparse
import json
from pathlib import Path

from q1_features.review_annotations import (
    analyze_annotation_agreement,
    export_multisegment_proposed_configs,
    finalize_annotation_gold,
    prepare_annotation_templates,
)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        description="Q1 blinded multi-segment and anchor-word review workflow"
    )
    sub = root.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare")
    prepare.add_argument("--run-dir", required=True)
    prepare.add_argument("--output-dir", required=True)
    prepare.add_argument("--risk-csv", required=True)
    prepare.add_argument("--baseline-alignment", required=True)
    prepare.add_argument("--seed", type=int, default=2026)
    prepare.add_argument("--force", action="store_true")

    analyze = sub.add_parser("analyze")
    analyze.add_argument("--run-dir", required=True)
    analyze.add_argument("--segments-a", required=True)
    analyze.add_argument("--segments-b", required=True)
    analyze.add_argument("--anchors-a", required=True)
    analyze.add_argument("--anchors-b", required=True)
    analyze.add_argument("--output-dir", required=True)
    analyze.add_argument("--endpoint-tolerance", type=float, default=0.20)
    analyze.add_argument("--tiou-threshold", type=float, default=0.80)

    finalize = sub.add_parser("finalize")
    finalize.add_argument("--run-dir", required=True)
    finalize.add_argument("--final-decisions", required=True)
    finalize.add_argument("--segments-a", required=True)
    finalize.add_argument("--segments-b", required=True)
    finalize.add_argument("--anchors-a", required=True)
    finalize.add_argument("--anchors-b", required=True)
    finalize.add_argument("--adjudicated-segments", required=True)
    finalize.add_argument("--adjudicated-anchors", required=True)
    finalize.add_argument("--agreement-dir", required=True)
    finalize.add_argument("--output-dir", required=True)
    finalize.add_argument("--endpoint-tolerance", type=float, default=0.20)
    finalize.add_argument("--tiou-threshold", type=float, default=0.80)

    export = sub.add_parser("export")
    export.add_argument("--run-dir", required=True)
    export.add_argument("--final-decisions", required=True)
    export.add_argument("--final-segments", required=True)
    export.add_argument("--baseline-alignment", required=True)
    export.add_argument("--output-dir", required=True)
    return root


def main() -> int:
    args = parser().parse_args()
    if args.command == "prepare":
        result = prepare_annotation_templates(
            Path(args.run_dir),
            Path(args.output_dir),
            Path(args.risk_csv),
            Path(args.baseline_alignment),
            seed=args.seed,
            force=args.force,
        )
    elif args.command == "analyze":
        result = analyze_annotation_agreement(
            Path(args.run_dir),
            Path(args.segments_a),
            Path(args.segments_b),
            Path(args.anchors_a),
            Path(args.anchors_b),
            Path(args.output_dir),
            endpoint_tolerance=args.endpoint_tolerance,
            tiou_threshold=args.tiou_threshold,
        )
    elif args.command == "finalize":
        result = finalize_annotation_gold(
            Path(args.run_dir),
            Path(args.final_decisions),
            Path(args.segments_a),
            Path(args.segments_b),
            Path(args.anchors_a),
            Path(args.anchors_b),
            Path(args.adjudicated_segments),
            Path(args.adjudicated_anchors),
            Path(args.agreement_dir),
            Path(args.output_dir),
            endpoint_tolerance=args.endpoint_tolerance,
            tiou_threshold=args.tiou_threshold,
        )
    else:
        result = export_multisegment_proposed_configs(
            Path(args.final_decisions),
            Path(args.final_segments),
            Path(args.baseline_alignment),
            Path(args.output_dir),
            Path(args.run_dir),
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
