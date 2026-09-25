from __future__ import annotations

import argparse
import sys

from .report import report_run
from .runner import (
    PipelineError, _selected, check_models, collect_run, load_run, prepare_run,
    run_pipeline, stage_align, stage_audio, stage_audit, stage_media, stage_pool,
    stage_text, stage_vision, validate_run,
)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="python -m q1_features")
    sub = root.add_subparsers(dest="command", required=True)
    commands = ["prepare", "check-models", "media", "text", "align", "audio", "vision", "audit", "pool", "collect", "validate", "report", "run"]
    for name in commands:
        p = sub.add_parser(name)
        p.add_argument("--config", required=True)
        p.add_argument("--run-dir", required=True)
        if name == "prepare":
            p.add_argument("--data-root", required=True)
            p.add_argument("--labels-xlsx")
        if name in {"check-models", "media", "text", "align", "audio", "vision", "audit", "pool", "validate", "run"}:
            p.add_argument("--ids-file")
        if name in {"check-models", "media", "text", "align", "audio", "vision", "audit", "pool", "run"}:
            p.add_argument("--resume", action="store_true")
        if name == "run":
            p.add_argument("--redo-stage", action="append", choices=["media", "text", "align", "audio", "vision"], default=[])
        if name == "vision":
            p.add_argument("--reuse-raw-csv", action="store_true")
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "prepare":
            prepare_run(args.config, args.run_dir, args.data_root, args.labels_xlsx)
            return 0
        cfg, run, manifest = load_run(args.config, args.run_dir)
        samples, selected_ids = _selected(manifest, getattr(args, "ids_file", None))
        resume = bool(getattr(args, "resume", False))
        if args.command == "check-models": check_models(cfg, run, samples)
        elif args.command == "media": stage_media(cfg, run, samples, resume=resume)
        elif args.command == "text": stage_text(cfg, run, samples, resume=resume)
        elif args.command == "align": stage_align(cfg, run, samples, resume=resume)
        elif args.command == "audio": stage_audio(cfg, run, samples, resume=resume)
        elif args.command == "vision": stage_vision(cfg, run, samples, resume=resume, reuse_raw_csv=args.reuse_raw_csv)
        elif args.command == "audit": stage_audit(cfg, run, samples)
        elif args.command == "pool": stage_pool(cfg, run, samples)
        elif args.command == "collect": collect_run(run, manifest)
        elif args.command == "validate":
            errors = validate_run(run, manifest, selected_ids)
            return 3 if errors else 0
        elif args.command == "report": report_run(run)
        elif args.command == "run":
            if args.redo_stage and not args.ids_file:
                raise PipelineError("--redo-stage requires --ids-file to make the affected sample set explicit")
            errors = run_pipeline(cfg, run, manifest, samples, selected_ids, resume=resume, redo_stages=set(args.redo_stage))
            return 3 if errors else 0
        return 0
    except (PipelineError, ValueError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
