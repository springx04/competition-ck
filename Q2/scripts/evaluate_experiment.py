"""Read out a named exploratory checkpoint; keep each evaluation in a separate directory."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import torch

from q2.__main__ import evaluate_best
from q2.evaluate import selection_key


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name")
    parser.add_argument("--variant", required=True)
    parser.add_argument("--seed", type=int, default=1111)
    parser.add_argument("--split", choices=("valid", "test"), default="test")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    path = root / "experiments" / args.name / "runs" / args.variant / f"seed_{args.seed}/best.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    output = root / "reports/exploratory" / args.name / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output.mkdir(parents=True)
    selection = {"variant": args.variant, "seed": args.seed, "checkpoint": str(path.relative_to(root)),
                 "epoch": checkpoint["epoch"], "class_bias": None, "split": args.split,
                 "independent_holdout": False, "selection_basis": "Saved best checkpoint selected on valid.",
                 "observation_history": "Adaptive exploration; test has already been observed in this project.",
                 "valid_selection_key": checkpoint["best_key"]}
    (output / "selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    rows, _ = evaluate_best(root, checkpoint["config"], args.variant, args.seed,
                            split=args.split, stress=False, output_dir=output, selection=selection)
    key = selection_key(rows, checkpoint["epoch"])
    result = {**selection, "directory": str(output.relative_to(root)),
              **dict(zip(("missing_macro_f1", "missing_mae", "clean_macro_f1", "clean_mae"), key[:4]))}
    (output / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
