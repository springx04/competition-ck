"""Collect completed and incomplete trials without mistaking valid for test."""
import csv
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from q2.masking import evaluation_grid


def metric_summary(path):
    with path.open(encoding="utf-8", newline="") as handle:
        rows = [r for r in csv.DictReader(handle) if r["scope"] == "all_samples"]
    names = {r["name"] for r in evaluation_grid()}
    main = [r for r in rows if r["scenario"] in names and not r.get("duplicate_of")]
    clean = next((r for r in rows if r["scenario"] == "clean"), None)
    if not main or clean is None:
        return None
    return {"main_scenarios": len(main), "n": int(clean["n"]),
            **{f"missing_{k}": float(np.mean([float(r[k]) for r in main]))
               for k in ("macro_f1", "mae", "accuracy", "weighted_f1")},
            **{f"clean_{k}": float(clean[k]) for k in
               ("macro_f1", "mae", "accuracy", "weighted_f1", "pearson")}}


def main():
    root = Path(__file__).resolve().parents[1]
    evaluations = []
    for p in sorted((root / "reports").rglob("selection.json")):
        if "final" in p.parts:
            continue
        s = json.loads(p.read_text(encoding="utf-8"))
        metrics = p.parent / "metrics_per_scenario.csv"
        if not metrics.exists() or not s.get("checkpoint"):
            continue
        stats = metric_summary(metrics)
        if stats is None:
            continue
        # Do not infer a split solely from a directory name.
        split = s.get("split")
        if split is None:
            split = "test" if stats["n"] == 727 else "valid" if stats["n"] == 728 else "unknown"
        evaluations.append({"checkpoint": s["checkpoint"], "split": split,
                            "class_bias": s.get("class_bias"),
                            "source": str(metrics.relative_to(root)), **stats})
    trials = []
    paths = list((root / "experiments").glob("*/runs/*/seed_*/best.pt"))
    paths += list((root / "runs_baseline_20260924").glob("*/seed_*/best.pt"))
    paths += list((root / "runs").glob("*/seed_*/best.pt"))
    for path in sorted(paths):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        conf = ckpt.get("config", {})
        history = path.parent / "history.csv"
        hist = list(csv.DictReader(history.open())) if history.exists() else []
        resource = path.parent / "resource_usage.json"
        rel = str(path.relative_to(root))
        valid_path = path.parent / "best_validation/metrics_per_scenario.csv"
        trials.append({"checkpoint": rel, "variant": path.parent.parent.name,
                       "seed": int(path.parent.name.split("_")[-1]),
                       "best_epoch": ckpt["epoch"],
                       "completed_epoch": int(hist[-1]["epoch"]) if hist else None,
                       "valid": metric_summary(valid_path) if valid_path.exists() else None,
                       "config": conf, "history": hist,
                       "resource": json.loads(resource.read_text()) if resource.exists() else None,
                       "evaluations": [r for r in evaluations if r["checkpoint"] == rel]})
    dest = root / "results/experiment_ledger.json"
    dest.parent.mkdir(exist_ok=True)
    # Historical checkpoints may be removed after their audit record is saved.
    # Preserve those records and refresh their available evaluation evidence.
    if dest.exists():
        archived = json.loads(dest.read_text(encoding="utf-8"))
        merged = {r["checkpoint"]: r for r in archived["trials"]}
        merged.update({r["checkpoint"]: r for r in trials})
        trials = list(merged.values())
        for row in trials:
            row["checkpoint_available"] = (root / row["checkpoint"]).is_file()
            row["evaluations"] = [r for r in evaluations if r["checkpoint"] == row["checkpoint"]]
    dest.write_text(json.dumps({"independent_holdout": False, "trials": trials,
                               "evaluations": evaluations}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"trials": len(trials), "evaluations": len(evaluations), "path": str(dest)}))


if __name__ == "__main__":
    main()
