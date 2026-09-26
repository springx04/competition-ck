"""Selection and result documents generated from completed run artifacts."""
import csv
import json
from pathlib import Path
import statistics

import numpy as np
import torch

from .model.network import Student


VARIANTS = ("full", "late_clean", "late_aug", "no_msd", "no_comp", "no_reliability",
            "no_cons", "no_span", "no_teacher", "uniform_spans")
SEEDS = (1111, 1112, 1113)


def read_run(root: Path, variant: str, seed: int):
    directory = Path(root) / "runs" / variant / f"seed_{seed}"
    last = torch.load(directory / "last.pt", map_location="cpu", weights_only=False)
    if last["epoch"] != 60:
        raise ValueError(f"incomplete run: {directory}, epoch {last['epoch']}")
    best = torch.load(directory / "best.pt", map_location="cpu", weights_only=False)
    with (directory / "best_validation/metrics_per_scenario.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    main = [r for r in rows if r["scope"] == "all_samples" and r["scenario"] != "clean"
            and not r["duplicate_of"] and not r["scenario"].startswith("TAV_")
            and not r["scenario"].endswith(("whole", "staggered_0.4"))]
    clean = next(r for r in rows if r["scenario"] == "clean")
    resource = json.loads((directory / "resource_usage.json").read_text(encoding="utf-8"))
    deployed_parameters = sum(parameter.numel() for parameter in
                              Student(variant, [1/3, 1/3, 1/3], 0,
                                      include_aux_heads=False).parameters())
    return {"variant": variant, "seed": seed, "best_epoch": best["epoch"],
            "missing_macro_f1": statistics.mean(float(r["macro_f1"]) for r in main),
            "missing_mae": statistics.mean(float(r["mae"]) for r in main),
            "clean_macro_f1": float(clean["macro_f1"]), "clean_mae": float(clean["mae"]),
            "training_seconds": resource["seconds"], "peak_memory_bytes": resource["peak_memory_bytes"],
            "parameter_count": resource["parameter_count"],
            "deployed_parameters": deployed_parameters,
            "checkpoint": str((directory / "best.pt").relative_to(root))}


def _dominates(a, b, tolerance=1e-6):
    terms = (("clean_macro_f1", 1), ("clean_mae", -1),
             ("missing_macro_f1", 1), ("missing_mae", -1))
    differences = [(a[key] - b[key]) * direction for key, direction in terms]
    return all(value >= -tolerance for value in differences) and any(value > tolerance for value in differences)


def select_final(root: Path):
    root = Path(root)
    runs = [read_run(root, variant, seed) for variant in VARIANTS for seed in SEEDS]
    candidates = []
    for variant in VARIANTS:
        rows = [row for row in runs if row["variant"] == variant]
        summary = {"variant": variant, "deployed_parameters": rows[0]["deployed_parameters"]}
        for key in ("missing_macro_f1", "missing_mae", "clean_macro_f1", "clean_mae"):
            summary[key] = statistics.mean(row[key] for row in rows)
            summary[key + "_std"] = statistics.stdev(row[key] for row in rows)
        candidates.append(summary)
    dominance = {b["variant"]: [a["variant"] for a in candidates if a is not b and _dominates(a, b)]
                 for b in candidates}
    survivors = [row for row in candidates if not dominance[row["variant"]]]
    survivors.sort(key=lambda row: (-row["missing_macro_f1"], row["missing_mae"],
                                  -row["clean_macro_f1"], row["clean_mae"],
                                  row["deployed_parameters"], row["variant"]))
    winner = survivors[0]["variant"]
    deployed = next(row for row in runs if row["variant"] == winner and row["seed"] == 1111)
    selection = {"runs": runs, "candidates": candidates, "dominated_by": dominance,
                 "ordered_survivors": [row["variant"] for row in survivors],
                 "variant": winner, "seed": 1111, "best_epoch": deployed["best_epoch"],
                 "checkpoint": deployed["checkpoint"]}
    path = root / "reports/selection.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8")
    return selection


def suite_progress(root: Path):
    rows = []
    for variant in VARIANTS:
        for seed in SEEDS:
            directory = Path(root) / "runs" / variant / f"seed_{seed}"
            last = directory / "last.pt"
            if last.exists():
                checkpoint = torch.load(last, map_location="cpu", weights_only=False)
                epoch = int(checkpoint["epoch"])
            else:
                epoch = 0
            complete = (epoch == 60 and (directory / "best.pt").exists()
                        and (directory / "best_validation/metrics_per_scenario.csv").exists()
                        and (directory / "resource_usage.json").exists())
            rows.append({"variant": variant, "seed": seed, "completed_epoch": epoch,
                         "status": "complete" if complete else "pending" if epoch == 0 else "incomplete"})
    path = Path(root) / "reports/suite_progress.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    return rows


def package_sizes(directory: Path, output_csv: Path):
    directory = Path(directory)
    rows = [{"path": str(path.relative_to(directory)), "bytes": path.stat().st_size,
             "purpose": "Q2 offline inference"} for path in directory.rglob("*") if path.is_file()]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    return sum(row["bytes"] for row in rows)


def generate_reports(root: Path):
    from .report_builder import build_reports
    return build_reports(root)
