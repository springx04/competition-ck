"""Exploratory test readout of a fixed three-seed grid-mix comparison."""
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics

import torch

from q2.__main__ import evaluate_best
from q2.evaluate import selection_key
from q2.analysis import video_bootstrap_mean_f1
from q2.masking import evaluation_grid


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / "reports" / ("gridmix_test_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    output.mkdir()
    controls = ["attn_tune_v3", "attn_cls_seed1112_v1", "attn_cls_seed1113_v1"]
    candidates = ["attn_gridmix_v1", "attn_gridmix_seed1112_v1", "attn_gridmix_seed1113_v1"]
    cohort = []
    for kind, names in (("original", controls), ("grid_mix", candidates)):
        for seed, name in zip((1111, 1112, 1113), names):
            path = root / "experiments" / name / "runs/late_attn_tune" / f"seed_{seed}/best.pt"
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            cohort.append({"name": name, "kind": kind, "variant": "late_attn_tune", "seed": seed,
                           "checkpoint": str(path.relative_to(root)), "epoch": checkpoint["epoch"],
                           "class_bias": None, "valid_selection_key": checkpoint["best_key"]})
    # Record the entire comparison before loading any test examples. Existing
    # test observations remain disclosed; this does not restore independence.
    manifest = {"split": "test", "independent_holdout": False,
                "reason": "User requested test comparison after earlier test observations; all three seeds reported.",
                "selection": "Previously saved valid-best checkpoint, no class bias, unchanged 72 main scenarios",
                "cohort": cohort}
    (output / "cohort.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    results = []
    for item in cohort:
        checkpoint = torch.load(root / item["checkpoint"], map_location="cpu", weights_only=False)
        rows, _ = evaluate_best(root, checkpoint["config"], item["variant"], item["seed"],
                                split="test", stress=False, output_dir=output / item["name"], selection=item)
        key = selection_key(rows, item["epoch"])
        result = {**item, **dict(zip(("missing_macro_f1", "missing_mae", "clean_macro_f1", "clean_mae"), key[:4]))}
        results.append(result)
        (output / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(json.dumps(result), flush=True)
    means = {kind: {metric: statistics.mean(r[metric] for r in results if r["kind"] == kind)
                    for metric in ("missing_macro_f1", "clean_macro_f1", "missing_mae", "clean_mae")}
             for kind in ("original", "grid_mix")}
    main_names = {r["name"] for r in evaluation_grid()}
    masks = json.loads((root / "data/masks/test/index.json").read_text())
    names = [r["name"] for r in masks if r["name"] in main_names and not r["duplicate_of"]]
    paired = [{"seed": seed, **video_bootstrap_mean_f1(output / candidate / "predictions.csv",
                output / control / "predictions.csv", names)}
              for seed, control, candidate in zip((1111, 1112, 1113), controls, candidates)]
    summary = {"independent_holdout": False, "means": means, "paired_video_bootstrap": paired}
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"directory": str(output), **summary}), flush=True)


if __name__ == "__main__":
    main()
