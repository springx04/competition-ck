"""Reproduce the fixed three-weight auxiliary study from completed test readouts."""
import csv
import json
from pathlib import Path

from q2.masking import evaluation_grid


def read_history(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main():
    root = Path(__file__).resolve().parents[1]
    scenarios = {row["name"] for row in evaluation_grid(stress=False)}
    results = []
    for weight in ("0", "0.05", "0.2"):
        name = f"attn_aux{weight}_v1"
        summaries = sorted((root / "reports/exploratory" / name).glob("*/summary.json"))
        evaluations = [json.loads(path.read_text(encoding="utf-8")) for path in summaries]
        result = next(row for row in reversed(evaluations) if row["split"] == "test")
        directory = root / result["directory"]
        rows = read_history(directory / "metrics_per_scenario.csv")
        main_rows = [row for row in rows if row["scope"] == "all_samples" and row["scenario"] in scenarios]
        if len(main_rows) != len(scenarios) or {row["scenario"] for row in main_rows} != scenarios:
            raise ValueError(f"incomplete main scenario grid: {directory}")
        clean = next(row for row in rows if row["scope"] == "all_samples" and row["scenario"] == "clean")
        result["missing_macro_f1"] = sum(float(row["macro_f1"]) for row in main_rows) / len(main_rows)
        result["clean_macro_f1"] = float(clean["macro_f1"])
        result["main_scenario_count"] = len(main_rows)
        result["auxiliary_weight"] = float(weight)
        run_dir = root / "experiments" / name / "runs/late_aux_tune/seed_1111"
        result["resource_usage"] = json.loads((run_dir / "resource_usage.json").read_text())
        results.append(result)

    reference = read_history(root / "experiments/attn_gridmix_reg1_v1/runs/late_attn_tune/seed_1111/history.csv")
    control = read_history(root / "experiments/attn_aux0_v1/runs/late_aux_tune/seed_1111/history.csv")
    if [row["epoch"] for row in reference] != [row["epoch"] for row in control]:
        raise ValueError("zero-weight control and reference epochs differ")
    output = {
        "independent_holdout": False,
        "selection": "Each saved best checkpoint was chosen on valid; all three weights received test evaluation.",
        "zero_weight_max_total_loss_delta": max(abs(float(a["total"]) - float(b["total"]))
                                                for a, b in zip(reference, control)),
        "results": results,
    }
    path = root / "reports/auxiliary_study_summary.json"
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output), flush=True)


if __name__ == "__main__":
    main()
