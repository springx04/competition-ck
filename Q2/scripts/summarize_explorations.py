"""Summarize retained validation runs without evaluating or reading test data."""
import csv
import json
from pathlib import Path
from q2.evaluate import better_key

root = Path(__file__).resolve().parents[1]
results = []
for history in sorted((root / "experiments").glob("*/runs/*/seed_*/history.csv")):
    rows = list(csv.DictReader(history.open(encoding="utf-8")))
    best = None
    for row in rows:
        if row["missing_macro_f1"]:
            candidate = tuple(float(row[key]) for key in
                ("missing_macro_f1", "missing_mae", "clean_macro_f1", "clean_mae")) + (int(row["epoch"]),)
            if better_key(candidate, best):
                best = candidate
    usage_path = history.parent / "resource_usage.json"
    usage = json.loads(usage_path.read_text()) if usage_path.exists() else None
    results.append({"experiment": history.parents[3].name,
                    "variant": history.parents[1].name, "seed": history.parent.name,
                    "completed": usage is not None,
                    "last_logged_epoch": int(rows[-1]["epoch"]) if rows else 0,
                    "selection_key": best, "resource_usage": usage,
                    "history": str(history.relative_to(root))})
output = root / "reports/optimization_experiments_20260924.json"
output.parent.mkdir(exist_ok=True)
output.write_text(json.dumps({"split": "valid", "results": results}, indent=2), encoding="utf-8")
for result in results:
    print(json.dumps(result, ensure_ascii=False))
