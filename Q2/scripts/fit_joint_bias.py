"""Reproduce the exploratory joint-model bias search using only saved valid logits."""
import csv
import itertools
import json
from pathlib import Path

import numpy as np

from q2.masking import evaluation_grid


def main():
    root = Path(__file__).resolve().parents[1]
    source = root / "experiments/attn_gridmix_reg1_v1/runs/late_attn_tune/seed_1111/best_validation/predictions.csv"
    with source.open() as handle:
        rows = list(csv.DictReader(handle))
    names = ["clean"] + [s["name"] for s in evaluation_grid()]
    grouped = [[r for r in rows if r["scenario"] == name] for name in names]
    logits = np.array([[json.loads(r["logits"]) for r in group] for group in grouped])
    labels = np.array([[int(r["class_id"]) for r in group] for group in grouped])
    observed = np.array([[sum(json.loads(r["u_counts"])) > sum(json.loads(r["p_counts"]))
                          for r in group] for group in grouped])
    best = None
    for a, b in itertools.product(np.arange(-.6, .601, .05), repeat=2):
        prediction = (logits + observed[..., None] * [a, b, 0]).argmax(-1)
        f1 = []
        for cls in (0, 1, 2):
            tp = ((labels == cls) & (prediction == cls)).sum(-1)
            denominator = (labels == cls).sum(-1) + (prediction == cls).sum(-1)
            f1.append(np.divide(2 * tp, denominator, out=np.zeros_like(tp, dtype=float), where=denominator > 0))
        scenario_f1 = np.mean(f1, axis=0)
        clean, missing = float(scenario_f1[0]), float(scenario_f1[1:].mean())
        candidate = ((clean + missing) / 2, missing, clean, float(a), float(b))
        if best is None or candidate > best:
            best = candidate
    result = {"source": str(source.relative_to(root)), "split": "valid", "independent_holdout": False,
              "objective": "half clean Macro-F1 plus half mean of 72 main-scenario Macro-F1",
              "tie_break": "descending missing F1, clean F1, first bias, second bias",
              "valid_missing": best[1], "valid_clean": best[2], "class_bias": [round(best[3], 2), round(best[4], 2), 0]}
    (root / "reports/joint_bias_selection.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
