"""Summarize observed test experiments and average uncorrected ensemble logits."""
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score

from q2.masking import evaluation_grid


def main():
    root = Path(__file__).resolve().parents[1]
    reports = root / "reports"
    names = ["clean"] + [r["name"] for r in evaluation_grid()]
    bias = np.array([-.1, .15, 0])  # Already chosen on seed1111 valid, never fit here.
    sources = ["gridmix_test_combo", "gridmix_reg1_test_seed1112", "gridmix_reg1_test_seed1113"]
    datasets = []
    for source in sources:
        with (reports / source / "predictions.csv").open() as handle:
            datasets.append({(r["scenario"], r["sample_id"]): r for r in csv.DictReader(handle)})
    metrics = {}
    for scenario in names:
        keys = sorted(k for k in datasets[0] if k[0] == scenario)
        logits = np.mean([[json.loads(ds[k]["logits"]) for k in keys] for ds in datasets], axis=0)
        # These sources contain uncorrected logits; apply bias exactly once,
        # and keep the same empty-content exception as the deployed model.
        observed = np.array([sum(json.loads(datasets[0][k]["u_counts"])) >
                             sum(json.loads(datasets[0][k]["p_counts"])) for k in keys])
        labels = [int(datasets[0][k]["class_id"]) for k in keys]
        predictions = (logits + observed[:, None] * bias).argmax(-1)
        metrics[scenario] = float(f1_score(labels, predictions, labels=[0, 1, 2], average="macro"))
        if scenario == "clean":
            clean_detail = {"per_class_f1": f1_score(labels, predictions, labels=[0, 1, 2], average=None).tolist(),
                            "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1, 2]).tolist()}
    result = {"independent_holdout": False, "ensemble": "equal mean of raw logits, not mean probabilities",
              "sources": sources, "class_bias": bias.tolist(), "bias_applications": 1,
              "missing_macro_f1": float(np.mean([metrics[s] for s in names[1:]])),
              "clean_macro_f1": metrics["clean"], "clean_details": clean_detail,
              "per_scenario": metrics}
    (reports / "joint_ensemble_corrected.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "per_scenario"}), flush=True)


if __name__ == "__main__":
    main()
