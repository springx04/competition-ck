"""Generate clean-test diagnostic figures from the exported Q2 inference package.

The script deliberately reloads the exported FP16-storage / FP32-compute
package, rather than a training checkpoint.  It evaluates only the official
attachment-2 test split and saves the per-sample outputs used by both figures.
"""
from __future__ import annotations

import argparse
import csv
import importlib
import json
from pathlib import Path
import sys
import tempfile
import zipfile

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, mean_absolute_error


ROOT = Path(__file__).resolve().parents[1]
CLASS_NAMES = ("Negative", "Neutral", "Positive")


def _bundle_directory(bundle_path: Path, temporary: str) -> Path:
    """Return a package directory, safely unpacking a delivery zip if needed."""
    if bundle_path.is_dir():
        return bundle_path
    if not zipfile.is_zipfile(bundle_path):
        raise ValueError(f"bundle must be a directory or zip file: {bundle_path}")
    target = Path(temporary)
    with zipfile.ZipFile(bundle_path) as archive:
        for member in archive.infolist():
            resolved = (target / member.filename).resolve()
            if target.resolve() not in resolved.parents and resolved != target.resolve():
                raise ValueError(f"unsafe archive path: {member.filename}")
        archive.extractall(target)
    candidates = [path for path in target.iterdir() if path.is_dir() and (path / "ensemble_config.json").is_file()]
    if len(candidates) != 1:
        raise ValueError("delivery zip must contain exactly one ensemble package directory")
    return candidates[0]


def _raw_batch(record: dict, indices: slice) -> dict[str, torch.Tensor]:
    return {
        key: torch.from_numpy(np.ascontiguousarray(record[key][indices]))
        for key in ("input_ids", "stored_attention", "token_type_ids", "audio", "vision")
    }


def _predict(bundle, record: dict, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    logits, scores = [], []
    for start in range(0, len(record["input_ids"]), batch_size):
        output = bundle.predict_raw(_raw_batch(record, slice(start, start + batch_size)), return_details=False)
        logits.append(output.logits.detach().cpu().numpy())
        scores.append(output.score.detach().cpu().numpy())
    return np.concatenate(logits), np.concatenate(scores)


def _plot_confusion(matrix: np.ndarray, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    row_totals = matrix.sum(axis=1, keepdims=True)
    rates = np.divide(matrix, row_totals, out=np.zeros_like(matrix, dtype=float), where=row_totals != 0)
    figure, axis = plt.subplots(figsize=(7.5, 6.2), constrained_layout=True)
    image = axis.imshow(rates, cmap="Blues", vmin=0, vmax=1)
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="Row-normalized proportion")
    axis.set_xticks(range(3), CLASS_NAMES)
    axis.set_yticks(range(3), CLASS_NAMES)
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")
    axis.set_title("Q2 clean test: classification confusion matrix (n = 727)")
    for row in range(3):
        for col in range(3):
            rate = rates[row, col]
            text_color = "white" if rate >= 0.52 else "#172554"
            axis.text(col, row, f"{matrix[row, col]}\n{rate:.1%}", ha="center", va="center",
                      color=text_color, fontsize=13, fontweight="bold" if row == col else "normal")
    axis.text(1, 3.06, "Each row sums to 100%; off-diagonal cells show the direction of errors.",
              ha="center", va="top", fontsize=10, color="#334155", transform=axis.transData)
    figure.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(figure)


def _plot_regression(true_score: np.ndarray, pred_score: np.ndarray, pearson: float, mae: float,
                     output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7.5, 6.2), constrained_layout=True)
    generator = np.random.default_rng(20260925)
    jittered_true = true_score + generator.uniform(-0.025, 0.025, size=len(true_score))
    axis.scatter(jittered_true, pred_score, s=23, alpha=0.38, color="#2563eb", edgecolors="none",
                 label="Test sample")
    lower, upper = -3.05, 3.05
    axis.plot((lower, upper), (lower, upper), color="#64748b", linewidth=1.5, linestyle="--",
              label="Ideal prediction")
    slope, intercept = np.polyfit(true_score, pred_score, deg=1)
    x_line = np.linspace(lower, upper, 100)
    axis.plot(x_line, slope * x_line + intercept, color="#dc2626", linewidth=2,
              label="Least-squares fit")
    axis.set(xlim=(lower, upper), ylim=(lower, upper), xlabel="True sentiment intensity",
             ylabel="Predicted sentiment intensity",
             title="Q2 clean test: regression prediction against ground truth")
    axis.set_aspect("equal", adjustable="box")
    axis.grid(alpha=0.22)
    axis.legend(loc="upper left")
    axis.text(0.97, 0.05, f"Pearson r = {pearson:.3f}\nMAE = {mae:.3f}", transform=axis.transAxes,
              ha="right", va="bottom", fontsize=11,
              bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#94a3b8"})
    figure.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT.parents[0] / "E题数据" / "附件2-数据集特征文件" / "aligned_50.pkl")
    parser.add_argument("--bundle", type=Path, default=ROOT / "delivery" / "q2_final.zip")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "final" / "figures")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    if not args.data.is_file():
        raise FileNotFoundError(args.data)
    if not args.bundle.exists():
        raise FileNotFoundError(args.bundle)

    with tempfile.TemporaryDirectory(prefix="q2_final_diagnostics_") as temporary:
        package = _bundle_directory(args.bundle, temporary)
        sys.path.insert(0, str(package / "src"))
        data_module = importlib.import_module("q2.data")
        ensemble_module = importlib.import_module("q2.ensemble")
        record = data_module.unpack_record(data_module.load_pickle(args.data)["test"], "attachment2")
        bundle = ensemble_module.load_ensemble(package, args.device)
        logits, pred_score = _predict(bundle, record, args.batch_size)

    true_class = record["class_id"].astype(np.int64, copy=False)
    true_score = record["score"].astype(float, copy=False)
    pred_class = logits.argmax(axis=1)
    matrix = confusion_matrix(true_class, pred_class, labels=(0, 1, 2))
    pearson = float(np.corrcoef(true_score, pred_score)[0, 1])
    mae = float(mean_absolute_error(true_score, pred_score))
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    _plot_confusion(matrix, output_dir / "test_clean_confusion_matrix.png")
    _plot_regression(true_score, pred_score, pearson, mae, output_dir / "test_clean_regression_scatter.png")

    with (output_dir / "test_clean_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("sample_id", "true_class", "pred_class", "true_score", "pred_score"))
        writer.writeheader()
        for index in range(len(true_class)):
            writer.writerow({"sample_id": record["id"][index], "true_class": int(true_class[index]),
                             "pred_class": int(pred_class[index]), "true_score": float(true_score[index]),
                             "pred_score": float(pred_score[index])})
    payload = {
        "source": "official attachment2 test split; reloaded exported q2_final package",
        "samples": int(len(true_class)), "class_order": list(CLASS_NAMES),
        "confusion_matrix_rows_true_columns_predicted": matrix.tolist(),
        "neutral_predictions": {CLASS_NAMES[index]: int(matrix[1, index]) for index in range(3)},
        "mae": mae, "pearson": pearson,
        "device": args.device,
    }
    (output_dir / "test_clean_diagnostics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
