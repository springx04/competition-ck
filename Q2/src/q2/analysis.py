"""Paired validity, deletion amount, and video-level uncertainty analyses."""
import ast
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .metrics import compute_metrics, degradation_metrics


MODALITY_INDEX = {"T": 0, "A": 1, "V": 2}
RATE_BINS = ((0, .2), (.2, .4), (.4, .6), (.6, .8), (.8, 1.0))


def _load_predictions(path: Path):
    by_scenario = defaultdict(dict)
    with Path(path).open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            for key in ("class_id", "pred_class"):
                row[key] = int(row[key])
            for key in ("true_score", "pred_score"):
                row[key] = float(row[key])
            for key in ("logits", "p_counts", "u_counts"):
                row[key] = ast.literal_eval(row[key])
            row["damaged"] = row["damaged"] == "True"
            by_scenario[row["scenario"]][row["sample_id"]] = row
    return by_scenario


def _metric(rows):
    if not rows:
        return None
    return compute_metrics([r["class_id"] for r in rows], [r["logits"] for r in rows],
                           [r["true_score"] for r in rows], [r["pred_score"] for r in rows])


def _rate(row, pattern):
    indices = [MODALITY_INDEX[c] for c in pattern]
    denominator = sum(row["u_counts"][m] for m in indices)
    return sum(row["p_counts"][m] for m in indices) / denominator if denominator else 0.0


def paired_condition_tables(predictions_csv: Path, output_dir: Path):
    """Compute same-sample comparisons before interpreting modality/position/length."""
    predictions = _load_predictions(predictions_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    patterns = ("T", "A", "V", "TA", "TV", "AV")
    positions = ("front", "middle", "rear")
    rhos = (.2, .4, .6, .8)
    name = lambda p, pos, r: f"{p}_{pos}_{r:.1f}"
    comparisons = []
    for pos in positions:
        for rho in rhos:
            comparisons.append(("modality", f"{pos}_{rho:.1f}", [(p, name(p, pos, rho)) for p in patterns]))
    for p in patterns:
        for rho in rhos:
            comparisons.append(("position", f"{p}_{rho:.1f}", [(pos, name(p, pos, rho)) for pos in positions]))
    for p in patterns:
        for pos in positions:
            comparisons.append(("span", f"{p}_{pos}", [(f"{rho:.1f}", name(p, pos, rho)) for rho in rhos]))
    paired = []
    for factor, group, conditions in comparisons:
        common = set.intersection(*(set(sample_id for sample_id, row in predictions[scenario].items()
                                            if row["damaged"]) for _, scenario in conditions))
        if not common:
            for condition, scenario in conditions:
                paired.append({"factor": factor, "group": group, "condition": condition,
                               "scenario": scenario, "n": 0, "macro_f1": "", "accuracy": "",
                               "mae": "", "pearson": "", "clean_macro_f1": "", "mean_actual_rate": ""})
            continue
        ids = sorted(common)
        clean = _metric([predictions["clean"][sample_id] for sample_id in ids])
        for condition, scenario in conditions:
            rows = [predictions[scenario][sample_id] for sample_id in ids]
            metric = _metric(rows)
            pattern = scenario.split("_")[0]
            paired.append({"factor": factor, "group": group, "condition": condition,
                           "scenario": scenario, "n": len(ids), "macro_f1": metric["macro_f1"],
                           "accuracy": metric["accuracy"], "mae": metric["mae"],
                           "pearson": metric["pearson"], "clean_macro_f1": clean["macro_f1"],
                           "mean_actual_rate": float(np.mean([_rate(row, pattern) for row in rows]))})
    with (output_dir / "paired_conditions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired[0]))
        writer.writeheader(); writer.writerows(paired)

    deletion = []
    for p in patterns:
        for pos in positions:
            for rho in rhos:
                scenario = name(p, pos, rho)
                rows = list(predictions[scenario].values())
                rates = np.asarray([_rate(row, p) for row in rows if row["damaged"]])
                deletion.append({"scenario": scenario, "n": len(rows),
                                 "damaged_n": len(rates), "no_new_damage_n": len(rows)-len(rates),
                                 "actual_rate_mean": float(rates.mean()) if len(rates) else "",
                                 "actual_rate_q25": float(np.quantile(rates, .25)) if len(rates) else "",
                                 "actual_rate_median": float(np.median(rates)) if len(rates) else "",
                                 "actual_rate_q75": float(np.quantile(rates, .75)) if len(rates) else ""})
    with (output_dir / "deletion_rates.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(deletion[0]))
        writer.writeheader(); writer.writerows(deletion)
    matched = []
    for pos in positions:
        for rho in rhos:
            for first, second in (("T", "A"), ("T", "V"), ("A", "V")):
                first_name, second_name = name(first, pos, rho), name(second, pos, rho)
                for lower, upper in RATE_BINS:
                    common = [sample_id for sample_id in predictions[first_name]
                              if lower < _rate(predictions[first_name][sample_id], first) <= upper
                              and lower < _rate(predictions[second_name][sample_id], second) <= upper]
                    for pattern, scenario in ((first, first_name), (second, second_name)):
                        rows = [predictions[scenario][sample_id] for sample_id in common]
                        metric = _metric(rows)
                        matched.append({"position": pos, "rho": rho, "comparison": first + "_vs_" + second,
                                        "rate_bin": f"({lower},{upper}]", "pattern": pattern,
                                        "n": len(rows), "macro_f1": metric["macro_f1"] if metric else "",
                                        "mae": metric["mae"] if metric else ""})
    with (output_dir / "matched_deletion_rates.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(matched[0]))
        writer.writeheader(); writer.writerows(matched)
    dual = []
    for pattern in ("TA", "TV", "AV"):
        first, second = [MODALITY_INDEX[c] for c in pattern]
        for pos in positions:
            for rho in rhos:
                scenario = name(pattern, pos, rho)
                rows = [row for row in predictions[scenario].values()
                        if row["p_counts"][first] > 0 and row["p_counts"][second] > 0]
                metric = _metric(rows)
                dual.append({"scenario": scenario, "both_modalities_damaged_n": len(rows),
                             "macro_f1": metric["macro_f1"] if metric else "",
                             "mae": metric["mae"] if metric else ""})
    with (output_dir / "dual_both_damaged.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dual[0]))
        writer.writeheader(); writer.writerows(dual)
    return {"paired_rows": len(paired), "deletion_rows": len(deletion),
            "matched_rows": len(matched), "dual_rows": len(dual)}


def video_bootstrap_difference(first_predictions: Path, second_predictions: Path,
                               scenario: str, metric_name="macro_f1", repeats=1000, seed=20260924):
    first = _load_predictions(first_predictions)[scenario]
    second = _load_predictions(second_predictions)[scenario]
    ids = sorted(set(first) & set(second))
    groups = defaultdict(list)
    for sample_id in ids:
        groups[first[sample_id]["video_id"]].append(sample_id)
    keys = list(groups)
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(repeats):
        selected = [sample_id for index in rng.integers(0, len(keys), size=len(keys))
                    for sample_id in groups[keys[index]]]
        first_metric = _metric([first[sample_id] for sample_id in selected])[metric_name]
        second_metric = _metric([second[sample_id] for sample_id in selected])[metric_name]
        if first_metric is not None and second_metric is not None:
            differences.append(first_metric - second_metric)
    return {"scenario": scenario, "metric": metric_name, "videos": len(keys),
            "repeats": len(differences), "difference_mean": float(np.mean(differences)),
            "ci_2_5": float(np.quantile(differences, .025)),
            "ci_97_5": float(np.quantile(differences, .975))}


def video_bootstrap_mean_f1(first_predictions: Path, second_predictions: Path,
                            scenario_names: list[str], repeats=1000, seed=20260924):
    """Paired source-video resampling of the equally weighted main-scenario F1."""
    first = _load_predictions(first_predictions)
    second = _load_predictions(second_predictions)
    sample_ids = sorted(set(first["clean"]) & set(second["clean"]))
    videos = sorted({first["clean"][sample_id]["video_id"] for sample_id in sample_ids})
    video_index = {video: index for index, video in enumerate(videos)}
    matrices = np.zeros((2, len(scenario_names), len(videos), 3, 3), dtype=np.int64)
    for s, scenario in enumerate(scenario_names):
        for sample_id in sample_ids:
            a, b = first[scenario][sample_id], second[scenario][sample_id]
            v = video_index[a["video_id"]]
            matrices[0, s, v, a["class_id"], a["pred_class"]] += 1
            matrices[1, s, v, b["class_id"], b["pred_class"]] += 1
    rng = np.random.default_rng(seed)
    sampled_counts = rng.multinomial(len(videos), [1 / len(videos)] * len(videos), size=repeats)
    confusion = np.einsum("rv,msvij->mrsij", sampled_counts, matrices)
    diagonal = np.diagonal(confusion, axis1=-2, axis2=-1)
    true_count = confusion.sum(axis=-1)
    pred_count = confusion.sum(axis=-2)
    f1 = np.divide(2 * diagonal, true_count + pred_count,
                   out=np.zeros_like(diagonal, dtype=np.float64), where=(true_count + pred_count) > 0)
    missing_macro = f1.mean(axis=-1).mean(axis=-1)
    differences = missing_macro[0] - missing_macro[1]
    return {"videos": len(videos), "scenarios": len(scenario_names), "repeats": repeats,
            "difference_mean": float(differences.mean()),
            "ci_2_5": float(np.quantile(differences, .025)),
            "ci_97_5": float(np.quantile(differences, .975))}


@torch.no_grad()
def reliability_analysis(root: Path, variant: str, seed: int, split="valid", batch_size=128):
    """Compare supervised error predictions with detached EMA clean targets."""
    from scipy.stats import spearmanr
    from .data import AlignedDataset, Normalizer, collate_raw
    from .masking import apply_span, evaluation_grid, make_span_mask
    from .model.network import Student, encode_view
    from .state import infer_state
    from .export import checkpoint_runtime_options, load_checkpoint_text_encoder

    root = Path(root)
    device = torch.device("cuda:0")
    checkpoint_path = root / "runs" / variant / f"seed_{seed}" / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    runtime = checkpoint_runtime_options(checkpoint, checkpoint_path)
    if checkpoint["teacher"] is None:
        return {"status": "no_ema_teacher"}
    normalizer = Normalizer.load(root / "data/processed/normalizer.npz")
    if runtime["clip_z"] is not None:
        normalizer.clip_z = runtime["clip_z"]
    student = Student(variant, normalizer.class_prior, normalizer.score_prior).to(device).eval()
    teacher = Student(variant, normalizer.class_prior, normalizer.score_prior).to(device).eval()
    student.load_state_dict(checkpoint["student"])
    teacher.load_state_dict(checkpoint["teacher"])
    if student.estimator is None:
        return {"status": "no_error_estimator"}
    encoder = load_checkpoint_text_encoder(root, variant, checkpoint, device)
    dataset = AlignedDataset(root / "data/processed" / split)
    clean_cache = np.load(root / ".cache/text" / split / "features.npy", mmap_mode="r")
    masks_dir = root / "data/masks" / split
    index = __import__("json").loads((masks_dir / "index.json").read_text(encoding="utf-8"))
    duplicates = {row["name"]: row["duplicate_of"] for row in index}
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_raw)
    predicted, actual = [], []
    for scenario in evaluation_grid():
        name = scenario["name"]
        if duplicates.get(name):
            continue
        descriptors = __import__("json").loads((masks_dir / f"{name}.json").read_text(encoding="utf-8"))
        scenario_path = root / ".cache/text" / split / "scenarios" / f"{name}.npy"
        scenario_cache = np.load(scenario_path, mmap_mode="r") if scenario_path.exists() else None
        offset = 0
        for batch in loader:
            n = len(batch["sample_id"])
            state0 = infer_state(batch)
            perturbation = make_span_mask(batch, state0, descriptors[offset:offset+n])
            damaged = apply_span(batch, perturbation)
            clean = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k,v in batch.items()}
            corrupt = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k,v in damaged.items()}
            cached = torch.from_numpy(np.asarray(clean_cache[offset:offset+n]).copy()).to(device)
            clean_input = encode_view(clean, encoder, normalizer, cached,
                                      cls_context=runtime["cls_context"])
            teacher_output = teacher(clean_input, return_details=True)
            corrupt_cached = torch.from_numpy(np.asarray(scenario_cache[offset:offset+n]).copy()).to(device) if scenario_cache is not None else cached
            damaged_input = encode_view(corrupt, encoder, normalizer, corrupt_cached,
                                        perturbation.P[:, :, 0].any(dim=1).to(device) if scenario_cache is None else None,
                                        cls_context=runtime["cls_context"])
            student_output = student(damaged_input, return_details=True)
            omega = perturbation.P.to(device) & state0.U.to(device) & student_output.B_comp
            if omega.any():
                residual = (F.layer_norm(student_output.F_hat.float(), (128,), eps=1e-5)
                            - F.layer_norm(teacher_output.F.detach().float(), (128,), eps=1e-5)).square().mean(-1)
                predicted.extend(student_output.e_hat[omega].cpu().tolist())
                actual.extend(residual[omega].cpu().tolist())
            offset += n
    if len(predicted) < 2:
        return {"status": "insufficient_omega", "count": len(predicted)}
    predicted, actual = np.asarray(predicted), np.asarray(actual)
    correlation = spearmanr(predicted, actual).statistic
    quantiles = np.unique(np.quantile(predicted, np.linspace(0, 1, 6)))
    bins = []
    for low, high in zip(quantiles[:-1], quantiles[1:]):
        chosen = (predicted >= low) & (predicted <= high if high == quantiles[-1] else predicted < high)
        bins.append({"lower": float(low), "upper": float(high), "count": int(chosen.sum()),
                     "actual_residual_mean": float(actual[chosen].mean()) if chosen.any() else None})
    return {"status": "complete", "count": len(predicted), "spearman": float(correlation),
            "mae": float(np.abs(predicted - actual).mean()), "bins": bins}
