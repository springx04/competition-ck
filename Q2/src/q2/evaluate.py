"""Fixed-mask evaluation on the official aligned position axis."""
from __future__ import annotations
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import AlignedDataset, collate_raw
from .masking import apply_span, evaluation_grid, make_span_mask
from .metrics import compute_metrics, degradation_metrics
from .model.network import encode_view
from .state import infer_state


def _device_batch(batch, device):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def make_fixed_masks(dataset: AlignedDataset, output_dir: Path, include_stress=True):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    grid = evaluation_grid() + (evaluation_grid(stress=True) if include_stress else [])
    signatures = []
    overview = []
    samples = [collate_raw([dataset[index]]) for index in range(len(dataset))]
    states = [infer_state(sample) for sample in samples]
    for scenario in grid:
        records = []
        all_p = []
        for index in range(len(dataset)):
            sample = samples[index]
            mask = make_span_mask(sample, states[index], [scenario])
            record = {"sample_id": sample["sample_id"][0], "sample_index": index,
                      **mask.descriptors[0], "original_counts": states[index].U[0].sum(dim=0).tolist()}
            records.append(record)
            all_p.append(mask.P[0].numpy())
        pattern = np.stack(all_p)
        duplicate = next((grid[j]["name"] for j, previous in enumerate(signatures)
                          if np.array_equal(pattern, previous)), None)
        signatures.append(pattern)
        for record in records:
            record["duplicate_of"] = duplicate
        (output_dir / f"{scenario['name']}.json").write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
        overview.append({"name": scenario["name"], "duplicate_of": duplicate,
                         "damaged_samples": int(pattern.any(axis=(1, 2)).sum()), "samples": len(dataset)})
    (output_dir / "index.json").write_text(json.dumps(overview, ensure_ascii=False, indent=2), encoding="utf-8")
    return overview


def _read_descriptors(mask_dir: Path, name: str):
    return json.loads((mask_dir / f"{name}.json").read_text(encoding="utf-8"))


@torch.no_grad()
def evaluate_model(student, text_encoder, normalizer, dataset: AlignedDataset, device,
                   mask_dir: Path, clean_cache: np.ndarray | None, batch_size=128,
                   output_dir: Path | None = None, stress=False, return_predictions=False,
                   cls_context=False, class_bias=None):
    student.eval()
    text_encoder.eval()
    if class_bias is not None:
        class_bias = np.asarray(class_bias, dtype=np.float32)
        if class_bias.shape != (3,) or not np.isfinite(class_bias).all():
            raise ValueError("class_bias must be a finite length-3 vector")
    scenarios = [{"name": "clean"}] + evaluation_grid() + (evaluation_grid(stress=True) if stress else [])
    index = json.loads((Path(mask_dir) / "index.json").read_text(encoding="utf-8"))
    duplicates = {entry["name"]: entry["duplicate_of"] for entry in index}
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_raw)
    metric_rows, prediction_rows = [], []
    # Scoped to this call only: an A/V-only perturbation cannot change BERT's
    # input. Reuse live clean features without persisting fine-tuned features.
    live_clean_cache = (np.empty((len(dataset), 50, 256), dtype=np.float32)
                        if clean_cache is None else None)
    scenario_outputs = {}
    for scenario in scenarios:
        name = scenario["name"]
        if name != "clean" and duplicates.get(name) in scenario_outputs:
            scenario_outputs[name] = scenario_outputs[duplicates[name]]
            continue
        descriptors = _read_descriptors(mask_dir, name) if name != "clean" else None
        text_affected = name != "clean" and "T" in scenario["pattern"]
        scenario_cache = None
        scenario_cache_new = False
        # None means the encoder was fine-tuned: every view must use its current
        # weights, including text-corrupted scenarios. Never open frozen caches.
        if text_affected and clean_cache is not None:
            cache_dir = Path(mask_dir).parents[2] / ".cache/text" / dataset.directory.name / "scenarios"
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = cache_dir / f"{name}.npy"
            if cache_path.exists():
                scenario_cache = np.load(cache_path, mmap_mode="r")
            else:
                scenario_cache = np.lib.format.open_memmap(cache_path, mode="w+", dtype="float32",
                                                           shape=(len(dataset), 50, 256))
                scenario_cache_new = True
        y_class, y_score, logits, scores, damaged, p_counts, u_counts, empty = [], [], [], [], [], [], [], []
        offset = 0
        for batch in loader:
            n = len(batch["sample_id"])
            clean = batch
            state0 = infer_state(clean)
            if descriptors is None:
                raw = clean
                p = torch.zeros_like(state0.U)
            else:
                perturbation = make_span_mask(clean, state0, descriptors[offset:offset+n])
                raw = apply_span(clean, perturbation)
                p = perturbation.P
            selected_cache = scenario_cache if scenario_cache is not None and not scenario_cache_new else clean_cache
            if clean_cache is None and name != "clean":
                selected_cache = live_clean_cache
            cache = (torch.from_numpy(np.asarray(selected_cache[offset:offset+n]).copy()).to(device)
                     if selected_cache is not None else None)
            raw_device = _device_batch(raw, device)
            model_input = encode_view(raw_device, text_encoder, normalizer, cache,
                                      p[:, :, 0].any(dim=1).to(device)
                                      if scenario_cache_new or (clean_cache is None and text_affected) else None,
                                      cls_context=cls_context)
            if live_clean_cache is not None and name == "clean":
                live_clean_cache[offset:offset+n] = model_input.text_features.cpu().numpy()
            if scenario_cache_new:
                scenario_cache[offset:offset+n] = model_input.text_features.cpu().numpy()
            output = student(model_input)
            if class_bias is not None:
                output.logits = output.logits + torch.as_tensor(
                    class_bias, device=output.logits.device, dtype=output.logits.dtype)
            y_class.extend(batch["class_id"].tolist())
            y_score.extend(batch["score"].tolist())
            logits.extend(output.logits.cpu().tolist())
            scores.extend(output.score.cpu().tolist())
            damaged.extend(p.any(dim=(1, 2)).tolist())
            p_counts.extend(p.sum(dim=1).tolist())
            u_counts.extend(state0.U.sum(dim=1).tolist())
            empty.extend((~output.J.any(dim=1)).cpu().tolist())
            offset += n
        if scenario_cache_new:
            scenario_cache.flush()
        result = {"class_id": np.asarray(y_class), "true_score": np.asarray(y_score),
                  "logits": np.asarray(logits), "pred_score": np.asarray(scores),
                  "damaged": np.asarray(damaged), "p_counts": np.asarray(p_counts),
                  "u_counts": np.asarray(u_counts), "empty": np.asarray(empty)}
        scenario_outputs[name] = result
    clean = scenario_outputs["clean"]
    clean_metrics = compute_metrics(clean["class_id"], clean["logits"], clean["true_score"], clean["pred_score"])
    for scenario in scenarios:
        name = scenario["name"]
        result = scenario_outputs[name]
        for scope in (("all_samples", "damaged_only", "original_visual_empty")
                      if name != "clean" else ("all_samples", "original_visual_empty")):
            selected = (np.ones(len(dataset), dtype=bool) if scope == "all_samples" else
                        result["damaged"] if scope == "damaged_only" else
                        result["u_counts"][:, 2] == 0)
            if not selected.any():
                continue
            current = compute_metrics(result["class_id"][selected], result["logits"][selected],
                                      result["true_score"][selected], result["pred_score"][selected])
            base = compute_metrics(clean["class_id"][selected], clean["logits"][selected],
                                   clean["true_score"][selected], clean["pred_score"][selected])
            metric_rows.append({"scenario": name, "scope": scope, "n": int(selected.sum()),
                                "no_new_damage": int((~result["damaged"]).sum()),
                                "empty_content": int(result["empty"][selected].sum()),
                                "original_modality_empty_T": int((result["u_counts"][selected, 0] == 0).sum()),
                                "original_modality_empty_A": int((result["u_counts"][selected, 1] == 0).sum()),
                                "original_modality_empty_V": int((result["u_counts"][selected, 2] == 0).sum()),
                                "duplicate_of": duplicates.get(name),
                                "metrics": current, "degradation": degradation_metrics(base, current)})
        if return_predictions or output_dir is not None:
            for i in range(len(dataset)):
                prediction_rows.append({"sample_id": dataset.metadata["id"][i], "video_id": dataset.metadata["video_id"][i],
                    "scenario": name, "class_id": int(result["class_id"][i]), "true_score": float(result["true_score"][i]),
                    "pred_class": int(result["logits"][i].argmax()), "pred_score": float(result["pred_score"][i]),
                    "logits": result["logits"][i].tolist(), "damaged": bool(result["damaged"][i]),
                    "p_counts": result["p_counts"][i].tolist(), "u_counts": result["u_counts"][i].tolist()})
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        flat = []
        for row in metric_rows:
            metric = row["metrics"]
            flat.append({k: v for k, v in row.items() if k not in ("metrics", "degradation")}
                        | {k: v for k, v in metric.items() if isinstance(v, (int, float, str)) or v is None}
                        | row["degradation"])
        with (output_dir / "metrics_per_scenario.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
            writer.writeheader(); writer.writerows(flat)
        with (output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(prediction_rows[0]))
            writer.writeheader(); writer.writerows(prediction_rows)
    return metric_rows, prediction_rows


def selection_key(rows, epoch):
    main = [r for r in rows if r["scope"] == "all_samples" and r["scenario"] != "clean"
            and r["duplicate_of"] is None and not r["scenario"].endswith(("whole", "staggered_0.4"))
            and not r["scenario"].startswith("TAV_")]
    clean = next(r for r in rows if r["scenario"] == "clean")
    return (float(np.mean([r["metrics"]["macro_f1"] for r in main])),
            float(np.mean([r["metrics"]["mae"] for r in main])),
            clean["metrics"]["macro_f1"], clean["metrics"]["mae"], epoch)


def better_key(candidate, current, tolerance=1e-6):
    if current is None:
        return True
    for index, direction in enumerate((1, -1, 1, -1)):
        delta = (candidate[index] - current[index]) * direction
        if delta > tolerance:
            return True
        if delta < -tolerance:
            return False
    return candidate[4] < current[4]
