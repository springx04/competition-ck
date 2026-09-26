"""Export, assess and verify the selected Q2 ensemble; report its actual stored weights."""
import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile

import numpy as np
import torch

from q2.analysis import paired_condition_tables, video_bootstrap_mean_f1
from q2.data import AlignedDataset, collate_raw, find_inputs
from q2.ensemble import export_ensemble, load_ensemble, load_training_ensemble
from q2.evaluate import evaluate_model, selection_key
from q2.export import predict_special
from q2.masking import apply_span, evaluation_grid, make_span_mask
from q2.metrics import compute_metrics
from q2.state import infer_state


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/final"
PACKAGE = ROOT / "delivery/q2_final"


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def prepare(selection):
    result = export_ensemble(ROOT, selection, PACKAGE)
    write(RESULTS / "export.json", result)
    write(RESULTS / "selection.json", selection)


def assess(selection):
    bundle = load_ensemble(PACKAGE, "cuda:0")
    resource = {}
    for split in ("valid", "test"):
        out = ROOT / "reports/final" / split
        out.mkdir(parents=True, exist_ok=True)
        dataset = AlignedDataset(ROOT / "data/processed" / split)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.perf_counter()
        rows, _ = evaluate_model(None, None, None, dataset, "cuda:0", ROOT / "data/masks" / split,
                                 None, batch_size=128, output_dir=out, stress=True, predictor=bundle)
        torch.cuda.synchronize()
        resource[split] = {"evaluation_seconds": time.perf_counter() - start,
                           "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                           "scenarios": 1 + len(evaluation_grid()) + len(evaluation_grid(stress=True))}
        write(RESULTS / split / "metrics.json", rows)
        write(RESULTS / split / "summary.json", dict(zip(
            ("missing_macro_f1", "missing_mae", "clean_macro_f1", "clean_mae"), selection_key(rows, 0)[:4])))
        paired_condition_tables(out / "predictions.csv", out)
        (RESULTS / split).mkdir(parents=True, exist_ok=True)
        for path in out.glob("*.csv"):
            if path.name != "predictions.csv":
                shutil.copy2(path, RESULTS / split / path.name)
    dataset = AlignedDataset(ROOT / "data/processed/test")
    from torch.utils.data import DataLoader
    empty_logits, empty_scores, labels, targets = [], [], [], []
    for batch in DataLoader(dataset, batch_size=128, collate_fn=collate_raw):
        desc = [{"pattern": "TAV", "position": "middle", "rho": 1.0} for _ in batch["sample_id"]]
        empty = apply_span(batch, make_span_mask(batch, infer_state(batch), desc))
        output = bundle.predict_raw(empty, return_details=False)
        if output.U.any():
            raise AssertionError("full-empty check retained observed content")
        empty_logits.extend(output.logits.cpu().tolist())
        empty_scores.extend(output.score.cpu().tolist())
        labels.extend(batch["class_id"].tolist())
        targets.extend(batch["score"].tolist())
    write(RESULTS / "test/all_empty.json", compute_metrics(labels, empty_logits, targets, empty_scores))
    raw = collate_raw([dataset[i] for i in range(32)])
    for _ in range(3):
        bundle.predict_raw(raw, return_details=False)
    torch.cuda.synchronize()
    times = []
    for _ in range(10):
        start = time.perf_counter()
        bundle.predict_raw(raw, return_details=False)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    resource["warm_batch32_latency"] = {"median_seconds": float(np.median(times)),
                                         "samples_per_second": 32 / float(np.median(times)),
                                         "repeats": 10, "includes_cpu_to_gpu": True}
    write(RESULTS / "inference_resources.json", resource)
    checkpoint = torch.load(ROOT / selection["members"][0]["checkpoint"], map_location="cpu", weights_only=False)
    inputs = find_inputs(Path(checkpoint["config"]["project"]["data_root"]))
    special_dir = inputs["special"][0].parent
    special_path = RESULTS / "q2_predictions_aligned.csv"
    predict_special(bundle, special_dir, special_path)
    (ROOT / "outputs").mkdir(exist_ok=True)
    for name in ("q2_predictions_aligned.csv", "q2_special_state.csv"):
        shutil.copy2(RESULTS / name, ROOT / "outputs" / name)
    shutil.copy2(special_path, PACKAGE / special_path.name)
    write(RESULTS / "special_input.json", {"directory": str(special_dir), "labels_available": False})


def verify(selection):
    dataset = AlignedDataset(ROOT / "data/processed/test")
    indices = list(range(8))
    zero = next(i for i in range(len(dataset)) if not dataset[i]["vision"].any())
    indices.append(zero)
    raw = collate_raw([dataset[i] for i in indices])
    descriptors = [{"pattern": "TAV", "position": "middle", "rho": 1.0} for _ in indices]
    damaged = apply_span(raw, make_span_mask(raw, infer_state(raw), descriptors))
    partial = apply_span(raw, make_span_mask(raw, infer_state(raw),
                        [{"pattern": p, "position": "middle", "rho": .4}
                         for p in ("T", "A", "V", "TA", "TV", "AV", "T", "V", "TV")]))
    expected = load_training_ensemble(ROOT, selection, "cuda:0")
    arrays = {}
    reference = {}
    for prefix, batch in (("clean", raw), ("partial", partial), ("empty", damaged)):
        for key in ("input_ids", "stored_attention", "token_type_ids", "audio", "vision"):
            arrays[prefix + "_" + key] = batch[key].numpy()
        output = expected.predict_raw(batch, return_details=False)
        reference[prefix] = (output.logits.cpu().numpy(), output.score.cpu().numpy())
    del expected
    with tempfile.TemporaryDirectory(prefix="q2-final-offline-") as td:
        td = Path(td)
        np.savez(td / "inputs.npz", **arrays)
        code = '''from pathlib import Path
import sys, numpy as np, torch
package, source, dest = sys.argv[1:]
sys.path.insert(0,str(Path(package)/"src"))
import q2
from q2.ensemble import load_ensemble
b=load_ensemble(Path(package),"cuda:0")
a=np.load(source); results={}
for prefix in ("clean","partial","empty"):
 raw={k:torch.from_numpy(a[prefix+"_"+k].copy()) for k in ("input_ids","stored_attention","token_type_ids","audio","vision")}
 o=b.predict_raw(raw,return_details=False)
 results[prefix+"_logits"]=o.logits.cpu().numpy();results[prefix+"_score"]=o.score.cpu().numpy()
np.savez(dest,**results)
print(q2.__file__)
'''
        env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "PYTHONPATH": ""}
        proc = subprocess.run([sys.executable, "-c", code, str(PACKAGE), str(td / "inputs.npz"),
                               str(td / "actual.npz")], cwd=td, env=env, capture_output=True, text=True, check=True)
        if not proc.stdout.strip().startswith(str(PACKAGE / "src")):
            raise AssertionError("isolated process imported external source code")
        actual = np.load(td / "actual.npz")
        comparisons = {}
        for prefix, (logits, score) in reference.items():
            comparisons[prefix] = {"max_logit_abs_difference": float(np.abs(logits - actual[prefix + "_logits"]).max()),
                                   "max_score_abs_difference": float(np.abs(score - actual[prefix + "_score"]).max()),
                                   "labels_equal": bool(np.array_equal(logits.argmax(-1), actual[prefix + "_logits"].argmax(-1)))}
        # Rounding differences are measured, not hidden; paper scores use the reloaded bundle.
        special_dir = json.loads((RESULTS / "special_input.json").read_text())["directory"]
        subprocess.run([sys.executable, str(PACKAGE / "run_inference.py"), "--input-dir", special_dir,
                        "--output", str(td / "special.csv"), "--device", "cuda:0"],
                       cwd=td, env=env, check=True, capture_output=True, text=True)
        original = list(csv.DictReader((RESULTS / "q2_predictions_aligned.csv").open()))
        isolated = list(csv.DictReader((td / "special.csv").open()))
        if len(original) != 30 or original != isolated:
            raise AssertionError("isolated special predictions differ from final bundle outputs")
        write(RESULTS / "offline_verification.json", {"passed": True, "isolated_import": proc.stdout.strip(),
              "special_rows": 30, "special_identical": True, "fp32_checkpoint_vs_fp16_storage": comparisons,
              "paper_metrics": "Full valid/test recomputation from the actual reloaded package"})
    sizes = [{"path": str(p.relative_to(PACKAGE)), "bytes": p.stat().st_size}
             for p in sorted(PACKAGE.rglob("*")) if p.is_file() and "__pycache__" not in p.parts]
    archive = PACKAGE.with_suffix(".zip")
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for entry in sizes:
            z.write(PACKAGE / entry["path"], "q2_final/" + entry["path"])
    total = sum(p["bytes"] for p in sizes)
    write(RESULTS / "package_sizes.json", {"directory_bytes_excluding_bytecode": total,
          "zip_bytes": archive.stat().st_size, "q2_only_under_50MB": total < 50_000_000,
          "whole_submission_size_verified": False, "files": sizes})


def analyze(selection):
    final_preds = ROOT / "reports/final/test/predictions.csv"
    names = [r["name"] for r in evaluation_grid()]
    comparisons = {}
    for name, path in (
        ("vs_grid50_single_seed1111", "reports/gridmix_test_combo/predictions.csv"),
        ("vs_historical_full_seed1111", "reports/final_controls/baseline_full_seed1111/predictions.csv")):
        comparisons[name] = {"missing": video_bootstrap_mean_f1(final_preds, ROOT / path, names),
                              "clean": video_bootstrap_mean_f1(final_preds, ROOT / path, ["clean"])}
    write(RESULTS / "paired_bootstrap.json", {"interpretation": "Conditional on fitted models; no multiplicity or adaptive-selection correction", **comparisons})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "assess", "verify", "analyze"))
    parser.add_argument("--selection", default="configs/final_selection.json")
    args = parser.parse_args()
    selection = json.loads((ROOT / args.selection).read_text(encoding="utf-8"))
    globals()[args.stage](selection)
    print(json.dumps({"completed": args.stage}), flush=True)


if __name__ == "__main__":
    main()
