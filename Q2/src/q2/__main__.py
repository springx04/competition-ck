"""Documented Q2 command line workflow."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys

import numpy as np
import torch

from .data import AlignedDataset, Normalizer, find_inputs, load_pickle, prepare_data, unpack_record
from .evaluate import evaluate_model, make_fixed_masks
from .export import export_bundle, load_training_best, predict_special, verify_export
from .masking import evaluation_grid
from .reporting import VARIANTS, SEEDS, generate_reports, select_final, suite_progress
from .text import cache_clean_text, load_text_encoder, prepare_model, tokenizer_check
from .trainer import train_one


def _read_config(path: Path):
    from .config import load_config
    return load_config(path)


def _root():
    return Path(__file__).resolve().parents[2]


def _source_data(config):
    return Path(config["project"]["data_root"])


def _selection(root, path):
    return json.loads((root / path).read_text(encoding="utf-8"))


def record_environment(root: Path):
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    package_names = ("torch", "numpy", "scipy", "pandas", "scikit-learn", "transformers",
                     "tokenizers", "huggingface-hub", "safetensors", "PyYAML", "tqdm",
                     "matplotlib", "pytest")
    lines = [f"Python: {sys.version}", f"Platform: {platform.platform()}",
             f"CPU: {platform.processor()}",
             f"Disk: {shutil.disk_usage(root)}", f"CUDA available: {torch.cuda.is_available()}"]
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        lines += [f"GPU: {props.name}", f"GPU total memory: {props.total_memory}",
                  f"GPU free/total: {torch.cuda.mem_get_info()}",
                  f"Torch CUDA: {torch.version.cuda}"]
    for name in package_names:
        lines.append(f"{name}: {importlib.metadata.version(name)}")
    for command in (("nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"),
                    ("free", "-h"), ("lscpu",)):
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        lines.append(" ".join(command) + ":\n" + result.stdout.strip())
    (reports / "environment.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    frozen = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True)
    (reports / "pip_freeze.txt").write_text(frozen.stdout, encoding="utf-8")


def inspect_data(root, config):
    inputs = find_inputs(_source_data(config))
    source = load_pickle(inputs["main"])
    inventory = {"main": str(inputs["main"]), "special_files": len(inputs["special"]), "splits": {}}
    ids_by_split = []
    videos_by_split = []
    for split in ("train", "valid", "test"):
        record = unpack_record(source[split], "attachment2")
        ids = record.get("id", [])
        videos = [item.split("$_$")[0] for item in ids]
        ids_by_split.append(set(ids)); videos_by_split.append(set(videos))
        inventory["splits"][split] = {"samples": len(record["input_ids"]),
            "visual_all_zero": int((record["vision"] == 0).all(axis=(1, 2)).sum()),
            "class_counts": np.bincount(record["class_id"], minlength=3).tolist(),
            "label_sign_mismatch": int((((record["class_id"] == 0) & (record["score"] >= 0))
                | ((record["class_id"] == 2) & (record["score"] <= 0))
                | ((record["class_id"] == 1) & (record["score"] != 0))).sum()),
            "shape": {key: list(record[key].shape) for key in ("input_ids", "audio", "vision")}}
    inventory["id_overlap"] = [len(ids_by_split[i] & ids_by_split[j]) for i,j in ((0,1),(0,2),(1,2))]
    inventory["video_overlap"] = [len(videos_by_split[i] & videos_by_split[j]) for i,j in ((0,1),(0,2),(1,2))]
    if [inventory["splits"][s]["samples"] for s in ("train", "valid", "test")] != [3395, 728, 727]:
        raise ValueError("official aligned split sizes differ from the reviewed input")
    (root / "reports").mkdir(exist_ok=True)
    (root / "reports/data_inventory.json").write_text(json.dumps(inventory, ensure_ascii=False, indent=2), encoding="utf-8")
    (root / "reports/data_inventory.md").write_text(
        "# Aligned input inventory\n\n" +
        "| Split | Samples | All-zero vision | Class 0/1/2 | Label sign mismatch |\n|---|---:|---:|---|---:|\n" +
        "".join(f"| {split} | {inventory['splits'][split]['samples']} | "
                f"{inventory['splits'][split]['visual_all_zero']} | "
                f"{inventory['splits'][split]['class_counts']} | "
                f"{inventory['splits'][split]['label_sign_mismatch']} |\n"
                for split in ("train", "valid", "test")) +
        f"\nID overlap: {inventory['id_overlap']}; source-video overlap: {inventory['video_overlap']}.\n",
        encoding="utf-8")
    return inventory


def verify(root, config):
    from .verification import verify_real_batch
    tests = subprocess.run([sys.executable, "-m", "pytest", "-q", str(root / "tests")],
                           cwd=root, capture_output=True, text=True)
    result = {"pytest_exit_code": tests.returncode, "pytest_stdout": tests.stdout,
              "pytest_stderr": tests.stderr}
    if tests.returncode != 0:
        raise RuntimeError(json.dumps(result, ensure_ascii=False))
    result["real_batch"] = verify_real_batch(root, config)
    (root / "reports/verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def evaluate_best(root, config, variant, seed, split="valid", stress=True, output_dir=None):
    path = root / "runs" / variant / f"seed_{seed}/best.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    normalizer = Normalizer.load(root / "data/processed/normalizer.npz")
    from .model.network import Student
    device = torch.device("cuda:0")
    student = Student(variant, normalizer.class_prior, normalizer.score_prior).to(device)
    student.load_state_dict(checkpoint["student"])
    frozen = load_text_encoder(root / "models/text_encoder", device)
    dataset = AlignedDataset(root / "data/processed" / split)
    cache = np.load(root / ".cache/text" / split / "features.npy", mmap_mode="r")
    return evaluate_model(student, frozen, normalizer, dataset, device, root / "data/masks" / split,
                          cache, batch_size=config["data"]["eval_batch_size"],
                          output_dir=output_dir, stress=stress, return_predictions=True)


def main():
    parser = argparse.ArgumentParser(prog="python -m q2")
    parser.add_argument("command", choices=("prepare-model", "inspect-data", "tokenizer-check", "prepare-data",
        "cache-text", "make-masks", "verify", "train", "train-suite", "evaluate-suite", "select-final",
        "evaluate-test", "predict-special", "export", "verify-export", "report"))
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--experiments", default="configs/experiments.yaml")
    parser.add_argument("--splits", nargs="+", default=["train", "valid"])
    parser.add_argument("--split", default="valid")
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--selection", default="reports/selection.json")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root = _root()
    config = _read_config(root / args.config)
    command = args.command
    result = None
    if command == "prepare-model":
        result = prepare_model(root / "models/bert_download", root / "models/text_encoder")
        record_environment(root)
    elif command == "inspect-data":
        result = inspect_data(root, config)
    elif command == "tokenizer-check":
        source = load_pickle(find_inputs(_source_data(config))["main"])
        record = unpack_record(source["train"], "attachment2")
        result = tokenizer_check(record, root / "models/text_encoder", root / "reports/tokenizer_compatibility.csv")
        (root / "reports/tokenizer_compatibility.md").write_text(
            "# Stored text/tokenizer compatibility\n\n" + json.dumps(result, indent=2) + "\n", encoding="utf-8")
        if result["unexplained"]:
            raise ValueError(f"{result['unexplained']} training samples have unexplained tokenization")
    elif command == "prepare-data":
        result = prepare_data(_source_data(config), root / "data/processed")
    elif command == "cache-text":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        encoder = load_text_encoder(root / "models/text_encoder", device)
        result = {}
        for split in args.splits:
            dataset = AlignedDataset(root / "data/processed" / split)
            result[split] = cache_clean_text(dataset, encoder, root / ".cache/text" / split,
                                             device, root / "models/text_encoder")
    elif command == "make-masks":
        result = make_fixed_masks(AlignedDataset(root / "data/processed" / args.split),
                                  root / "data/masks" / args.split)
    elif command == "verify":
        result = verify(root, config)
    elif command == "train":
        if args.variant is None or args.seed is None:
            parser.error("train requires --variant and --seed")
        result = train_one(root, config, args.variant, args.seed, args.resume)
    elif command == "train-suite":
        from .config import load_experiments
        variants = load_experiments(root / args.experiments)
        for variant in variants:
            for seed in config["train"]["seeds"]:
                progress = next(r for r in suite_progress(root) if r["variant"] == variant and r["seed"] == seed)
                if progress["status"] == "complete":
                    continue
                train_one(root, config, variant, seed, resume=args.resume or progress["completed_epoch"] > 0)
                suite_progress(root)
        result = suite_progress(root)
    elif command == "evaluate-suite":
        from .analysis import paired_condition_tables, reliability_analysis
        from .config import load_experiments
        variants = load_experiments(root / args.experiments)
        for variant in variants:
            for seed in SEEDS:
                out = root / "reports/valid" / variant / f"seed_{seed}"
                evaluate_best(root, config, variant, seed, output_dir=out)
                paired_condition_tables(out / "predictions.csv", out)
                if variant in ("full", "no_msd", "no_cons", "no_span", "uniform_spans"):
                    diagnosis = reliability_analysis(root, variant, seed)
                    (out / "reliability.json").write_text(json.dumps(diagnosis, indent=2), encoding="utf-8")
        result = {"evaluated_runs": 30}
    elif command == "select-final":
        result = select_final(root)
        from .analysis import video_bootstrap_mean_f1
        index = json.loads((root / "data/masks/valid/index.json").read_text(encoding="utf-8"))
        main_names = {item["name"] for item in evaluation_grid()}
        scenarios = [item["name"] for item in index if item["name"] in main_names and not item["duplicate_of"]]
        comparisons = []
        for first, second in dict.fromkeys((("full", "late_aug"), ("full", "no_comp"),
                                                (result["variant"], "late_aug"))):
            if first == second:
                continue
            a = root / "reports/valid" / first / "seed_1111/predictions.csv"
            b = root / "reports/valid" / second / "seed_1111/predictions.csv"
            if a.exists() and b.exists():
                comparisons.append({"first": first, "second": second,
                    **video_bootstrap_mean_f1(a, b, scenarios)})
        (root / "reports/bootstrap_comparisons.json").write_text(
            json.dumps(comparisons, ensure_ascii=False, indent=2), encoding="utf-8")
    elif command == "evaluate-test":
        from .analysis import paired_condition_tables
        selection = _selection(root, args.selection)
        result, _ = evaluate_best(root, config, selection["variant"], selection["seed"],
                                  split="test", output_dir=root / "reports/test")
        paired_condition_tables(root / "reports/test/predictions.csv", root / "reports/test")
        result = {"scenarios": len(result)}
    elif command == "predict-special":
        selection = _selection(root, args.selection)
        bundle = load_training_best(root, selection)
        inputs = find_inputs(_source_data(config))
        result = {"rows": len(predict_special(bundle, inputs["special"][0].parent,
            root / "outputs/q2_predictions_aligned.csv"))}
    elif command == "export":
        result = export_bundle(root, _selection(root, args.selection))
    elif command == "verify-export":
        inputs = find_inputs(_source_data(config))
        result = verify_export(root, _selection(root, args.selection), inputs["special"][0].parent)
        (root / "reports/export_verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    elif command == "report":
        result = generate_reports(root)
    print(json.dumps(result, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
