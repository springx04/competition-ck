"""Self-contained offline inference bundle and fixed special predictions."""
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from .data import Normalizer, load_pickle, unpack_record
from .model.network import Student, TUNED_VARIANTS, encode_view
from .text import load_text_encoder
from .calibration import apply_class_bias, validate_class_bias


CLASS_NAMES = ("Negative", "Neutral", "Positive")
_FP32_EXPORT_TOLERANCE = 1e-5
_FP16_TEXT_EXPORT_TOLERANCE = 1e-3

# These are the only historical exploratory runs known to have been trained
# while the temporary unconditional CLS term was present.  Checkpoints from
# any other legacy run keep the original, token-only frontend semantics.
LEGACY_CLS_CONTEXT_RUNS = frozenset({
    "pool_clsctx_v3",
    "pool_pw15_v3",
    "full_tune_v3",
    "full_tune_stress_v3",
    "attn_tune_v3",
    "pool_low_lr5e5_30_v3",
    "attn_40_v4",
})


def checkpoint_runtime_options(checkpoint: dict, checkpoint_path: Path | None = None,
                               selection: dict | None = None) -> dict:
    """Resolve frontend options saved by a checkpoint, with a narrow legacy map."""
    config = checkpoint.get("config") or {}
    text_config = config.get("text") or {}
    data_config = config.get("data") or {}
    cls_context = text_config.get("cls_context")
    if cls_context is None:
        names = set()
        output_root = config.get("project", {}).get("output_root")
        if output_root:
            names.add(Path(str(output_root)).name)
        if checkpoint_path is not None:
            path = Path(checkpoint_path)
            names.update(parent.name for parent in path.parents)
        cls_context = any(name in LEGACY_CLS_CONTEXT_RUNS for name in names)
    evaluation_config = config.get("evaluation") or {}
    bias = (selection or {}).get("class_bias", evaluation_config.get("class_bias"))
    return {"cls_context": bool(cls_context), "clip_z": data_config.get("clip_z"),
            "class_bias": validate_class_bias(bias)}


def load_checkpoint_text_encoder(root: Path, variant: str, checkpoint: dict, device):
    """Load the base encoder and restore the state saved by tuned runs."""
    # The on-disk model is FP16, while encode_text allocates FP32 features.
    # Normalize the loaded module to the declared compute dtype before loading
    # a tuned checkpoint state or running inference.
    encoder = load_text_encoder(Path(root) / "models/text_encoder", torch.device(device)).float()
    if variant in TUNED_VARIANTS:
        state = checkpoint.get("text_encoder")
        if state is None:
            raise ValueError(f"{variant} checkpoint is missing text_encoder weights")
        encoder.load_state_dict(state)
        encoder.eval()
    return encoder


class Bundle:
    def __init__(self, student, text_encoder, normalizer, device, cls_context=False,
                 class_bias=None):
        self.student = student.eval()
        self.text_encoder = text_encoder.eval()
        self.normalizer = normalizer
        self.device = torch.device(device)
        self.cls_context = bool(cls_context)
        self.class_bias = validate_class_bias(class_bias)

    @torch.no_grad()
    def predict_raw(self, raw_batch: dict, return_details=True):
        raw = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in raw_batch.items()}
        model_input = encode_view(raw, self.text_encoder, self.normalizer,
                                  cls_context=self.cls_context)
        output = self.student(model_input, return_details=return_details)
        if self.class_bias is not None:
            output.logits = apply_class_bias(output.logits, model_input.U, self.class_bias)
        return output


def load_bundle(directory: Path, device="cpu") -> Bundle:
    directory = Path(directory)
    config = __import__("yaml").safe_load((directory / "model_config.yaml").read_text(encoding="utf-8"))
    normalizer = Normalizer.load(directory / "normalizer.npz")
    data_config = config.get("data") or {}
    if data_config.get("clip_z") is not None:
        normalizer.clip_z = data_config["clip_z"]
    student = Student(config["variant"], normalizer.class_prior, normalizer.score_prior,
                      include_aux_heads=False)
    weights = load_file(str(directory / "student.safetensors"), device="cpu")
    student.load_state_dict(weights, strict=True)
    student = student.to(device).eval()
    text_encoder = load_text_encoder(directory / "assets/text_encoder", torch.device(device)).float()
    text_config = config.get("text") or {}
    evaluation_config = config.get("evaluation") or {}
    return Bundle(student, text_encoder, normalizer, device,
                  cls_context=bool(text_config.get("cls_context", False)),
                  class_bias=evaluation_config.get("class_bias"))


def load_training_best(root: Path, selection: dict, device="cuda:0", round_text_for_export=False) -> Bundle:
    normalizer = Normalizer.load(Path(root) / "data/processed/normalizer.npz")
    checkpoint_path = Path(root) / selection["checkpoint"]
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    runtime = checkpoint_runtime_options(checkpoint, checkpoint_path, selection)
    if runtime["clip_z"] is not None:
        normalizer.clip_z = runtime["clip_z"]
    student = Student(selection["variant"], normalizer.class_prior, normalizer.score_prior)
    student.load_state_dict(checkpoint["student"])
    student = student.to(device).eval()
    encoder = load_checkpoint_text_encoder(root, selection["variant"], checkpoint, device)
    if round_text_for_export:
        # Match the FP16-on-disk text weights used by export_bundle, while
        # keeping inference itself in the declared FP32 compute dtype.
        encoder.half().float()
    return Bundle(student, encoder, normalizer, device,
                  cls_context=runtime["cls_context"],
                  class_bias=runtime["class_bias"])


def raw_from_record(record: dict):
    return {key: torch.from_numpy(np.array(record[key], copy=True)) for key in
            ("input_ids", "stored_attention", "token_type_ids", "audio", "vision")}


def predict_special(bundle: Bundle, attachment3_dir: Path, output_csv: Path, batch_size=128):
    files = sorted(Path(attachment3_dir).glob("附件3_*.pkl"))
    if len(files) != 30:
        raise ValueError(f"expected 30 special aligned files, found {len(files)}")
    rows = []
    state_rows = []
    for path in files:
        record = unpack_record(load_pickle(path), "attachment3")
        if len(record["input_ids"]) != 1:
            raise ValueError(f"special sample does not contain exactly one example: {path}")
        output = bundle.predict_raw(raw_from_record(record), return_details=True)
        probabilities = output.logits.softmax(dim=-1)[0].cpu().tolist()
        rows.append({"sample_id": path.stem, "feature_version": "aligned",
                     "pred_label": CLASS_NAMES[int(np.argmax(probabilities))],
                     "pred_score": float(output.score[0].cpu()),
                     "prob_negative": probabilities[0], "prob_neutral": probabilities[1],
                     "prob_positive": probabilities[2]})
        state_rows.append({"sample_id": path.stem, "time_axis": output.time_axis,
                           "structural_positions": int(output.J[0].sum().cpu()),
                           "observed_text_positions": int(output.U[0, :, 0].sum().cpu()),
                           "observed_audio_positions": int(output.U[0, :, 1].sum().cpu()),
                           "observed_visual_positions": int(output.U[0, :, 2].sum().cpu()),
                           "compensated_text_positions": int(output.B_comp[0, :, 0].sum().cpu()),
                           "compensated_audio_positions": int(output.B_comp[0, :, 1].sum().cpu()),
                           "compensated_visual_positions": int(output.B_comp[0, :, 2].sum().cpu())})
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    with output_csv.with_name("q2_special_state.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(state_rows[0]))
        writer.writeheader(); writer.writerows(state_rows)
    return rows


INFERENCE_SCRIPT = '''"""Offline Q2 aligned attachment3 inference."""
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from q2.export import load_bundle, predict_special

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", default="q2_predictions_aligned.csv")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    bundle = load_bundle(ROOT, args.device)
    predict_special(bundle, Path(args.input_dir), Path(args.output), args.batch_size)
    print(f"q2 module: {sys.modules['q2'].__file__}")

if __name__ == "__main__":
    main()
'''


def export_bundle(root: Path, selection: dict):
    root = Path(root)
    dest = root / "delivery/q2_inference"
    if dest.exists():
        shutil.rmtree(dest)
    (dest / "assets").mkdir(parents=True)
    normalizer = Normalizer.load(root / "data/processed/normalizer.npz")
    checkpoint_path = root / selection["checkpoint"]
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    runtime = checkpoint_runtime_options(checkpoint, checkpoint_path, selection)
    if runtime["clip_z"] is not None:
        normalizer.clip_z = runtime["clip_z"]
    student = Student(selection["variant"], normalizer.class_prior, normalizer.score_prior,
                      include_aux_heads=False)
    state = {key: value for key, value in checkpoint["student"].items()
             if not key.startswith(("decomposition.aux_class.", "decomposition.aux_score.", "auxiliary_classifiers."))}
    student.load_state_dict(state, strict=True)
    save_file({key: value.contiguous() for key, value in state.items()}, str(dest / "student.safetensors"))
    shutil.copy2(root / "data/processed/normalizer.npz", dest / "normalizer.npz")
    shutil.copytree(root / "models/text_encoder", dest / "assets/text_encoder")
    if selection["variant"] in TUNED_VARIANTS:
        encoder = load_checkpoint_text_encoder(root, selection["variant"], checkpoint, torch.device("cpu"))
        # Keep the package's FP16-on-disk / FP32-at-runtime text contract.
        encoder.half().save_pretrained(dest / "assets/text_encoder", safe_serialization=True)
    shutil.copytree(root / "src/q2", dest / "src/q2", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copytree(root / "licenses", dest / "licenses")
    for name in ("THIRD_PARTY.md", "requirements.txt"):
        shutil.copy2(root / name, dest / name)
    shutil.copy2(root / "outputs/q2_predictions_aligned.csv", dest / "q2_predictions_aligned.csv")
    (dest / "model_config.yaml").write_text(
        __import__("yaml").safe_dump({
            "variant": selection["variant"],
            "text": {"cls_context": runtime["cls_context"]},
            "data": {"clip_z": runtime["clip_z"]},
            "evaluation": {"class_bias": runtime["class_bias"]},
        }), encoding="utf-8")
    (dest / "run_inference.py").write_text(INFERENCE_SCRIPT, encoding="utf-8")
    (dest / "README.md").write_text("# Q2 offline inference\n\nRun `python run_inference.py --input-dir PATH --output q2_predictions_aligned.csv --device cuda:0`. The bundle contains the selected student and the complete frozen text encoder.\n", encoding="utf-8")
    archive = root / "delivery/q2_inference.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        for path in dest.rglob("*"):
            if path.is_file():
                handle.write(path, path.relative_to(dest.parent))
    return {"directory": str(dest), "zip": str(archive),
            "directory_bytes": sum(path.stat().st_size for path in dest.rglob("*") if path.is_file()),
            "zip_bytes": archive.stat().st_size}


def verify_export(root: Path, selection: dict, attachment3_dir: Path, device="cuda:0"):
    """Run package entry point in an isolated cwd with offline HF settings."""
    root = Path(root)
    package = root / "delivery/q2_inference"
    with tempfile.TemporaryDirectory(prefix="q2-offline-") as temp:
        output = Path(temp) / "predictions.csv"
        from .data import AlignedDataset, collate_raw
        from .masking import apply_span, make_span_mask
        from .state import infer_state
        valid = AlignedDataset(root / "data/processed/valid")
        selected = list(range(7))
        zero_vision = next(i for i in range(len(valid)) if not valid[i]["vision"].any())
        selected.append(zero_vision)
        clean = collate_raw([valid[i] for i in selected])
        patterns = ("T", "T", "T", "A", "A", "V", "V", "TV")
        descriptors = [{"pattern": p, "position": "middle", "rho": .4} for p in patterns]
        damaged = apply_span(clean, make_span_mask(clean, infer_state(clean), descriptors))
        reference_bundle = load_training_best(root, selection, device)
        export_reference_bundle = load_training_best(root, selection, device,
                                                      round_text_for_export=True)
        expected = []
        export_expected = []
        for raw in (clean, damaged):
            predicted = reference_bundle.predict_raw(raw, return_details=False)
            expected.append((predicted.logits.cpu().numpy(), predicted.score.cpu().numpy()))
            predicted = export_reference_bundle.predict_raw(raw, return_details=False)
            export_expected.append((predicted.logits.cpu().numpy(), predicted.score.cpu().numpy()))
        arrays = {}
        for prefix, raw in (("clean", clean), ("damaged", damaged)):
            for key in ("input_ids", "stored_attention", "token_type_ids", "audio", "vision"):
                arrays[prefix + "_" + key] = raw[key].numpy()
        input_npz = Path(temp) / "valid_cases.npz"
        np.savez(input_npz, **arrays)
        package_output = Path(temp) / "package_valid.npz"
        script = '''from pathlib import Path
import numpy as np
import sys
import torch
package, input_file, output_file, device = sys.argv[1:]
sys.path.insert(0, str(Path(package) / "src"))
import q2
from q2.export import load_bundle
bundle = load_bundle(Path(package), device)
source = np.load(input_file)
results = {}
for prefix in ("clean", "damaged"):
    raw = {key: torch.from_numpy(source[prefix + "_" + key].copy()) for key in
           ("input_ids", "stored_attention", "token_type_ids", "audio", "vision")}
    output = bundle.predict_raw(raw, return_details=False)
    results[prefix + "_logits"] = output.logits.cpu().numpy()
    results[prefix + "_score"] = output.score.cpu().numpy()
np.savez(output_file, **results)
print(q2.__file__)
'''
        env = os.environ.copy()
        env.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "PYTHONPATH": ""})
        valid_process = subprocess.run([sys.executable, "-c", script, str(package),
                                        str(input_npz), str(package_output), device],
                                       cwd=temp, env=env, capture_output=True, text=True, check=True)
        if not valid_process.stdout.strip().startswith(str(package / "src")):
            raise AssertionError(f"offline process imported unexpected q2 module: {valid_process.stdout}")
        with np.load(package_output) as actual:
            valid_asset_differences = []
            valid_rounding_differences = []
            valid_asset_labels = []
            valid_rounding_labels = []
            for i, prefix in enumerate(("clean", "damaged")):
                package_logits = actual[prefix + "_logits"]
                package_score = actual[prefix + "_score"]
                valid_asset_differences.extend((
                    np.max(np.abs(package_logits - export_expected[i][0])),
                    np.max(np.abs(package_score - export_expected[i][1])),
                ))
                valid_rounding_differences.extend((
                    np.max(np.abs(export_expected[i][0] - expected[i][0])),
                    np.max(np.abs(export_expected[i][1] - expected[i][1])),
                ))
                valid_asset_labels.append(np.array_equal(package_logits.argmax(axis=-1),
                                                          export_expected[i][0].argmax(axis=-1)))
                valid_rounding_labels.append(np.array_equal(export_expected[i][0].argmax(axis=-1),
                                                            expected[i][0].argmax(axis=-1)))
        completed = subprocess.run([sys.executable, str(package / "run_inference.py"),
                                    "--input-dir", str(attachment3_dir), "--output", str(output),
                                    "--device", device], cwd=temp, env=env,
                                   capture_output=True, text=True, check=True)
        with output.open(encoding="utf-8", newline="") as handle:
            exported = list(csv.DictReader(handle))
        rounded_output = Path(temp) / "rounded_predictions.csv"
        predict_special(export_reference_bundle, attachment3_dir, rounded_output)
        with rounded_output.open(encoding="utf-8", newline="") as handle:
            export_reference = list(csv.DictReader(handle))
        with (root / "outputs/q2_predictions_aligned.csv").open(encoding="utf-8", newline="") as handle:
            original = list(csv.DictReader(handle))
        prediction_fields = ("pred_score", "prob_negative", "prob_neutral", "prob_positive")
        asset_differences = [abs(float(a[key]) - float(b[key]))
                             for a, b in zip(exported, export_reference)
                             for key in prediction_fields]
        rounding_differences = [abs(float(a[key]) - float(b[key]))
                                for a, b in zip(export_reference, original)
                                for key in prediction_fields]
        labels_equal = (len(exported) == len(export_reference)
                        and all(a["pred_label"] == b["pred_label"]
                                for a, b in zip(exported, export_reference)))
        training_labels_equal = (len(export_reference) == len(original)
                                 and all(a["pred_label"] == b["pred_label"]
                                         for a, b in zip(export_reference, original)))
        comparison_tolerance = _FP32_EXPORT_TOLERANCE
        rounding_tolerance = _FP16_TEXT_EXPORT_TOLERANCE
        result = {"module_path_output": completed.stdout.strip(),
                  "valid_module_path": valid_process.stdout.strip(), "valid_cases": selected,
                  "valid_max_absolute_difference": float(max(valid_asset_differences)), "rows": len(exported),
                  "max_absolute_difference": max(asset_differences),
                  "comparison_tolerance": comparison_tolerance,
                  "valid_label_agreement": all(valid_asset_labels),
                  "labels_equal": labels_equal,
                  "fp16_valid_max_absolute_difference": float(max(valid_rounding_differences)),
                  "fp16_max_absolute_difference": max(rounding_differences),
                  "fp16_comparison_tolerance": rounding_tolerance,
                  "training_valid_label_agreement": all(valid_rounding_labels),
                  "training_labels_equal": training_labels_equal,
                  "passed": len(exported) == 30 and len(export_reference) == 30
                  and labels_equal and all(valid_asset_labels)
                  and max(asset_differences) <= comparison_tolerance
                  and max(valid_asset_differences) <= comparison_tolerance}
        if not result["passed"]:
            raise AssertionError(result)
        return result
