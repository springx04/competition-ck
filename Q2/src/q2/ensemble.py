"""Offline ensembles of Q2 students.

The ensemble is deliberately a thin deployment layer around :mod:`q2.export`.
Each member still owns its own student and text encoder.  Predictions are
combined only at the raw-logit and regression-score level; details such as
the observed-state and compensation tensors are copied from the first
member and are therefore not ensemble-level explanations.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from safetensors.torch import load_file, save_file
from transformers import BertConfig, BertModel

from .calibration import apply_class_bias, validate_class_bias
from .data import Normalizer
from .export import Bundle, checkpoint_runtime_options, load_training_best
from .model.network import Student


ENSEMBLE_CONFIG = "ensemble_config.json"
_TEXT_WEIGHT_NAMES = {
    "model.safetensors",
    "pytorch_model.bin",
    "tf_model.h5",
    "flax_model.msgpack",
}


class EnsembleBundle:
    """A collection of uncalibrated :class:`~q2.export.Bundle` members.

    ``predict_raw`` averages every member's raw logits and scores.  The
    optional class bias is applied exactly once to that average, and only for
    samples with at least one observed modality.  The first member's output
    supplies all state and diagnostic fields because those fields do not have
    a defined ensemble interpretation.
    """

    def __init__(self, members: Sequence[Bundle], device="cpu", class_bias=None):
        if not members:
            raise ValueError("an ensemble must contain at least one member")
        self.members = list(members)
        self.device = torch.device(device)
        self.class_bias = validate_class_bias(class_bias)

    @torch.no_grad()
    def predict_raw(self, raw_batch: dict, return_details=True):
        outputs = [member.predict_raw(raw_batch, return_details=return_details)
                   for member in self.members]
        first = outputs[0]
        logits = torch.stack([output.logits for output in outputs], dim=0).mean(dim=0)
        score = torch.stack([output.score for output in outputs], dim=0).mean(dim=0)
        if self.class_bias is not None:
            logits = apply_class_bias(logits, first.U, self.class_bias)
        return _replace_prediction(first, logits=logits, score=score)


def _replace_prediction(output: Any, *, logits: torch.Tensor, score: torch.Tensor):
    """Copy an output while replacing only its two averaged predictions."""
    if dataclasses.is_dataclass(output):
        return dataclasses.replace(output, logits=logits, score=score)
    result = copy.copy(output)
    result.logits = logits
    result.score = score
    return result


def _selection_members(selection: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    if not isinstance(selection, Mapping):
        raise TypeError("selection must be a mapping")
    members = selection.get("members")
    if not isinstance(members, Sequence) or isinstance(members, (str, bytes)) or not members:
        raise ValueError("selection.members must contain at least one member")
    checked = []
    for index, member in enumerate(members):
        if not isinstance(member, Mapping):
            raise TypeError(f"selection.members[{index}] must be a mapping")
        for key in ("variant", "checkpoint"):
            if key not in member:
                raise ValueError(f"selection.members[{index}] is missing {key}")
        checked.append(member)
    return checked


def _export_members(config: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    members = config.get("members")
    if not isinstance(members, Sequence) or isinstance(members, (str, bytes)) or not members:
        raise ValueError("ensemble config members must contain at least one member")
    for index, member in enumerate(members):
        if not isinstance(member, Mapping):
            raise TypeError(f"ensemble config members[{index}] must be a mapping")
        for key in ("variant", "student", "text_encoder"):
            if key not in member:
                raise ValueError(f"ensemble config members[{index}] is missing {key}")
    return list(members)


def load_training_ensemble(root: Path, selection: dict, device="cuda:0",
                           round_text_for_export=False) -> EnsembleBundle:
    """Load the selected training checkpoints as an uncalibrated ensemble.

    The checkpoint's own historical ``evaluation.class_bias`` is explicitly
    suppressed for every member.  ``selection.class_bias`` is retained by the
    returned ensemble and is applied after averaging.
    """
    root = Path(root)
    members = []
    for member in _selection_members(selection):
        member_selection = dict(member)
        # A member must return raw logits even if an old checkpoint happened to
        # contain a deployment bias.  The selection-level bias is the single
        # calibration applied by EnsembleBundle.
        member_selection["class_bias"] = None
        members.append(load_training_best(root, member_selection, device,
                                          round_text_for_export=round_text_for_export))
    return EnsembleBundle(members, device=device,
                          class_bias=selection.get("class_bias"))


def _student_state(student: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Return the FP32 deployment state, omitting optional auxiliary heads."""
    return {
        key: value.detach().cpu().float().contiguous()
        for key, value in student.state_dict().items()
        if not key.startswith(("decomposition.aux_class.",
                               "decomposition.aux_score.",
                               "auxiliary_classifiers."))
    }


def _encoder_state(encoder: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu() for key, value in encoder.state_dict().items()}


def _fp16_tensor(value: torch.Tensor) -> torch.Tensor:
    # BERT's position/token-type id buffers are integer tensors and must keep
    # their dtype for strict reload; learned floating-point weights are FP16.
    return (value.half() if value.is_floating_point() else value.clone()).contiguous()


def _copy_text_assets(source: Path, destination: Path, encoder: Any) -> None:
    """Copy config/tokenizer files while excluding every base model weight."""
    source = Path(source)
    destination.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        for path in source.rglob("*"):
            if not path.is_file() or path.name in _TEXT_WEIGHT_NAMES:
                continue
            relative = path.relative_to(source)
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    config_path = destination / "config.json"
    if not config_path.is_file():
        config = getattr(encoder, "config", None)
        if config is None or not hasattr(config, "to_json_file"):
            raise FileNotFoundError("text encoder export is missing BertConfig/config.json")
        config.to_json_file(config_path)


def _source_text_dir(root: Path, checkpoint: Mapping[str, Any]) -> Path:
    config = checkpoint.get("config") or {}
    text_config = config.get("text") or {}
    text_dir = text_config.get("model_dir") or "models/text_encoder"
    return root / text_dir


def _load_checkpoint(root: Path, member: Mapping[str, Any]) -> tuple[Path, dict]:
    checkpoint_path = Path(root) / member["checkpoint"]
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    return checkpoint_path, checkpoint


def _shared_and_private_encoder_states(encoders: Sequence[torch.nn.Module]):
    states = [_encoder_state(encoder) for encoder in encoders]
    if not states:
        raise ValueError("an ensemble must contain at least one text encoder")
    shared_keys = sorted(key for key in states[0] if key.startswith("embeddings."))
    if not shared_keys:
        raise ValueError("text encoder has no embeddings.* parameters to share")
    for index, state in enumerate(states[1:], start=1):
        if set(state) != set(states[0]):
            raise ValueError(f"text encoder state keys differ for member {index}")
        for key in shared_keys:
            # This is intentionally exact.  Sharing rounded or merely close
            # embeddings would silently load the wrong tuned model.
            if not torch.equal(states[0][key], state[key]):
                raise ValueError(f"text encoder embeddings differ for member {index}: {key}")
    shared = {key: _fp16_tensor(states[0][key]) for key in shared_keys}
    private = []
    for state in states:
        private.append({key: _fp16_tensor(value)
                        for key, value in state.items() if key not in shared_keys})
    return shared, private


def _save_state(path: Path, state: Mapping[str, torch.Tensor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file({key: value.contiguous() for key, value in state.items()}, str(path))


def _jsonable_runtime(checkpoint_path: Path, checkpoint: Mapping[str, Any],
                      selection: Mapping[str, Any]) -> dict[str, Any]:
    runtime = checkpoint_runtime_options(checkpoint, checkpoint_path,
                                         {"class_bias": None})
    return {"cls_context": bool(runtime["cls_context"]),
            "clip_z": runtime["clip_z"]}


def export_ensemble(root: Path, selection: dict, dest: Path):
    """Export an offline ensemble directory without overwriting ``dest``.

    Students are stored as FP32 safetensors.  Each tuned BERT member stores
    its non-embedding weights as FP16, while exact shared ``embeddings.*``
    tensors are stored once.  The destination contains no training
    checkpoints and can be loaded with :func:`load_ensemble` offline.
    """
    root = Path(root)
    dest = Path(dest)
    if dest.exists():
        raise FileExistsError(f"refusing to overwrite existing ensemble destination: {dest}")
    selected_members = _selection_members(selection)

    # Load the original FP32 checkpoints and verify shared tensors before any
    # destination files are created.  The exact comparison must happen before
    # FP16 export rounding, otherwise two different embeddings could collapse
    # to the same half-precision values and be incorrectly shared.
    bundle = load_training_ensemble(root, selection, device="cpu",
                                    round_text_for_export=False)
    encoders = [member.text_encoder for member in bundle.members]
    shared_embeddings, private_encoders = _shared_and_private_encoder_states(encoders)

    checkpoint_info = []
    for member in selected_members:
        checkpoint_path, checkpoint = _load_checkpoint(root, member)
        checkpoint_info.append((checkpoint_path, checkpoint,
                                _jsonable_runtime(checkpoint_path, checkpoint, selection)))

    assets_text = dest / "assets" / "text_encoder"
    assets_text.mkdir(parents=True, exist_ok=True)
    first_source = _source_text_dir(root, checkpoint_info[0][1])
    _copy_text_assets(first_source, assets_text, encoders[0])
    _save_state(assets_text / "shared_embeddings.safetensors", shared_embeddings)

    members_config = []
    for index, (member, private_state, member_info) in enumerate(
            zip(selected_members, private_encoders, checkpoint_info)):
        student_path = dest / "students" / f"student_{index}.safetensors"
        text_path = assets_text / f"member_{index}.safetensors"
        _save_state(student_path, _student_state(bundle.members[index].student))
        _save_state(text_path, private_state)
        student_rel = student_path.relative_to(dest).as_posix()
        text_rel = text_path.relative_to(dest).as_posix()
        checkpoint_path, _checkpoint, runtime = member_info
        members_config.append({
            "variant": member["variant"],
            "seed": member.get("seed"),
            "source_checkpoint": str(member["checkpoint"]),
            "student": student_rel,
            "text_encoder": text_rel,
            "text": runtime,
        })

    shutil.copy2(root / "data/processed/normalizer.npz", dest / "normalizer.npz")
    source_files = ("THIRD_PARTY.md", "requirements.txt")
    for name in source_files:
        path = root / name
        if path.is_file():
            shutil.copy2(path, dest / name)
    source_code = root / "src/q2"
    if source_code.is_dir():
        shutil.copytree(source_code, dest / "src/q2",
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    licenses = root / "licenses"
    if licenses.is_dir():
        shutil.copytree(licenses, dest / "licenses")

    config = {
        "format_version": 1,
        "normalizer": "normalizer.npz",
        "text": {
            "directory": "assets/text_encoder",
            "shared_embeddings": "assets/text_encoder/shared_embeddings.safetensors",
        },
        "class_bias": validate_class_bias(selection.get("class_bias")),
        "evaluation": {"class_bias": validate_class_bias(selection.get("class_bias"))},
        "members": members_config,
    }
    (dest / ENSEMBLE_CONFIG).write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (dest / "run_inference.py").write_text(INFERENCE_SCRIPT, encoding="utf-8")
    (dest / "README.md").write_text(
        "# Q2 offline ensemble inference\n\n"
        "Run `python run_inference.py --input-dir PATH --output "
        "q2_predictions_aligned.csv --device cpu`.\n",
        encoding="utf-8")
    return {
        "directory": str(dest),
        "config": str(dest / ENSEMBLE_CONFIG),
        "members": len(members_config),
        "directory_bytes": sum(path.stat().st_size for path in dest.rglob("*")
                                if path.is_file()),
        "source_checkpoints": [item["source_checkpoint"] for item in members_config],
    }


def _load_export_text_encoder(directory: Path, member_config: Mapping[str, Any], device):
    text_directory = directory / member_config.get("text", {}).get(
        "directory", "assets/text_encoder")
    config = BertConfig.from_pretrained(text_directory, local_files_only=True)
    encoder = BertModel(config, add_pooling_layer=False)
    shared_path = directory / member_config.get("text", {}).get(
        "shared_embeddings", "assets/text_encoder/shared_embeddings.safetensors")
    private_path = directory / member_config["text_encoder"]
    shared = load_file(str(shared_path), device="cpu")
    private = load_file(str(private_path), device="cpu")
    if set(shared).intersection(private):
        raise ValueError("exported shared and private text encoder keys overlap")
    state = {key: (value.float() if value.is_floating_point() else value)
             for key, value in {**shared, **private}.items()}
    # strict=True is part of the offline package contract: a missing or extra
    # BERT tensor must fail loudly instead of producing a subtly different model.
    encoder.load_state_dict(state, strict=True)
    return encoder.to(device).float().requires_grad_(False).eval()


def load_ensemble(directory: Path, device="cpu") -> EnsembleBundle:
    """Load an exported ensemble with no network or original checkpoint."""
    directory = Path(directory)
    config_path = directory / ENSEMBLE_CONFIG
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    normalizer_path = directory / config.get("normalizer", "normalizer.npz")
    members = []
    text_config = config.get("text") or {}
    for member_config in _export_members(config):
        normalizer = Normalizer.load(normalizer_path)
        runtime = member_config.get("text") or {}
        if runtime.get("clip_z") is not None:
            normalizer.clip_z = runtime["clip_z"]
        student = Student(member_config["variant"], normalizer.class_prior,
                          normalizer.score_prior, include_aux_heads=False)
        state = load_file(str(directory / member_config["student"]), device="cpu")
        student.load_state_dict(state, strict=True)
        student = student.to(device).eval()
        text_encoder = _load_export_text_encoder(
            directory,
            {"text_encoder": member_config["text_encoder"],
             "text": {"directory": text_config.get("directory", "assets/text_encoder"),
                       "shared_embeddings": text_config.get(
                           "shared_embeddings",
                           "assets/text_encoder/shared_embeddings.safetensors")}},
            device,
        )
        members.append(Bundle(student, text_encoder, normalizer, device,
                              cls_context=bool(runtime.get("cls_context", False)),
                              class_bias=None))
    evaluation = config.get("evaluation") or {}
    class_bias = evaluation.get("class_bias", config.get("class_bias"))
    return EnsembleBundle(members, device=device,
                          class_bias=class_bias)


INFERENCE_SCRIPT = '''"""Offline Q2 ensemble aligned attachment3 inference."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from q2.ensemble import load_ensemble
from q2.export import predict_special


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", default="q2_predictions_aligned.csv")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    bundle = load_ensemble(ROOT, args.device)
    predict_special(bundle, Path(args.input_dir), Path(args.output), args.batch_size)


if __name__ == "__main__":
    main()
'''
