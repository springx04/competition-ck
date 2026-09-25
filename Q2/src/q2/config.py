"""Strict loading and path resolution for the Q2 YAML configuration."""

from __future__ import annotations

from numbers import Real
import math
from pathlib import Path
from typing import Any, Mapping

import yaml


FIXED_VARIANTS = (
    "full",
    "late_clean",
    "late_aug",
    "no_msd",
    "no_comp",
    "no_reliability",
    "no_cons",
    "no_span",
    "no_teacher",
    "uniform_spans",
)


_TEXT_MODEL_LAYERS = {
    "google/bert_uncased_L-4_H-256_A-4": 4,
    "google/bert_uncased_L-8_H-256_A-4": 8,
}


class ConfigError(ValueError):
    """Raised when a Q2 config does not match the declared schema."""


_SCHEMA: dict[str, tuple[str, ...]] = {
    "root": ("project", "data", "text", "model", "loss", "train", "evaluation"),
    "project": ("name", "data_root", "output_root"),
    "data": (
        "feature_version",
        "seq_len",
        "modality_order",
        "dims",
        "num_workers",
        "pin_memory",
        "train_batch_size",
        "eval_batch_size",
        "drop_last",
    ),
    "text": (
        "model_dir",
        "model_id",
        "frozen",
        "compute_dtype",
        "stored_dtype",
        "attention_implementation",
    ),
    "model": (
        "hidden_dim",
        "encoder_layers",
        "num_heads",
        "ffn_dim",
        "dropout",
        "layer_norm_eps",
        "compensation_rounds",
        "shared_compensator",
        "source_classes",
        "q_dim",
        "num_classes",
        "class_names",
        "score_min",
        "score_max",
    ),
    "loss": (
        "regression_weight",
        "huber_delta",
        "msd_weight",
        "shared_temperature",
        "specific_inner_weight",
        "unimodal_inner_weight",
        "span_weight",
        "calibration_weight",
        "consistency_weight",
        "ramp_epochs",
    ),
    "train": (
        "epochs",
        "warmup_epochs",
        "optimizer",
        "learning_rate",
        "min_learning_rate",
        "betas",
        "adam_eps",
        "weight_decay",
        "gradient_clip_norm",
        "ema_decay",
        "seeds",
        "eval_epochs",
        "amp",
        "tf32",
        "compile",
    ),
    "evaluation": (
        "main_patterns",
        "main_positions",
        "main_rhos",
        "bootstrap_repeats",
        "bootstrap_seed",
        "deployed_seed",
    ),
}


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{name} must be a mapping")
    return dict(value)


def _check_keys(value: Mapping[str, Any], name: str, expected: tuple[str, ...]) -> None:
    actual = set(value)
    allowed = set(expected)
    optional = ({"learning_rate", "unfrozen_layers", "cls_context", "train_embeddings"} if name == "text" else
                {"class_weight_power", "late_unimodal_weight"} if name == "loss" else
                {"clip_z", "video_sampling_power"} if name == "data" else
                {"stress_text", "grid_mix", "grid_mix_probability", "teacher_targets"} if name == "train" else
                {"class_bias"} if name == "evaluation" else set())
    unknown = sorted(actual - allowed - optional)
    missing = sorted(allowed - actual)
    if unknown:
        raise ConfigError(f"{name} contains unknown key(s): {', '.join(unknown)}")
    if missing:
        raise ConfigError(f"{name} is missing required key(s): {', '.join(missing)}")


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ConfigError(f"{name} must be a non-empty string")
    return value


def _bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise ConfigError(f"{name} must be a boolean")
    return value


def _integer(value: Any, name: str, *, minimum: int | None = None) -> int:
    if type(value) is not int:
        raise ConfigError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise ConfigError(f"{name} must be >= {minimum}")
    return value


def _number(value: Any, name: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ConfigError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ConfigError(f"{name} must be finite")
    if minimum is not None and result < minimum:
        raise ConfigError(f"{name} must be >= {minimum}")
    return result


def _list(value: Any, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ConfigError(f"{name} must be a list")
    return value


def _equal_list(value: Any, expected: list[Any], name: str) -> None:
    actual = _list(value, name)
    if actual != expected:
        raise ConfigError(f"{name} must equal {expected!r}; got {actual!r}")


def _validate_values(config: Mapping[str, Any]) -> None:
    project = config["project"]
    _string(project["name"], "project.name")
    _string(project["data_root"], "project.data_root")
    _string(project["output_root"], "project.output_root")

    data = config["data"]
    if data["feature_version"] != "aligned":
        raise ConfigError("data.feature_version must be 'aligned'")
    _integer(data["seq_len"], "data.seq_len", minimum=1)
    _equal_list(data["modality_order"], ["text", "audio", "vision"], "data.modality_order")
    _equal_list(data["dims"], [256, 74, 35], "data.dims")
    _integer(data["num_workers"], "data.num_workers", minimum=0)
    if "clip_z" in data:
        _number(data["clip_z"], "data.clip_z", minimum=0.0)
    if "video_sampling_power" in data:
        _number(data["video_sampling_power"], "data.video_sampling_power", minimum=0.0)
    _bool(data["pin_memory"], "data.pin_memory")
    _integer(data["train_batch_size"], "data.train_batch_size", minimum=1)
    _integer(data["eval_batch_size"], "data.eval_batch_size", minimum=1)
    _bool(data["drop_last"], "data.drop_last")

    text = config["text"]
    _string(text["model_dir"], "text.model_dir")
    model_layers = (_TEXT_MODEL_LAYERS.get(text["model_id"])
                    if isinstance(text["model_id"], str) else None)
    if model_layers is None:
        raise ConfigError("text.model_id must be a supported 256-dimensional Google BERT")
    _bool(text["frozen"], "text.frozen")
    if "cls_context" in text:
        _bool(text["cls_context"], "text.cls_context")
    if "train_embeddings" in text:
        _bool(text["train_embeddings"], "text.train_embeddings")
    if "learning_rate" in text:
        _number(text["learning_rate"], "text.learning_rate", minimum=0.0)
    if "unfrozen_layers" in text:
        if type(text["unfrozen_layers"]) is not int or not 0 <= text["unfrozen_layers"] <= model_layers:
            raise ConfigError(
                f"text.unfrozen_layers must be an integer from 0 through {model_layers}"
            )
    if text["compute_dtype"] != "float32" or text["stored_dtype"] != "float16":
        raise ConfigError("text.compute_dtype/stored_dtype must be float32/float16")
    if text["attention_implementation"] != "eager":
        raise ConfigError("text.attention_implementation must be eager")

    model = config["model"]
    for key in ("hidden_dim", "encoder_layers", "num_heads", "ffn_dim", "compensation_rounds", "source_classes", "q_dim", "num_classes"):
        _integer(model[key], f"model.{key}", minimum=1)
    if model["hidden_dim"] != 128 or model["encoder_layers"] != 2 or model["num_heads"] != 4 or model["ffn_dim"] != 256:
        raise ConfigError("model hidden dimension, layer, head, or FFN settings do not match Q2")
    if model["compensation_rounds"] != 1 or model["shared_compensator"] is not True:
        raise ConfigError("model compensation must be one shared round")
    if model["source_classes"] != 3 or model["q_dim"] != 5 or model["num_classes"] != 3:
        raise ConfigError("model source_classes/q_dim/num_classes do not match Q2")
    _number(model["dropout"], "model.dropout", minimum=0.0)
    _number(model["layer_norm_eps"], "model.layer_norm_eps", minimum=0.0)
    _equal_list(model["class_names"], ["Negative", "Neutral", "Positive"], "model.class_names")
    score_min = _number(model["score_min"], "model.score_min")
    score_max = _number(model["score_max"], "model.score_max")
    if score_min >= score_max:
        raise ConfigError("model.score_min must be less than model.score_max")

    loss = config["loss"]
    if "late_unimodal_weight" in loss:
        _number(loss["late_unimodal_weight"], "loss.late_unimodal_weight", minimum=0.0)
    if "class_weight_power" in loss:
        _number(loss["class_weight_power"], "loss.class_weight_power", minimum=0.0)
    for key in (
        "regression_weight",
        "huber_delta",
        "msd_weight",
        "shared_temperature",
        "specific_inner_weight",
        "unimodal_inner_weight",
        "span_weight",
        "calibration_weight",
        "consistency_weight",
    ):
        _number(loss[key], f"loss.{key}", minimum=0.0)
    _integer(loss["ramp_epochs"], "loss.ramp_epochs", minimum=0)

    train = config["train"]
    if "teacher_targets" in train:
        _string(train["teacher_targets"], "train.teacher_targets")
    if "stress_text" in train:
        _bool(train["stress_text"], "train.stress_text")
    if "grid_mix" in train:
        _bool(train["grid_mix"], "train.grid_mix")
    if "grid_mix_probability" in train:
        _number(train["grid_mix_probability"], "train.grid_mix_probability", minimum=0.0)
        if train["grid_mix_probability"] > 1:
            raise ConfigError("train.grid_mix_probability must be at most 1")
    for key in ("epochs", "warmup_epochs"):
        _integer(train[key], f"train.{key}", minimum=1)
    if train["optimizer"] != "adamw":
        raise ConfigError("train.optimizer must be adamw")
    for key in ("learning_rate", "min_learning_rate", "adam_eps", "weight_decay", "gradient_clip_norm", "ema_decay"):
        _number(train[key], f"train.{key}", minimum=0.0)
    if train["min_learning_rate"] > train["learning_rate"]:
        raise ConfigError("train.min_learning_rate must not exceed train.learning_rate")
    _equal_list(train["betas"], [0.9, 0.999], "train.betas")
    seeds = _list(train["seeds"], "train.seeds")
    if not seeds or any(type(seed) is not int for seed in seeds):
        raise ConfigError("train.seeds must be a non-empty list of integers")
    eval_epochs = _list(train["eval_epochs"], "train.eval_epochs")
    if any(type(epoch) is not int or epoch < 1 or epoch > train["epochs"] for epoch in eval_epochs):
        raise ConfigError("train.eval_epochs must contain valid epoch integers")
    for key in ("amp", "tf32", "compile"):
        _bool(train[key], f"train.{key}")
    if train["amp"] or train["tf32"] or train["compile"]:
        raise ConfigError("train.amp, train.tf32, and train.compile must be false for Q2")

    evaluation = config["evaluation"]
    _equal_list(evaluation["main_patterns"], ["T", "A", "V", "TA", "TV", "AV"], "evaluation.main_patterns")
    _equal_list(evaluation["main_positions"], ["front", "middle", "rear"], "evaluation.main_positions")
    _equal_list(evaluation["main_rhos"], [0.2, 0.4, 0.6, 0.8], "evaluation.main_rhos")
    _integer(evaluation["bootstrap_repeats"], "evaluation.bootstrap_repeats", minimum=1)
    _integer(evaluation["bootstrap_seed"], "evaluation.bootstrap_seed")
    _integer(evaluation["deployed_seed"], "evaluation.deployed_seed")
    if "class_bias" in evaluation:
        bias = _list(evaluation["class_bias"], "evaluation.class_bias")
        if len(bias) != 3:
            raise ConfigError("evaluation.class_bias must contain exactly 3 numbers")
        for index, value in enumerate(bias):
            _number(value, f"evaluation.class_bias[{index}]")
    if evaluation["deployed_seed"] not in train["seeds"]:
        raise ConfigError("evaluation.deployed_seed must be one of train.seeds")


def _resolve_path(value: str, root: Path) -> str:
    path = Path(value).expanduser()
    return str(path if path.is_absolute() else (root / path).resolve())


def resolve_config_paths(config: Mapping[str, Any], root: str | Path) -> dict[str, Any]:
    """Return a deep copy with Q2-relative path fields made absolute."""
    resolved = _deep_copy(dict(config))
    q2_root = Path(root).expanduser().resolve()
    resolved["project"]["data_root"] = _resolve_path(resolved["project"]["data_root"], q2_root)
    resolved["project"]["output_root"] = _resolve_path(resolved["project"]["output_root"], q2_root)
    resolved["text"]["model_dir"] = _resolve_path(resolved["text"]["model_dir"], q2_root)
    return resolved


def _deep_copy(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _deep_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_deep_copy(item) for item in value]
    return value


def validate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and return a detached plain dictionary."""
    root = _mapping(config, "config")
    _check_keys(root, "config", _SCHEMA["root"])
    for section in _SCHEMA["root"]:
        section_value = _mapping(root[section], section)
        _check_keys(section_value, section, _SCHEMA[section])
        root[section] = section_value
    _validate_values(root)
    return _deep_copy(root)


def load_config(path: str | Path = "configs/default.yaml") -> dict[str, Any]:
    """Load, strictly validate, and resolve a Q2 default YAML config."""
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    validated = validate_config(loaded)
    return resolve_config_paths(validated, config_path.parent.parent)


def load_experiments(path: str | Path = "configs/experiments.yaml") -> list[str]:
    """Load the fixed ten-variant experiment list in its declared order."""
    experiment_path = Path(path).expanduser().resolve()
    if not experiment_path.is_file():
        raise FileNotFoundError(experiment_path)
    with experiment_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    root = _mapping(loaded, "experiments")
    _check_keys(root, "experiments", ("variants",))
    variants = _list(root["variants"], "experiments.variants")
    if variants != list(FIXED_VARIANTS):
        raise ConfigError(
            "experiments.variants must contain the fixed Q2 variants in order: "
            f"{list(FIXED_VARIANTS)!r}; got {variants!r}"
        )
    return list(FIXED_VARIANTS)


__all__ = [
    "ConfigError",
    "FIXED_VARIANTS",
    "load_config",
    "load_experiments",
    "resolve_config_paths",
    "validate_config",
]
