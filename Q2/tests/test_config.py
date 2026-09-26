from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from q2.config import ConfigError, load_config, validate_config  # noqa: E402


def test_cls_context_is_optional_and_defaults_to_disabled_in_legacy_configs():
    config = load_config(Path(__file__).resolve().parents[1] / "configs/default.yaml")
    config["text"].pop("cls_context", None)
    validated = validate_config(config)
    assert "cls_context" not in validated["text"]


def test_cls_context_must_be_boolean_when_declared():
    config = load_config(Path(__file__).resolve().parents[1] / "configs/default.yaml")
    config["text"]["cls_context"] = "true"
    with pytest.raises(ConfigError, match=r"text\.cls_context must be a boolean"):
        validate_config(config)


def test_class_bias_accepts_exactly_three_finite_numbers():
    config = load_config(Path(__file__).resolve().parents[1] / "configs/default.yaml")
    config["evaluation"]["class_bias"] = [-0.3, 0.025, 0.0]
    assert validate_config(config)["evaluation"]["class_bias"] == [-0.3, 0.025, 0.0]
    config["evaluation"]["class_bias"] = [0.0, 1.0]
    with pytest.raises(ConfigError, match="exactly 3"):
        validate_config(config)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_class_bias_rejects_nonfinite_values(value):
    config = load_config(Path(__file__).resolve().parents[1] / "configs/default.yaml")
    config["evaluation"]["class_bias"] = [0.0, value, 0.0]
    with pytest.raises(ConfigError, match="must be finite"):
        validate_config(config)


def test_train_embeddings_requires_boolean():
    config = load_config(Path(__file__).resolve().parents[1] / "configs/default.yaml")
    config["text"]["train_embeddings"] = True
    validate_config(config)
    config["text"]["train_embeddings"] = "true"
    with pytest.raises(ConfigError, match="text.train_embeddings must be a boolean"):
        validate_config(config)


def test_text_stress_training_option_requires_boolean():
    config = load_config(Path(__file__).resolve().parents[1] / "configs/default.yaml")
    config["train"]["stress_text"] = True
    validate_config(config)
    config["train"]["stress_text"] = "false"
    with pytest.raises(ConfigError, match="train.stress_text must be a boolean"):
        validate_config(config)


@pytest.mark.parametrize("model_id,max_layers", [
    ("google/bert_uncased_L-4_H-256_A-4", 4),
    ("google/bert_uncased_L-8_H-256_A-4", 8),
])
def test_unfrozen_layers_limit_matches_selected_text_model(model_id, max_layers):
    config = load_config(Path(__file__).resolve().parents[1] / "configs/default.yaml")
    config["text"]["model_id"] = model_id
    for value in (0, max_layers):
        config["text"]["unfrozen_layers"] = value
        assert validate_config(config)["text"]["unfrozen_layers"] == value
    config["text"]["unfrozen_layers"] = max_layers + 1
    with pytest.raises(ConfigError, match=rf"from 0 through {max_layers}"):
        validate_config(config)


@pytest.mark.parametrize("probability", [0.0, 1.0])
def test_grid_mix_probability_accepts_closed_interval_endpoints(probability):
    config = load_config(Path(__file__).resolve().parents[1] / "configs/default.yaml")
    config["train"]["grid_mix_probability"] = probability
    assert validate_config(config)["train"]["grid_mix_probability"] == probability
