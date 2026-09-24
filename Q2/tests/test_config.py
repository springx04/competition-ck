from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from q2.config import ConfigError, load_config, validate_config  # noqa: E402


def test_cls_context_is_optional_and_defaults_to_disabled_in_legacy_configs():
    config = load_config(Path(__file__).resolve().parents[1] / "configs/default.yaml")
    config["text"].pop("cls_context")
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
