from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import q2.__main__ as cli  # noqa: E402
import q2.export as export  # noqa: E402
from q2.model.network import TUNED_VARIANTS  # noqa: E402


class _Normalizer:
    class_prior = [0.3, 0.3, 0.4]
    score_prior = 0.0


class _Encoder:
    def __init__(self):
        self.loaded = None
        self.saved = []

    def load_state_dict(self, state):
        self.loaded = state

    def eval(self):
        return self

    def float(self):
        return self

    def half(self):
        return self

    def save_pretrained(self, directory, safe_serialization=True):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "tuned-state.txt").write_text(repr(self.loaded), encoding="utf-8")
        self.saved.append((directory, safe_serialization))


class _Student:
    def __init__(self, *args, **kwargs):
        self.loaded = None

    def to(self, device):
        return self

    def eval(self):
        return self

    def load_state_dict(self, state, **kwargs):
        self.loaded = state


def test_bundle_forwards_restored_cls_context_to_frontend(monkeypatch):
    class _ForwardingStudent:
        def eval(self):
            return self

        def __call__(self, model_input, return_details=True):
            return model_input

    calls = {}
    monkeypatch.setattr(
        export,
        "encode_view",
        lambda raw, encoder, normalizer, **kwargs: calls.update(kwargs) or kwargs,
    )
    bundle = export.Bundle(_ForwardingStudent(), _Encoder(), _Normalizer(), "cpu",
                           cls_context=True)

    assert bundle.predict_raw({}, return_details=False)["cls_context"] is True
    assert calls["cls_context"] is True


@pytest.mark.parametrize("variant", TUNED_VARIANTS)
def test_evaluate_best_restores_tuned_state_uses_output_root_and_disables_cache(monkeypatch, tmp_path, variant):
    output_root = tmp_path / "training-output"
    checkpoint = {"student": {"student": torch.tensor(1)}, "text_encoder": {"bert": torch.tensor(2)}}
    encoder = _Encoder()
    calls = {}

    monkeypatch.setattr(cli.torch, "load", lambda path, **kwargs: calls.setdefault("checkpoint_path", Path(path)) and checkpoint)
    monkeypatch.setattr(cli.Normalizer, "load", lambda path: _Normalizer())
    monkeypatch.setattr(cli, "AlignedDataset", lambda path: object())
    monkeypatch.setattr(cli, "load_checkpoint_text_encoder",
                        lambda root, actual_variant, actual_checkpoint, device:
                        calls.update(encoder=(root, actual_variant, actual_checkpoint, device)) or encoder)
    monkeypatch.setattr(cli, "evaluate_model", lambda *args, **kwargs: calls.update(evaluate=(args, kwargs)) or ([], []))
    monkeypatch.setattr(cli.np, "load", lambda *args, **kwargs: pytest.fail("tuned evaluation opened a text cache"))
    monkeypatch.setattr("q2.model.network.Student", _Student)

    config = {"project": {"output_root": str(output_root)}, "data": {"eval_batch_size": 7}}
    cli.evaluate_best(tmp_path, config, variant, 1111, stress=False)

    assert calls["checkpoint_path"] == output_root / "runs" / variant / "seed_1111" / "best.pt"
    assert calls["encoder"][1] == variant
    assert calls["evaluate"][0][6] is None
    assert calls["evaluate"][1]["batch_size"] == 7


def test_load_training_best_restores_checkpoint_text_encoder(monkeypatch, tmp_path):
    encoder = _Encoder()
    state = {"bert": torch.tensor(3)}
    checkpoint = {"student": {"student": torch.tensor(1)}, "text_encoder": state,
                  "config": {"text": {"cls_context": True}, "data": {"clip_z": 2.5}}}
    monkeypatch.setattr(export.Normalizer, "load", lambda path: _Normalizer())
    monkeypatch.setattr(export.torch, "load", lambda path, **kwargs: checkpoint)
    monkeypatch.setattr(export, "Student", _Student)
    monkeypatch.setattr(export, "load_text_encoder", lambda path, device: encoder)

    bundle = export.load_training_best(
        tmp_path,
        {"variant": "late_tune_aug", "checkpoint": "runs/late_tune_aug/seed_1111/best.pt"},
        device="cpu",
    )

    assert bundle.text_encoder is encoder
    assert encoder.loaded is state
    assert bundle.cls_context is True
    assert bundle.normalizer.clip_z == 2.5


@pytest.mark.parametrize("selected_bias", [[-.3, .025, 0], None])
def test_selection_checkpoint_and_bias_agree_in_evaluation_and_prediction(monkeypatch, tmp_path, selected_bias):
    checkpoint = {"student": {}, "text_encoder": {},
                  "config": {"evaluation": {"class_bias": [1, 2, 3]}}}
    paths, calls = [], []
    monkeypatch.setattr(cli.torch, "load", lambda path, **kwargs: paths.append(Path(path)) or checkpoint)
    monkeypatch.setattr(cli.Normalizer, "load", lambda path: _Normalizer())
    monkeypatch.setattr(cli, "AlignedDataset", lambda path: object())
    monkeypatch.setattr(cli, "load_checkpoint_text_encoder", lambda *args: _Encoder())
    monkeypatch.setattr(export, "load_checkpoint_text_encoder", lambda *args: _Encoder())
    monkeypatch.setattr(cli, "evaluate_model", lambda *args, **kwargs: calls.append(kwargs) or ([], []))
    monkeypatch.setattr("q2.model.network.Student", _Student)
    monkeypatch.setattr(export, "Student", _Student)
    selection = {"variant": "late_attn_tune", "seed": 1111,
                 "checkpoint": "experiments/chosen/best.pt", "class_bias": selected_bias}
    config = {"project": {"output_root": "wrong-directory"}, "data": {"eval_batch_size": 4},
              "evaluation": {"class_bias": [99, 99, 99]}}
    cli.evaluate_best(tmp_path, config, selection["variant"], 1111, selection=selection)
    bundle = export.load_training_best(tmp_path, selection, device="cpu")
    assert paths == [tmp_path / selection["checkpoint"]] * 2
    assert calls[0]["class_bias"] == bundle.class_bias
    if selected_bias is None:
        assert bundle.class_bias is None
    else:
        assert bundle.class_bias == pytest.approx(selected_bias)


def test_old_checkpoint_without_frontend_option_defaults_to_token_only(monkeypatch, tmp_path):
    encoder = _Encoder()
    checkpoint = {"student": {"student": torch.tensor(1)}, "text_encoder": {"bert": torch.tensor(3)}}
    monkeypatch.setattr(export.Normalizer, "load", lambda path: _Normalizer())
    monkeypatch.setattr(export.torch, "load", lambda path, **kwargs: checkpoint)
    monkeypatch.setattr(export, "Student", _Student)
    monkeypatch.setattr(export, "load_text_encoder", lambda path, device: encoder)

    bundle = export.load_training_best(
        tmp_path,
        {"variant": "late_tune_aug", "checkpoint": "runs/late_tune_aug/seed_1111/best.pt"},
        device="cpu",
    )

    assert bundle.cls_context is False
    assert not hasattr(bundle.normalizer, "clip_z")


@pytest.mark.parametrize("variant", TUNED_VARIANTS)
def test_tuned_loading_rejects_checkpoint_without_text_state(monkeypatch, tmp_path, variant):
    monkeypatch.setattr(export, "load_text_encoder", lambda path, device: _Encoder())

    with pytest.raises(ValueError, match="missing text_encoder weights"):
        export.load_checkpoint_text_encoder(tmp_path, variant, {}, "cpu")


def test_export_bundle_serializes_tuned_encoder_into_directory_and_zip(monkeypatch, tmp_path):
    root = tmp_path
    for directory in (root / "data/processed", root / "models/text_encoder", root / "src/q2",
                      root / "licenses", root / "outputs"):
        directory.mkdir(parents=True)
    (root / "data/processed/normalizer.npz").write_bytes(b"normalizer")
    (root / "models/text_encoder/base.txt").write_text("base", encoding="utf-8")
    (root / "src/q2/module.py").write_text("# package", encoding="utf-8")
    (root / "licenses/EBMC-LICENSE").write_text("license", encoding="utf-8")
    (root / "THIRD_PARTY.md").write_text("third party", encoding="utf-8")
    (root / "requirements.txt").write_text("numpy", encoding="utf-8")
    (root / "outputs/q2_predictions_aligned.csv").write_text("sample_id\n", encoding="utf-8")

    encoder = _Encoder()
    checkpoint = {"student": {"student": torch.tensor(1)}, "text_encoder": {"bert": torch.tensor(4)},
                  "config": {"text": {"cls_context": True}, "data": {"clip_z": 3.0}}}
    monkeypatch.setattr(export.Normalizer, "load", lambda path: _Normalizer())
    monkeypatch.setattr(export.torch, "load", lambda path, **kwargs: checkpoint)
    monkeypatch.setattr(export, "Student", _Student)
    monkeypatch.setattr(export, "save_file", lambda state, path: Path(path).write_bytes(b"student"))
    monkeypatch.setattr(export, "load_checkpoint_text_encoder", lambda *args, **kwargs: encoder)

    result = export.export_bundle(
        root,
        {"variant": "late_tune_aug", "checkpoint": "runs/late_tune_aug/seed_1111/best.pt",
         "class_bias": [-.3, .025, 0]},
    )

    marker = Path(result["directory"]) / "assets/text_encoder/tuned-state.txt"
    assert marker.read_text(encoding="utf-8")
    model_config = __import__("yaml").safe_load(
        (Path(result["directory"]) / "model_config.yaml").read_text(encoding="utf-8"))
    assert model_config["text"]["cls_context"] is True
    assert model_config["data"]["clip_z"] == 3.0
    assert model_config["evaluation"]["class_bias"] == pytest.approx([-.3, .025, 0])
    with export.zipfile.ZipFile(result["zip"]) as archive:
        assert "q2_inference/assets/text_encoder/tuned-state.txt" in archive.namelist()
