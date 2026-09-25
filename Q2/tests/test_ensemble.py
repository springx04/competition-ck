from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from safetensors.torch import load_file
from transformers import BertConfig, BertModel


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import q2.ensemble as ensemble  # noqa: E402
from q2.data import Normalizer  # noqa: E402
from q2.model.network import Student  # noqa: E402


class _Member:
    def __init__(self, output):
        self.output = output
        self.calls = []

    def predict_raw(self, raw_batch, return_details=True):
        self.calls.append(return_details)
        return self.output


def _output(logits, score, U, marker):
    return SimpleNamespace(
        logits=torch.tensor(logits, dtype=torch.float32),
        score=torch.tensor(score, dtype=torch.float32),
        U=U,
        marker=marker,
        source=torch.tensor([[0]], dtype=torch.long),
    )


def test_ensemble_averages_raw_predictions_and_biases_only_observed_rows():
    U = torch.tensor([[[False, False, False]], [[True, False, False]]])
    members = [
        _Member(_output([[1, 2, 3], [2, 4, 6]], [1, 2], U, "first")),
        _Member(_output([[3, 4, 5], [4, 6, 8]], [3, 4], U, "second")),
        _Member(_output([[5, 6, 7], [6, 8, 10]], [5, 6], U, "third")),
    ]

    result = ensemble.EnsembleBundle(members, device="cpu",
                                     class_bias=[.1, .2, .3]).predict_raw(
                                         {"unused": "raw"}, return_details=False)

    assert torch.allclose(result.logits[0], torch.tensor([3., 4., 5.]))
    assert torch.allclose(result.logits[1], torch.tensor([4.1, 6.2, 8.3]))
    assert torch.equal(result.score, torch.tensor([3., 4.]))
    assert result.marker == "first"
    assert result.source.item() == 0
    assert all(member.calls == [False] for member in members)


def test_load_training_ensemble_suppresses_member_bias(monkeypatch, tmp_path):
    calls = []

    class Dummy:
        def predict_raw(self, raw_batch, return_details=True):
            raise AssertionError("not called")

    def fake_load(root, member, device, round_text_for_export=False):
        calls.append((root, member, device, round_text_for_export))
        return Dummy()

    monkeypatch.setattr(ensemble, "load_training_best", fake_load)
    selection = {
        "members": [
            {"variant": "late_attn_tune", "seed": 1111, "checkpoint": "a.pt"},
            {"variant": "late_attn_tune", "seed": 1112, "checkpoint": "b.pt"},
        ],
        "class_bias": [-.3, .025, 0.0],
    }

    bundle = ensemble.load_training_ensemble(tmp_path, selection, "cpu", True)

    assert bundle.class_bias == pytest.approx([-.3, .025, 0.0])
    assert [call[1]["class_bias"] for call in calls] == [None, None]
    assert all(call[3] for call in calls)


def _make_export_root(tmp_path, mismatched=False):
    root = Path(tmp_path)
    (root / "data/processed").mkdir(parents=True)
    (root / "models/text_encoder").mkdir(parents=True)
    (root / "src/q2").mkdir(parents=True)
    (root / "licenses").mkdir(parents=True)
    (root / "src/q2/__init__.py").write_text("", encoding="utf-8")
    (root / "src/q2/ensemble.py").write_text("# copied package source\n", encoding="utf-8")
    (root / "licenses/BERT-LICENSE").write_text("license\n", encoding="utf-8")
    (root / "THIRD_PARTY.md").write_text("third party\n", encoding="utf-8")
    (root / "requirements.txt").write_text("torch\n", encoding="utf-8")

    normalizer = Normalizer(
        np.zeros(74, dtype=np.float32), np.ones(74, dtype=np.float32),
        np.zeros(35, dtype=np.float32), np.ones(35, dtype=np.float32),
        0, 0, np.array([.2, .3, .5], dtype=np.float32), 0.25,
    )
    normalizer.save(root / "data/processed/normalizer.npz")

    config = BertConfig(
        vocab_size=32, hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
        intermediate_size=32, max_position_embeddings=16, type_vocab_size=2,
    )
    base = BertModel(config, add_pooling_layer=False).eval()
    base.save_pretrained(root / "models/text_encoder", safe_serialization=True)
    (root / "models/text_encoder/vocab.txt").write_text(
        "\n".join(f"token{i}" for i in range(32)) + "\n", encoding="utf-8")

    selection_members = []
    for index, seed in enumerate((1111, 1112, 1113)):
        encoder = BertModel(config, add_pooling_layer=False).eval()
        encoder.load_state_dict(base.state_dict(), strict=True)
        if mismatched and index == 1:
            with torch.no_grad():
                encoder.embeddings.word_embeddings.weight[0, 0] += 1
        student = Student("late_attn_tune", normalizer.class_prior,
                          normalizer.score_prior, include_aux_heads=False).eval()
        checkpoint_path = root / f"runs/late_attn_tune/seed_{seed}/best.pt"
        checkpoint_path.parent.mkdir(parents=True)
        torch.save({
            "student": student.state_dict(),
            "text_encoder": encoder.state_dict(),
            "config": {"text": {"model_dir": "models/text_encoder"},
                       "data": {"clip_z": 2.0}},
        }, checkpoint_path)
        selection_members.append({
            "variant": "late_attn_tune", "seed": seed,
            "checkpoint": str(checkpoint_path.relative_to(root)),
        })
    return root, {"members": selection_members, "class_bias": [-.3, .025, 0.0]}, base


def test_export_reload_shares_embeddings_and_restores_fp32_bert(tmp_path):
    root, selection, _base = _make_export_root(tmp_path)
    destination = root / "delivery/ensemble"

    result = ensemble.export_ensemble(root, selection, destination)
    assert result["members"] == 3
    assert (destination / "run_inference.py").is_file()
    assert (destination / "licenses/BERT-LICENSE").is_file()
    assert not any(destination.rglob("best.pt"))

    config = json.loads((destination / "ensemble_config.json").read_text(encoding="utf-8"))
    assert [member["source_checkpoint"] for member in config["members"]] == [
        member["checkpoint"] for member in selection["members"]
    ]
    shared = load_file(str(destination / "assets/text_encoder/shared_embeddings.safetensors"))
    private = load_file(str(destination / "assets/text_encoder/member_0.safetensors"))
    assert all(value.dtype == torch.float16 for key, value in shared.items()
               if value.is_floating_point())
    assert all(value.dtype == torch.float16 for key, value in private.items()
               if value.is_floating_point())
    students = load_file(str(destination / "students/student_0.safetensors"))
    assert all(value.dtype == torch.float32 for value in students.values())

    loaded = ensemble.load_ensemble(destination, device="cpu")
    assert len(loaded.members) == 3
    assert loaded.class_bias == pytest.approx(selection["class_bias"])
    assert all(all(value.dtype == torch.float32
                   for value in member.text_encoder.state_dict().values()
                   if value.is_floating_point()) for member in loaded.members)
    assert all(member.text_encoder.training is False for member in loaded.members)

    with pytest.raises(FileExistsError):
        ensemble.export_ensemble(root, selection, destination)


def test_export_rejects_nonidentical_tuned_embeddings_before_creating_destination(tmp_path):
    root, selection, _base = _make_export_root(tmp_path, mismatched=True)
    destination = root / "delivery/ensemble"

    with pytest.raises(ValueError, match="embeddings differ"):
        ensemble.export_ensemble(root, selection, destination)
    assert not destination.exists()
