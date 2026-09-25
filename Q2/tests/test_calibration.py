from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from q2 import evaluate, export
from q2.data import Normalizer
from q2.model.network import Student


class EmptyAndObservedDataset:
    metadata = {"id": ["empty", "observed"], "video_id": ["a", "b"]}

    def __len__(self):
        return 2

    def __getitem__(self, index):
        # MASK slots preserve structure, but contain no observed text.
        ids = torch.tensor([101, 103, 102] + [0] * 47)
        audio = torch.zeros(50, 74)
        if index == 1:
            audio[1] = 1
        return {"input_ids": ids, "stored_attention": (ids != 0).long(),
                "token_type_ids": torch.zeros_like(ids), "audio": audio,
                "vision": torch.zeros(50, 35), "sample_id": self.metadata["id"][index],
                "sample_index": index, "class_id": torch.tensor(index),
                "score": torch.tensor(float(index))}


def test_evaluation_and_bundle_keep_empty_prior_and_correct_observed_rows(tmp_path, monkeypatch):
    import json
    from q2.data import collate_raw
    monkeypatch.setattr(evaluate, "evaluation_grid", lambda **kwargs: [])
    (tmp_path / "index.json").write_text(json.dumps([]))
    norm = Normalizer(np.zeros(74, dtype=np.float32), np.ones(74, dtype=np.float32),
                      np.zeros(35, dtype=np.float32), np.ones(35, dtype=np.float32),
                      0, 0, np.array([.2, .3, .5], dtype=np.float32), .4)
    class Encoder(torch.nn.Module):
        def forward(self, input_ids, **kwargs):
            return SimpleNamespace(last_hidden_state=torch.ones(*input_ids.shape, 256))
    encoder = Encoder()
    student = Student("late_attn_tune", norm.class_prior, norm.score_prior).eval()
    data = EmptyAndObservedDataset()
    bias = [-.3, .025, 0]
    metrics, predictions = evaluate.evaluate_model(student, encoder, norm, data, "cpu", tmp_path,
                                              None, return_predictions=True, class_bias=bias)
    assert metrics[0]["empty_content"] == 1
    raw = collate_raw([data[0], data[1]])
    plain = export.Bundle(student, encoder, norm, "cpu").predict_raw(raw)
    corrected = export.Bundle(student, encoder, norm, "cpu", class_bias=bias).predict_raw(raw)
    assert raw["input_ids"][0, 1] == 103
    assert torch.allclose(corrected.logits[0].softmax(-1), torch.tensor([.2, .3, .5]))
    assert corrected.score[0].item() == pytest.approx(.4)
    assert torch.equal(corrected.logits[0], plain.logits[0])
    assert torch.allclose(corrected.logits[1], plain.logits[1] + torch.tensor(bias))
    assert np.allclose([row["logits"] for row in predictions], corrected.logits.numpy())


@pytest.mark.parametrize("bias", [[0, 1], [0, float("nan"), 0], [0, float("inf"), 0]])
def test_bundle_rejects_invalid_bias(bias):
    with pytest.raises(ValueError, match="finite length-3"):
        export.Bundle(torch.nn.Identity(), torch.nn.Identity(), None, "cpu", class_bias=bias)
