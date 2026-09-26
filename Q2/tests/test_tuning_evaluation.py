from pathlib import Path
from types import SimpleNamespace
import sys
import json

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from q2 import evaluate
from q2.data import Normalizer


def test_raw_predictor_receives_actual_corruption_and_owns_calibration(tmp_path, monkeypatch):
    import pytest
    grid = [{"name": "T_middle_0.4", "pattern": "T", "rho": .4, "position": "middle"}]
    monkeypatch.setattr(evaluate, "evaluation_grid", lambda **kwargs: grid)

    class Dataset:
        directory = tmp_path / "data/processed/test"
        metadata = {"id": ["a", "b", "c"], "video_id": ["a", "b", "c"]}

        def __len__(self):
            return 3

        def __getitem__(self, index):
            ids = torch.tensor([101, 201, 202, 203, 204, 205, 102] + [0] * 43)
            return {"input_ids": ids, "stored_attention": (ids != 0).long(),
                    "token_type_ids": torch.zeros_like(ids), "audio": torch.zeros(50, 74),
                    "vision": torch.zeros(50, 35), "class_id": torch.tensor(index),
                    "score": torch.tensor(float(index - 1)), "sample_id": self.metadata["id"][index],
                    "sample_index": index}

    class Predictor:
        def __init__(self):
            self.calls = []

        def predict_raw(self, raw, return_details=False):
            self.calls.append(raw["input_ids"].clone())
            value = (raw["input_ids"] == 103).sum(-1).float()
            return SimpleNamespace(logits=torch.stack((value, -value, value * 0), -1), score=value)

    data, predictor = Dataset(), Predictor()
    masks = tmp_path / "masks"
    evaluate.make_fixed_masks(data, masks, include_stress=False)
    monkeypatch.setattr(evaluate, "encode_view", lambda *a, **kw: pytest.fail("predictor encoded twice"))
    _, predictions = evaluate.evaluate_model(None, None, None, data, "cpu", masks, None,
                                             predictor=predictor, return_predictions=True)
    assert len(predictor.calls) == 2
    assert not (predictor.calls[0] == 103).any()
    assert (predictor.calls[1] == 103).any()
    assert all(row["damaged"] for row in predictions if row["scenario"] != "clean")
    with pytest.raises(ValueError, match="owns its calibration"):
        evaluate.evaluate_model(None, None, None, data, "cpu", masks, None,
                                predictor=predictor, class_bias=[0, 1, 0])
from q2.trainer import learning_rate, optimizer_for, set_learning_rate, _rng_state, _restore_rng
from q2.masking import sample_train_descriptor
from q2.data import Normalizer


def test_live_evaluation_uses_current_encoder_for_clean_and_corrupt(tmp_path, monkeypatch):
    grid = [{"name": "T_middle_0.4", "pattern": "T", "rho": .4, "position": "middle"}]
    monkeypatch.setattr(evaluate, "evaluation_grid", lambda **kwargs: grid)

    class Dataset:
        directory = tmp_path / "data/processed/valid"
        metadata = {"id": ["a", "b", "c"], "video_id": ["a", "b", "c"]}

        def __len__(self):
            return 3

        def __getitem__(self, index):
            ids = torch.tensor([101, 201, 202, 203, 204, 205, 102] + [0] * 43)
            return {"input_ids": ids, "stored_attention": (ids != 0).long(),
                    "token_type_ids": torch.zeros_like(ids), "audio": torch.zeros(50, 74),
                    "vision": torch.zeros(50, 35), "class_id": torch.tensor(index),
                    "score": torch.tensor(float(index - 1)), "sample_id": self.metadata["id"][index],
                    "sample_index": index}

    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))
            self.calls = []

        def forward(self, input_ids, **kwargs):
            self.calls.append(input_ids.clone())
            return SimpleNamespace(last_hidden_state=self.weight * torch.ones(*input_ids.shape, 256))

    class Student(torch.nn.Module):
        def forward(self, batch):
            score = batch.text_features.mean((1, 2))
            return SimpleNamespace(logits=torch.stack((score, -score, score * 0), -1),
                                   score=score, J=batch.J)

    data = Dataset()
    mask_dir = tmp_path / "data/masks/valid"
    evaluate.make_fixed_masks(data, mask_dir, include_stress=False)
    # Reading any persisted feature cache would fail this test.
    monkeypatch.setattr(np, "load", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("stale cache read")))
    norm = Normalizer(np.zeros(74), np.ones(74), np.zeros(35), np.ones(35), 0, 0,
                      np.ones(3) / 3, 0)
    encoder = Encoder()
    _, first = evaluate.evaluate_model(Student(), encoder, norm, data, "cpu", mask_dir,
                                       None, return_predictions=True)
    with torch.no_grad():
        encoder.weight.fill_(2)
    _, second = evaluate.evaluate_model(Student(), encoder, norm, data, "cpu", mask_dir,
                                        None, return_predictions=True)
    assert len(encoder.calls) == 4
    assert not (encoder.calls[0] == 103).any()
    assert (encoder.calls[1] == 103).any()
    assert all(np.allclose(np.array(b["logits"]), 2 * np.array(a["logits"]))
               for a, b in zip(first, second))
    assert not (tmp_path / ".cache").exists()


def test_optimizer_group_scales_are_persistent_and_rng_resume_matches():
    optimizer = optimizer_for(torch.nn.Linear(3, 2), torch.nn.Linear(3, 2), text_lr=3e-5)
    assert [group["lr_scale"] for group in optimizer.param_groups] == [1, 1, .1]
    rates = [learning_rate(e, 20, 2) for e in range(1, 21)]
    assert rates[-1] == 3e-5
    assert rates[-2] > rates[-1]
    for rate in rates:
        set_learning_rate(optimizer, rate)
    assert np.allclose([group["lr"] for group in optimizer.param_groups], [3e-5, 3e-5, 3e-6])
    generator = torch.Generator().manual_seed(123)
    state = _rng_state(generator)
    expected = torch.rand(5, generator=generator)
    _restore_rng(state, generator)
    assert torch.equal(torch.rand(5, generator=generator), expected)


def test_short_curriculum_reaches_severe_dual_modality_missing():
    descriptors = [sample_train_descriptor(1111, 20, i, total_epochs=20, warmup_epochs=2)
                   for i in range(100)]
    assert any(len(d["pattern"]) == 2 and d["rho"] > .6 for d in descriptors)
    assert all(sample_train_descriptor(1111, 2, i, total_epochs=20, warmup_epochs=2)["rho"] == 0
               for i in range(10))
    # Original 60-epoch settings retain their early single-modality stage.
    early = [sample_train_descriptor(1111, 15, i) for i in range(100)]
    assert all(len(d["pattern"]) <= 1 and d["rho"] <= .2 for d in early)
    stress = [sample_train_descriptor(1111, 10, i, total_epochs=20, warmup_epochs=2,
                                      stress_text=True) for i in range(200)]
    assert sum(d["pattern"] == "T" for d in stress) > 50
    assert min(d["rho"] for d in stress) >= .35


def test_train_only_zscore_clipping_preserves_missing_zero_rows():
    normalizer = Normalizer(np.zeros(1), np.ones(1), np.zeros(1), np.ones(1), 1, 1,
                            np.ones(3) / 3, 0)
    normalizer.clip_z = 3
    raw = {"audio": torch.tensor([[[10.0], [0.0]]]),
           "vision": torch.zeros(1, 2, 1)}
    transformed = normalizer.transform(raw, "audio")
    assert transformed[0, 0, 0].item() == 3
    assert transformed[0, 1, 0].item() == 0
