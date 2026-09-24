from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from q2.losses import compute_losses, masked_sample_modality_mean
from q2.data import Normalizer
from q2.masking import apply_span, make_span_mask
from q2.model.network import ModelInput, Student, encode_view
from q2.model.compensation import SharedCompensator
from q2.model.encoders import sinusoidal_positions
from q2.state import infer_state
from q2.trainer import update_ema


def _case(batch_size=2):
    torch.manual_seed(7)
    ids = torch.zeros(batch_size, 50, dtype=torch.long)
    ids[:, 0] = 101
    ids[:, 1:7] = torch.tensor([100, 201, 202, 203, 204, 205])
    ids[:, 7] = 102
    attention = torch.zeros_like(ids)
    attention[:, :8] = 1
    audio = torch.zeros(batch_size, 50, 74)
    vision = torch.zeros(batch_size, 50, 35)
    audio[:, 1:7] = torch.randn(batch_size, 6, 74)
    vision[:, 1:7] = torch.randn(batch_size, 6, 35)
    raw = {"input_ids": ids, "stored_attention": attention,
           "token_type_ids": torch.zeros_like(ids), "audio": audio, "vision": vision}
    return raw


def _model_input(raw):
    state = infer_state(raw)
    text = torch.randn(raw["input_ids"].shape[0], 50, 256)
    text *= state.U[:, :, 0, None]
    return ModelInput(text, raw["audio"], raw["vision"], state.U, state.J, state.q)


class RecordingTextEncoder:
    def __init__(self):
        self.calls = []

    def __call__(self, input_ids, attention_mask, token_type_ids):
        self.calls.append((input_ids.clone(), attention_mask.clone(), token_type_ids.clone()))
        return SimpleNamespace(last_hidden_state=torch.nn.functional.one_hot(
            input_ids % 256, num_classes=256).float())


def _normalizer():
    return Normalizer(np.zeros(74, np.float32), np.ones(74, np.float32),
                      np.zeros(35, np.float32), np.ones(35, np.float32),
                      1, 1, np.array([.3, .3, .4], np.float32), 0.0)


def test_empty_content_uses_train_prior_without_nan():
    raw = _case(1)
    raw["input_ids"].zero_()
    raw["stored_attention"].zero_()
    raw["audio"].zero_()
    raw["vision"].zero_()
    model = Student("full", [0.2, 0.3, 0.5], 0.25).eval()
    with torch.no_grad():
        output = model(_model_input(raw), return_details=True)
    assert torch.allclose(output.logits.softmax(-1)[0], torch.tensor([.2, .3, .5]), atol=1e-6)
    assert output.score.shape == (1,)
    assert output.score.item() == pytest.approx(.25)
    assert not output.B_comp.any()
    assert torch.isfinite(output.logits).all()


def test_calibration_gradient_is_confined_to_estimator():
    raw = _case()
    state = infer_state(raw)
    mask = make_span_mask(raw, state, [{"pattern": "T", "position": "middle", "rho": .4}] * 2)
    damaged = apply_span(raw, mask)
    model = Student("full", [.3, .3, .4], 0).eval()
    output = model(_model_input(damaged), return_details=True)
    assert output.B_comp.any()
    output.e_hat.sum().backward()
    non_estimator = [name for name, p in model.named_parameters()
                     if not name.startswith("estimator.") and p.grad is not None]
    assert not non_estimator, non_estimator
    assert any(p.grad is not None for p in model.estimator.parameters())


def test_task_gradient_does_not_train_estimator():
    raw = _case()
    state = infer_state(raw)
    mask = make_span_mask(raw, state, [{"pattern": "T", "position": "middle", "rho": .4}] * 2)
    damaged = apply_span(raw, mask)
    model = Student("full", [.3, .3, .4], 0).eval()
    output = model(_model_input(damaged), return_details=True)
    (output.logits.square().sum() + output.score.square().sum()).backward()
    assert all(p.grad is None for p in model.estimator.parameters())
    assert any(p.grad is not None for p in model.compensator.parameters())


def test_span_aggregation_weights_sample_modality_pairs_equally():
    values = torch.zeros(2, 4, 3)
    omega = torch.zeros(2, 4, 3, dtype=torch.bool)
    values[0, 0, 0] = 2
    omega[0, 0, 0] = True
    values[1, :3, 1] = torch.tensor([4., 6., 8.])
    omega[1, :3, 1] = True
    assert masked_sample_modality_mean(values, omega).item() == pytest.approx(4.0)


def test_ema_parameters_and_buffers():
    student = Student("no_comp", [.3, .3, .4], 0)
    teacher = Student("no_comp", [.3, .3, .4], 0)
    with torch.no_grad():
        next(student.parameters()).fill_(2)
        next(teacher.parameters()).fill_(1)
    update_ema(teacher, student, .99)
    assert next(teacher.parameters()).flatten()[0].item() == pytest.approx(1.01)


def test_masked_source_keys_have_zero_attention():
    raw = _case(1)
    raw["vision"].zero_()
    state = infer_state(raw)
    mask = make_span_mask(raw, state, [{"pattern": "T", "position": "middle", "rho": .4}])
    damaged = apply_span(raw, mask)
    model = Student("full", [.3, .3, .4], 0).eval()
    with torch.no_grad():
        output = model(_model_input(damaged), return_details=True)
    assert output.attention is not None
    invalid_keys = ~output.U.permute(0, 2, 1).reshape(1, 150)
    assert torch.count_nonzero(output.attention[..., invalid_keys[0]]) == 0
    assert output.time_axis == "official_aligned_positions"
    assert output.source_positions.shape == (150, 2)
    assert torch.equal(output.source_positions[0], torch.tensor([0, 0]))
    assert torch.equal(output.source_positions[50], torch.tensor([1, 0]))
    assert torch.equal(output.source_positions[149], torch.tensor([2, 49]))
    assert output.q.shape == (1, 50, 3, 5)
    assert torch.isfinite(output.logits).all()


def test_text_damage_reencodes_masked_ids_before_student_forward():
    raw = _case(1)
    mask = make_span_mask(raw, infer_state(raw), [{"pattern": "T", "position": "middle", "rho": .4}])
    damaged = apply_span(raw, mask)
    encoder = RecordingTextEncoder()
    view = encode_view(damaged, encoder, _normalizer())
    sent_ids, sent_attention, _ = encoder.calls[0]
    assert mask.P[0, :, 0].any()
    assert torch.all(sent_ids[mask.P[:, :, 0]] == 103)
    assert torch.all(sent_attention[mask.P[:, :, 0]] == 0)
    assert torch.all(view.text_features[mask.P[:, :, 0]] == 0)
    assert torch.equal(damaged["stored_attention"], raw["stored_attention"])


def test_deleted_values_cannot_change_predictions_through_frontend():
    raw = _case(1)
    mask = make_span_mask(raw, infer_state(raw), [{"pattern": "TAV", "position": "middle", "rho": .4}])
    changed = {key: value.clone() for key, value in raw.items()}
    changed["input_ids"][mask.P[:, :, 0]] = 250
    changed["audio"][mask.P[:, :, 1]] = 99
    changed["vision"][mask.P[:, :, 2]] = -99
    first, second = apply_span(raw, mask), apply_span(changed, mask)
    encoder = RecordingTextEncoder()
    normalizer = _normalizer()
    student = Student("full", [.3, .3, .4], 0).eval()
    with torch.no_grad():
        a = student(encode_view(first, encoder, normalizer))
        b = student(encode_view(second, encoder, normalizer))
    assert torch.allclose(a.logits, b.logits, atol=1e-5)
    assert torch.allclose(a.score, b.score, atol=1e-5)


def test_unavailable_encoded_placeholders_cannot_reactivate_attention_keys():
    raw = _case(1)
    raw["vision"].zero_()
    mask = make_span_mask(raw, infer_state(raw), [{"pattern": "TA", "position": "middle", "rho": .4}])
    view = _model_input(apply_span(raw, mask))
    altered = ModelInput(view.text_features.clone(), view.audio_norm.clone(),
                         view.vision_norm.clone(), view.U, view.J, view.q)
    altered.text_features[~view.U[:, :, 0]] = 1000
    altered.audio_norm[~view.U[:, :, 1]] = -1000
    altered.vision_norm[~view.U[:, :, 2]] = 2000
    model = Student("full", [.3, .3, .4], 0).eval()
    with torch.no_grad():
        original = model(view, return_details=True)
        changed = model(altered, return_details=True)
    assert torch.allclose(original.logits, changed.logits, atol=1e-5)
    assert torch.allclose(original.score, changed.score, atol=1e-5)
    assert torch.allclose(original.attention, changed.attention, atol=1e-5)


def test_compensation_is_one_round_and_independent_of_target_order():
    torch.manual_seed(14)
    raw = _case(1)
    mask = make_span_mask(raw, infer_state(raw), [{"pattern": "TA", "position": "middle", "rho": .4}])
    state = infer_state(apply_span(raw, mask))
    shape = (1, 50, 3, 128)
    c = torch.randn(shape) * state.U[..., None]
    s = torch.randn(shape) * state.U[..., None]
    f = torch.randn(shape) * state.U[..., None]
    module = SharedCompensator().eval()
    position = sinusoidal_positions(50, 128, torch.device("cpu"))
    embeddings = torch.randn(3, 128)
    with torch.no_grad():
        forward = module(c, s, f, state.U, state.J, position, embeddings,
                         return_attention=True, target_order=(0, 1, 2))
        reversed_order = module(c, s, f, state.U, state.J, position, embeddings,
                                return_attention=True, target_order=(2, 1, 0))
    for first, second in zip(forward, reversed_order):
        assert torch.allclose(first, second, atol=1e-5)
    valid_keys = state.U.permute(0, 2, 1).reshape(1, 150)
    assert torch.count_nonzero(forward[2][..., ~valid_keys[0]]) == 0
