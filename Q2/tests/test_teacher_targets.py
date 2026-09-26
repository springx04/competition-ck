from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from q2.losses import compute_losses  # noqa: E402
from q2.teacher_targets import TrainTeacherTargets  # noqa: E402


SAMPLE_IDS = ["train-0", "train-1", "train-2"]
LOGITS = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float32)
SCORES = np.array([-1.0, 0.0, 1.0], dtype=np.float32)


def _write_targets(path, *, source_split="train", sample_ids=SAMPLE_IDS,
                   logits=LOGITS, score=SCORES):
    np.savez(path, source_split=np.array(source_split), sample_id=np.asarray(sample_ids),
             logits=np.asarray(logits), score=np.asarray(score))


def test_teacher_targets_reject_wrong_split_order_and_length(tmp_path):
    path = tmp_path / "teacher.npz"

    _write_targets(path, source_split="valid")
    with pytest.raises(ValueError, match="train"):
        TrainTeacherTargets.load(path, SAMPLE_IDS)

    _write_targets(path)
    with pytest.raises(ValueError, match="IDs/order"):
        TrainTeacherTargets.load(path, [SAMPLE_IDS[1], SAMPLE_IDS[0], SAMPLE_IDS[2]])
    with pytest.raises(ValueError, match="IDs/order"):
        TrainTeacherTargets.load(path, SAMPLE_IDS[:-1])


@pytest.mark.parametrize(
    "field,value",
    [
        ("logits", np.zeros((3, 2), dtype=np.float32)),
        ("score", np.zeros((3, 1), dtype=np.float32)),
        ("logits", np.array([[np.nan, 0.0, 0.0]] * 3, dtype=np.float32)),
        ("score", np.array([0.0, np.inf, 0.0], dtype=np.float32)),
    ],
)
def test_teacher_targets_reject_invalid_shape_or_nonfinite_values(tmp_path, field, value):
    path = tmp_path / "teacher.npz"
    arrays = {"logits": LOGITS, "score": SCORES}
    arrays[field] = value
    _write_targets(path, logits=arrays["logits"], score=arrays["score"])

    with pytest.raises(ValueError):
        TrainTeacherTargets.load(path, SAMPLE_IDS)


def test_teacher_target_batch_preserves_reordered_duplicates_and_is_detached(tmp_path):
    path = tmp_path / "teacher.npz"
    _write_targets(path)
    targets = TrainTeacherTargets.load(path, SAMPLE_IDS)

    batch = targets.batch([2, 0, 2], device="cpu")

    assert torch.equal(batch.logits, torch.tensor(LOGITS[[2, 0, 2]]))
    assert torch.equal(batch.score, torch.tensor(SCORES[[2, 0, 2]]))
    assert not batch.logits.requires_grad
    assert not batch.score.requires_grad
    assert batch.F is None


def _consistency_case():
    clean = SimpleNamespace(
        logits=torch.tensor([[1.0, -1.0, 0.0], [-0.2, 0.5, 0.1]], requires_grad=True),
        score=torch.tensor([0.2, -0.4], requires_grad=True),
        U=torch.ones(2, 1, 3, dtype=torch.bool),
        B_comp=torch.zeros(2, 1, 3, dtype=torch.bool),
        F_hat=None,
        unimodal_logits=None,
    )
    corrupt = SimpleNamespace(
        logits=torch.tensor([[-0.8, 1.1, 0.2], [0.6, -0.7, 0.3]], requires_grad=True),
        score=torch.tensor([-0.7, 0.8], requires_grad=True),
        U=clean.U,
        B_comp=clean.B_comp,
        F_hat=None,
        unimodal_logits=None,
    )
    teacher = SimpleNamespace(
        logits=torch.tensor([[1.5, -0.3, 0.0], [-0.4, 0.2, 1.2]], requires_grad=True),
        score=torch.tensor([0.7, -0.2], requires_grad=True),
        F=None,
    )
    labels = {"class_id": torch.tensor([0, 2]), "score": torch.tensor([0.0, 0.5])}
    perturbation = SimpleNamespace(P=torch.zeros_like(clean.U))
    state0 = SimpleNamespace(U=clean.U)
    student = SimpleNamespace(variant="late_attn_tune", use_msd=False)
    return clean, corrupt, teacher, labels, perturbation, state0, student


def _compute_consistency_loss(*, epoch, consistency_weight=None):
    case = _consistency_case()
    clean, corrupt, teacher, labels, perturbation, state0, student = case
    kwargs = {}
    if consistency_weight is not None:
        kwargs["consistency_weight"] = consistency_weight
    loss = compute_losses(
        clean,
        corrupt,
        teacher,
        labels,
        perturbation,
        state0,
        epoch=epoch,
        student=student,
        warmup_epochs=5,
        ramp_epochs=10,
        **kwargs,
    )
    return loss, case


def test_consistency_is_zero_during_warmup():
    loss, _ = _compute_consistency_loss(epoch=5, consistency_weight=0.8)

    assert loss.terms["consistency"].item() == 0.0
    assert torch.allclose(loss.total, loss.terms["task_clean"])


def test_consistency_weight_scales_post_warmup_ramped_term_and_defaults_to_point_two():
    default_loss, _ = _compute_consistency_loss(epoch=10)
    explicit_default, _ = _compute_consistency_loss(epoch=10, consistency_weight=0.2)
    zero_weight, _ = _compute_consistency_loss(epoch=10, consistency_weight=0.0)
    high_weight, _ = _compute_consistency_loss(epoch=10, consistency_weight=0.8)

    assert torch.allclose(default_loss.total, explicit_default.total)
    delta = high_weight.total - zero_weight.total
    expected = 0.5 * 0.8 * high_weight.terms["consistency"]
    assert delta.item() > 0
    assert delta.item() == pytest.approx(expected.item())


def test_consistency_detaches_teacher_and_backpropagates_through_corrupt_logits():
    loss, case = _compute_consistency_loss(epoch=10, consistency_weight=0.8)
    loss.terms["consistency"].backward()
    _, corrupt, teacher, _, _, _, _ = case

    assert teacher.logits.grad is None
    assert teacher.score.grad is None
    assert corrupt.logits.grad is not None
    assert corrupt.score.grad is not None
    assert torch.count_nonzero(corrupt.logits.grad) > 0
    assert torch.count_nonzero(corrupt.score.grad) > 0
