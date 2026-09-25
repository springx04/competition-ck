from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import q2.export as export  # noqa: E402
from q2.losses import unimodal_classification_loss  # noqa: E402
from q2.model.network import ModelInput, Student  # noqa: E402


def _model_input(batch_size=2, length=5, U=None):
    if U is None:
        U = torch.ones(batch_size, length, 3, dtype=torch.bool)
    return ModelInput(
        torch.randn(batch_size, length, 256),
        torch.randn(batch_size, length, 74),
        torch.randn(batch_size, length, 35),
        U,
        U.any(dim=-1),
        torch.zeros(batch_size, length, 3, 5),
    )


def _aux_output(U, logits=None, main_logits=None):
    if logits is None:
        logits = torch.randn(U.shape[0], 3, 3, requires_grad=True)
    if main_logits is None:
        main_logits = torch.zeros(U.shape[0], 3, requires_grad=True)
    return SimpleNamespace(logits=main_logits, unimodal_logits=logits, U=U)


def test_missing_modalities_do_not_contribute_auxiliary_loss_or_gradient():
    U = torch.tensor(
        [
            [[True, True, False], [True, False, False]],
            [[False, False, True], [False, False, True]],
        ]
    )
    class_id = torch.tensor([0, 2])
    aux_logits = torch.randn(2, 3, 3, requires_grad=True)
    output = _aux_output(U, aux_logits)

    loss = unimodal_classification_loss(output, class_id)
    changed = aux_logits.detach().clone()
    observed = U.any(dim=1)
    missing = (~observed).unsqueeze(-1).expand_as(changed)
    changed[missing] += 1000
    changed_loss = unimodal_classification_loss(_aux_output(U, changed), class_id)

    assert torch.allclose(loss, changed_loss)
    loss.backward()
    assert torch.count_nonzero(aux_logits.grad[missing]) == 0
    assert torch.count_nonzero(aux_logits.grad[~missing]) > 0


def test_auxiliary_loss_gives_each_available_modality_equal_weight():
    U = torch.tensor(
        [
            [[True, True, True], [True, True, True]],
            [[True, False, True], [True, False, True]],
            [[True, False, False], [True, False, False]],
        ]
    )
    class_id = torch.tensor([0, 1, 2])
    torch.manual_seed(31)
    aux_logits = torch.randn(3, 3, 3)
    output = _aux_output(U, aux_logits)

    actual = unimodal_classification_loss(output, class_id)
    observed = U.any(dim=1)
    per_modality = torch.stack(
        [F.cross_entropy(aux_logits[observed[:, m], m], class_id[observed[:, m]])
         for m in range(3)]
    )
    expected = per_modality.mean()
    pairwise_mean = torch.cat(
        [F.cross_entropy(aux_logits[observed[:, m], m], class_id[observed[:, m]], reduction="none")
         for m in range(3)]
    ).mean()

    assert torch.allclose(actual, expected)
    assert not torch.allclose(actual, pairwise_mean)


def test_auxiliary_loss_uses_weighted_sample_mean_after_reduction_none():
    U = torch.ones(2, 1, 3, dtype=torch.bool)
    class_id = torch.tensor([0, 1])
    logits_for_each_modality = torch.tensor(
        [[torch.log(torch.tensor(2.0)), 0.0, 0.0],
         [0.0, torch.log(torch.tensor(3.0)), 0.0]]
    )
    aux_logits = logits_for_each_modality[:, None, :].expand(2, 3, 3)
    class_weight = torch.tensor([1.0, 2.0, 4.0])

    actual = unimodal_classification_loss(
        _aux_output(U, aux_logits), class_id, class_weight=class_weight
    )
    expected = (torch.log(torch.tensor(2.0))
                + 2 * torch.log(torch.tensor(5.0 / 3.0))) / 2

    assert actual.item() == pytest.approx(expected.item())


def test_auxiliary_loss_backpropagates_into_late_encoders():
    torch.manual_seed(32)
    model = Student("late_aux_tune", [.2, .3, .5], 0.0).eval()
    output = model(_model_input(batch_size=3), return_details=True)
    assert output.unimodal_logits is not None

    unimodal_classification_loss(output, torch.tensor([0, 1, 2])).backward()

    encoder_gradient = sum(
        parameter.grad.abs().sum()
        for parameter in model.encoders.parameters()
        if parameter.grad is not None
    )
    head_gradient = sum(
        parameter.grad.abs().sum()
        for parameter in model.auxiliary_classifiers.parameters()
        if parameter.grad is not None
    )
    assert encoder_gradient > 0
    assert head_gradient > 0


def test_auxiliary_logits_are_materialized_only_when_details_are_requested():
    torch.manual_seed(33)
    model = Student("late_aux_tune", [.2, .3, .5], 0.0).eval()
    inputs = _model_input()

    plain = model(inputs, return_details=False)
    detailed = model(inputs, return_details=True)

    assert plain.unimodal_logits is None
    assert detailed.unimodal_logits.shape == (2, 3, 3)
    assert torch.equal(plain.logits, detailed.logits)
    assert torch.equal(plain.score, detailed.score)


def test_auxiliary_heads_preserve_shared_seeded_model_and_eval_predictions():
    seed = 34
    torch.manual_seed(seed)
    control = Student("late_attn_tune", [.2, .3, .5], 0.0).eval()
    torch.manual_seed(seed)
    with_heads = Student("late_aux_tune", [.2, .3, .5], 0.0).eval()
    torch.manual_seed(seed)
    without_heads = Student("late_aux_tune", [.2, .3, .5], 0.0, include_aux_heads=False).eval()

    assert with_heads.auxiliary_classifiers is not None
    assert without_heads.auxiliary_classifiers is None
    assert set(control.state_dict()) == set(without_heads.state_dict())
    for key, value in control.state_dict().items():
        assert torch.equal(value, with_heads.state_dict()[key])
        assert torch.equal(value, without_heads.state_dict()[key])

    inputs = _model_input()
    control_output = control(inputs, return_details=True)
    with_heads_output = with_heads(inputs, return_details=True)
    without_heads_output = without_heads(inputs, return_details=True)
    for field in ("logits", "score"):
        assert torch.equal(getattr(control_output, field), getattr(with_heads_output, field))
        assert torch.equal(getattr(with_heads_output, field), getattr(without_heads_output, field))
    assert without_heads_output.unimodal_logits is None


def test_auxiliary_head_initialization_preserves_public_rng_state():
    seed = 341
    torch.manual_seed(seed)
    Student("late_aux_tune", [.2, .3, .5], 0.0, include_aux_heads=False)
    after_without_heads = torch.rand(8)

    torch.manual_seed(seed)
    Student("late_aux_tune", [.2, .3, .5], 0.0, include_aux_heads=True)
    after_with_heads = torch.rand(8)

    assert torch.equal(after_without_heads, after_with_heads)


def test_all_empty_auxiliary_loss_uses_main_logits_zero_fallback():
    U = torch.zeros(2, 4, 3, dtype=torch.bool)
    main_logits = torch.randn(2, 3, requires_grad=True)
    aux_logits = torch.randn(2, 3, 3, requires_grad=True)
    output = _aux_output(U, aux_logits, main_logits)

    loss = unimodal_classification_loss(output, torch.tensor([0, 1]))

    assert loss.item() == 0.0
    loss.backward()
    assert main_logits.grad is not None
    assert torch.count_nonzero(main_logits.grad) == 0
    assert aux_logits.grad is None


class _ExportNormalizer:
    class_prior = [0.2, 0.3, 0.5]
    score_prior = 0.0


class _ExportEncoder:
    def half(self):
        return self

    def save_pretrained(self, directory, safe_serialization=True):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "mock-encoder.txt").write_text(str(safe_serialization), encoding="utf-8")


def test_export_bundle_removes_late_auxiliary_heads_from_saved_student(monkeypatch, tmp_path):
    root = Path(tmp_path)
    for directory in (
        root / "data/processed",
        root / "models/text_encoder",
        root / "src/q2",
        root / "licenses",
        root / "outputs",
        root / "runs/late_aux_tune/seed_1111",
    ):
        directory.mkdir(parents=True)
    (root / "data/processed/normalizer.npz").write_bytes(b"normalizer")
    (root / "models/text_encoder/base.txt").write_text("base", encoding="utf-8")
    (root / "src/q2/module.py").write_text("# package", encoding="utf-8")
    (root / "licenses/EBMC-LICENSE").write_text("license", encoding="utf-8")
    (root / "THIRD_PARTY.md").write_text("third party", encoding="utf-8")
    (root / "requirements.txt").write_text("numpy", encoding="utf-8")
    (root / "outputs/q2_predictions_aligned.csv").write_text("sample_id\n", encoding="utf-8")

    torch.manual_seed(35)
    trained = Student("late_aux_tune", _ExportNormalizer.class_prior, 0.0)
    checkpoint = {
        "student": trained.state_dict(),
        "text_encoder": {"mock": torch.tensor(1)},
        "config": {},
    }
    checkpoint_path = root / "runs/late_aux_tune/seed_1111/best.pt"
    torch.save(checkpoint, checkpoint_path)
    assert any(key.startswith("auxiliary_classifiers.") for key in checkpoint["student"])

    monkeypatch.setattr(export.Normalizer, "load", lambda path: _ExportNormalizer())
    monkeypatch.setattr(export, "load_checkpoint_text_encoder", lambda *args, **kwargs: _ExportEncoder())

    result = export.export_bundle(
        root,
        {"variant": "late_aux_tune", "checkpoint": "runs/late_aux_tune/seed_1111/best.pt"},
    )

    saved = export.load_file(str(Path(result["directory"]) / "student.safetensors"), device="cpu")
    assert not any(key.startswith("auxiliary_classifiers.") for key in saved)
    stripped = Student("late_aux_tune", _ExportNormalizer.class_prior, 0.0, include_aux_heads=False)
    stripped.load_state_dict(saved, strict=True)
