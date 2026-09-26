from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from q2.masking import apply_span, make_span_mask  # noqa: E402
from q2.state import CLS, MASK, PAD, SEP, UNK, infer_state  # noqa: E402


def _raw_batch(
    input_ids: torch.Tensor,
    stored_attention: torch.Tensor,
    audio: torch.Tensor | None = None,
    vision: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    batch, length = input_ids.shape
    return {
        "input_ids": input_ids.clone(),
        "stored_attention": stored_attention.clone(),
        "token_type_ids": torch.zeros((batch, length), dtype=torch.long),
        "audio": audio.clone() if audio is not None else torch.zeros((batch, length, 74)),
        "vision": vision.clone() if vision is not None else torch.zeros((batch, length, 35)),
    }


def test_infer_state_separates_special_tokens_unknown_and_masked_slots() -> None:
    ids = torch.tensor(
        [[PAD, CLS, 17, UNK, MASK, SEP, PAD, PAD, PAD, PAD, PAD, PAD]],
        dtype=torch.long,
    )
    stored = torch.tensor([[0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]], dtype=torch.long)

    state = infer_state(_raw_batch(ids, stored))

    expected_text_slot = torch.tensor(
        [[False, False, True, True, True, False, False, False, False, False, False, False]]
    )
    expected_text_observed = torch.tensor(
        [[False, False, True, True, False, False, False, False, False, False, False, False]]
    )
    assert torch.equal(state.text_slot, expected_text_slot)
    assert torch.equal(state.U[0, :, 0], expected_text_observed[0])
    assert torch.equal(state.J[0], expected_text_slot[0])

    # UNK is content; CLS and SEP are structural BERT tokens only.
    assert bool(state.U[0, 3, 0])
    assert not bool(state.J[0, 1])
    assert not bool(state.J[0, 5])
    # MASK remains a structural slot, but is not an observed text value.
    assert bool(state.J[0, 4])
    assert not bool(state.U[0, 4, 0])

    expected_bert_attention = torch.tensor(
        [[False, True, True, True, False, True, False, False, False, False, False, False]]
    )
    assert torch.equal(state.bert_attention, expected_bert_attention)
    assert torch.isfinite(state.q).all()


def test_infer_state_empty_j_returns_zero_q_without_nan() -> None:
    ids = torch.zeros((1, 12), dtype=torch.long)
    stored = torch.zeros((1, 12), dtype=torch.long)

    state = infer_state(_raw_batch(ids, stored))

    assert not state.J.any()
    assert not state.U.any()
    assert torch.equal(state.q, torch.zeros_like(state.q))
    assert torch.isfinite(state.q).all()


def test_make_span_mask_preserves_original_indices_and_skips_natural_holes() -> None:
    ids = torch.full((1, 12), 17, dtype=torch.long)
    stored = torch.ones((1, 12), dtype=torch.long)
    audio = torch.zeros((1, 12, 74), dtype=torch.float32)
    for position, value in ((3, 1.0), (5, 2.0), (6, 3.0)):
        audio[0, position, 0] = value
    raw = _raw_batch(ids, stored, audio=audio)
    state0 = infer_state(raw)

    perturbation = make_span_mask(
        raw,
        state0,
        [{"pattern": "A", "intervals": {"A": [3, 7]}, "rho": 0.4}],
    )

    expected_audio_p = torch.zeros(12, dtype=torch.bool)
    expected_audio_p[[3, 5, 6]] = True
    assert torch.equal(perturbation.P[0, :, 1], expected_audio_p)
    assert not bool(perturbation.P[0, 4, 1])
    descriptor = perturbation.descriptors[0]
    assert descriptor["intervals"] == {"A": [3, 7]}
    assert descriptor["new_count"] == {"A": 3}
    # Actual deletion rate is relative to the original availability of A.
    assert descriptor["actual_rate"] == pytest.approx(1.0)


def test_apply_span_does_not_mutate_raw_input() -> None:
    ids = torch.full((1, 12), 17, dtype=torch.long)
    stored = torch.ones((1, 12), dtype=torch.long)
    audio = torch.zeros((1, 12, 74), dtype=torch.float32)
    vision = torch.zeros((1, 12, 35), dtype=torch.float32)
    audio[0, 2:4, 0] = torch.tensor([1.0, 2.0])
    vision[0, 2:4, 0] = torch.tensor([3.0, 4.0])
    raw = _raw_batch(ids, stored, audio=audio, vision=vision)
    state0 = infer_state(raw)
    perturbation = make_span_mask(
        raw,
        state0,
        [{"pattern": "TAV", "intervals": {"T": [2, 4], "A": [2, 4], "V": [2, 4]}}],
    )
    before = {key: value.clone() for key, value in raw.items()}

    corrupted = apply_span(raw, perturbation)

    for key, value in before.items():
        assert torch.equal(raw[key], value), key
    assert torch.equal(corrupted["input_ids"][0, 2:4], torch.full((2,), MASK, dtype=torch.long))
    assert torch.equal(corrupted["stored_attention"], raw["stored_attention"])
    assert torch.equal(corrupted["token_type_ids"], raw["token_type_ids"])
    assert torch.count_nonzero(corrupted["audio"][0, 2:4]) == 0
    assert torch.count_nonzero(corrupted["vision"][0, 2:4]) == 0
