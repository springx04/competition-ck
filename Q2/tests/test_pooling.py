from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from q2.model.pooling import consecutive_run_weights
from q2.model.network import ModelInput, Student


def test_identical_run_weights_use_only_observed_rows_and_break_at_gaps():
    x = torch.tensor([[[1.], [1.], [8.], [1.], [2.], [2.]]])
    mask = torch.tensor([[True, True, False, True, True, True]])
    weights = consecutive_run_weights(x, mask)
    assert torch.equal(weights, torch.tensor([[.5, .5, 0, 1, .5, .5]]))
    x[:, 2] = 1  # A hidden value must not bridge the two runs.
    assert torch.equal(consecutive_run_weights(x, mask), weights)
    assert torch.equal(consecutive_run_weights(x, torch.zeros_like(mask)), torch.zeros_like(weights))


def _av_case(rows):
    length = len(rows)
    x = torch.tensor(rows, dtype=torch.float32)[None, :, None]
    u = torch.zeros(1, length, 3, dtype=torch.bool)
    u[:, :, 1:] = True
    return ModelInput(torch.zeros(1, length, 256), x.expand(1, length, 74),
                      x.expand(1, length, 35), u, u.any(-1), torch.zeros(1, length, 3, 5))


def test_runweighted_pooling_preserves_prediction_when_av_row_is_repeated():
    model = Student("late_attn_runweight_tune", [.2, .3, .5], 0).eval()
    with torch.no_grad():
        short = model(_av_case([1, 2]))
        repeated = model(_av_case([1, 1, 1, 2]))
    assert torch.allclose(short.logits, repeated.logits, atol=1e-6)
    assert torch.allclose(short.score, repeated.score, atol=1e-6)
    assert repeated.U.sum() == 8  # Actual observations and official indices survive.


def test_runweight_keeps_text_weighting_and_observation_metadata():
    torch.manual_seed(11)
    control = Student("late_attn_tune", [.2, .3, .5], 0).eval()
    weighted = Student("late_attn_runweight_tune", [.2, .3, .5], 0).eval()
    weighted.load_state_dict(control.state_dict())
    case = _av_case([1, 2, 3, 4])  # No duplicate AV rows, so their weights stay 1.
    case.U[:, :, 0] = True
    case.U[:, 2, 1:] = False
    case.text_features[:] = torch.tensor([1, 1, 2, 3])[None, :, None]
    with torch.no_grad():
        plain = control(case, return_details=True)
        output = weighted(case, return_details=True)
    # Repeated text rows must still have ordinary text attention weights.
    assert torch.allclose(plain.logits, output.logits, atol=1e-6)
    for field in ("U", "J", "source", "source_positions"):
        assert torch.equal(getattr(plain, field), getattr(output, field))
    assert output.time_axis == "official_aligned_positions"
