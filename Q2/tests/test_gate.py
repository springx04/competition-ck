from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from q2.model.network import ModelInput, Student


def _input(flags):
    u = torch.zeros(2, 4, 3, dtype=torch.bool)
    for m, value in enumerate(flags):
        u[:, :, m] = value
    return ModelInput(torch.randn(2, 4, 256), torch.randn(2, 4, 74), torch.randn(2, 4, 35),
                      u, u.any(-1), torch.zeros(2, 4, 3, 5))


def test_reliability_gate_handles_each_availability_pattern_and_empty_prior():
    torch.manual_seed(12)
    model = Student("late_gate_tune", [.2, .3, .5], .25).eval()
    with torch.no_grad():
        for flags in ((True, True, True), (False, True, True),
                      (True, False, True), (True, True, False), (False, False, True)):
            output = model(_input(flags))
            assert torch.isfinite(output.logits).all()
            assert torch.isfinite(output.score).all()
        empty = _input((False, False, False))
        output = model(empty)
    assert torch.allclose(output.logits.softmax(-1), torch.tensor([[.2, .3, .5]]).expand(2, -1))
    assert torch.all(output.score == .25)
