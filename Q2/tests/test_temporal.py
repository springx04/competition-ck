from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from q2.model.temporal import ObservedGRU
from q2.model.network import ModelInput, Student


def test_gru_ignores_holes_preserves_order_and_scatter_positions():
    torch.manual_seed(8)
    model = ObservedGRU(8).eval()
    features = torch.randn(2, 5, 8, requires_grad=True)
    observed = torch.tensor([[False, True, False, True, True], [False] * 5])
    output = model(features, observed)
    compact = model(features[:1, [1, 3, 4]], torch.ones(1, 3, dtype=torch.bool))
    assert torch.allclose(output[0, [1, 3, 4]], compact[0], atol=1e-6)
    assert torch.count_nonzero(output[~observed]) == 0
    dirty = features.detach().clone()
    dirty[~observed] = 9999
    assert torch.allclose(model(dirty, observed), output, atol=1e-6)
    output[..., 0].sum().backward()
    assert torch.count_nonzero(features.grad[~observed]) == 0
    assert model.gru.weight_ih_l0.grad.abs().sum() > 0


def test_gru_student_keeps_empty_prior_and_shared_initialization():
    torch.manual_seed(9)
    control = Student("late_attn_tune", [.2, .3, .5], .25)
    torch.manual_seed(9)
    model = Student("late_gru_tune", [.2, .3, .5], .25).eval()
    for key, tensor in control.state_dict().items():
        assert torch.equal(tensor, model.state_dict()[key])
    u = torch.zeros(2, 50, 3, dtype=torch.bool)
    inputs = ModelInput(torch.zeros(2, 50, 256), torch.zeros(2, 50, 74),
                        torch.zeros(2, 50, 35), u, torch.ones(2, 50, dtype=torch.bool),
                        torch.zeros(2, 50, 3, 5))
    output = model(inputs)
    assert torch.allclose(output.logits.softmax(-1), torch.tensor([[.2, .3, .5]]).expand(2, -1))
    assert torch.all(output.score == .25)
