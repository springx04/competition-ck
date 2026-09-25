from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from q2.model.network import ModelInput, Student


def _case(text=True):
    u = torch.zeros(2, 4, 3, dtype=torch.bool)
    u[:, :, 1:] = True
    if text:
        u[:, :, 0] = True
    x = torch.randn(2, 4, 256)
    return ModelInput(x, torch.randn(2, 4, 74), torch.randn(2, 4, 35), u,
                      u.any(-1), torch.zeros(2, 4, 3, 5))


def test_av_branch_only_changes_text_missing_rows_and_keeps_empty_fallback():
    torch.manual_seed(4)
    ordinary = Student("late_attn_tune", [.2, .3, .5], .25).eval()
    model = Student("late_av_tune", [.2, .3, .5], .25).eval()
    model.load_state_dict(ordinary.state_dict(), strict=False)
    with torch.no_grad():
        ordinary_out = ordinary(_case(True))
        mixed = _case(True)
        mixed.U[1, :, 0] = False
        mixed_out = model(mixed)
        ordinary_mixed = ordinary(mixed)
        empty = ModelInput(torch.zeros(1, 4, 256), torch.zeros(1, 4, 74),
                           torch.zeros(1, 4, 35), torch.zeros(1, 4, 3, dtype=torch.bool),
                           torch.ones(1, 4, dtype=torch.bool), torch.zeros(1, 4, 3, 5))
        empty_out = model(empty)
    assert torch.allclose(mixed_out.logits[0], ordinary_mixed.logits[0])
    assert not torch.allclose(mixed_out.logits[1], ordinary_mixed.logits[1])
    assert torch.allclose(empty_out.logits.softmax(-1), torch.tensor([[.2, .3, .5]]))
    assert empty_out.score.item() == .25
