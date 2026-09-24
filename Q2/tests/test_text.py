from pathlib import Path
from types import SimpleNamespace
import sys
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from q2.state import infer_state
from q2.text import encode_text


class _Encoder:
    def __call__(self, input_ids, attention_mask, token_type_ids):
        # CLS row is 7; token rows equal their token id.
        out = input_ids.float().unsqueeze(-1).expand(*input_ids.shape, 256).clone()
        out[:, 0] = 7
        return SimpleNamespace(last_hidden_state=out)


def test_cls_context_is_disabled_by_default_for_legacy_semantics():
    ids = torch.tensor([[101, 201, 202, 102, 0, 0]])
    raw = {"input_ids": ids, "stored_attention": (ids != 0).long(),
           "token_type_ids": torch.zeros_like(ids), "audio": torch.zeros(1, 6, 1),
           "vision": torch.zeros(1, 6, 1)}
    output = encode_text(raw, _Encoder(), infer_state(raw))
    assert torch.all(output[0, 0] == 0)
    assert torch.all(output[0, 1] == 201)
    assert torch.all(output[0, 2] == 202)
    assert torch.all(output[0, 3:] == 0)


def test_cls_context_is_added_only_to_observed_text_positions():
    ids = torch.tensor([[101, 201, 202, 102, 0, 0]])
    raw = {"input_ids": ids, "stored_attention": (ids != 0).long(),
           "token_type_ids": torch.zeros_like(ids), "audio": torch.zeros(1, 6, 1),
           "vision": torch.zeros(1, 6, 1)}
    output = encode_text(raw, _Encoder(), infer_state(raw), cls_context=True)
    assert torch.all(output[0, 0] == 0)
    assert torch.all(output[0, 1] == 208)
    assert torch.all(output[0, 2] == 209)
    assert torch.all(output[0, 3:] == 0)
