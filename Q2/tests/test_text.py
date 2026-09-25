from pathlib import Path
from types import SimpleNamespace
import sys
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from q2.state import infer_state
from q2.text import encode_text
from q2.text import configure_text_training
from transformers import BertConfig, BertModel
import pytest


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


@pytest.mark.parametrize("train_embeddings", [False, True])
def test_selected_bert_components_receive_gradients_and_updates(train_embeddings):
    model = BertModel(BertConfig(vocab_size=32, hidden_size=16, num_hidden_layers=2,
                                num_attention_heads=2, intermediate_size=32),
                      add_pooling_layer=False)
    configure_text_training(model, unfrozen_layers=1, train_embeddings=train_embeddings)
    assert not any(p.requires_grad for p in model.encoder.layer[0].parameters())
    assert all(p.requires_grad for p in model.encoder.layer[1].parameters())
    before = model.embeddings.word_embeddings.weight.detach().clone()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.01)
    model(input_ids=torch.tensor([[1, 2, 3]])).last_hidden_state[..., 0].sum().backward()
    assert (model.embeddings.word_embeddings.weight.grad is not None) == train_embeddings
    assert model.encoder.layer[1].attention.self.query.weight.grad is not None
    optimizer.step()
    assert torch.equal(before, model.embeddings.word_embeddings.weight) != train_embeddings


def test_configure_text_training_rejects_more_layers_than_model_has():
    model = BertModel(BertConfig(vocab_size=32, hidden_size=16, num_hidden_layers=2,
                                num_attention_heads=2, intermediate_size=32),
                      add_pooling_layer=False)
    with pytest.raises(ValueError, match=r"unfrozen_layers=3 exceeds .* 2 encoder layers"):
        configure_text_training(model, unfrozen_layers=3)


def test_configure_text_training_can_unfreeze_all_encoder_layers():
    model = BertModel(BertConfig(vocab_size=32, hidden_size=16, num_hidden_layers=8,
                                num_attention_heads=2, intermediate_size=32),
                      add_pooling_layer=False)
    configure_text_training(model, unfrozen_layers=8)
    assert all(p.requires_grad for p in model.encoder.parameters())
