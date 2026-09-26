"""Frozen, locally stored BERT frontend and tokenizer compatibility checks."""
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import torch
from transformers import BertModel, BertTokenizerFast
from huggingface_hub import snapshot_download

from .state import infer_state


MODEL_ID = "google/bert_uncased_L-4_H-256_A-4"


def prepare_model(download_dir: Path, output_dir: Path) -> dict:
    download_dir, output_dir = Path(download_dir), Path(output_dir)
    download_dir.mkdir(parents=True, exist_ok=True)
    if not all((download_dir / name).exists() for name in ("config.json", "vocab.txt", "pytorch_model.bin")):
        snapshot_download(repo_id=MODEL_ID, local_dir=download_dir,
                          allow_patterns=["config.json", "vocab.txt", "pytorch_model.bin", "README.md"])
    tokenizer = BertTokenizerFast(vocab_file=str(download_dir / "vocab.txt"), do_lower_case=True)
    if (tokenizer.pad_token_id, tokenizer.unk_token_id, tokenizer.cls_token_id,
        tokenizer.sep_token_id, tokenizer.mask_token_id, len(tokenizer)) != (0, 100, 101, 102, 103, 30522):
        raise ValueError("specified tokenizer has unexpected special IDs or vocabulary size")
    model = BertModel.from_pretrained(download_dir, add_pooling_layer=False,
                                     attn_implementation="eager", torch_dtype=torch.float32)
    if model.config.hidden_size != 256 or model.config.num_hidden_layers != 4:
        raise ValueError("unexpected BERT architecture")
    model.requires_grad_(False).eval()
    output_dir.mkdir(parents=True, exist_ok=True)
    model.half().save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    reloaded = load_text_encoder(output_dir, torch.device("cpu"))
    with torch.no_grad():
        result = reloaded(input_ids=torch.tensor([[101, 2023, 102]]), attention_mask=torch.ones(1, 3, dtype=torch.long))
    if result.last_hidden_state.shape != (1, 3, 256):
        raise ValueError("reloaded BERT output shape mismatch")
    return {"model_id": MODEL_ID, "parameters": sum(p.numel() for p in reloaded.parameters()),
            "stored_weight_bytes": (output_dir / "model.safetensors").stat().st_size}


def load_text_encoder(path: Path, device: torch.device) -> BertModel:
    model = BertModel.from_pretrained(Path(path), add_pooling_layer=False, local_files_only=True,
                                     attn_implementation="eager", torch_dtype=torch.float32)
    return model.to(device).requires_grad_(False).eval()


@torch.no_grad()
def encode_text(raw: dict, model: BertModel, state=None, batch_size: int = 64) -> torch.Tensor:
    state = state or infer_state(raw)
    ids = raw["input_ids"].clone()
    ids[raw["stored_attention"] == 0] = 0
    output = torch.zeros(*ids.shape, 256, dtype=torch.float32, device=ids.device)
    valid = torch.where(state.bert_attention.any(dim=1))[0]
    for indices in valid.split(batch_size):
        representation = model(input_ids=ids[indices],
                               attention_mask=state.bert_attention[indices].long(),
                               token_type_ids=raw["token_type_ids"][indices]).last_hidden_state
        output[indices] = representation * state.U[indices, :, 0, None]
    return output


def tokenizer_check(record: dict, model_dir: Path, output_csv: Path) -> dict:
    tokenizer = BertTokenizerFast.from_pretrained(model_dir, local_files_only=True)
    rows = []
    explanations = {"standard50": 0, "prefix50": 0, "unexplained": 0}
    for i, (sample_id, raw_text) in enumerate(zip(record["id"], record["raw_text"])):
        stored = record["input_ids"][i]
        attention = record["stored_attention"][i]
        token_types = record["token_type_ids"][i]
        standard = tokenizer(raw_text, add_special_tokens=True, truncation=True, max_length=50,
                             padding="max_length")
        full = tokenizer(raw_text, add_special_tokens=True, truncation=False)
        prefix_ids = full["input_ids"][:50]
        prefix_ids += [0] * (50 - len(prefix_ids))
        prefix_attention = [int(token != 0) for token in prefix_ids]
        standard_ok = (np.array_equal(stored, standard["input_ids"])
                       and np.array_equal(attention, standard["attention_mask"])
                       and np.array_equal(token_types, standard["token_type_ids"]))
        prefix_ok = (np.array_equal(stored, prefix_ids) and np.array_equal(attention, prefix_attention)
                     and np.array_equal(token_types, np.zeros(50, dtype=np.int64)))
        mismatch = np.where(stored != np.asarray(standard["input_ids"]))[0]
        first = int(mismatch[0]) if len(mismatch) else None
        rows.append({"sample_id": sample_id, "effective_length": int(attention.sum()),
                     "standard50": standard_ok, "prefix50": prefix_ok,
                     "first_mismatch": first,
                     "stored_token": tokenizer.convert_ids_to_tokens(int(stored[first])) if first is not None else "",
                     "standard_token": tokenizer.convert_ids_to_tokens(int(standard["input_ids"][first])) if first is not None else ""})
        explanations["standard50" if standard_ok else "prefix50" if prefix_ok else "unexplained"] += 1
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return explanations


def cache_clean_text(dataset, encoder, output_dir: Path, device: torch.device,
                     model_dir: Path, batch_size: int = 64) -> dict:
    from torch.utils.data import DataLoader
    from .data import collate_raw
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = np.lib.format.open_memmap(output_dir / "features.npy", mode="w+", dtype="float32",
                                       shape=(len(dataset), 50, 256))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_raw)
    offset = 0
    for batch in loader:
        raw = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        features = encode_text(raw, encoder).cpu().numpy()
        result[offset:offset+len(features)] = features
        offset += len(features)
    result.flush()
    metadata = {"model_dir": str(model_dir), "model_id": MODEL_ID,
                "tokenizer": "BertTokenizerFast", "samples": len(dataset),
                "sample_ids": dataset.metadata["id"], "created_at": datetime.now(timezone.utc).isoformat()}
    (output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")
    return {"samples": len(dataset), "shape": list(result.shape)}
