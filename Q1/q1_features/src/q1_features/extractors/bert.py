from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np


class BertExtractionError(RuntimeError):
    pass


def _word_value(word: Any, name: str) -> Any:
    return getattr(word, name) if hasattr(word, name) else word[name]


class BertWordEncoder:
    """One frozen BERT instance reused for an entire extraction stage."""

    def __init__(self, bert_dir: str | Path, device: str) -> None:
        import torch
        from transformers import BertModel, BertTokenizerFast

        self.torch = torch
        self.device = str(device)
        self.tokenizer = BertTokenizerFast.from_pretrained(str(bert_dir), local_files_only=True)
        self.model = BertModel.from_pretrained(str(bert_dir), local_files_only=True)
        self.model.eval().requires_grad_(False)
        self.model.to(device=self.device, dtype=torch.float32)

    def close(self) -> None:
        model = getattr(self, "model", None)
        if model is not None:
            model.to("cpu")
            del self.model
        if self.device.startswith("cuda") and self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()

    def encode(
        self,
        words: Sequence[Any],
        *,
        content_tokens_per_chunk: int = 510,
        chunk_stride: int = 382,
    ) -> tuple[np.ndarray, list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        """Encode parent words from one tokenization with overlap averaging."""
        torch, tokenizer, model, device = self.torch, self.tokenizer, self.model, self.device
        if not words:
            return np.empty((0, model.config.hidden_size), np.float32), [], [], []
        raw_words = [_word_value(w, "raw_word") for w in words]
        encoded = tokenizer(
        raw_words, is_split_into_words=True, add_special_tokens=False,
        truncation=False, return_offsets_mapping=True,
        return_attention_mask=False, return_token_type_ids=False,
        )
        token_ids = list(encoded["input_ids"])
        offsets = list(encoded["offset_mapping"])
        word_ids = encoded.word_ids()
        token_rows: list[dict[str, Any]] = []
        per_word: list[list[int]] = [[] for _ in words]
        for token_index, (token_id, word_id, offset) in enumerate(zip(token_ids, word_ids, offsets)):
            if word_id is None:
                continue
            char_start = int(_word_value(words[word_id], "char_start")) + int(offset[0])
            char_end = int(_word_value(words[word_id], "char_start")) + int(offset[1])
            per_word[word_id].append(token_index)
            token_rows.append({
                "token_index": token_index, "token_id": int(token_id),
                "token": tokenizer.convert_ids_to_tokens(int(token_id)),
                "word_id": int(word_id), "parent_offset_start": int(offset[0]),
                "parent_offset_end": int(offset[1]), "char_start": char_start,
                "char_end": char_end, "is_unk": int(token_id) == tokenizer.unk_token_id,
            })

        length = len(token_ids)
        chunks: list[tuple[int, int]] = []
        start = 0
        while start < length:
            end = min(length, start + content_tokens_per_chunk)
            chunks.append((start, end))
            if end == length:
                break
            start += chunk_stride
        sums = np.zeros((length, model.config.hidden_size), dtype=np.float64)
        counts = np.zeros(length, dtype=np.int64)
        chunk_rows: list[dict[str, Any]] = []
        with torch.inference_mode():
            for chunk_id, (begin, end) in enumerate(chunks):
                ids = [tokenizer.cls_token_id] + token_ids[begin:end] + [tokenizer.sep_token_id]
                tensor = torch.tensor([ids], dtype=torch.long, device=device)
                outputs = model(
                    input_ids=tensor,
                    attention_mask=torch.ones_like(tensor),
                    token_type_ids=torch.zeros_like(tensor),
                ).last_hidden_state[0, 1:-1].detach().cpu().numpy()
                sums[begin:end] += outputs.astype(np.float64)
                counts[begin:end] += 1
                chunk_rows.append({"chunk_id": chunk_id, "token_start": begin, "token_end": end, "input_length": len(ids)})
        if length and np.any(counts == 0):
            raise BertExtractionError("chunk plan left content tokens unencoded")
        token_vectors = (sums / counts[:, None]).astype(np.float32) if length else np.empty((0, model.config.hidden_size), np.float32)
        word_vectors = np.zeros((len(words), model.config.hidden_size), dtype=np.float32)
        word_status: list[dict[str, Any]] = []
        for word_id, indices in enumerate(per_word):
            status = "ok" if indices else "empty"
            if indices:
                word_vectors[word_id] = token_vectors[indices].astype(np.float64).mean(axis=0).astype(np.float32)
            word_status.append({"word_id": word_id, "bert_piece_indices": indices, "semantic_status": status})
        return word_vectors, token_rows, chunk_rows, word_status


def encode_words(
    words: Sequence[Any], bert_dir: str | Path, device: str, *,
    content_tokens_per_chunk: int = 510, chunk_stride: int = 382,
) -> tuple[np.ndarray, list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Compatibility wrapper for one-off callers; stages should reuse the class."""
    encoder = BertWordEncoder(bert_dir, device)
    try:
        return encoder.encode(words, content_tokens_per_chunk=content_tokens_per_chunk, chunk_stride=chunk_stride)
    finally:
        encoder.close()
