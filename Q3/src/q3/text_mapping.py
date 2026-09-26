"""Small, dependency-free helpers for connecting BERT tokens to raw text.

The stored Q2 text input is the source of truth for token positions.  These
helpers deliberately keep special tokens (whose offset is ``(0, 0)``)
unexplained instead of assigning them to the first word.
"""

import html
import re


_WORD_RE = re.compile(r"\S+")


def _as_list(value):
    """Convert common tensor/array/list values to a plain list."""

    if value is None:
        return []
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    while isinstance(value, (list, tuple)) and len(value) == 1 and isinstance(value[0], (list, tuple)):
        value = value[0]
    return list(value) if isinstance(value, (list, tuple)) else [value]


def _offset_pair(value):
    try:
        values = _as_list(value)
        if len(values) < 2:
            return 0, 0
        return int(values[0]), int(values[1])
    except (TypeError, ValueError, OverflowError):
        return 0, 0


def map_tokens(raw_text, offsets, ids, attention, token_type, observed, token_strings=None):
    """Return JSON-friendly token, character and whitespace-word rows.

    ``offsets`` follows the tokenizer's half-open character convention.  A
    token is mapped to a word only when its non-empty span intersects exactly
    one whitespace-delimited raw word.  This permits BERT subwords to share a
    word while leaving special/padding tokens explicitly unexplained.
    """

    raw_text = "" if raw_text is None else str(raw_text)
    offsets = _as_list(offsets)
    ids = _as_list(ids)
    attention = _as_list(attention)
    token_type = _as_list(token_type)
    observed = _as_list(observed)
    token_strings = None if token_strings is None else _as_list(token_strings)
    count = max(len(offsets), len(ids), len(attention), len(token_type), len(observed))
    words = [(match.start(), match.end(), match.group(0)) for match in _WORD_RE.finditer(raw_text)]
    rows = []
    for token_index in range(count):
        start, end = _offset_pair(offsets[token_index] if token_index < len(offsets) else (0, 0))
        valid_span = 0 <= start <= end <= len(raw_text)
        if not valid_span:
            start, end = 0, 0
        hits = [
            word_id
            for word_id, (word_start, word_end, _word) in enumerate(words)
            if end > start and end > word_start and start < word_end
        ]
        token_id = ids[token_index] if token_index < len(ids) else None
        try:
            token_id = int(token_id)
        except (TypeError, ValueError, OverflowError):
            token_id = None
        attention_value = attention[token_index] if token_index < len(attention) else 0
        type_value = token_type[token_index] if token_index < len(token_type) else 0
        observed_value = observed[token_index] if token_index < len(observed) else bool(attention_value)
        try:
            attention_value = int(attention_value)
        except (TypeError, ValueError, OverflowError):
            attention_value = 0
        try:
            type_value = int(type_value)
        except (TypeError, ValueError, OverflowError):
            type_value = 0
        row = {
            "official_t": int(token_index),
            "token_id": token_id,
            "stored_attention": attention_value,
            "token_type": type_value,
            "is_observed": bool(observed_value),
            "is_unknown": token_id == 100,
            "char_start": int(start),
            "char_end": int(end),
            "char_span": [int(start), int(end)] if end > start else None,
            "word_id": hits[0] if len(hits) == 1 else None,
            "word_ids": [int(item) for item in hits],
            "token_mapping_status": "matched" if len(hits) == 1 else "unexplained",
        }
        if token_strings is not None and token_index < len(token_strings):
            value = token_strings[token_index]
            row["token"] = None if value is None else str(value)
        rows.append(row)
    return rows


def highlight(text, start, end):
    """HTML-highlight one raw half-open character span."""

    text = "" if text is None else str(text)
    start = max(0, min(len(text), int(start)))
    end = max(start, min(len(text), int(end)))
    return html.escape(text[:start]) + "<mark>" + html.escape(text[start:end]) + "</mark>" + html.escape(text[end:])
