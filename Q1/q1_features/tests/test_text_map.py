from __future__ import annotations

import sys
from pathlib import Path

import pytest


SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SOURCE_ROOT))

from q1_features.text_map import (  # noqa: E402
    build_words,
    map_bert_offsets,
    merge_bert_chunk_vectors,
    normalize_ctc_text,
    plan_bert_chunks,
    pool_bert_vectors_by_word,
    prepare_ctc_rows,
)


def test_build_words_preserves_exact_ranges_and_punctuation_attachment() -> None:
    raw_text = "  Hello,\t  don’t...   WORLD!  "
    words = build_words(raw_text)

    assert [word.raw_word for word in words] == ["Hello,", "don’t...", "WORLD!"]
    assert [raw_text[word.char_start : word.char_end] for word in words] == [
        word.raw_word for word in words
    ]
    assert [(word.char_start, word.char_end) for word in words] == [
        (2, 8),
        (11, 19),
        (22, 28),
    ]
    assert [word.word_id for word in words] == [0, 1, 2]
    assert all(word.attached_to_word_id is None for word in words)

    leading = build_words("... hello !!!")
    assert [word.attached_to_word_id for word in leading] == [1, None, 1]

    only_punctuation = build_words("... !!!")
    assert [word.attached_to_word_id for word in only_punctuation] == [None, None]


def test_bert_offsets_add_parent_word_start_and_drop_special_tokens() -> None:
    words = build_words("Hi  WORLD!")
    encoded = {
        "input_ids": [101, 11, 12, 13, 102],
        "word_ids": [None, 0, 1, 1, None],
        "offset_mapping": [(0, 0), (0, 2), (0, 4), (4, 5), (0, 0)],
        "tokens": ["[CLS]", "hi", "world", "!", "[SEP]"],
    }

    pieces = map_bert_offsets(words, encoded)
    assert [(piece.piece_index, piece.word_id) for piece in pieces] == [
        (0, 0),
        (1, 1),
        (2, 1),
    ]
    assert [piece.global_char_range for piece in pieces] == [(0, 2), (4, 8), (8, 9)]
    assert pieces[0].parent_offset == (0, 2)
    assert pieces[0].char_start == words[0].char_start


def test_bert_chunk_plan_covers_long_tail_without_redundant_final_chunk() -> None:
    chunks = plan_bert_chunks(1000)
    assert chunks == [(0, 510), (382, 892), (764, 1000)]
    assert chunks[-1].content_end == 1000
    assert all(chunk.input_length <= 512 for chunk in chunks)
    assert plan_bert_chunks(0) == []
    assert plan_bert_chunks(892) == [(0, 510), (382, 892)]


def test_overlapping_chunk_vectors_are_averaged_once_per_global_piece() -> None:
    chunks = plan_bert_chunks(4, content_tokens_per_chunk=3, chunk_stride=2)
    assert chunks == [(0, 3), (2, 4)]
    merged = merge_bert_chunk_vectors(
        [[[1.0], [2.0], [3.0]], [[30.0], [40.0]]],
        chunks,
    )
    assert [float(vector[0]) for vector in merged] == [1.0, 2.0, 16.5, 40.0]

    words = build_words("alpha beta")
    # Three pieces belong to alpha, one to beta.  Word pooling must first use
    # the distinct global piece vectors, then give the two words equal weight
    # when a caller later applies equal-duration word intervals.
    class Piece:
        def __init__(self, index: int, word_id: int) -> None:
            self.piece_index = index
            self.word_id = word_id

    pieces = [Piece(0, 0), Piece(1, 0), Piece(2, 0), Piece(3, 1)]
    pooled = pool_bert_vectors_by_word(
        words,
        pieces,  # type: ignore[arg-type]
        [[1.0], [2.0], [3.0], [10.0]],
    )
    assert pooled.shape == (2, 1)
    assert pooled[:, 0].tolist() == [2.0, 10.0]


def test_ctc_normalization_covers_quotes_numbers_separators_and_unsupported() -> None:
    assert normalize_ctc_text("Don’t").ctc_text == "DON'T"
    assert normalize_ctc_text("state-of-the-art/AI").ctc_text == "STATE OF THE ART AI"

    year = normalize_ctc_text("2,020")
    assert year.ctc_text == "TWO THOUSAND AND TWENTY"
    assert year.normalization_assumed is True

    decimal_percent = normalize_ctc_text("3.14%")
    assert decimal_percent.ctc_text == "THREE POINT ONE FOUR PERCENT"
    assert "number_decimal_percent" in decimal_percent.transformations

    ordinal = normalize_ctc_text("21st")
    assert ordinal.ctc_text == "TWENTY FIRST"
    assert "number_ordinal" in ordinal.transformations

    punctuation = normalize_ctc_text("!!!")
    assert punctuation.ctc_text == ""
    assert punctuation.status == "no_spoken_content"

    unsupported = normalize_ctc_text("hello🙂")
    assert unsupported.ctc_text == "HELLO🙂"
    assert unsupported.status == "unsupported_text"
    assert unsupported.contains_unsupported is True
    assert "🙂" in unsupported.unsupported_characters


def test_ctc_rows_keep_explicit_word_mapping_and_do_not_swallow_unknown() -> None:
    class FakeSentencePiece:
        def encode(self, text: str, out_type: type[str] = str) -> list[str]:
            return {
                "HELLO": ["▁HELLO"],
                "WORLD": ["▁MISSING"],
                "!!!": [],
            }[text]

    words = build_words("hello !!! world")
    prepared = prepare_ctc_rows(
        words,
        FakeSentencePiece(),
        ["<blank>", "<unk>", "▁HELLO"],
    )

    assert prepared.ctc_texts == ["HELLO", "WORLD"]
    assert prepared.ctc_row_to_word_id == [0, 2]
    assert prepared.token_arrays == [[2], [1]]
    assert prepared.contains_unk_word_ids == {2}
    assert prepared.skipped_word_ids == {1}
    # This is the invariant consumed by ctc_segmentation callers.
    assert [words[word_id].raw_word for word_id in prepared.ctc_row_to_word_id] == [
        "hello",
        "world",
    ]


@pytest.mark.parametrize("bad_length", [-1, 1.5])
def test_chunk_plan_rejects_invalid_lengths(bad_length: object) -> None:
    with pytest.raises(ValueError):
        plan_bert_chunks(bad_length)  # type: ignore[arg-type]
