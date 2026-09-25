"""Pure text mapping helpers for the Q1 feature pipeline.

The functions in this module deliberately stop at the boundary between text
bookkeeping and model execution.  They do not load a tokenizer, a BERT model,
or a SentencePiece model.  A caller supplies the output of those components
and this module keeps the original word IDs and character coordinates
traceable.

The implementation follows sections 9, 10.1, and 18 of
``E题_Q1_服务器具体实现说明_v1_Agent执行版.md``:

* words are made from non-whitespace ``re.finditer`` matches;
* tokenizer offsets are parent-word offsets and are translated to source
  string offsets exactly once;
* BERT content chunks contain at most 510 tokens and advance by 382;
* CTC text normalization is conservative and records unsupported content;
* CTC rows are kept separate from the complete word list through an explicit
  ``ctc_row_to_word_id`` mapping.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field, replace
from numbers import Integral
from typing import Any, Iterable, Iterator, Mapping, NamedTuple, Sequence


DEFAULT_BERT_CONTENT_TOKENS = 510
DEFAULT_BERT_CHUNK_STRIDE = 382


_CURLY_APOSTROPHES = {
    "\u2018": "'",  # left single quotation mark
    "\u2019": "'",  # right single quotation mark
    "\u201a": "'",  # single low-9 quotation mark
    "\u201b": "'",  # single high-reversed-9 quotation mark
    "\u02bb": "'",  # modifier letter turned comma
    "\u02bc": "'",  # modifier letter apostrophe
    "\u2032": "'",  # prime, frequently used as a curly apostrophe
    "\u2035": "'",  # reversed prime
    "\uff07": "'",  # fullwidth apostrophe
}

_SEPARATOR_CHARS = {
    "-",
    "\u2010",  # hyphen
    "\u2011",  # non-breaking hyphen
    "\u2012",  # figure dash
    "\u2013",  # en dash
    "\u2014",  # em dash
    "\u2015",  # horizontal bar
    "\u2212",  # minus sign
    "/",
    "\\",
    "\u2044",  # fraction slash
    "\uff0f",  # fullwidth solidus
}

# The expression is intentionally limited to expressions that can be read
# without guessing a date, currency, or an alphanumeric identifier.  Leading
# and trailing punctuation remains in the string and is handled by the later
# conservative punctuation pass.
_NUMBER_EXPRESSION = re.compile(
    r"(?<![A-Za-z0-9])"
    r"(?P<number>\d[\d,]*(?:\.\d+)?(?:st|nd|rd|th)?%?)"
    r"(?![A-Za-z0-9])",
    re.IGNORECASE,
)


@dataclass
class Word:
    """One original non-whitespace word segment.

    ``char_start`` and ``char_end`` are Python Unicode code-point offsets into
    the original string.  The first four fields are the stable identity of a
    word; all later fields are derived bookkeeping and can be filled by later
    pipeline stages.
    """

    word_id: int
    raw_word: str
    char_start: int
    char_end: int
    bert_piece_indices: list[int] = field(default_factory=list)
    ctc_text: str = ""
    semantic_status: str = "empty"
    alignment_status: str = "pending"
    attached_to_word_id: int | None = None
    normalization_assumed: bool = False
    normalization_types: tuple[str, ...] = ()
    normalization_status: str = "unprocessed"
    contains_unsupported: bool = False
    contains_unk: bool = False

    @property
    def is_pure_punctuation(self) -> bool:
        """Whether this word consists only of Unicode punctuation characters."""

        return is_pure_punctuation(self.raw_word)

    @property
    def unsupported_text(self) -> bool:
        """Compatibility/readability alias for ``contains_unsupported``."""

        return self.contains_unsupported


def is_pure_punctuation(text: str) -> bool:
    """Return true only for a non-empty Unicode-punctuation-only string.

    Symbols such as ``$`` and ``+`` are not silently classified as harmless
    punctuation.  If they occur inside a word they remain visible to the CTC
    normalizer and are marked unsupported.
    """

    return bool(text) and all(unicodedata.category(ch).startswith("P") for ch in text)


def build_words(raw_text: str) -> list[Word]:
    """Build the complete, source-ordered word list from ``raw_text``.

    The regular expression and the range assignment are deliberately kept
    together: no case conversion, punctuation stripping, or whitespace
    normalization is allowed to alter a source range.
    """

    if not isinstance(raw_text, str):
        raise TypeError("raw_text must be a str")

    words = [
        Word(
            word_id=word_id,
            raw_word=match.group(0),
            char_start=match.start(),
            char_end=match.end(),
        )
        for word_id, match in enumerate(re.finditer(r"\S+", raw_text))
    ]

    # A punctuation-only segment is retained as a word ID, but its source
    # reading anchor is a nearby spoken word.  The search is over the full
    # original word list so consecutive punctuation segments follow the same
    # left-first rule and never get attached to another punctuation segment.
    non_punctuation_ids = [
        word.word_id for word in words if not word.is_pure_punctuation
    ]
    for word in words:
        if not word.is_pure_punctuation:
            continue

        left = next(
            (candidate for candidate in reversed(non_punctuation_ids) if candidate < word.word_id),
            None,
        )
        if left is not None:
            word.attached_to_word_id = left
            continue

        word.attached_to_word_id = next(
            (
                candidate
                for candidate in non_punctuation_ids
                if candidate > word.word_id
            ),
            None,
        )

    return words


@dataclass(frozen=True)
class BertPiece:
    """A content tokenizer piece with both parent and source offsets."""

    piece_index: int
    token_index: int
    token_id: int
    token: str
    word_id: int
    offset_start: int
    offset_end: int
    char_start: int
    char_end: int
    is_unknown: bool = False

    @property
    def global_index(self) -> int:
        return self.piece_index

    @property
    def parent_offset(self) -> tuple[int, int]:
        return self.offset_start, self.offset_end

    @property
    def word_offset(self) -> tuple[int, int]:
        return self.offset_start, self.offset_end

    @property
    def global_char_range(self) -> tuple[int, int]:
        return self.char_start, self.char_end

    @property
    def piece_text(self) -> str:
        return self.token


def _unwrap_singleton(value: Any) -> Any:
    """Unwrap a batch-of-one value while leaving a normal list unchanged."""

    if isinstance(value, (list, tuple)) and len(value) == 1:
        first = value[0]
        if isinstance(first, (list, tuple)):
            return first
    return value


def _encoded_value(encoded: Any, key: str, default: Any = None) -> Any:
    if isinstance(encoded, Mapping):
        return encoded.get(key, default)
    try:
        return encoded[key]
    except (KeyError, IndexError, TypeError, AttributeError):
        return getattr(encoded, key, default)


def _encoded_word_ids(encoded: Any) -> Any:
    method = getattr(encoded, "word_ids", None)
    if callable(method):
        try:
            return method()
        except TypeError:
            return method(batch_index=0)
    return _encoded_value(encoded, "word_ids")


def _encoded_tokens(encoded: Any) -> Any:
    method = getattr(encoded, "tokens", None)
    if callable(method):
        try:
            return method()
        except TypeError:
            return method(batch_index=0)
    return _encoded_value(encoded, "tokens")


def map_bert_offsets(
    words: Sequence[Word],
    input_ids_or_encoded: Any,
    word_ids: Sequence[int | None] | None = None,
    offset_mapping: Sequence[Sequence[int]] | None = None,
    tokens: Sequence[str] | None = None,
) -> list[BertPiece]:
    """Map pre-tokenized BERT offsets to global source character offsets.

    The function accepts either the four tokenizer outputs
    ``(input_ids, word_ids, offset_mapping, tokens)`` or a Hugging Face-like
    encoded object as the second argument.  In pre-tokenized mode offsets are
    local to ``words[word_id].raw_word``; only ``char_start`` is added.  Items
    with ``word_id is None`` are special tokens and are intentionally omitted
    from the returned content-piece list.
    """

    if word_ids is None and offset_mapping is None:
        encoded = input_ids_or_encoded
        input_ids = _unwrap_singleton(_encoded_value(encoded, "input_ids"))
        word_ids = _unwrap_singleton(_encoded_word_ids(encoded))
        offset_mapping = _unwrap_singleton(_encoded_value(encoded, "offset_mapping"))
        if tokens is None:
            tokens = _unwrap_singleton(_encoded_tokens(encoded))
    else:
        input_ids = _unwrap_singleton(input_ids_or_encoded)

    if input_ids is None or word_ids is None or offset_mapping is None:
        raise ValueError("input_ids, word_ids, and offset_mapping are required")

    input_ids = list(input_ids)
    word_ids = list(word_ids)
    offset_mapping = list(offset_mapping)
    if not (len(input_ids) == len(word_ids) == len(offset_mapping)):
        raise ValueError("BERT tokenizer outputs have inconsistent lengths")

    if tokens is None:
        tokens = [str(token_id) for token_id in input_ids]
    else:
        tokens = list(tokens)
        if len(tokens) != len(input_ids):
            raise ValueError("tokens and input_ids have inconsistent lengths")

    pieces: list[BertPiece] = []
    for token_index, (token_id, parent_word_id, raw_offset, token) in enumerate(
        zip(input_ids, word_ids, offset_mapping, tokens)
    ):
        if parent_word_id is None:
            # CLS/SEP/PAD (or any tokenizer special) has no source word and no
            # time anchor.  It must never enter a word mean.
            continue
        if not isinstance(parent_word_id, Integral) or not 0 <= parent_word_id < len(words):
            raise ValueError(f"invalid BERT word_id at token {token_index}: {parent_word_id!r}")
        if len(raw_offset) != 2:
            raise ValueError(f"invalid offset mapping at token {token_index}: {raw_offset!r}")

        offset_start, offset_end = int(raw_offset[0]), int(raw_offset[1])
        parent_word = words[parent_word_id]
        if not (0 <= offset_start <= offset_end <= len(parent_word.raw_word)):
            raise ValueError(
                "BERT offset is outside its parent word: "
                f"word_id={parent_word_id}, offset={(offset_start, offset_end)!r}, "
                f"word_length={len(parent_word.raw_word)}"
            )

        pieces.append(
            BertPiece(
                piece_index=len(pieces),
                token_index=token_index,
                token_id=int(token_id),
                token=str(token),
                word_id=int(parent_word_id),
                offset_start=offset_start,
                offset_end=offset_end,
                char_start=parent_word.char_start + offset_start,
                char_end=parent_word.char_start + offset_end,
                is_unknown=str(token).upper() == "[UNK]",
            )
        )
    return pieces


def map_bert_offset_mapping(
    words: Sequence[Word],
    input_ids: Sequence[int],
    word_ids: Sequence[int | None],
    offset_mapping: Sequence[Sequence[int]],
    tokens: Sequence[str] | None = None,
) -> list[BertPiece]:
    """Explicit-output alias for :func:`map_bert_offsets`."""

    return map_bert_offsets(words, input_ids, word_ids, offset_mapping, tokens)


def extract_bert_pieces(words: Sequence[Word], encoded: Any) -> list[BertPiece]:
    """Extract content pieces from a tokenizer/BatchEncoding object."""

    return map_bert_offsets(words, encoded)


def piece_indices_by_word(
    words: Sequence[Word], pieces: Iterable[BertPiece]
) -> list[list[int]]:
    """Return global content-piece indices grouped by original word ID."""

    grouped = [[] for _ in words]
    for piece in pieces:
        if not 0 <= piece.word_id < len(words):
            raise ValueError(f"piece has invalid word_id: {piece.word_id}")
        if piece.piece_index not in grouped[piece.word_id]:
            grouped[piece.word_id].append(piece.piece_index)
    return grouped


def with_bert_piece_indices(
    words: Sequence[Word], pieces: Iterable[BertPiece]
) -> list[Word]:
    """Return copies of ``words`` carrying the grouped global piece indices."""

    grouped = piece_indices_by_word(words, pieces)
    return [replace(word, bert_piece_indices=indices) for word, indices in zip(words, grouped)]


class BertChunk(NamedTuple):
    """Half-open global content-token range covered by one BERT chunk."""

    content_start: int
    content_end: int

    @property
    def input_length(self) -> int:
        return self.content_end - self.content_start + 2


def plan_bert_chunks(
    content_length: int,
    content_tokens_per_chunk: int = DEFAULT_BERT_CONTENT_TOKENS,
    chunk_stride: int = DEFAULT_BERT_CHUNK_STRIDE,
) -> list[BertChunk]:
    """Plan complete overlapping BERT content ranges.

    ``content_end`` is exclusive.  The loop stops as soon as a chunk reaches
    the global tail; this avoids adding a redundant chunk that is completely
    covered by the already planned final chunk.
    """

    if not isinstance(content_length, int) or content_length < 0:
        raise ValueError("content_length must be a non-negative integer")
    if not isinstance(content_tokens_per_chunk, int) or content_tokens_per_chunk <= 0:
        raise ValueError("content_tokens_per_chunk must be positive")
    if not isinstance(chunk_stride, int) or chunk_stride <= 0:
        raise ValueError("chunk_stride must be positive")
    if chunk_stride > content_tokens_per_chunk:
        raise ValueError("chunk_stride cannot exceed content_tokens_per_chunk")

    if content_length == 0:
        return []

    chunks: list[BertChunk] = []
    start = 0
    while True:
        end = min(start + content_tokens_per_chunk, content_length)
        chunks.append(BertChunk(start, end))
        if end >= content_length:
            break
        next_start = start + chunk_stride
        if next_start <= start:
            raise ValueError("chunk_stride must advance chunk starts")
        start = next_start
    return chunks


def bert_chunk_plan(
    content_length: int,
    content_tokens_per_chunk: int = DEFAULT_BERT_CONTENT_TOKENS,
    chunk_stride: int = DEFAULT_BERT_CHUNK_STRIDE,
) -> list[BertChunk]:
    """Naming alias for :func:`plan_bert_chunks`."""

    return plan_bert_chunks(content_length, content_tokens_per_chunk, chunk_stride)


def merge_bert_chunk_vectors(
    chunk_vectors: Sequence[Sequence[Sequence[float]]],
    chunks: Sequence[Sequence[int] | BertChunk],
    content_length: int | None = None,
) -> list[Any]:
    """Average repeated global subword vectors across overlapping chunks.

    The returned list is indexed by global content-piece index.  Accumulation
    uses float64 and the final values use float32 when NumPy is available,
    matching the numerical rule in section 9.2 without requiring NumPy just
    to import this text-mapping module.
    """

    if len(chunk_vectors) != len(chunks):
        raise ValueError("chunk_vectors and chunks have inconsistent lengths")

    normalized_chunks = [(int(chunk[0]), int(chunk[1])) for chunk in chunks]
    if content_length is None:
        content_length = max((end for _, end in normalized_chunks), default=0)
    if content_length < 0:
        raise ValueError("content_length must be non-negative")

    dimension: int | None = None
    sums: list[list[float]] = [[0.0] * 0 for _ in range(content_length)]
    counts = [0] * content_length
    for vectors, (start, end) in zip(chunk_vectors, normalized_chunks):
        if not (0 <= start <= end <= content_length):
            raise ValueError(f"invalid chunk range: {(start, end)!r}")
        if len(vectors) != end - start:
            raise ValueError(
                f"chunk vector length {len(vectors)} does not match range {(start, end)!r}"
            )
        for local_index, vector in enumerate(vectors):
            values = [float(value) for value in vector]
            if dimension is None:
                dimension = len(values)
                sums = [[0.0] * dimension for _ in range(content_length)]
            elif len(values) != dimension:
                raise ValueError("BERT chunk vectors have inconsistent dimensions")
            global_index = start + local_index
            for component, value in enumerate(values):
                sums[global_index][component] += value
            counts[global_index] += 1

    if any(count == 0 for count in counts):
        missing = [index for index, count in enumerate(counts) if count == 0]
        raise ValueError(f"BERT chunk plan leaves global pieces uncovered: {missing[:5]!r}")

    merged = [
        [value / count for value in row]
        for row, count in zip(sums, counts)
    ]
    try:
        import numpy as np  # type: ignore

        return [np.asarray(vector, dtype=np.float32) for vector in merged]
    except ImportError:
        return merged


def pool_bert_vectors_by_word(
    words: Sequence[Word],
    pieces: Sequence[BertPiece],
    piece_vectors: Sequence[Sequence[float]],
    hidden_size: int | None = None,
) -> Any:
    """Mean each distinct global content piece into its parent word.

    Overlap has already been collapsed by global piece index before this
    function is called.  Thus a word split into three subwords is averaged
    over three pieces, not over the number of overlapping chunk occurrences.
    """

    if len(pieces) != len(piece_vectors):
        raise ValueError("pieces and piece_vectors have inconsistent lengths")
    grouped = piece_indices_by_word(words, pieces)
    if hidden_size is None:
        hidden_size = len(piece_vectors[0]) if piece_vectors else 0

    # Be defensive if a caller supplies one occurrence per chunk instead of
    # the already merged global-piece table: average occurrences by global
    # piece first, then pool each distinct piece into its parent word.
    occurrence_sums: dict[int, list[float]] = {}
    occurrence_counts: dict[int, int] = {}
    piece_word_ids: dict[int, int] = {}
    for piece, vector in zip(pieces, piece_vectors):
        if len(vector) != hidden_size:
            raise ValueError("BERT vectors have inconsistent dimensions")
        if piece.piece_index in piece_word_ids and piece_word_ids[piece.piece_index] != piece.word_id:
            raise ValueError(f"global piece {piece.piece_index} has multiple parent words")
        piece_word_ids[piece.piece_index] = piece.word_id
        row = occurrence_sums.setdefault(piece.piece_index, [0.0] * hidden_size)
        for component, value in enumerate(vector):
            row[component] += float(value)
        occurrence_counts[piece.piece_index] = occurrence_counts.get(piece.piece_index, 0) + 1
    by_index = {
        index: [value / occurrence_counts[index] for value in values]
        for index, values in occurrence_sums.items()
    }

    result: list[list[float]] = []
    for indices in grouped:
        if not indices:
            result.append([0.0] * hidden_size)
            continue
        sums = [0.0] * hidden_size
        distinct_indices = list(dict.fromkeys(indices))
        for index in distinct_indices:
            if index not in by_index:
                raise ValueError(f"missing vector for global piece {index}")
            vector = by_index[index]
            if len(vector) != hidden_size:
                raise ValueError("BERT vectors have inconsistent dimensions")
            for component, value in enumerate(vector):
                sums[component] += float(value)
        result.append([value / len(distinct_indices) for value in sums])

    try:
        import numpy as np  # type: ignore

        return np.asarray(result, dtype=np.float32)
    except ImportError:
        return result


def _replace_curly_apostrophes(text: str) -> str:
    return "".join(_CURLY_APOSTROPHES.get(char, char) for char in text)


def _ascii_cardinal_under_1000(number: int) -> str:
    ones = (
        "zero",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
        "eleven",
        "twelve",
        "thirteen",
        "fourteen",
        "fifteen",
        "sixteen",
        "seventeen",
        "eighteen",
        "nineteen",
    )
    tens = (
        "",
        "",
        "twenty",
        "thirty",
        "forty",
        "fifty",
        "sixty",
        "seventy",
        "eighty",
        "ninety",
    )
    if number < 20:
        return ones[number]
    if number < 100:
        return tens[number // 10] + (f"-{ones[number % 10]}" if number % 10 else "")
    remainder = number % 100
    if remainder:
        return f"{ones[number // 100]} hundred and {_ascii_cardinal_under_1000(remainder)}"
    return f"{ones[number // 100]} hundred"


def _fallback_cardinal(number: int) -> str:
    if number < 0:
        return "minus " + _fallback_cardinal(-number)
    if number < 1000:
        return _ascii_cardinal_under_1000(number)

    scales = (
        (10**12, "trillion"),
        (10**9, "billion"),
        (10**6, "million"),
        (10**3, "thousand"),
    )
    parts: list[str] = []
    remainder = number
    for scale_value, scale_name in scales:
        if remainder >= scale_value:
            groups, remainder = divmod(remainder, scale_value)
            parts.append(f"{_fallback_cardinal(groups)} {scale_name}")
    if remainder:
        if parts and remainder < 100:
            parts.append("and " + _fallback_cardinal(remainder))
        else:
            parts.append(_fallback_cardinal(remainder))
    return " ".join(parts)


def _fallback_ordinal(number: int) -> str:
    irregular = {
        0: "zeroth",
        1: "first",
        2: "second",
        3: "third",
        4: "fourth",
        5: "fifth",
        6: "sixth",
        7: "seventh",
        8: "eighth",
        9: "ninth",
        10: "tenth",
        11: "eleventh",
        12: "twelfth",
        13: "thirteenth",
        14: "fourteenth",
        15: "fifteenth",
        16: "sixteenth",
        17: "seventeenth",
        18: "eighteenth",
        19: "nineteenth",
    }
    if number in irregular:
        return irregular[number]
    if number < 0:
        return "minus " + _fallback_ordinal(-number)
    if number < 100:
        tens = {
            20: "twentieth",
            30: "thirtieth",
            40: "fortieth",
            50: "fiftieth",
            60: "sixtieth",
            70: "seventieth",
            80: "eightieth",
            90: "ninetieth",
        }
        if number in tens:
            return tens[number]
        return _fallback_cardinal(number // 10 * 10) + "-" + _fallback_ordinal(number % 10)
    if number % 1000 == 0:
        return _fallback_cardinal(number // 1000) + " thousandth"
    if number >= 1000:
        prefix, remainder = divmod(number, 1000)
        prefix_words = _fallback_cardinal(prefix) + " thousand"
        if remainder < 100:
            return prefix_words + " and " + _fallback_ordinal(remainder)
        return prefix_words + " " + _fallback_ordinal(remainder)
    if number % 100 == 0:
        return _fallback_cardinal(number // 100) + " hundredth"
    return _fallback_cardinal(number // 100 * 100) + " and " + _fallback_ordinal(number % 100)


def _num2words(number: int, *, ordinal: bool = False) -> str:
    """Use num2words when present, with a deterministic local fallback."""

    try:
        from num2words import num2words as external_num2words  # type: ignore
    except ImportError:
        external_num2words = None

    if external_num2words is not None:
        if ordinal:
            return str(external_num2words(number, lang="en", to="ordinal"))
        return str(external_num2words(number, lang="en", to="cardinal"))
    return _fallback_ordinal(number) if ordinal else _fallback_cardinal(number)


def _valid_integer_part(integer_part: str) -> bool:
    if not integer_part:
        return False
    if "," not in integer_part:
        return integer_part.isdigit()
    return bool(re.fullmatch(r"\d{1,3}(?:,\d{3})+", integer_part))


def _number_expression_to_words(expression: str) -> tuple[str, str]:
    """Return lower-case spoken words and the transformation kind."""

    percent = expression.endswith("%")
    core = expression[:-1] if percent else expression
    ordinal_match = re.search(r"(st|nd|rd|th)$", core, flags=re.IGNORECASE)
    ordinal = ordinal_match is not None
    if ordinal:
        core = core[: -len(ordinal_match.group(1))]

    if "." in core:
        integer_part, fractional_part = core.split(".", 1)
        if not _valid_integer_part(integer_part) or not fractional_part.isdigit():
            raise ValueError(f"unsupported decimal expression: {expression!r}")
        integer = int(integer_part.replace(",", ""))
        words = _num2words(integer)
        words += " point " + " ".join(_num2words(int(digit)) for digit in fractional_part)
        kind = "decimal"
    else:
        if not _valid_integer_part(core):
            raise ValueError(f"unsupported numeric expression: {expression!r}")
        integer = int(core.replace(",", ""))
        words = _num2words(integer, ordinal=ordinal)
        kind = "ordinal" if ordinal else "cardinal"

    if percent:
        words += " percent"
        kind = "decimal_percent" if kind == "decimal" else "percent"
    return words, kind


def _expand_number_expressions(text: str, transformations: list[str]) -> str:
    def replace(match: re.Match[str]) -> str:
        expression = match.group("number")
        try:
            spoken, kind = _number_expression_to_words(expression)
        except ValueError:
            return expression
        transformations.append(f"number_{kind}")
        transformations.append("normalization_assumed")
        if "," in expression:
            transformations.append("thousands_separator_removed")
        if "." in expression:
            transformations.append("decimal_point_expanded")
        return f" {spoken} "

    return _NUMBER_EXPRESSION.sub(replace, text)


def _is_ascii_letter(char: str) -> bool:
    return "A" <= char <= "Z" or "a" <= char <= "z"


def _clean_ctc_characters(text: str, transformations: list[str]) -> tuple[str, tuple[str, ...]]:
    output: list[str] = []
    unsupported: list[str] = []
    punctuation_removed = False
    separators_seen = False

    for char in text:
        if char.isspace():
            output.append(" ")
        elif _is_ascii_letter(char):
            output.append(char.upper())
        elif char == "'":
            # Keep it for one pass; outer/standalone apostrophes are removed
            # below, while contractions such as DON'T remain intact.
            output.append(char)
        elif char in _SEPARATOR_CHARS:
            output.append(" ")
            separators_seen = True
        elif unicodedata.category(char).startswith("P"):
            output.append(" ")
            punctuation_removed = True
        else:
            # Digits that escaped numeric recognition, non-Latin letters, and
            # symbols are meaningful uncertainty.  Preserve them in ctc_text
            # so a later SentencePiece/UNK decision cannot hide the source.
            output.append(char)
            unsupported.append(char)

    cleaned = "".join(output)
    cleaned_without_outer_apostrophes = re.sub(
        r"(?<![A-Za-z])'+|'+(?![A-Za-z])", " ", cleaned
    )
    if cleaned_without_outer_apostrophes != cleaned:
        punctuation_removed = True
    cleaned = re.sub(r"\s+", " ", cleaned_without_outer_apostrophes).strip()

    if separators_seen:
        transformations.append("separators_to_spaces")
    if punctuation_removed:
        transformations.append("punctuation_removed")
    if any("'" in part for part in re.findall(r"[A-Za-z']+", cleaned)):
        transformations.append("apostrophe_preserved")

    # Keep first-seen order while making the diagnostic stable and compact.
    unique_unsupported = tuple(dict.fromkeys(unsupported))
    if unique_unsupported:
        transformations.append("unsupported_text")
    return cleaned, unique_unsupported


@dataclass(frozen=True)
class CtcNormalization:
    """Pure result of normalizing one original word for CTC alignment."""

    raw_text: str
    ctc_text: str
    status: str
    normalization_assumed: bool = False
    transformations: tuple[str, ...] = ()
    unsupported_characters: tuple[str, ...] = ()
    contains_unsupported: bool = False

    @property
    def no_spoken_content(self) -> bool:
        return self.status == "no_spoken_content"

    @property
    def unsupported_text(self) -> bool:
        return self.contains_unsupported

    @property
    def normalization_types(self) -> tuple[str, ...]:
        return self.transformations


def normalize_ctc_text(raw_word: str | Word) -> CtcNormalization:
    """Normalize one word conservatively for the English CTC target.

    Numeric replacements happen before generic punctuation removal.  The
    returned normalized text never silently discards a meaningful unsupported
    character; such characters stay in ``ctc_text`` and set
    ``contains_unsupported``/``status``.
    """

    if isinstance(raw_word, Word):
        raw_text = raw_word.raw_word
    elif isinstance(raw_word, str):
        raw_text = raw_word
    else:
        raise TypeError("raw_word must be a str or Word")

    transformations: list[str] = []
    nfkd_text = unicodedata.normalize("NFKD", raw_text)
    without_marks = "".join(
        char for char in nfkd_text if not unicodedata.category(char).startswith("M")
    )
    if without_marks != raw_text:
        transformations.append("nfkd_without_combining_marks")

    quote_text = _replace_curly_apostrophes(without_marks)
    if quote_text != without_marks:
        transformations.append("curly_apostrophe_to_ascii")

    expanded = _expand_number_expressions(quote_text, transformations)
    cleaned, unsupported_characters = _clean_ctc_characters(expanded, transformations)
    # The execution specification treats every word containing digits as an
    # assumed reading, including an alphanumeric/otherwise unsupported token
    # for which no safe numeric expansion was possible.
    if any(char.isdigit() for char in quote_text):
        if "normalization_assumed" not in transformations:
            transformations.append("normalization_assumed")
    transformations = list(dict.fromkeys(transformations))
    normalization_assumed = "normalization_assumed" in transformations

    if not cleaned:
        status = "no_spoken_content"
    elif unsupported_characters:
        status = "unsupported_text"
    else:
        status = "ok"

    return CtcNormalization(
        raw_text=raw_text,
        ctc_text=cleaned,
        status=status,
        normalization_assumed=normalization_assumed,
        transformations=tuple(transformations),
        unsupported_characters=unsupported_characters,
        contains_unsupported=bool(unsupported_characters),
    )


def normalize_word_ctc(word: Word) -> Word:
    """Return a copy of ``word`` carrying its pure CTC normalization result."""

    result = normalize_ctc_text(word)
    return replace(
        word,
        ctc_text=result.ctc_text,
        normalization_assumed=result.normalization_assumed,
        normalization_types=result.transformations,
        normalization_status=result.status,
        contains_unsupported=result.contains_unsupported,
    )


def normalize_words_ctc(words: Sequence[Word]) -> list[Word]:
    """Purely normalize a complete word list without changing word IDs."""

    return [normalize_word_ctc(word) for word in words]


@dataclass(frozen=True)
class CtcPieceMapping:
    """SentencePiece strings mapped into the ESPnet token-list ID space."""

    token_ids: tuple[int, ...]
    pieces: tuple[str, ...]
    contains_unk: bool


def _token_list_mapping(token_list: Sequence[str] | Mapping[str, int]) -> dict[str, int]:
    if isinstance(token_list, Mapping):
        return {str(piece): int(token_id) for piece, token_id in token_list.items()}
    return {str(piece): token_id for token_id, piece in enumerate(token_list)}


def _find_unk_id(
    token_list: Sequence[str] | Mapping[str, int],
    mapping: Mapping[str, int],
    unk_id: int | None,
) -> int | None:
    if unk_id is not None:
        return int(unk_id)
    for candidate in ("<unk>", "<UNK>", "<unknown>", "<UNKNOWN>"):
        if candidate in mapping:
            return int(mapping[candidate])
    # Some SentencePiece/ESPnet token lists use a case variant with metadata
    # around it.  Resolve it without assuming a fixed ID beyond the explicit
    # token-list mapping.
    for piece, token_id in mapping.items():
        if piece.strip("<>").lower() in {"unk", "unknown"}:
            return int(token_id)
    return None


def map_ctc_pieces_to_ids(
    pieces: Sequence[str],
    token_list: Sequence[str] | Mapping[str, int],
    *,
    unk_id: int | None = None,
) -> CtcPieceMapping:
    """Map SentencePiece strings through the complete ESPnet token list.

    SentencePiece integer IDs are intentionally ignored.  Missing pieces and
    explicit ``<unk>`` pieces receive the model UNK ID and set
    ``contains_unk`` so the downstream alignment stage can retain a candidate
    but reject it as an accepted word interval.
    """

    mapping = _token_list_mapping(token_list)
    resolved_unk_id = _find_unk_id(token_list, mapping, unk_id)
    token_ids: list[int] = []
    contains_unk = False
    normalized_pieces: list[str] = []
    for piece in pieces:
        normalized_piece = str(piece)
        normalized_pieces.append(normalized_piece)
        if normalized_piece in mapping:
            token_id = int(mapping[normalized_piece])
            token_ids.append(token_id)
            if token_id == resolved_unk_id or normalized_piece.strip("<>").lower() in {
                "unk",
                "unknown",
            }:
                contains_unk = True
        else:
            if resolved_unk_id is None:
                raise ValueError("token_list has no <unk> entry; pass unk_id explicitly")
            token_ids.append(resolved_unk_id)
            contains_unk = True
    return CtcPieceMapping(tuple(token_ids), tuple(normalized_pieces), contains_unk)


def _sentencepiece_encode(sentencepiece_model: Any, text: str) -> list[str]:
    if hasattr(sentencepiece_model, "encode"):
        try:
            pieces = sentencepiece_model.encode(text, out_type=str)
        except TypeError:
            pieces = sentencepiece_model.encode(text)
    elif callable(sentencepiece_model):
        pieces = sentencepiece_model(text)
    else:
        raise TypeError("sentencepiece_model must provide encode() or be callable")
    if isinstance(pieces, str):
        return [pieces]
    return [str(piece) for piece in pieces]


@dataclass
class CtcPreparation:
    """Rows passed to ``ctc_segmentation`` plus the explicit source mapping.

    The object behaves as a three-item sequence for convenient unpacking:
    ``token_arrays, ctc_texts, ctc_row_to_word_id = result``.
    """

    token_arrays: list[list[int]]
    ctc_texts: list[str]
    ctc_row_to_word_id: list[int]
    normalizations: dict[int, CtcNormalization] = field(default_factory=dict)
    contains_unk_word_ids: set[int] = field(default_factory=set)
    unsupported_word_ids: set[int] = field(default_factory=set)
    skipped_word_ids: set[int] = field(default_factory=set)

    @property
    def no_spoken_content_word_ids(self) -> set[int]:
        return {
            word_id
            for word_id, normalization in self.normalizations.items()
            if normalization.no_spoken_content
        }

    def __iter__(self) -> Iterator[Any]:
        yield self.token_arrays
        yield self.ctc_texts
        yield self.ctc_row_to_word_id

    def __len__(self) -> int:
        return 3

    def __getitem__(self, index: int) -> Any:
        values = (self.token_arrays, self.ctc_texts, self.ctc_row_to_word_id)
        return values[index]


def prepare_ctc_rows(
    words: Sequence[Word],
    sentencepiece_model: Any,
    token_list: Sequence[str] | Mapping[str, int],
    *,
    unk_id: int | None = None,
) -> CtcPreparation:
    """Prepare non-empty CTC rows while preserving original word identity.

    A pure-punctuation/no-spoken-content word is intentionally absent from
    ``token_arrays``.  Every other attempted row remains represented, even if
    one of its pieces maps to UNK.  Consumers must use
    ``ctc_row_to_word_id[row]`` instead of zipping CTC output with all words.
    """

    token_arrays: list[list[int]] = []
    ctc_texts: list[str] = []
    ctc_row_to_word_id: list[int] = []
    normalizations: dict[int, CtcNormalization] = {}
    contains_unk_word_ids: set[int] = set()
    unsupported_word_ids: set[int] = set()
    skipped_word_ids: set[int] = set()

    for word in words:
        normalization = normalize_ctc_text(word)
        normalizations[word.word_id] = normalization
        if normalization.contains_unsupported:
            unsupported_word_ids.add(word.word_id)
        if normalization.no_spoken_content:
            skipped_word_ids.add(word.word_id)
            continue

        pieces = _sentencepiece_encode(sentencepiece_model, normalization.ctc_text)
        pieces = [
            piece
            for piece in pieces
            if piece not in {"<sos>", "<eos>", "<sos/eos>", "<pad>", "<blank>"}
        ]
        if not pieces:
            # Do not pass an empty array to prepare_token_list, but keep the
            # original word ID in diagnostics through skipped_word_ids.
            skipped_word_ids.add(word.word_id)
            continue

        mapped = map_ctc_pieces_to_ids(pieces, token_list, unk_id=unk_id)
        token_arrays.append(list(mapped.token_ids))
        ctc_texts.append(normalization.ctc_text)
        ctc_row_to_word_id.append(word.word_id)
        if mapped.contains_unk:
            contains_unk_word_ids.add(word.word_id)

    return CtcPreparation(
        token_arrays=token_arrays,
        ctc_texts=ctc_texts,
        ctc_row_to_word_id=ctc_row_to_word_id,
        normalizations=normalizations,
        contains_unk_word_ids=contains_unk_word_ids,
        unsupported_word_ids=unsupported_word_ids,
        skipped_word_ids=skipped_word_ids,
    )


def prepare_ctc_inputs(
    words: Sequence[Word],
    sentencepiece_model: Any,
    token_list: Sequence[str] | Mapping[str, int],
    *,
    unk_id: int | None = None,
) -> CtcPreparation:
    """Naming alias for :func:`prepare_ctc_rows`."""

    return prepare_ctc_rows(words, sentencepiece_model, token_list, unk_id=unk_id)


def prepare_ctc_token_rows(
    words: Sequence[Word],
    sentencepiece_model: Any,
    token_list: Sequence[str] | Mapping[str, int],
    *,
    unk_id: int | None = None,
) -> CtcPreparation:
    """Naming alias emphasizing that rows are token-ID arrays."""

    return prepare_ctc_rows(words, sentencepiece_model, token_list, unk_id=unk_id)


__all__ = [
    "BertChunk",
    "BertPiece",
    "CtcNormalization",
    "CtcPieceMapping",
    "CtcPreparation",
    "DEFAULT_BERT_CHUNK_STRIDE",
    "DEFAULT_BERT_CONTENT_TOKENS",
    "Word",
    "bert_chunk_plan",
    "build_words",
    "extract_bert_pieces",
    "is_pure_punctuation",
    "map_bert_offset_mapping",
    "map_bert_offsets",
    "map_ctc_pieces_to_ids",
    "merge_bert_chunk_vectors",
    "normalize_ctc_text",
    "normalize_word_ctc",
    "normalize_words_ctc",
    "piece_indices_by_word",
    "plan_bert_chunks",
    "pool_bert_vectors_by_word",
    "prepare_ctc_inputs",
    "prepare_ctc_rows",
    "prepare_ctc_token_rows",
    "with_bert_piece_indices",
]
