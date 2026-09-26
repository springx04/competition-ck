import numpy as np
from q3.text_mapping import map_tokens, highlight

def test_unicode_offsets_and_repeated_words():
    text="go 😀 go"
    rows=map_tokens(text,[(0,2),(3,4),(5,7)],[1,100,1],[1,1,1],[0,0,0],[1,1,1])
    assert [row["word_id"] for row in rows]==[0,1,2]
    assert "<mark>😀</mark>" in highlight(text,3,4)


def test_special_and_padding_offsets_do_not_attach_to_first_word():
    rows = map_tokens(
        "hello",
        [(0, 0), (0, 5), (0, 0)],
        [101, 42, 0],
        [1, 1, 0],
        [0, 0, 0],
        [0, 1, 0],
        ["[CLS]", "hello", "[PAD]"],
    )
    assert rows[0]["word_id"] is None
    assert rows[2]["word_id"] is None
    assert rows[1]["word_id"] == 0
    assert rows[1]["char_span"] == [0, 5]
