import numpy as np
from q3.text_mapping import map_tokens, highlight

def test_unicode_offsets_and_repeated_words():
    text="go 😀 go"
    rows=map_tokens(text,[(0,2),(3,4),(5,7)],[1,100,1],[1,1,1],[0,0,0],[1,1,1])
    assert [row["word_id"] for row in rows]==[0,1,2]
    assert "<mark>😀</mark>" in highlight(text,3,4)
