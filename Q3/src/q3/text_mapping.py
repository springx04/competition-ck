import html
import re

def map_tokens(raw_text, offsets, ids, attention, token_type, observed):
    words = [(match.start(), match.end()) for match in re.finditer(r"\S+", raw_text or "")]
    rows = []
    for t, (start, end) in enumerate(offsets):
        hits = [i for i, (a, b) in enumerate(words) if end > a and start < b] if end > start else []
        rows.append({"official_t": t, "token_id": int(ids[t]), "stored_attention": int(attention[t]), "token_type": int(token_type[t]), "is_observed": bool(observed[t]), "is_unknown": int(ids[t]) == 100, "char_start": int(start), "char_end": int(end), "word_id": hits[0] if len(hits) == 1 else None, "token_mapping_status": "matched" if len(hits) == 1 else "unexplained"})
    return rows

def highlight(text, start, end):
    return html.escape(text[:start]) + "<mark>" + html.escape(text[start:end]) + "</mark>" + html.escape(text[end:])
