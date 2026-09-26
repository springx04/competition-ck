"""Build traceable text/CTC/material mappings for one Q3 case.

This module intentionally uses only the Python standard library.  Text
positions are accepted only after the tokenizer reproduces all three stored
arrays (IDs, attention and token type IDs).  CTC rows remain algorithmic
evidence until an existing review row has a reviewer, a parseable review time,
and valid half-open time bounds.  Audio/visual rows remain unresolved unless
an explicit provenance source reference is present.
"""

from __future__ import annotations

import csv
import json
import math
import re
from datetime import datetime
from pathlib import Path

from .text_mapping import _as_list, map_tokens


def unresolved_mapping(reason="official A/V provenance not supplied"):
    return {"status": "unresolved", "reason": reason}


_MISSING = {"", "none", "null", "nan", "na", "n/a", "unknown", "unresolved", "pending"}
_MATCH_REVIEW = {"confirmed_match", "confirmed", "verified", "approved", "accepted", "pass", "match"}
_MISMATCH_REVIEW = {"confirmed_mismatch", "mismatch", "rejected", "reject", "fail"}


def _get(value, key, default=None):
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _first(*values):
    for value in values:
        if value is not None:
            return value
    return None


def _plain(value):
    """Make values from numpy/torch/csv inputs JSON serializable."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if hasattr(value, "item"):
        try:
            return _plain(value.item())
        except (ValueError, TypeError):
            pass
    if hasattr(value, "tolist"):
        try:
            return _plain(value.tolist())
        except (ValueError, TypeError):
            pass
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return str(value)


def _int_list(value):
    result = []
    for item in _as_list(value):
        try:
            result.append(int(item))
        except (TypeError, ValueError, OverflowError):
            result.append(None)
    return result


def _raw_mapping(sample, original):
    raw = _get(sample, "raw", None)
    if raw is None and isinstance(sample, dict):
        raw = sample
    if raw is None:
        raw = {}
    original_raw = _get(original, "raw", None) or {}

    def field(*names):
        for source in (raw, original_raw, original):
            if source is None:
                continue
            for name in names:
                value = _get(source, name, None)
                if value is not None:
                    return value
        return None

    return {
        "input_ids": _int_list(field("input_ids", "ids", "text_input_ids")),
        "stored_attention": _int_list(field("stored_attention", "attention_mask", "attention", "text_attention")),
        "token_type_ids": _int_list(field("token_type_ids", "token_type", "text_token_type_ids")),
    }


def _raw_text(sample, original):
    return _first(_get(sample, "raw_text", None), _get(sample, "text", None), _get(original, "raw_text", None), _get(original, "text", None))


def _encoded_get(encoded, *keys):
    for key in keys:
        if isinstance(encoded, dict) and key in encoded:
            return encoded[key]
        try:
            return encoded[key]
        except (KeyError, TypeError, IndexError):
            pass
        value = getattr(encoded, key, None)
        if value is not None:
            return value
    return None


def _tokenize(tokenizer, raw_text, length):
    kwargs = {
        "add_special_tokens": True,
        "truncation": True,
        "return_offsets_mapping": True,
    }
    if length:
        kwargs.update({"max_length": int(length), "padding": "max_length"})
    try:
        return tokenizer(raw_text, **kwargs)
    except TypeError:
        # Lightweight test tokenizers sometimes do not accept all optional
        # Hugging Face keyword arguments.  Keep the same semantic request.
        kwargs.pop("padding", None)
        kwargs.pop("truncation", None)
        try:
            return tokenizer(raw_text, **kwargs)
        except TypeError:
            kwargs.pop("max_length", None)
            return tokenizer(raw_text, **kwargs)


def _same_ints(left, right):
    return bool(left) and len(left) == len(right) and all(a is not None and a == b for a, b in zip(left, right))


def _offsets_valid(offsets, text_length, expected_length):
    if len(offsets) != expected_length:
        return False
    for value in offsets:
        values = _as_list(value)
        if len(values) < 2:
            return False
        try:
            start, end = int(values[0]), int(values[1])
        except (TypeError, ValueError, OverflowError):
            return False
        if start < 0 or end < start or end > text_length:
            return False
    return True


def _observed(original, attention, length):
    value = _get(original, "U0", None)
    if value is not None:
        value = _as_list(value)
        result = []
        for row in value[:length]:
            row = _as_list(row)
            result.append(bool(row[0]) if row else False)
        if len(result) == length:
            return result
    return [bool(item) for item in (attention[:length] + [0] * length)[:length]]


def _path(config, *locations):
    value = None
    for location in locations:
        current = config
        if isinstance(location, str):
            location = (location,)
        for key in location:
            if not isinstance(current, dict):
                current = None
                break
            current = current.get(key)
        if current is not None:
            value = current
            break
    if value is None:
        return None
    path = Path(str(value))
    if path.is_absolute():
        return path
    root = None
    for location in (("project", "root"), ("root",)):
        current = config
        for key in location:
            current = current.get(key) if isinstance(current, dict) else None
        if current is not None:
            root = Path(str(current))
            break
    return (root / path).resolve() if root is not None else path.resolve()


def _sample_id(sample, original):
    return str(_first(_get(sample, "sample_id", None), _get(sample, "id", None), _get(original, "sample_id", None), ""))


def _read_jsonl(path):
    rows = []
    errors = []
    if path is None or not path.is_file():
        return rows, ["ctc_file_missing" if path is None else f"ctc_file_missing:{path}"]
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                    if isinstance(item, dict):
                        rows.append(item)
                    else:
                        errors.append(f"ctc_line_not_object:{line_number}")
                except json.JSONDecodeError:
                    errors.append(f"ctc_invalid_json:{line_number}")
    except OSError as exc:
        errors.append(f"ctc_read_error:{exc}")
    return rows, errors


def _read_table(path):
    if path is None or not path.is_file():
        return [], ["file_missing" if path is None else f"file_missing:{path}"]
    try:
        suffix = path.suffix.lower()
        if suffix in {".json", ".jsonl"}:
            if suffix == ".json":
                value = json.loads(path.read_text(encoding="utf-8"))
                return (value if isinstance(value, list) else [value]), []
            return _read_jsonl(path)
        with path.open(encoding="utf-8-sig", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)], []
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return [], [f"file_read_error:{exc}"]


def _text_value(row, *keys):
    for key in keys:
        value = row.get(key) if isinstance(row, dict) else None
        if value is not None and str(value).strip() != "":
            return str(value).strip()
    return ""


def _float_value(row, *keys):
    value = _text_value(row, *keys)
    if not value:
        return None
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError, OverflowError):
        return None


def _valid_bounds(start, end):
    return start is not None and end is not None and math.isfinite(start) and math.isfinite(end) and start >= 0 and end > start


def _valid_timestamp(value):
    if not value:
        return False
    text = str(value).strip()
    try:
        datetime.fromisoformat(text.replace("Z", "+00:00"))
        return True
    except ValueError:
        return False


def _load_reviews(config):
    path = _path(config, ("alignment", "review_file"), ("review_file",))
    rows, errors = _read_table(path)
    indexed = {}
    for row in rows:
        sid = _text_value(row, "sample_id", "id")
        wid = _text_value(row, "word_id")
        if sid and wid:
            indexed[(sid, wid)] = row
    return path, indexed, errors


def _load_provenance(config):
    path = _path(config, ("alignment", "av_provenance_file"), ("av_provenance_file",))
    rows, errors = _read_table(path)
    return path, rows, errors


def _load_ctc(config, sample_id):
    run = _path(config, ("output", "special_run"), ("special_run",), ("project", "special_run"))
    path = None if run is None else run / "alignment" / str(sample_id) / "words.jsonl"
    return path, *_read_jsonl(path)


def _reviewed_time(row, auto_start, auto_end):
    manual_start = _float_value(row, "review_start_common", "review_start", "gold_start", "start_reviewed")
    manual_end = _float_value(row, "review_end_common", "review_end", "gold_end", "end_reviewed")
    if (manual_start is None) != (manual_end is None):
        return None, None, False, "review_bounds_incomplete"
    start, end = (manual_start, manual_end) if manual_start is not None else (auto_start, auto_end)
    return start, end, _valid_bounds(start, end), None if _valid_bounds(start, end) else "invalid_time_bounds"


def _ctc_row(row, review):
    item = {str(key): _plain(value) for key, value in row.items()}
    auto_start = _float_value(row, "start_common", "start", "auto_start_common")
    auto_end = _float_value(row, "end_common", "end", "auto_end_common")
    item["candidate_start_common"] = auto_start
    item["candidate_end_common"] = auto_end
    item["start_common"] = auto_start
    item["end_common"] = auto_end
    item["algorithm_status"] = _text_value(row, "algorithm_status") or "unknown"
    item["review_status"] = _text_value(review or {}, "review_status") or _text_value(row, "review_status") or "pending"
    item["reviewer"] = _text_value(review or {}, "reviewer")
    item["reviewed_at"] = _text_value(review or {}, "reviewed_at", "review_time")
    item["time_source"] = "algorithm"
    item["verified"] = False
    item["material_status"] = "unresolved"
    item["material_reason"] = "no valid human review"
    if review:
        review_word = _text_value(review, "raw_word", "word")
        current_word = _text_value(row, "raw_word", "word")
        if review_word and current_word and review_word != current_word:
            item["material_reason"] = "review_raw_word_mismatch"
        else:
            start, end, bounds_ok, bounds_reason = _reviewed_time(review, auto_start, auto_end)
            review_status = str(item["review_status"]).strip().lower()
            reviewer_ok = bool(item["reviewer"])
            timestamp_ok = _valid_timestamp(item["reviewed_at"])
            if review_status in _MATCH_REVIEW and reviewer_ok and timestamp_ok and bounds_ok:
                item["start_common"], item["end_common"] = start, end
                item["time_source"] = "human_review" if _float_value(review, "review_start_common", "review_start", "gold_start", "start_reviewed") is not None else "algorithm_confirmed_by_human"
                item["verified"] = True
                item["material_status"] = "verified"
                item["material_reason"] = "human_review_with_valid_bounds"
            elif review_status in _MISMATCH_REVIEW and reviewer_ok and timestamp_ok and bounds_ok:
                item["start_common"], item["end_common"] = start, end
                item["time_source"] = "human_review"
                item["material_status"] = "reviewed_mismatch"
                item["material_reason"] = "human_review_rejected_candidate"
            elif review_status not in _MISSING:
                item["material_reason"] = bounds_reason or "reviewer_or_review_time_missing_or_invalid"
    if item["material_status"] == "unresolved":
        if str(item["algorithm_status"]).strip().lower() in {"accepted", "algorithm_accepted"} and _valid_bounds(auto_start, auto_end):
            item["material_status"] = "algorithm_accepted"
            item["material_reason"] = "algorithm_candidate_not_human_verified"
        elif str(item["algorithm_status"]).strip().lower() in {"rejected", "algorithm_rejected"}:
            item["material_status"] = "algorithm_rejected"
            item["material_reason"] = "algorithm_candidate_rejected"
    return item


def _provenance_rows(rows, sample_id, selection_id, modality):
    candidates = []
    for row in rows:
        sid = _text_value(row, "sample_id", "id")
        if sid and sid != str(sample_id):
            continue
        row_selection = _text_value(row, "selection_id", "evidence_id")
        row_modality = _selection_path({"path": _text_value(row, "modality", "path", "kind")})
        if row_selection and row_selection != str(selection_id):
            continue
        if row_modality and row_modality not in {str(modality).upper(), "J"}:
            continue
        source_ref = _text_value(row, "source_ref", "source", "source_id", "feature_source")
        if source_ref.lower() in _MISSING:
            continue
        candidates.append({str(key): _plain(value) for key, value in row.items()})
    return candidates


def _selection_path(selection):
    value = _text_value(selection, "path", "modality")
    if value:
        value = value.upper()
        return {"TEXT": "T", "AUDIO": "A", "SPEECH": "A", "VISUAL": "V", "VIDEO": "V"}.get(value, value)
    match = re.search(r"(?:^|\.)([TAVJ])(?:\.|$)", _text_value(selection, "selection_id"))
    return match.group(1) if match else ""


def _selection_intervals(selection):
    result = []
    for interval in _as_list(selection.get("intervals", []) if isinstance(selection, dict) else []):
        values = _as_list(interval)
        if len(values) < 2:
            continue
        try:
            start, end = int(values[0]), int(values[1])
        except (TypeError, ValueError, OverflowError):
            continue
        if end > start:
            result.append([start, end])
    return result


def _selection_tokens(selection, modality_index, token_count):
    atoms = []
    if isinstance(selection, dict):
        for value in _as_list(selection.get("delete_j", [])):
            try:
                atom = int(value)
            except (TypeError, ValueError, OverflowError):
                continue
            if atom >= 0 and atom % 3 == modality_index:
                token = atom // 3
                if token < token_count:
                    atoms.append(token)
    if not atoms and 'delete_j' not in selection:
        for start, end in _selection_intervals(selection):
            atoms.extend(range(max(0, start), min(token_count, end)))
    return sorted(set(atoms))


def _text_fragments(raw_text, token_rows, token_indices):
    selected = [row for row in token_rows if row.get("official_t") in set(token_indices) and row.get("char_end", 0) > row.get("char_start", 0) and row.get("word_id") is not None]
    selected.sort(key=lambda row: row["official_t"])
    groups = []
    for row in selected:
        if groups and row["official_t"] == groups[-1]["token_indices"][-1] + 1:
            groups[-1]["token_indices"].append(row["official_t"])
            groups[-1]["word_ids"].extend(row.get("word_ids", []))
            groups[-1]["end"] = row["char_end"]
        else:
            groups.append({"start": row["char_start"], "end": row["char_end"], "token_indices": [row["official_t"]], "word_ids": list(row.get("word_ids", []))})
    fragments = []
    for group in groups:
        start, end = int(group["start"]), int(group["end"])
        fragments.append({
            "text": raw_text[start:end],
            "char_span": [start, end],
            "token_indices": sorted(set(group["token_indices"])),
            "word_ids": sorted(set(int(item) for item in group["word_ids"])),
        })
    return fragments


def _selection_evidence(config, sample_id, selection, token_rows, raw_text, ctc_rows, reviews, provenance_rows):
    selection_id = _text_value(selection, "selection_id")
    path = _selection_path(selection)
    modalities = tuple("TAV"[m] for m in range(3) if any(int(a)%3==m for a in selection.get("delete_j",[]))) if path == "J" else tuple(item for item in path if item in "TAV")
    if not modalities:
        modalities = (path,) if path in "TAV" else tuple()
    modality_status = {}
    text_fragments = []
    word_ids = []
    word_times = []
    text_tokens = _selection_tokens(selection, 0, len(token_rows)) if "T" in modalities else []
    if "T" in modalities:
        text_fragments = _text_fragments(raw_text, token_rows, text_tokens)
        word_ids = sorted({int(item) for fragment in text_fragments for item in fragment["word_ids"]})
        for row in ctc_rows:
            try:
                row_word_id = int(row.get("word_id"))
            except (TypeError, ValueError, OverflowError):
                continue
            if row_word_id in word_ids:
                word_times.append(_ctc_row(row, reviews.get((str(sample_id), str(row_word_id)))))
        if not token_rows:
            modality_status["T"] = {"status": "unresolved", "reason": "token_rows_missing", "verified": False}
        elif text_fragments:
            modality_status["T"] = {"status": "resolved", "reason": "tokenizer_offsets_to_raw_text", "verified": True}
        else:
            modality_status["T"] = {"status": "unresolved", "reason": "selected_tokens_have_no_raw_character_span", "verified": False}
    for modality in "AV":
        if modality not in modalities:
            continue
        matches = _provenance_rows(provenance_rows, sample_id, selection_id, modality)
        source_refs = [_text_value(row, "source_ref", "source", "source_id", "feature_source") for row in matches]
        source_refs = [value for value in source_refs if value and value.lower() not in _MISSING]
        if source_refs:
            modality_status[modality] = {"status": "provenance_linked", "reason": "explicit_source_ref", "source_ref": source_refs[0], "verified": False, "provenance": matches}
        else:
            modality_status[modality] = {"status": "unresolved", "reason": "official_A/V_source_ref_missing", "source_ref": None, "verified": False, "provenance": []}
    statuses = [item["status"] for item in modality_status.values()]
    if statuses and all(item in {"resolved", "provenance_linked"} for item in statuses):
        status = "resolved"
    elif not statuses:
        status = "unresolved"
    else:
        status = "partial" if any(item in {"resolved", "provenance_linked", "partial"} for item in statuses) else "unresolved"
    return {
        "selection_id": selection_id,
        "path": path,
        "intervals": _selection_intervals(selection),
        "token_indices": text_tokens,
        "selected_text": " ".join(fragment["text"] for fragment in text_fragments),
        "text_fragments": text_fragments,
        "char_spans": [fragment["char_span"] for fragment in text_fragments],
        "word_ids": word_ids,
        "word_times": word_times,
        "ctc_candidate_count": len(word_times),
        "modality_status": modality_status,
        "material_status": status,
        "status": status,
        "verified": all(item.get("verified", False) for item in modality_status.values()) if modality_status else False,
    }


def build_mapping(config, sample, original, selections, tokenizer):
    """Build one JSON-serialisable mapping record.

    The function does not modify review or provenance files.  It reads the
    configured CTC and review artifacts and reports their independent status.
    """

    config = config or {}
    original = original or {}
    sample_id = _sample_id(sample, original)
    raw_text = _raw_text(sample, original)
    raw = _raw_mapping(sample, original)
    raw_ids = raw["input_ids"]
    raw_attention = raw["stored_attention"]
    raw_type = raw["token_type_ids"]
    validation = {
        "raw_text_present": raw_text is not None,
        "ids_match": False,
        "attention_match": False,
        "token_type_match": False,
        "offsets_valid": False,
        "exact": False,
        "errors": [],
    }
    encoded = None
    generated_ids, generated_attention, generated_type, offsets = [], [], [], []
    token_strings = None
    if raw_text is None:
        validation["errors"].append("raw_text_missing")
    elif tokenizer is None:
        validation["errors"].append("tokenizer_missing")
    else:
        try:
            encoded = _tokenize(tokenizer, str(raw_text), len(raw_ids))
            generated_ids = _int_list(_encoded_get(encoded, "input_ids"))
            generated_attention = _int_list(_encoded_get(encoded, "attention_mask", "stored_attention"))
            generated_type = _int_list(_encoded_get(encoded, "token_type_ids", "token_type"))
            offsets = _as_list(_encoded_get(encoded, "offset_mapping", "offsets"))
            validation["ids_match"] = _same_ints(generated_ids, raw_ids)
            validation["attention_match"] = _same_ints(generated_attention, raw_attention)
            validation["token_type_match"] = _same_ints(generated_type, raw_type)
            if not generated_ids:
                validation["errors"].append("tokenizer_ids_missing")
            if not generated_attention:
                validation["errors"].append("tokenizer_attention_missing")
            if not generated_type:
                validation["errors"].append("tokenizer_token_type_missing")
            validation["offsets_valid"] = _offsets_valid(offsets, len(str(raw_text)), len(raw_ids))
            if len(offsets) != len(raw_ids):
                validation["errors"].append("offset_length_mismatch")
            elif not validation["offsets_valid"]:
                validation["errors"].append("offsets_invalid")
        except Exception as exc:  # tokenizers vary in their optional kwargs
            validation["errors"].append(f"tokenizer_error:{exc}")
    validation["exact"] = bool(validation["raw_text_present"] and validation["ids_match"] and validation["attention_match"] and validation["token_type_match"] and validation["offsets_valid"])
    if raw_text is None:
        raw_text = ""
    row_length = len(raw_ids) or len(generated_ids)
    observed = _observed(original, raw_attention, row_length)
    if encoded is not None:
        try:
            token_strings = [str(item) for item in _as_list(tokenizer.convert_ids_to_tokens(generated_ids))]
        except (AttributeError, TypeError, ValueError):
            token_strings = None
    token_rows = map_tokens(raw_text, offsets if validation["exact"] else [], raw_ids or generated_ids, raw_attention or generated_attention, raw_type or generated_type, observed, token_strings if validation["exact"] else None)

    ctc_path, ctc_rows, ctc_errors = _load_ctc(config, sample_id)
    mismatched_ctc_rows = [row for row in ctc_rows if _text_value(row, "sample_id", "id") and _text_value(row, "sample_id", "id") != str(sample_id)]
    if mismatched_ctc_rows:
        ctc_errors.append("ctc_sample_id_mismatch")
        ctc_rows = [row for row in ctc_rows if row not in mismatched_ctc_rows]
    review_path, reviews, review_errors = _load_reviews(config)
    provenance_path, provenance_rows, provenance_errors = _load_provenance(config)
    evidence = {}
    for selection in selections or []:
        if not isinstance(selection, dict):
            continue
        item = _selection_evidence(config, sample_id, selection, token_rows, raw_text, ctc_rows, reviews, provenance_rows)
        if selection.get("status") == "not_applicable" or not selection.get("delete_j"):
            item.update(status="not_applicable", material_status="not_applicable", verified=False,
                        reason=selection.get("reason") or "no_selected_observations",
                        token_indices=[], selected_text="", text_fragments=[], char_spans=[],
                        word_ids=[], word_times=[], modality_status={})
        evidence[item["selection_id"]] = item

    text_selection_ids = [key for key, value in evidence.items() if "T" in value.get("modality_status", {})]
    av_selection_ids = [key for key, value in evidence.items() if any(modality in value.get("modality_status", {}) for modality in "AV")]
    token_mapped = sum(row.get("token_mapping_status") == "matched" for row in token_rows)
    ctc_candidate_count = sum(len(value.get("word_times", [])) for value in evidence.values())
    all_word_times = [_ctc_row(row,reviews.get((str(sample_id),str(row.get("word_id"))))) for row in ctc_rows]
    ctc_verified_count = len({item["word_id"] for item in all_word_times if item.get("verified",False)})
    av_resolved = sum(any(value.get("modality_status", {}).get(modality, {}).get("status") == "provenance_linked" for modality in "AV") for value in evidence.values())
    coverage = {
        "token_denominator": len(raw_ids),
        "token_rows": len(token_rows),
        "token_mapped": int(token_mapped),
        "text_selection_denominator": len(text_selection_ids),
        "text_selection_with_text": sum(bool(evidence[key].get("text_fragments")) for key in text_selection_ids),
        "ctc_word_denominator": len(ctc_rows),
        "ctc_candidates_for_selected_text": int(ctc_candidate_count),
        "ctc_word_verified": int(ctc_verified_count),
        "av_selection_denominator": len(av_selection_ids),
        "av_selection_provenance_linked": int(av_resolved),
    }
    if not validation["exact"]:
        status = "unresolved"
    elif ctc_verified_count == len(ctc_rows) and all(value.get("verified") or value.get("status") == "not_applicable" for value in evidence.values()):
        status = "complete"
    elif any(value.get("status") == "resolved" for value in evidence.values()):
        status = "partial"
    else:
        status = "unresolved"
    coverage["status"] = status
    return {
        "sample_id": sample_id,
        "raw_text": raw_text,
        "token_rows": token_rows,
        "tokenizer_validation": validation,
        "ctc": {"path": str(ctc_path) if ctc_path is not None else None, "word_times": all_word_times, "errors": ctc_errors},
        "review": {"path": str(review_path) if review_path is not None else None, "errors": review_errors},
        "av_provenance": {"path": str(provenance_path) if provenance_path is not None else None, "errors": provenance_errors},
        "evidence": evidence,
        "coverage": coverage,
        "status": status,
    }
