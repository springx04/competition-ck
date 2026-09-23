from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable


def word_error_counts(reference: str, hypothesis: str) -> tuple[int, int, int, float]:
    ref = reference.split()
    hyp = hypothesis.split()
    # DP cells store (cost, S, D, I); deterministic tie order prefers
    # substitution, deletion, insertion.
    dp: list[list[tuple[int, int, int, int]]] = [[(0, 0, 0, 0) for _ in range(len(hyp) + 1)] for _ in range(len(ref) + 1)]
    for i in range(1, len(ref) + 1):
        dp[i][0] = (i, 0, i, 0)
    for j in range(1, len(hyp) + 1):
        dp[0][j] = (j, 0, 0, j)
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
                continue
            sub = dp[i - 1][j - 1]
            delete = dp[i - 1][j]
            insert = dp[i][j - 1]
            choices = [
                (sub[0] + 1, sub[1] + 1, sub[2], sub[3]),
                (delete[0] + 1, delete[1], delete[2] + 1, delete[3]),
                (insert[0] + 1, insert[1], insert[2], insert[3] + 1),
            ]
            dp[i][j] = min(choices, key=lambda x: (x[0], -x[1], -x[2], -x[3]))
    _, substitutions, deletions, insertions = dp[-1][-1]
    wer = (substitutions + deletions + insertions) / max(1, len(ref))
    return substitutions, deletions, insertions, wer


def load_reviews(path: str | Path, sample_id: str | None = None) -> list[dict[str, str]]:
    target = Path(path)
    if not target.is_file():
        return []
    with target.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return rows if sample_id is None else [r for r in rows if r.get("sample_id") == sample_id]


def audit_pairing(
    *, diagnostic_wer: float | None, aligned_word_fraction: float | None,
    automatic_issues: Iterable[dict[str, Any]], reviews: Iterable[dict[str, str]],
    wer_warn: float = 0.65, aligned_fraction_warn: float = 0.80,
) -> dict[str, Any]:
    issues = list(automatic_issues)
    triggered = bool(issues)
    if diagnostic_wer is not None and diagnostic_wer > wer_warn:
        triggered = True
        issues.append({"issue_type": "suspected_text_audio_mismatch", "evidence": f"diagnostic_wer={diagnostic_wer:.6f}"})
    if aligned_word_fraction is not None and aligned_word_fraction < aligned_fraction_warn:
        triggered = True
        issues.append({"issue_type": "low_aligned_word_fraction", "evidence": f"aligned_word_fraction={aligned_word_fraction:.6f}"})
    review_rows = list(reviews)
    mismatch = any(r.get("review_status") == "confirmed_mismatch" or r.get("action") == "quarantine_pairing" for r in review_rows)
    release = bool(review_rows) and all(r.get("review_status") == "confirmed_match" and r.get("action") == "release_pairing" for r in review_rows)
    unresolved = any(r.get("review_status") == "unresolved" or r.get("action") == "keep_unresolved" for r in review_rows)
    if mismatch:
        status, paired = "confirmed_mismatch", False
    elif release:
        status, paired = "reviewed_match", True
    elif unresolved:
        status, paired = "unverifiable", False
    elif triggered:
        status, paired = "suspected", False
    else:
        status, paired = "no_issue_detected", True
    return {"pairing_status": status, "paired_use": paired, "issues": issues, "review_scope": review_rows}

