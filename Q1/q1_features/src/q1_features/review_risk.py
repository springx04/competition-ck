from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Sequence

import yaml

from .storage import read_json, read_jsonl, write_csv, write_json


RISK_FIELDS = [
    "sample_index", "sample_id", "video_id", "diagnostic_wer",
    "aligned_word_fraction", "whisperx_wer", "wer_percentile",
    "unaligned_percentile", "whisperx_wer_percentile", "risk_score",
    "risk_components", "existing_ctc_alert", "auxiliary_complete",
]


def percentile_ranks(values: Sequence[float]) -> list[float]:
    """Return [0, 1] average-tie percentile ranks.

    A singleton receives rank 0.5. Non-finite values are rejected so missing
    auxiliary scores cannot silently enter the ranking.
    """
    if not values:
        return []
    if any(not math.isfinite(value) for value in values):
        raise ValueError("percentile ranks require finite values")
    if len(values) == 1:
        return [0.5]
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    position = 0
    while position < len(order):
        end = position + 1
        while end < len(order) and values[order[end]] == values[order[position]]:
            end += 1
        average_position = (position + end - 1) / 2.0
        percentile = average_position / (len(values) - 1)
        for ordered_index in order[position:end]:
            ranks[ordered_index] = percentile
        position = end
    return ranks


def _read_optional_whisperx(path: str | Path | None) -> dict[str, float]:
    if path is None:
        return {}
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    result: dict[str, float] = {}
    for row in rows:
        sample_id = row.get("sample_id", "").strip()
        if not sample_id or sample_id in result:
            raise ValueError(f"invalid or duplicate WhisperX sample_id: {sample_id!r}")
        value = float(row.get("whisperx_wer", ""))
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"invalid whisperx_wer for {sample_id}: {value}")
        result[sample_id] = value
    return result


def build_audio_risk_ranking(
    run_dir: str | Path,
    output_csv: str | Path,
    *,
    whisperx_csv: str | Path | None = None,
) -> dict[str, Any]:
    run = Path(run_dir).resolve()
    manifest = read_jsonl(run / "manifest.jsonl")
    with (run / "config.yaml").open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    wer_warn = float(config["quality"]["diagnostic_wer_warn"])
    aligned_warn = float(config["quality"]["aligned_word_fraction_warn"])
    whisperx = _read_optional_whisperx(whisperx_csv)
    expected_ids = {str(row["sample_id"]) for row in manifest}
    extra = sorted(set(whisperx) - expected_ids)
    if extra:
        raise ValueError(f"WhisperX rows contain unknown sample IDs: {extra}")

    rows: list[dict[str, Any]] = []
    for item in manifest:
        index = int(item["sample_index"])
        sample_dir = run / "samples" / f"{index:06d}"
        diagnostic = read_json(sample_dir / "diagnostic.json", {})
        words = read_jsonl(sample_dir / "words.jsonl")
        spoken = [word for word in words if word.get("ctc_text")]
        accepted = [word for word in spoken if word.get("accepted_interval")]
        fraction = len(accepted) / max(1, len(spoken)) if spoken else 1.0
        wer = float(diagnostic["diagnostic_wer"])
        sample_id = str(item["sample_id"])
        rows.append({
            "sample_index": index,
            "sample_id": sample_id,
            "video_id": str(item["video_id"]),
            "diagnostic_wer": wer,
            "aligned_word_fraction": fraction,
            "whisperx_wer": whisperx.get(sample_id),
            "existing_ctc_alert": wer > wer_warn or fraction < aligned_warn,
        })

    wer_ranks = percentile_ranks([row["diagnostic_wer"] for row in rows])
    unaligned_ranks = percentile_ranks([1.0 - row["aligned_word_fraction"] for row in rows])
    whisper_ranks: dict[str, float] = {}
    if whisperx:
        ids = [row["sample_id"] for row in rows if row["sample_id"] in whisperx]
        ranked = percentile_ranks([whisperx[sample_id] for sample_id in ids])
        whisper_ranks = dict(zip(ids, ranked))

    for index, row in enumerate(rows):
        components = [wer_ranks[index], unaligned_ranks[index]]
        whisper_rank = whisper_ranks.get(row["sample_id"])
        if whisper_rank is not None:
            components.append(whisper_rank)
        row.update({
            "wer_percentile": wer_ranks[index],
            "unaligned_percentile": unaligned_ranks[index],
            "whisperx_wer_percentile": whisper_rank if whisper_rank is not None else "",
            "risk_score": sum(components) / len(components),
            "risk_components": len(components),
            "auxiliary_complete": len(components) == 3,
        })
    rows.sort(key=lambda row: (-float(row["risk_score"]), int(row["sample_index"])))
    write_csv(output_csv, rows, RISK_FIELDS)
    metadata = {
        "sample_count": len(rows),
        "ctc_alert_count": sum(bool(row["existing_ctc_alert"]) for row in rows),
        "whisperx_row_count": len(whisperx),
        "complete_three_component_count": sum(bool(row["auxiliary_complete"]) for row in rows),
        "formula": "mean(percentile(diagnostic_wer), percentile(1-aligned_word_fraction), optional percentile(whisperx_wer))",
        "warning": "This is review prioritization only and must not create confirmed decisions.",
    }
    write_json(Path(output_csv).with_suffix(".metadata.json"), metadata)
    return metadata

