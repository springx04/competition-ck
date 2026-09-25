from __future__ import annotations

import csv
import json
import math
import random
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from .storage import read_json, read_jsonl, write_csv, write_json


TRANSCRIPT_STATUSES = {"match", "mismatch", "unverifiable"}
VISUAL_STATUSES = {
    "active_face_identified", "no_visible_target", "multiple_ambiguous", "unverifiable",
}
REVIEW_FIELDS = [
    "review_order", "sample_index", "sample_id", "video_id", "video_path",
    "audio_path", "raw_text", "words_path", "vision_rows_path", "frames_dir",
    "duration_s", "reviewer_id", "transcript_status", "transcript_start",
    "transcript_end", "visual_status", "visual_start", "visual_end", "face_id",
    "episode_id", "confidence", "evidence_note", "review_seconds", "submitted_at_utc",
]
ADJUDICATION_FIELDS = [
    "sample_index", "sample_id", "video_id", "needs_adjudication", "reasons",
    "reviewer_a_transcript", "reviewer_b_transcript", "reviewer_a_visual",
    "reviewer_b_visual", "final_transcript_status", "transcript_start",
    "transcript_end", "final_visual_status", "visual_start", "visual_end",
    "face_id", "episode_id", "adjudicator", "evidence_note", "review_seconds",
    "submitted_at_utc",
]
ALIGNMENT_FIELDS = [
    "sample_id", "issue_type", "start", "end", "review_status", "reviewer",
    "evidence", "action",
]
FACE_FIELDS = [
    "sample_id", "start", "end", "face_id", "episode_id", "reviewer", "evidence_note",
]


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _manifest(run_dir: Path) -> list[dict[str, Any]]:
    return read_jsonl(run_dir / "manifest.jsonl")


def _run_data_root(run_dir: Path) -> Path:
    import yaml

    with (run_dir / "config.yaml").open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    data_root = cfg.get("paths", {}).get("data_root")
    if not data_root:
        raise ValueError("run config does not record paths.data_root")
    return Path(data_root)


def _guard_new_directory(path: Path, force: bool) -> None:
    if path.exists() and any(path.iterdir()) and not force:
        raise FileExistsError(f"review workspace is not empty: {path}; pass --force to replace templates")
    path.mkdir(parents=True, exist_ok=True)


def prepare_workspace(
    run_dir: str | Path, output_dir: str | Path, *, seed: int = 2026, force: bool = False,
) -> dict[str, Any]:
    run = Path(run_dir).resolve()
    output = Path(output_dir).resolve()
    _guard_new_directory(output, force)
    data_root = _run_data_root(run)
    manifest = _manifest(run)
    if len(manifest) != 100:
        raise ValueError(f"expected 100 manifest rows, found {len(manifest)}")
    if len({row["sample_id"] for row in manifest}) != len(manifest):
        raise ValueError("manifest sample_id values are not unique")

    packets: list[dict[str, Any]] = []
    zero_face_ids: list[str] = []
    for item in manifest:
        index = int(item["sample_index"])
        sample_dir = run / "samples" / f"{index:06d}"
        media = read_json(sample_dir / "media.json", {})
        vision_path = sample_dir / "vision_diagnostic.json"
        vision = read_json(vision_path, {})
        if vision_path.is_file() and int(vision.get("successful_candidate_count", 0) or 0) == 0:
            zero_face_ids.append(str(item["sample_id"]))
        packets.append({
            "sample_index": index,
            "sample_id": item["sample_id"],
            "video_id": item["video_id"],
            "video_path": str((data_root / item["video_relpath"]).resolve()),
            "audio_path": str((sample_dir / "audio_16k.wav").resolve()),
            "raw_text": item["raw_text"],
            "words_path": str((sample_dir / "words.jsonl").resolve()),
            "vision_rows_path": str((sample_dir / "vision_rows.jsonl").resolve()),
            "frames_dir": str((sample_dir / "frames").resolve()),
            "duration_s": media.get("duration", ""),
        })

    for reviewer_id, reviewer_seed in (("reviewer_a", seed), ("reviewer_b", seed + 1)):
        ordered = list(packets)
        random.Random(reviewer_seed).shuffle(ordered)
        rows = []
        for order, item in enumerate(ordered, start=1):
            rows.append({
                **item, "review_order": order, "reviewer_id": reviewer_id,
                "transcript_status": "", "transcript_start": "", "transcript_end": "",
                "visual_status": "", "visual_start": "", "visual_end": "",
                "face_id": "", "episode_id": "", "confidence": "",
                "evidence_note": "", "review_seconds": "", "submitted_at_utc": "",
            })
        write_csv(output / f"{reviewer_id}.csv", rows, REVIEW_FIELDS)

    write_csv(output / "packet_index.csv", packets, REVIEW_FIELDS[1:12])
    write_json(output / "experiment_manifest.json", {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run), "data_root": str(data_root), "seed": seed,
        "sample_count": len(manifest), "video_count": len({r["video_id"] for r in manifest}),
        "reviewer_a_order_seed": seed, "reviewer_b_order_seed": seed + 1,
        "zero_face_sample_ids": sorted(zero_face_ids),
        "blind_fields_excluded": ["label", "paired_use", "pairing_status", "automatic_issues", "auxiliary_scores"],
    })
    baseline = output / "baseline"
    baseline.mkdir(exist_ok=True)
    for source in (
        run / "reports" / "quality.csv",
        run / "reports" / "alignment_issues.csv",
        Path("configs/alignment_review.csv").resolve(),
        Path("configs/target_face_segments.csv").resolve(),
    ):
        if source.is_file():
            shutil.copy2(source, baseline / source.name)
    return {"samples": len(manifest), "zero_face": len(zero_face_ids), "output_dir": str(output)}


def validate_review_rows(
    rows: Sequence[dict[str, str]], *, expected_ids: set[str] | None = None,
    require_complete: bool = True, require_provenance: bool = True,
) -> list[str]:
    errors: list[str] = []
    ids = [row.get("sample_id", "").strip() for row in rows]
    duplicates = sorted(sample_id for sample_id, count in Counter(ids).items() if sample_id and count > 1)
    if duplicates:
        errors.append(f"duplicate sample_id rows: {duplicates}")
    if expected_ids is not None:
        missing = sorted(expected_ids - set(ids))
        extra = sorted(set(ids) - expected_ids)
        if missing: errors.append(f"missing sample IDs: {missing}")
        if extra: errors.append(f"unexpected sample IDs: {extra}")
    for line, row in enumerate(rows, start=2):
        sample_id = row.get("sample_id", "").strip() or f"line {line}"
        transcript = row.get("transcript_status", "").strip()
        visual = row.get("visual_status", "").strip()
        if require_complete and transcript not in TRANSCRIPT_STATUSES:
            errors.append(f"{sample_id}: invalid transcript_status={transcript!r}")
        elif transcript and transcript not in TRANSCRIPT_STATUSES:
            errors.append(f"{sample_id}: invalid transcript_status={transcript!r}")
        if require_complete and visual not in VISUAL_STATUSES:
            errors.append(f"{sample_id}: invalid visual_status={visual!r}")
        elif visual and visual not in VISUAL_STATUSES:
            errors.append(f"{sample_id}: invalid visual_status={visual!r}")
        confidence = row.get("confidence", "").strip()
        if require_complete and confidence not in {"1", "2", "3"}:
            errors.append(f"{sample_id}: confidence must be 1, 2, or 3")
        if require_complete and require_provenance and not row.get("reviewer_id", "").strip():
            errors.append(f"{sample_id}: reviewer_id is required")
        if require_complete and not row.get("evidence_note", "").strip():
            errors.append(f"{sample_id}: evidence_note is required")
        review_seconds = row.get("review_seconds", "").strip()
        if require_complete and require_provenance:
            try:
                if not math.isfinite(float(review_seconds)) or float(review_seconds) <= 0:
                    raise ValueError
            except ValueError:
                errors.append(f"{sample_id}: review_seconds must be a positive finite number")
            submitted = row.get("submitted_at_utc", "").strip()
            try:
                datetime.fromisoformat(submitted.replace("Z", "+00:00"))
            except ValueError:
                errors.append(f"{sample_id}: submitted_at_utc must be ISO-8601")
        for prefix in ("transcript", "visual"):
            start, end = row.get(f"{prefix}_start", "").strip(), row.get(f"{prefix}_end", "").strip()
            if bool(start) != bool(end):
                errors.append(f"{sample_id}: {prefix}_start/end must both be set or blank")
            if start and end:
                try:
                    if not (math.isfinite(float(start)) and math.isfinite(float(end)) and float(end) > float(start)):
                        raise ValueError
                except ValueError:
                    errors.append(f"{sample_id}: invalid {prefix} interval {start!r}-{end!r}")
        if transcript in {"mismatch", "unverifiable"} and not row.get("transcript_start", "").strip():
            errors.append(f"{sample_id}: {transcript} requires transcript interval")
        if visual and not row.get("visual_start", "").strip():
            errors.append(f"{sample_id}: {visual} requires visual interval")
        if visual == "active_face_identified":
            if not row.get("visual_start", "").strip():
                errors.append(f"{sample_id}: identified face requires visual interval")
            if not row.get("face_id", "").strip() or not row.get("episode_id", "").strip():
                errors.append(f"{sample_id}: identified face requires face_id and episode_id")
    return errors


def cohen_kappa(a: Sequence[str], b: Sequence[str]) -> float:
    if len(a) != len(b) or not a:
        raise ValueError("kappa requires two non-empty equally sized sequences")
    observed = sum(x == y for x, y in zip(a, b)) / len(a)
    ca, cb = Counter(a), Counter(b)
    labels = set(ca) | set(cb)
    expected = sum((ca[label] / len(a)) * (cb[label] / len(b)) for label in labels)
    return 1.0 if expected == 1.0 and observed == 1.0 else (observed - expected) / (1.0 - expected)


def krippendorff_alpha_nominal(*ratings: Sequence[str]) -> float:
    if len(ratings) < 2 or not ratings[0] or any(len(row) != len(ratings[0]) for row in ratings):
        raise ValueError("alpha requires at least two equally sized non-empty rating sequences")
    disagreements = comparisons = 0
    for values in zip(*ratings):
        for i in range(len(values)):
            for j in range(i + 1, len(values)):
                comparisons += 1
                disagreements += values[i] != values[j]
    observed = disagreements / comparisons
    counts = Counter(value for rating in ratings for value in rating)
    total = sum(counts.values())
    expected = sum(count * (total - count) for count in counts.values()) / (total * (total - 1))
    return 1.0 if expected == 0.0 and observed == 0.0 else 1.0 - observed / expected



def _percentile(values: Sequence[float], q: float) -> float:
    ordered = sorted(value for value in values if math.isfinite(value))
    if not ordered:
        return float("nan")
    position = (len(ordered) - 1) * q
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def grouped_bootstrap_agreement(
    a_rows: Sequence[dict[str, str]], b_rows: Sequence[dict[str, str]], field: str,
    *, iterations: int = 10_000, seed: int = 2026,
) -> dict[str, Any]:
    a = {row["sample_id"]: row for row in a_rows}
    b = {row["sample_id"]: row for row in b_rows}
    groups: dict[str, list[str]] = {}
    for sample_id, row in a.items():
        groups.setdefault(row["video_id"], []).append(sample_id)
    group_ids = sorted(groups)
    rng = random.Random(seed)
    kappas: list[float] = []
    alphas: list[float] = []
    for _ in range(iterations):
        sampled = [rng.choice(group_ids) for _ in group_ids]
        ids = [sample_id for group in sampled for sample_id in groups[group]]
        left, right = [a[x][field] for x in ids], [b[x][field] for x in ids]
        kappas.append(cohen_kappa(left, right))
        alphas.append(krippendorff_alpha_nominal(left, right))
    return {
        "iterations": iterations, "seed": seed, "group_count": len(group_ids),
        "cohen_kappa_95ci": [_percentile(kappas, 0.025), _percentile(kappas, 0.975)],
        "krippendorff_alpha_95ci": [_percentile(alphas, 0.025), _percentile(alphas, 0.975)],
    }

def analyze_agreement(
    reviewer_a: str | Path, reviewer_b: str | Path, run_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    run = Path(run_dir).resolve()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    expected = {str(row["sample_id"]) for row in _manifest(run)}
    a_rows, b_rows = _read_csv(reviewer_a), _read_csv(reviewer_b)
    errors = validate_review_rows(a_rows, expected_ids=expected) + validate_review_rows(b_rows, expected_ids=expected)
    if errors:
        raise ValueError("review validation failed:\n" + "\n".join(errors))
    a = {row["sample_id"]: row for row in a_rows}
    b = {row["sample_id"]: row for row in b_rows}
    ordered_ids = sorted(expected)
    zero_face = set(read_json(Path(reviewer_a).resolve().parent / "experiment_manifest.json", {}).get("zero_face_sample_ids", []))
    metrics: dict[str, Any] = {}
    for field in ("transcript_status", "visual_status"):
        av, bv = [a[x][field] for x in ordered_ids], [b[x][field] for x in ordered_ids]
        metrics[field] = {
            "n": len(av), "raw_agreement": sum(x == y for x, y in zip(av, bv)) / len(av),
            "cohen_kappa": cohen_kappa(av, bv),
            "krippendorff_alpha_nominal": krippendorff_alpha_nominal(av, bv),
            "grouped_bootstrap": grouped_bootstrap_agreement(a_rows, b_rows, field),
        }
    adjudication: list[dict[str, Any]] = []
    for sample_id in ordered_ids:
        left, right = a[sample_id], b[sample_id]
        reasons = []
        if left["transcript_status"] != right["transcript_status"]: reasons.append("transcript_disagreement")
        if left["visual_status"] != right["visual_status"]: reasons.append("visual_disagreement")
        if left["transcript_status"] == right["transcript_status"] == "mismatch" and (
            left["transcript_start"] != right["transcript_start"]
            or left["transcript_end"] != right["transcript_end"]
        ):
            reasons.append("transcript_interval_disagreement")
        if left["visual_status"] == right["visual_status"] == "active_face_identified" and any(
            left[field] != right[field]
            for field in ("visual_start", "visual_end", "face_id", "episode_id")
        ):
            reasons.append("visual_detail_disagreement")
        if "1" in {left["confidence"], right["confidence"]}: reasons.append("low_confidence")
        if "unverifiable" in {left["transcript_status"], right["transcript_status"], left["visual_status"], right["visual_status"]}:
            reasons.append("unverifiable")
        if sample_id in zero_face: reasons.append("automatic_zero_face")
        if not reasons:
            continue
        adjudication.append({
            "sample_index": left["sample_index"], "sample_id": sample_id,
            "video_id": left["video_id"], "needs_adjudication": "true",
            "reasons": ";".join(reasons),
            "reviewer_a_transcript": left["transcript_status"],
            "reviewer_b_transcript": right["transcript_status"],
            "reviewer_a_visual": left["visual_status"], "reviewer_b_visual": right["visual_status"],
            "final_transcript_status": "", "transcript_start": "", "transcript_end": "",
            "final_visual_status": "", "visual_start": "", "visual_end": "",
            "face_id": "", "episode_id": "", "adjudicator": "", "evidence_note": "",
            "review_seconds": "", "submitted_at_utc": "",
        })
    write_json(output / "agreement.json", {"metrics": metrics, "threshold": 0.80, "passes": all(v["cohen_kappa"] >= 0.80 and v["krippendorff_alpha_nominal"] >= 0.80 for v in metrics.values()), "adjudication_count": len(adjudication)})
    write_csv(output / "adjudication_required.csv", adjudication, ADJUDICATION_FIELDS)
    return {"metrics": metrics, "adjudication_count": len(adjudication)}


def _decision_to_review(status: str) -> tuple[str, str]:
    if status in {"match", "active_face_identified"}:
        return "confirmed_match", "release_pairing"
    if status in {"mismatch", "no_visible_target"}:
        return "confirmed_mismatch", "quarantine_pairing"
    return "unresolved", "keep_unresolved"


def export_proposed_configs(
    final_decisions_csv: str | Path, baseline_alignment_csv: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    decisions = _read_csv(final_decisions_csv)
    required = {str(row["sample_id"]) for row in decisions}
    normalized = []
    for row in decisions:
        normalized.append({
            "sample_id": row.get("sample_id", ""),
            "transcript_status": row.get("final_transcript_status", ""),
            "transcript_start": row.get("transcript_start", ""),
            "transcript_end": row.get("transcript_end", ""),
            "visual_status": row.get("final_visual_status", ""),
            "visual_start": row.get("visual_start", ""), "visual_end": row.get("visual_end", ""),
            "face_id": row.get("face_id", ""), "episode_id": row.get("episode_id", ""),
            "confidence": "3", "evidence_note": row.get("evidence_note", ""),
        })
    errors = validate_review_rows(normalized, expected_ids=required, require_provenance=False)
    for row in decisions:
        if not row.get("adjudicator", "").strip(): errors.append(f"{row.get('sample_id')}: adjudicator is required")
    if errors:
        raise ValueError("adjudication validation failed:\n" + "\n".join(errors))
    by_id = {row["sample_id"]: row for row in decisions}
    baseline = _read_csv(baseline_alignment_csv)
    alignment: list[dict[str, Any]] = []
    covered_types: set[tuple[str, str]] = set()
    for row in baseline:
        decision = by_id.get(row["sample_id"])
        if decision is None:
            alignment.append(row)
            continue
        issue_type = row["issue_type"]
        visual_issue = issue_type in {"identity_unknown", "vision_no_face", "human_identity_unverifiable"}
        final = decision["final_visual_status"] if visual_issue else decision["final_transcript_status"]
        review_status, action = _decision_to_review(final)
        alignment.append({
            **row, "review_status": review_status, "action": action,
            "reviewer": decision["adjudicator"],
            "evidence": f"double-review adjudication: {decision['evidence_note']}",
        })
        covered_types.add((row["sample_id"], issue_type))
    for sample_id, decision in sorted(by_id.items()):
        transcript = decision["final_transcript_status"]
        visual = decision["final_visual_status"]
        if transcript != "match" and not any(sid == sample_id and "text" in issue.lower() for sid, issue in covered_types):
            status, action = _decision_to_review(transcript)
            alignment.append({
                "sample_id": sample_id, "issue_type": "human_text_audio_mismatch" if transcript == "mismatch" else "human_text_audio_unverifiable",
                "start": decision["transcript_start"], "end": decision["transcript_end"],
                "review_status": status, "reviewer": decision["adjudicator"],
                "evidence": f"double-review adjudication: {decision['evidence_note']}", "action": action,
            })
        if visual == "no_visible_target" and (sample_id, "vision_no_face") not in covered_types:
            alignment.append({
                "sample_id": sample_id, "issue_type": "vision_no_face",
                "start": decision["visual_start"], "end": decision["visual_end"],
                "review_status": "confirmed_mismatch", "reviewer": decision["adjudicator"],
                "evidence": f"natural visual missing after double review: {decision['evidence_note']}",
                "action": "quarantine_pairing",
            })
        elif visual in {"multiple_ambiguous", "unverifiable"} and (sample_id, "identity_unknown") not in covered_types:
            alignment.append({
                "sample_id": sample_id, "issue_type": "human_identity_unverifiable",
                "start": decision["visual_start"], "end": decision["visual_end"],
                "review_status": "unresolved", "reviewer": decision["adjudicator"],
                "evidence": f"double-review adjudication: {decision['evidence_note']}",
                "action": "keep_unresolved",
            })
    faces = [{
        "sample_id": sample_id, "start": row["visual_start"], "end": row["visual_end"],
        "face_id": row["face_id"], "episode_id": row["episode_id"],
        "reviewer": row["adjudicator"], "evidence_note": row["evidence_note"],
    } for sample_id, row in sorted(by_id.items()) if row["final_visual_status"] == "active_face_identified"]
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "alignment_review.proposed.csv", alignment, ALIGNMENT_FIELDS)
    write_csv(output / "target_face_segments.proposed.csv", faces, FACE_FIELDS)
    return {"alignment_rows": len(alignment), "face_segments": len(faces), "output_dir": str(output.resolve())}



def finalize_decisions(
    reviewer_a: str | Path, reviewer_b: str | Path, adjudication_csv: str | Path,
    run_dir: str | Path, output_path: str | Path,
) -> dict[str, Any]:
    expected = {str(row["sample_id"]) for row in _manifest(Path(run_dir))}
    a_rows, b_rows = _read_csv(reviewer_a), _read_csv(reviewer_b)
    errors = validate_review_rows(a_rows, expected_ids=expected) + validate_review_rows(b_rows, expected_ids=expected)
    if errors:
        raise ValueError("review validation failed:\n" + "\n".join(errors))
    a = {row["sample_id"]: row for row in a_rows}
    b = {row["sample_id"]: row for row in b_rows}
    adjudication = {row["sample_id"]: row for row in _read_csv(adjudication_csv)}
    zero_face = {
        str(row["sample_id"])
        for row in _manifest(Path(run_dir))
        if (diagnostic_path := Path(run_dir) / "samples" / f"{int(row['sample_index']):06d}" / "vision_diagnostic.json").is_file()
        and int(read_json(diagnostic_path, {}).get("successful_candidate_count", 0) or 0) == 0
    }
    final_rows: list[dict[str, Any]] = []
    for sample_id in sorted(expected):
        left, right = a[sample_id], b[sample_id]
        reasons = []
        if left["transcript_status"] != right["transcript_status"]: reasons.append("transcript_disagreement")
        if left["visual_status"] != right["visual_status"]: reasons.append("visual_disagreement")
        if left["transcript_status"] == right["transcript_status"] == "mismatch" and (
            left["transcript_start"] != right["transcript_start"] or left["transcript_end"] != right["transcript_end"]
        ): reasons.append("transcript_interval_disagreement")
        if left["visual_status"] == right["visual_status"] == "active_face_identified" and any(
            left[field] != right[field] for field in ("visual_start", "visual_end", "face_id", "episode_id")
        ): reasons.append("visual_detail_disagreement")
        if "1" in {left["confidence"], right["confidence"]}: reasons.append("low_confidence")
        if "unverifiable" in {left["transcript_status"], right["transcript_status"], left["visual_status"], right["visual_status"]}:
            reasons.append("unverifiable")
        if sample_id in zero_face:
            reasons.append("automatic_zero_face")
        source = adjudication.get(sample_id) if reasons else None
        if reasons and source is None:
            errors.append(f"{sample_id}: adjudication required for {reasons}")
            continue
        if source is not None:
            if not source.get("adjudicator", "").strip() or not source.get("evidence_note", "").strip():
                errors.append(f"{sample_id}: adjudicator and evidence_note are required")
                continue
            transcript_status = source.get("final_transcript_status", "")
            visual_status = source.get("final_visual_status", "")
            row = {
                "sample_index": left["sample_index"], "sample_id": sample_id, "video_id": left["video_id"],
                "final_transcript_status": transcript_status,
                "transcript_start": source.get("transcript_start", ""), "transcript_end": source.get("transcript_end", ""),
                "final_visual_status": visual_status,
                "visual_start": source.get("visual_start", ""), "visual_end": source.get("visual_end", ""),
                "face_id": source.get("face_id", ""), "episode_id": source.get("episode_id", ""),
                "adjudicator": source.get("adjudicator", ""),
                "evidence_note": source.get("evidence_note", ""),
                "decision_source": "adjudication",
            }
        else:
            row = {
                "sample_index": left["sample_index"], "sample_id": sample_id, "video_id": left["video_id"],
                "final_transcript_status": left["transcript_status"],
                "transcript_start": left["transcript_start"], "transcript_end": left["transcript_end"],
                "final_visual_status": left["visual_status"],
                "visual_start": left["visual_start"], "visual_end": left["visual_end"],
                "face_id": left["face_id"], "episode_id": left["episode_id"],
                "adjudicator": "independent_agreement",
                "evidence_note": f"reviewer_a: {left['evidence_note']} | reviewer_b: {right['evidence_note']}",
                "decision_source": "independent_agreement",
            }
        check = {
            "sample_id": sample_id, "transcript_status": row["final_transcript_status"],
            "transcript_start": row["transcript_start"], "transcript_end": row["transcript_end"],
            "visual_status": row["final_visual_status"], "visual_start": row["visual_start"],
            "visual_end": row["visual_end"], "face_id": row["face_id"], "episode_id": row["episode_id"],
            "confidence": "3", "evidence_note": row["evidence_note"],
        }
        errors.extend(validate_review_rows([check], expected_ids={sample_id}, require_provenance=False))
        final_rows.append(row)
    if errors:
        raise ValueError("finalization failed:\n" + "\n".join(errors))
    fields = [
        "sample_index", "sample_id", "video_id", "final_transcript_status", "transcript_start",
        "transcript_end", "final_visual_status", "visual_start", "visual_end", "face_id",
        "episode_id", "adjudicator", "evidence_note", "decision_source",
    ]
    write_csv(output_path, final_rows, fields)
    return {"final_rows": len(final_rows), "adjudicated_rows": sum(row["decision_source"] == "adjudication" for row in final_rows)}
