from __future__ import annotations

import csv
import os
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from .storage import load_npz, read_csv, read_json, read_jsonl, write_csv


def package_sizes(root: str | Path) -> list[dict[str, Any]]:
    root_path = Path(root)
    rows: list[dict[str, Any]] = []
    for path in sorted(p for p in root_path.rglob("*") if p.is_file()):
        rel = path.relative_to(root_path).as_posix()
        if rel.startswith(("models/", "third_party/", "env/python/", "env/openface/")):
            category = "external_runtime"
        elif "/frames/" in rel or rel.endswith("audio_16k.wav") or "/openface/" in rel:
            category = "server_working_material"
        else:
            category = "q1_compact_candidate"
        rows.append({"category": category, "path": rel, "bytes": path.stat().st_size})
    return rows


def write_timeline(sample_dir: Path, destination: Path) -> None:
    status = read_json(sample_dir / "status.json", {})
    media = read_json(sample_dir / "media.json", {})
    words = read_jsonl(sample_dir / "words.jsonl")
    duration = float(media.get("duration", 0.0) or 0.0)
    lines = [
        f"# Timeline: {status.get('sample_id', sample_dir.name)}", "",
        f"D={duration:.6f}s; pairing={status.get('pairing_status', 'unknown')}; paired_use={status.get('paired_use', False)}", "",
        "| word_id | original | accepted interval | candidate | reasons |",
        "|---:|---|---|---|---|",
    ]
    for word in words:
        accepted = word.get("accepted_interval")
        candidate = word.get("candidate_interval")
        lines.append(f"| {word.get('word_id')} | {word.get('raw_word','')} | {accepted} | {candidate} | {', '.join(word.get('alignment_reasons', []))} |")
    lines.extend(["", "## 50 windows", "", "| k | start | end | T/A/V observed | coverage | reasons |", "|---:|---:|---:|---|---|---|"])
    for row in read_jsonl(sample_dir / "bin_quality.jsonl"):
        modalities = [row.get(name, {}) for name in ("text", "audio", "vision")]
        obs = "/".join(str(int(bool(x.get("observed")))) for x in modalities)
        cov = "/".join(f"{float(x.get('coverage',0)):.3f}" for x in modalities)
        reasons = "; ".join(f"{name}:{','.join(row.get(name,{}).get('reasons',[]))}" for name in ("text", "audio", "vision") if row.get(name,{}).get("reasons"))
        lines.append(f"| {row.get('bin_index')} | {row.get('start',0):.6f} | {row.get('end',0):.6f} | {obs} | {cov} | {reasons} |")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")


def report_run(run_dir: str | Path) -> Path:
    run = Path(run_dir)
    manifest = read_jsonl(run / "manifest.jsonl")
    statuses = [read_json(run / "samples" / f"{int(row['sample_index']):06d}" / "status.json", {}) for row in manifest]
    counts = Counter(s.get("processing_status", "not_processed") for s in statuses)
    pair_counts = Counter(s.get("pairing_status", "unknown") for s in statuses)
    rss_values = []
    cuda_values = []
    for status in statuses:
        for stage in status.get("stages", {}).values():
            resources = stage.get("resources", {})
            if resources.get("process_peak_rss_kb") is not None: rss_values.append(int(resources["process_peak_rss_kb"]))
            if resources.get("cuda_peak_allocated_bytes") is not None: cuda_values.append(int(resources["cuda_peak_allocated_bytes"]))
    for row in manifest:
        sample_dir = run / "samples" / f"{int(row['sample_index']):06d}"
        if sample_dir.exists():
            write_timeline(sample_dir, run / "reports" / "timelines" / f"{int(row['sample_index']):06d}.md")
    size_rows = package_sizes(run.parent.parent if run.parent.name == "runs" else run)
    size_rows.append({"category": "q2_q3_reserved", "path": "(not implemented in Q1; budget reservation unresolved)", "bytes": 0})
    write_csv(run / "reports" / "package_size.csv", size_rows, ["category", "path", "bytes"])
    compact_bytes = sum(int(r["bytes"]) for r in size_rows if r["category"] == "q1_compact_candidate")
    feature_path = run / "features" / "q1_compact50.npz"
    modality_lines: list[str] = []
    max_bin_width: float | str = "unavailable"
    if feature_path.is_file():
        arrays = load_npz(feature_path)
        observed = arrays["observed_mask"]
        valid = arrays["valid_mask"]
        denominator = max(1, int(valid.sum()))
        for index, name in enumerate(("text", "audio", "vision")):
            modality_lines.append(f"- {name} observed windows / valid windows: {int(observed[:,:,index].sum())}/{denominator} ({observed[:,:,index].sum()/denominator:.4%})")
        widths = arrays["time_intervals"][:, :, 1] - arrays["time_intervals"][:, :, 0]
        max_bin_width = float(widths.max(initial=0.0))
    issue_rows = read_csv(run / "reports" / "alignment_issues.csv") if (run / "reports" / "alignment_issues.csv").is_file() else []
    issue_ids = sorted({row["sample_id"] for row in issue_rows})

    hand_example = "No accepted cross-window word was available."
    for manifest_row in manifest:
        sample_dir = run / "samples" / f"{int(manifest_row['sample_index']):06d}"
        map_path, native_path, compact_path = sample_dir / "map_text.npz", sample_dir / "native.npz", sample_dir / "compact50.npz"
        if not (map_path.is_file() and native_path.is_file() and compact_path.is_file()):
            continue
        mapping, native, compact = load_npz(map_path), load_npz(native_path), load_npz(compact_path)
        occurrences: dict[int, list[int]] = {}
        for k in range(50):
            for pos in range(int(mapping["indptr"][k]), int(mapping["indptr"][k+1])):
                occurrences.setdefault(int(mapping["source_ids"][pos]), []).append(k)
        candidate = next(((sid, bins) for sid, bins in occurrences.items() if len(set(bins)) > 1), None)
        if candidate is None:
            continue
        sid, bins = candidate
        k = bins[0]
        left, right = int(mapping["indptr"][k]), int(mapping["indptr"][k+1])
        ids = mapping["source_ids"][left:right].astype(int)
        overlaps = mapping["overlap_s"][left:right]
        weights = mapping["weights"][left:right]
        value0 = float(sum(float(weight) * float(native["word_features"][source, 0]) for source, weight in zip(ids, weights)))
        hand_example = (
            f"sample={manifest_row['sample_id']}, cross-window word_id={sid}, bins={sorted(set(bins))}; "
            f"for bin {k}: source_ids={ids.tolist()}, overlap_s={overlaps.tolist()}, "
            f"weights=overlap/sum={weights.tolist()}, first_dimension={value0:.8g}, "
            f"stored={float(compact['text'][k,0]):.8g}."
        )
        break
    summary = [
        "# Q1 feature extraction report", "",
        f"- Manifest rows: {len(manifest)}",
        f"- Processing: {dict(counts)}",
        f"- Pairing: {dict(pair_counts)}",
        f"- Alignment issue sample IDs: {issue_ids}",
        f"- Compact-candidate bytes: {compact_bytes}",
        f"- Maximum 50-bin width: {max_bin_width} seconds",
        f"- Maximum recorded process peak RSS: {max(rss_values) if rss_values else 'unavailable'} kB",
        f"- Maximum recorded PyTorch CUDA allocation: {max(cuda_values) if cuda_values else 'unavailable'} bytes",
        "- Submission weight policy: unresolved",
        "- Human audiovisual review: only rows explicitly present in alignment_review.csv are treated as reviewed.",
        "- Pretrained BERT/Conformer weights and server working material are excluded from the compact-candidate count; this is not a claim that the complete competition submission is below 50,000,000 bytes.",
        "",
        "## Modality availability",
        "",
        *modality_lines,
        "",
        "## Cross-window source-weight example",
        "",
        hand_example,
    ]
    output = run / "reports" / "summary.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(summary) + "\n", encoding="utf-8")
    return output
