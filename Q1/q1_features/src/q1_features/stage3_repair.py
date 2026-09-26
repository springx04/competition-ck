from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .alignment_repair import av_sync_evidence, best_assignment


MAPPING_STATUSES = {"NOT_SUSPECTED", "MAPPING_STRONG", "MAPPING_WEAK", "MAPPING_UNRESOLVED"}
STAGE3_VISUAL_STATUSES = {
    "VERIFIED_ACTIVE", "VERIFIED_CONTINUITY", "AMBIGUOUS", "MISSING", "UNRESOLVED"
}
RESOLUTION_STATUSES = {
    "RESOLVED_VALID", "RESOLVED_PARTIAL", "RESOLVED_CONFLICT", "RESOLVED_MISSING", "UNRESOLVED"
}


def choose_mapping_candidate(
    sample_id: str,
    assignments: Mapping[str, str],
    score_pairs: Mapping[str, tuple[float, float]],
) -> str:
    """Choose the consensus off-diagonal candidate without modifying any mapping."""
    votes = Counter(value for value in assignments.values() if value != sample_id)
    if not votes:
        return sample_id
    top_count = max(votes.values())
    tied = sorted(candidate for candidate, count in votes.items() if count == top_count)
    if len(tied) == 1:
        return tied[0]
    # Tie break is deterministic and evidence-based: summed advantage for metrics
    # whose Hungarian assignment proposed that candidate.
    def support(candidate: str) -> float:
        return sum(
            score_pairs[name][1] - score_pairs[name][0]
            for name, assigned in assignments.items() if assigned == candidate
        )
    return max(tied, key=lambda candidate: (support(candidate), candidate))


def classify_mapping_evidence(
    *,
    sample_id: str,
    candidate_id: str,
    assignments: Mapping[str, str],
    score_pairs: Mapping[str, tuple[float, float]],
    global_true_rank: int,
    global_candidate_rank: int,
    strong_vote_min: int = 3,
    strong_advantage_min: float = 0.10,
    strong_token_score_min: float = 0.35,
    weak_vote_min: int = 2,
    weak_advantage_min: float = 0.05,
    weak_absolute_score_min: float = 0.35,
) -> tuple[str, str]:
    """Combine independent ASR/metric assignments into a frozen mapping status."""
    if candidate_id == sample_id:
        return "MAPPING_UNRESOLVED", "no_stable_off_diagonal_candidate"
    votes = sum(value == candidate_id for value in assignments.values())
    advantages = {name: pair[1] - pair[0] for name, pair in score_pairs.items()}
    strong_metrics = sum(value >= strong_advantage_min for value in advantages.values())
    weak_metrics = sum(value >= weak_advantage_min for value in advantages.values())
    whisper_token_support = assignments.get("whisper_token") == candidate_id
    ctc_token_support = assignments.get("ctc_token") == candidate_id
    both_token_scores = (
        score_pairs["whisper_token"][1] >= strong_token_score_min
        and score_pairs["ctc_token"][1] >= strong_token_score_min
    )
    global_support = global_candidate_rank < global_true_rank
    if (
        votes >= strong_vote_min
        and whisper_token_support
        and ctc_token_support
        and strong_metrics >= strong_vote_min
        and both_token_scores
        and global_support
    ):
        return "MAPPING_STRONG", "independent_ASRs_and_metrics_support_same_off_diagonal_mapping"
    maximum_candidate_score = max(pair[1] for pair in score_pairs.values())
    if votes >= weak_vote_min and weak_metrics >= weak_vote_min and maximum_candidate_score >= weak_absolute_score_min:
        reasons = []
        if not (whisper_token_support and ctc_token_support):
            reasons.append("ASR_token_assignments_disagree")
        if strong_metrics < strong_vote_min:
            reasons.append("off_diagonal_advantage_not_consistently_strong")
        if not global_support:
            reasons.append("global_retrieval_does_not_prefer_candidate")
        return "MAPPING_WEAK", ";".join(reasons) or "partial_off_diagonal_support"
    return "MAPPING_UNRESOLVED", "assignments_unstable_or_absolute_similarity_too_low"


def mapping_proposal_row(
    sample_id: str, candidate_id: str, mapping_status: str, reason: str
) -> dict[str, Any]:
    if mapping_status not in MAPPING_STATUSES:
        raise ValueError(f"invalid mapping status: {mapping_status}")
    return {
        "sample_id": sample_id,
        "proposed_audio_sample_id": candidate_id if mapping_status == "MAPPING_STRONG" else "",
        "mapping_status": mapping_status,
        "proposal_only": 1,
        "manifest_modified": 0,
        "reason": reason,
    }


def _parse_json_mapping(value: Any) -> dict[int, float | None]:
    if isinstance(value, Mapping):
        raw = value
    else:
        raw = json.loads(str(value) or "{}")
    output = {}
    for key, item in raw.items():
        output[int(key)] = None if item is None else float(item)
    return output


def _visibility_components(
    source_rows: Sequence[Mapping[str, Any]], max_identity_gap: float
) -> tuple[dict[int, list[dict[str, Any]]], dict[tuple[int, int], int]]:
    by_cluster: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in source_rows:
        cluster = row.get("cluster_id")
        confidence = float(row.get("confidence", 0.0) or 0.0)
        if cluster in (None, "") or confidence < 0.80:
            continue
        by_cluster[int(cluster)].append(row)
    components: dict[int, list[dict[str, Any]]] = defaultdict(list)
    row_component: dict[tuple[int, int], int] = {}
    for cluster, rows in by_cluster.items():
        ordered = sorted(rows, key=lambda row: (float(row["start"]), float(row["end"]), int(row["source_id"])))
        current = None
        for row in ordered:
            start, end = float(row["start"]), float(row["end"])
            if current is None or start - float(current["end"]) > max_identity_gap:
                current = {"component_id": len(components[cluster]), "start": start, "end": end, "source_ids": []}
                components[cluster].append(current)
            else:
                current["end"] = max(float(current["end"]), end)
            current["source_ids"].append(int(row["source_id"]))
            row_component[(cluster, int(row["source_id"]))] = int(current["component_id"])
    return dict(components), row_component


def _component_for_window(
    cluster: int, start: float, end: float, components: Mapping[int, Sequence[Mapping[str, Any]]]
) -> int | None:
    matches = []
    for component in components.get(cluster, []):
        overlap = min(end, float(component["end"])) - max(start, float(component["start"]))
        if overlap > 0:
            matches.append((overlap, int(component["component_id"])))
    return max(matches)[1] if matches else None


def build_stage3_visual_timeline(
    *,
    sample_id: str,
    duration: float,
    stage2_timeline: Sequence[Mapping[str, Any]],
    source_rows: Sequence[Mapping[str, Any]],
    partition_stable: bool,
    av_percentiles: Mapping[int, float],
    max_identity_gap: float = 1.0,
    stride_s: float = 0.25,
    logit_threshold: float = 0.0,
    anchor_min_positive_windows: int = 2,
    anchor_neighbor_gap_s: float = 0.50,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Use TalkNet anchors and bounded ArcFace continuity for local visual masks."""
    if duration < 0 or max_identity_gap < 0 or stride_s <= 0:
        raise ValueError("invalid Stage3 visual parameters")
    rows = sorted(stage2_timeline, key=lambda row: float(row["start"]))
    components, _ = _visibility_components(source_rows, max_identity_gap)
    prepared = []
    for index, row in enumerate(rows):
        start = float(row["start"])
        analysis_end = float(row["end"])
        cell_end = min(duration, start + stride_s)
        visible = []
        component_by_cluster = {}
        for cluster in sorted(components):
            component_id = _component_for_window(cluster, start, analysis_end, components)
            if component_id is not None:
                visible.append(cluster)
                component_by_cluster[cluster] = component_id
        logits = _parse_json_mapping(row.get("cluster_logits", "{}"))
        active = [
            cluster for cluster in visible
            if logits.get(cluster) is not None and float(logits[cluster]) >= logit_threshold
        ]
        prepared.append({
            "index": index, "start": start, "analysis_end": analysis_end, "end": cell_end,
            "visible": visible, "components": component_by_cluster, "logits": logits, "active": active,
        })

    positive_by_component: dict[tuple[int, int], list[int]] = defaultdict(list)
    for row in prepared:
        if len(row["active"]) == 1:
            cluster = row["active"][0]
            component = row["components"].get(cluster)
            if component is not None:
                positive_by_component[(cluster, component)].append(int(row["index"]))
    anchors: dict[tuple[int, int], list[int]] = defaultdict(list)
    for key, indices in positive_by_component.items():
        run = []
        for index in indices:
            if not run or prepared[index]["start"] - prepared[run[-1]]["start"] <= anchor_neighbor_gap_s + 1e-9:
                run.append(index)
            else:
                if len(run) >= anchor_min_positive_windows:
                    anchors[key].extend(run)
                run = [index]
        if len(run) >= anchor_min_positive_windows:
            anchors[key].extend(run)

    def clear_path(cluster: int, anchor_index: int, target_index: int) -> bool:
        left, right = sorted((anchor_index, target_index))
        for row in prepared[left + 1:right + 1]:
            active = row["active"]
            if len(active) > 1 or (len(active) == 1 and active[0] != cluster):
                return False
        return True

    anchor_stats = []
    clusters = sorted(set(components) | {cluster for row in prepared for cluster in row["logits"]})
    for cluster in clusters:
        positive_rows = [row for row in prepared if row["logits"].get(cluster) is not None and row["logits"][cluster] >= logit_threshold]
        visible_rows = [row for row in prepared if cluster in row["visible"]]
        values = [float(row["logits"][cluster]) for row in positive_rows]
        anchored_components = sorted(component for (key_cluster, component) in anchors if key_cluster == cluster)
        anchor_stats.append({
            "sample_id": sample_id, "cluster_id": cluster,
            "positive_window_count": len(positive_rows),
            "positive_duration": len(positive_rows) * stride_s,
            "median_positive_logit": float(np.median(values)) if values else "",
            "max_logit": max(values) if values else "",
            "positive_window_fraction": len(positive_rows) / max(1, len(visible_rows)),
            "anchor_component_count": len(anchored_components),
            "anchored_components": json.dumps(anchored_components, separators=(",", ":")),
        })

    output = []
    for row in prepared:
        visible, active = row["visible"], row["active"]
        anchor_identity: int | str = ""
        propagated_identity: int | str = ""
        propagation_source_time: float | str = ""
        continuity = 0
        quality_flag = ""
        if not visible:
            status = "MISSING"
        elif len(active) > 1:
            status = "AMBIGUOUS"
        elif len(active) == 1:
            status = "VERIFIED_ACTIVE"
            cluster = active[0]
            component = row["components"].get(cluster)
            if component is not None and (cluster, component) in anchors:
                anchor_identity = cluster
        elif not partition_stable:
            status = "UNRESOLVED"
            quality_flag = "arcface_partition_unstable"
        else:
            candidates = []
            for cluster in visible:
                component = row["components"].get(cluster)
                if component is None:
                    continue
                reachable = [
                    index for index in anchors.get((cluster, component), [])
                    if clear_path(cluster, index, int(row["index"]))
                ]
                if reachable:
                    nearest = min(reachable, key=lambda index: abs(prepared[index]["start"] - row["start"]))
                    candidates.append((cluster, nearest))
            if len(candidates) == 1:
                cluster, anchor_index = candidates[0]
                status = "VERIFIED_CONTINUITY"
                anchor_identity = cluster
                propagated_identity = cluster
                propagation_source_time = float(prepared[anchor_index]["start"])
                continuity = 1
            elif len(candidates) > 1 or len(visible) > 1:
                status = "AMBIGUOUS"
            else:
                status = "UNRESOLVED"
        selected = active[0] if status == "VERIFIED_ACTIVE" else (propagated_identity if status == "VERIFIED_CONTINUITY" else None)
        av_flag = av_sync_evidence(av_percentiles.get(int(selected)) if selected not in (None, "") else None)
        if selected not in (None, "") and av_flag == "conflict":
            quality_flag = ";".join(filter(None, [quality_flag, "talknet_or_continuity_av_sync_conflict"]))
        output.append({
            "sample_id": sample_id, "start": row["start"], "end": row["end"],
            "analysis_window_end": row["analysis_end"],
            "visible_identity_clusters": json.dumps(visible, separators=(",", ":")),
            "talknet_active_clusters": json.dumps(active, separators=(",", ":")),
            "anchor_identity": anchor_identity,
            "propagated_identity": propagated_identity,
            "propagation_source_time": propagation_source_time,
            "arcface_continuity": continuity,
            "av_sync_flag": av_flag,
            "segment_status": status,
            "selected_identity": "" if selected is None else selected,
            "quality_flag": quality_flag,
        })
    return output, anchor_stats


def summarize_stage3_visual(
    timeline: Sequence[Mapping[str, Any]], stage2_usable_fraction: float
) -> dict[str, Any]:
    durations = Counter()
    for row in timeline:
        duration = max(0.0, float(row["end"]) - float(row["start"]))
        durations[str(row["segment_status"])] += duration
    total = sum(durations.values()) or 1.0
    direct = durations["VERIFIED_ACTIVE"] / total
    continuity = durations["VERIFIED_CONTINUITY"] / total
    ambiguous = durations["AMBIGUOUS"] / total
    missing = durations["MISSING"] / total
    unresolved = durations["UNRESOLVED"] / total
    usable = direct + continuity
    if usable >= 0.80 and ambiguous <= 0.10 and unresolved <= 0.10:
        status = "FULLY_VERIFIED"
    elif usable > 0:
        status = "PARTIALLY_VERIFIED"
    elif ambiguous > 0:
        status = "AMBIGUOUS"
    elif missing >= 1.0 - 1e-9:
        status = "MISSING"
    else:
        status = "UNRESOLVED"
    return {
        "stage2_usable_fraction": float(stage2_usable_fraction),
        "stage3_visual_status": status,
        "direct_talknet_verified_fraction": direct,
        "continuity_recovered_fraction": continuity,
        "ambiguous_fraction": ambiguous,
        "missing_fraction": missing,
        "unresolved_fraction": unresolved,
        "total_usable_visual_fraction": usable,
    }


def classify_resolution_status(
    *,
    semantic_status: str,
    temporal_status: str,
    visual_status: str,
    mapping_status: str = "NOT_SUSPECTED",
) -> tuple[str, str]:
    if mapping_status == "MAPPING_STRONG":
        return "RESOLVED_CONFLICT", "stable_off_diagonal_mapping_evidence_has_explicit_safe_mask_action"
    # Confirmed physical absence has a complete, deterministic treatment (missing/mask),
    # even when another modality also carries a diagnostic warning. The independent
    # semantic and mapping axes remain in the exported table and are never erased.
    if visual_status == "MISSING":
        return "RESOLVED_MISSING", "visual_modality_is_confirmed_naturally_missing"
    if mapping_status in {"MAPPING_WEAK", "MAPPING_UNRESOLVED"}:
        return "UNRESOLVED", "clip_mapping_evidence_is_not_stable_enough"
    if semantic_status == "UNRESOLVED":
        return "UNRESOLVED", "semantic_pairing_remains_unresolved"
    if visual_status == "UNRESOLVED":
        return "UNRESOLVED", "target_visual_identity_remains_unresolved"
    if semantic_status == "CONFLICT":
        return "RESOLVED_CONFLICT", "semantic_or_mapping_conflict_has_explicit_safe_mask_action"
    if temporal_status == "FAILED":
        return "UNRESOLVED", "official_text_time_alignment_failed"
    if temporal_status == "VERIFIED" and visual_status == "FULLY_VERIFIED":
        return "RESOLVED_VALID", "semantic_time_and_visual_evidence_are_sufficient"
    return "RESOLVED_PARTIAL", "usable_evidence_is_partial_and_unreliable_regions_are_explicitly_masked"


def assert_status_sets(mapping_status: str, visual_status: str, resolution_status: str) -> None:
    if mapping_status not in MAPPING_STATUSES:
        raise ValueError(mapping_status)
    if visual_status not in {"FULLY_VERIFIED", "PARTIALLY_VERIFIED", "AMBIGUOUS", "MISSING", "UNRESOLVED"}:
        raise ValueError(visual_status)
    if resolution_status not in RESOLUTION_STATUSES:
        raise ValueError(resolution_status)



def assert_semantic_audio_mask(semantic_statuses: Sequence[str], audio_mask: np.ndarray) -> None:
    statuses = np.asarray(semantic_statuses)
    masks = np.asarray(audio_mask)
    if len(statuses) != len(masks):
        raise ValueError("semantic/audio length mismatch")
    unsafe = np.isin(statuses, ["CONFLICT", "UNRESOLVED"])
    if np.any(masks[unsafe] != 0):
        raise ValueError("semantic conflict or unresolved item has an audio mask")


def assert_candidate_path_safe(candidate: Path, protected: Sequence[Path]) -> None:
    resolved = candidate.resolve()
    if any(resolved == path.resolve() for path in protected):
        raise ValueError(f"candidate path would overwrite protected input: {resolved}")


def verify_expected_sha256(path: Path, expected_sha256: str) -> bool:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest() == expected_sha256


def assert_source_traceability(
    sample_indices: Sequence[int], source_ids: Sequence[int], source_rows: Sequence[Mapping[str, Any]]
) -> None:
    known = {(int(row["sample_index"]), int(row["source_id"])) for row in source_rows}
    missing = [
        (int(sample_index), int(source_id))
        for sample_index, source_id in zip(sample_indices, source_ids, strict=True)
        if (int(sample_index), int(source_id)) not in known
    ]
    if missing:
        raise ValueError(f"untraceable source references: {missing[:3]}")
