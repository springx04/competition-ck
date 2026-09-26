"""Build the final Q2 result tables and figures from stored evidence.

The final evaluator writes the raw ``evaluate_model`` rows to
``results/final/{valid,test}/metrics.json``.  Those rows deliberately keep
classification and regression metrics nested, so this script reads that
format directly and does not re-evaluate a checkpoint or infer missing
numbers.  Missing artifacts are shown as ``—`` in the Markdown report and
are recorded in the report notes.

Run from the Q2 project root (or pass ``--root``)::

    python scripts/build_final_tables.py

Only the report and the PNG/SVG figures are written.  The source JSON, CSV
and ledger files are never modified.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
MISSING = "—"
MAIN_PATTERNS = ("T", "A", "V", "TA", "TV", "AV")
MAIN_POSITIONS = ("front", "middle", "rear")
MAIN_RHOS = ("0.2", "0.4", "0.6", "0.8")
CLASS_NAMES = ("Negative", "Neutral", "Positive")
METRICS = ("accuracy", "macro_f1", "weighted_f1", "mae", "pearson")
def _is_mapping(value: Any) -> bool:
    return isinstance(value, Mapping)


def _read_json(path: Path, notes: list[str]) -> Any:
    """Read one JSON artifact, returning ``None`` for absent/bad evidence."""

    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        notes.append(f"无法读取 `{path.as_posix()}`：{exc}")
        return None


def _read_csv(path: Path, notes: list[str]) -> list[dict[str, str]]:
    if not path.exists():
        return []
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    except (OSError, UnicodeError, csv.Error) as exc:
        notes.append(f"无法读取 `{path.as_posix()}`：{exc}")
        return []


def _metrics_rows(payload: Any, path: Path, notes: list[str]) -> list[dict[str, Any]]:
    """Normalize the final metrics JSON to its raw row-list contract."""

    if payload is None:
        return []
    rows = payload
    if _is_mapping(payload):
        for key in ("rows", "metrics", "metric_rows", "results"):
            if isinstance(payload.get(key), list):
                rows = payload[key]
                break
    if not isinstance(rows, list):
        notes.append(f"`{path.as_posix()}` 不是 evaluate_model 行列表，已跳过。")
        return []
    result = [row for row in rows if isinstance(row, dict)]
    if len(result) != len(rows):
        notes.append(f"`{path.as_posix()}` 含有非对象行，已跳过这些行。")
    return result


def _number(value: Any) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return float(value)
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _integer(value: Any) -> int | None:
    number = _number(value)
    return int(number) if number is not None else None


def _fmt(value: Any, digits: int = 4, signed: bool = False) -> str:
    number = _number(value)
    if number is None:
        return MISSING
    return f"{number:+.{digits}f}" if signed else f"{number:.{digits}f}"


def _fmt_count(value: Any, digits: int = 1) -> str:
    number = _number(value)
    if number is None:
        return MISSING
    if float(number).is_integer():
        return str(int(number))
    return f"{number:.{digits}f}"


def _mean(values: Iterable[Any]) -> float | None:
    numbers = [number for value in values if (number := _number(value)) is not None]
    return statistics.fmean(numbers) if numbers else None


def _fmt_mean_sd(values: Iterable[Any], digits: int = 4) -> str:
    numbers = [number for value in values if (number := _number(value)) is not None]
    if not numbers:
        return MISSING
    mean = statistics.fmean(numbers)
    sd = statistics.stdev(numbers) if len(numbers) >= 2 else None
    return f"{mean:.{digits}f} ± {sd:.{digits}f}" if sd is not None else f"{mean:.{digits}f} ± {MISSING}"


def _cell(value: Any) -> str:
    """Escape a value for a Markdown table without changing its meaning."""

    return str(value).replace("|", r"\|").replace("\n", " ")


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> list[str]:
    result = ["| " + " | ".join(_cell(header) for header in headers) + " |",
              "|" + "|".join("---" for _ in headers) + "|"]
    result.extend("| " + " | ".join(_cell(value) for value in row) + " |" for row in rows)
    return result


def _metric_payload(row: Mapping[str, Any]) -> Mapping[str, Any]:
    payload = row.get("metrics")
    return payload if _is_mapping(payload) else row


def _value(row: Mapping[str, Any], name: str) -> Any:
    payload = _metric_payload(row)
    aliases = {
        "pearson": ("pearson", "pcc"),
        "accuracy": ("accuracy", "acc"),
        "macro_f1": ("macro_f1", "macro-F1", "macrof1"),
        "weighted_f1": ("weighted_f1", "weighted-F1", "weightedf1"),
        "mae": ("mae", "MAE"),
    }
    for key in aliases.get(name, (name,)):
        if key in payload:
            return payload[key]
        if key in row and payload is not row:
            return row[key]
    return None


def _degradation_value(row: Mapping[str, Any], name: str) -> Any:
    degradation = row.get("degradation")
    if _is_mapping(degradation) and name in degradation:
        return degradation[name]
    return row.get(name)


def _class_payload(row: Mapping[str, Any], class_name: str) -> Mapping[str, Any]:
    payload = _metric_payload(row)
    per_class = payload.get("per_class") if _is_mapping(payload) else None
    if not _is_mapping(per_class):
        per_class = row.get("per_class") if _is_mapping(row.get("per_class")) else {}
    candidates = (class_name, class_name.lower(), str(CLASS_NAMES.index(class_name)))
    for key in candidates:
        value = per_class.get(key)
        if _is_mapping(value):
            return value
    return {}


def _class_value(row: Mapping[str, Any], class_name: str, metric: str) -> Any:
    payload = _class_payload(row, class_name)
    if metric in payload:
        return payload[metric]
    # A few older audit writers used flat class_0_f1 / negative_f1 fields.
    index = CLASS_NAMES.index(class_name)
    aliases = (
        f"{class_name.lower()}_{metric}",
        f"class_{index}_{metric}",
        f"{metric}_{class_name.lower()}",
        f"{metric}_class_{index}",
    )
    for key in aliases:
        if key in row:
            return row[key]
        if key in _metric_payload(row):
            return _metric_payload(row)[key]
    return None


def _scope_rows(rows: Sequence[Mapping[str, Any]], scope: str = "all_samples") -> list[dict[str, Any]]:
    return [dict(row) for row in rows if row.get("scope", scope) == scope]


def _scenario_name(row: Mapping[str, Any]) -> str:
    return str(row.get("scenario", ""))


def _scenario_parts(name: str) -> tuple[str, str, str] | None:
    parts = name.split("_")
    if len(parts) != 3 or parts[0] not in MAIN_PATTERNS or parts[1] not in MAIN_POSITIONS:
        return None
    try:
        rho = f"{float(parts[2]):.1f}"
    except ValueError:
        return None
    if rho not in MAIN_RHOS:
        return None
    return parts[0], parts[1], rho


def _is_main_scenario(name: str) -> bool:
    return _scenario_parts(name) is not None


def _sort_scenario(name: str) -> tuple[int, int, float, str]:
    parts = _scenario_parts(name)
    if parts is None:
        return (999, 999, 999.0, name)
    pattern, position, rho = parts
    return (MAIN_PATTERNS.index(pattern), MAIN_POSITIONS.index(position), float(rho), name)


def _clean_and_main(rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    scoped = _scope_rows(rows)
    clean = next((row for row in scoped if _scenario_name(row) == "clean"), None)
    main = [row for row in scoped if _is_main_scenario(_scenario_name(row))]
    main.sort(key=lambda row: _sort_scenario(_scenario_name(row)))
    return clean, main


def _deletion_map(deletion_rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    return {str(row.get("scenario", "")): row for row in deletion_rows if row.get("scenario")}


def _summary_rows(rows: Sequence[Mapping[str, Any]]) -> list[list[str]]:
    clean, main = _clean_and_main(rows)
    result: list[list[str]] = []
    for label, group, count in (("clean", [clean] if clean else [], 1),
                                ("72-scenario mean", main, len(main))):
        if not group:
            result.append([label, MISSING, str(count) if label != "clean" else "1", MISSING] + [MISSING] * 5)
            continue
        values = [_mean(_value(row, metric) for row in group) for metric in METRICS]
        n = _mean(_integer(row.get("n")) for row in group)
        result.append([label, _scenario_name(group[0]) if label == "clean" else "T/A/V/TA/TV/AV × 3 × 4",
                       str(count), _fmt_count(n)] + [_fmt(value) for value in values])
    return result


def _per_class_rows(rows: Sequence[Mapping[str, Any]]) -> list[list[str]]:
    clean, main = _clean_and_main(rows)
    result: list[list[str]] = []
    for condition, group in (("clean", [clean] if clean else []), ("72-scenario mean", main)):
        for class_name in CLASS_NAMES:
            result.append([condition, class_name,
                           _fmt(_mean(_class_value(row, class_name, "precision") for row in group)),
                           _fmt(_mean(_class_value(row, class_name, "recall") for row in group)),
                           _fmt(_mean(_class_value(row, class_name, "f1") for row in group)),
                           _fmt_count(_mean(_class_value(row, class_name, "support") for row in group))])
    return result


def _grouped_rows(rows: Sequence[Mapping[str, Any]], deletion: Sequence[Mapping[str, Any]]) -> list[list[str]]:
    _, main = _clean_and_main(rows)
    deletion_by_name = _deletion_map(deletion)
    groups: list[tuple[str, str, list[dict[str, Any]]]] = []
    for pattern in MAIN_PATTERNS:
        groups.append(("modality", pattern, [row for row in main if _scenario_parts(_scenario_name(row))[0] == pattern]))
    for position in MAIN_POSITIONS:
        groups.append(("position", position, [row for row in main if _scenario_parts(_scenario_name(row))[1] == position]))
    for rho in MAIN_RHOS:
        groups.append(("rho", rho, [row for row in main if _scenario_parts(_scenario_name(row))[2] == rho]))
    result: list[list[str]] = []
    for factor, label, group in groups:
        actual_rates = [deletion_by_name.get(_scenario_name(row), {}).get("actual_rate_mean") for row in group]
        result.append([factor, label, str(len(group)), _fmt_count(_mean(row.get("n") for row in group)),
                       *[_fmt(_mean(_value(row, metric) for row in group)) for metric in METRICS],
                       _fmt(_mean(actual_rates))])
    return result


def _scenario_rows(rows: Sequence[Mapping[str, Any]], deletion: Sequence[Mapping[str, Any]],
                   names: set[str] | None = None) -> list[list[str]]:
    scoped = _scope_rows(rows)
    selected = [row for row in scoped if row.get("scenario") != "clean"
                and (names is None or str(row.get("scenario")) in names)]
    selected.sort(key=lambda row: (_sort_scenario(_scenario_name(row)), _scenario_name(row)))
    deletion_by_name = _deletion_map(deletion)
    result = []
    for row in selected:
        name = _scenario_name(row)
        rate = deletion_by_name.get(name, {})
        result.append([name, _fmt_count(row.get("n")), _fmt(_value(row, "accuracy")),
                       _fmt(_value(row, "macro_f1")), _fmt(_value(row, "weighted_f1")),
                       _fmt(_value(row, "mae")), _fmt(_value(row, "pearson")),
                       _fmt(_degradation_value(row, "delta_macro_f1"), signed=True),
                       _fmt(_degradation_value(row, "delta_mae"), signed=True),
                       _fmt_count(rate.get("damaged_n")), _fmt(rate.get("actual_rate_mean"))])
    return result


def _paired_summary(rows: Sequence[Mapping[str, Any]]) -> list[list[str]]:
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row.get("factor", "")), str(row.get("condition", "")))].append(row)
    result: list[list[str]] = []
    for (factor, condition), group in sorted(groups.items()):
        valid = [row for row in group if _number(row.get("macro_f1")) is not None]
        result.append([factor, condition, str(len(valid)), _fmt_count(_mean(row.get("n") for row in valid)),
                       _fmt(_mean(row.get("macro_f1") for row in valid)),
                       _fmt(_mean(row.get("mae") for row in valid)),
                       _fmt(_mean(row.get("mean_actual_rate") for row in valid))])
    return result


def _matched_summary(rows: Sequence[Mapping[str, Any]]) -> list[list[str]]:
    groups: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row.get("comparison", "")), str(row.get("rate_bin", "")),
                str(row.get("pattern", "")))].append(row)
    result = []
    for (comparison, rate_bin, pattern), group in sorted(groups.items()):
        valid = [row for row in group if _number(row.get("macro_f1")) is not None]
        result.append([comparison, rate_bin, pattern, str(len(valid)),
                       _fmt_count(_mean(row.get("n") for row in valid)),
                       _fmt(_mean(row.get("macro_f1") for row in valid)),
                       _fmt(_mean(row.get("mae") for row in valid))])
    return result


def _dual_summary(rows: Sequence[Mapping[str, Any]]) -> list[list[str]]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        name = str(row.get("scenario", ""))
        groups[name.split("_", 1)[0]].append(row)
    result = []
    for pattern in ("TA", "TV", "AV"):
        group = groups.get(pattern, [])
        valid = [row for row in group if _number(row.get("macro_f1")) is not None]
        result.append([pattern, str(len(valid)), _fmt_count(_mean(row.get("both_modalities_damaged_n") for row in valid)),
                       _fmt(_mean(row.get("macro_f1") for row in valid)),
                       _fmt(_mean(row.get("mae") for row in valid))])
    return result


def _payload_metric_row(payload: Mapping[str, Any]) -> list[str]:
    return [_fmt(_value(payload, metric)) for metric in METRICS]


def _special_path(root: Path) -> Path | None:
    for path in (root / "results/final/q2_predictions_aligned.csv",
                 root / "outputs/q2_predictions_aligned.csv",
                 root / "results/final/special30.csv"):
        if path.exists():
            return path
    return None


def _special_rows(rows: Sequence[Mapping[str, Any]]) -> list[list[str]]:
    aliases = {
        "sample_id": ("sample_id", "id", "name"),
        "pred_label": ("pred_label", "label", "predicted_label"),
        "pred_score": ("pred_score", "score", "predicted_score"),
        "prob_negative": ("prob_negative", "p_negative", "prob_0"),
        "prob_neutral": ("prob_neutral", "p_neutral", "prob_1"),
        "prob_positive": ("prob_positive", "p_positive", "prob_2"),
    }
    def get(row: Mapping[str, Any], key: str) -> Any:
        for alias in aliases[key]:
            if alias in row:
                return row[alias]
        return None
    result = []
    for row in rows:
        result.append([get(row, "sample_id") or MISSING, get(row, "pred_label") or MISSING,
                       _fmt(get(row, "pred_score"), 5), _fmt(get(row, "prob_negative")),
                       _fmt(get(row, "prob_neutral")), _fmt(get(row, "prob_positive"))])
    return result


def _lower(value: Any) -> str:
    return str(value or "").lower().replace("\\", "/")


def _trial_text(trial: Mapping[str, Any]) -> str:
    config = trial.get("config") if _is_mapping(trial.get("config")) else {}
    project = config.get("project") if _is_mapping(config.get("project")) else {}
    text = config.get("text") if _is_mapping(config.get("text")) else {}
    train = config.get("train") if _is_mapping(config.get("train")) else {}
    parts = [trial.get("checkpoint"), trial.get("variant"), project.get("output_root"),
             text.get("model_id"), text.get("model_dir"), text.get("pooling"),
             text.get("representation"), train.get("pooling")]
    return " ".join(_lower(part) for part in parts)


def _is_bert8(trial: Mapping[str, Any]) -> bool:
    text = _trial_text(trial)
    config = trial.get("config") if _is_mapping(trial.get("config")) else {}
    text_cfg = config.get("text") if _is_mapping(config.get("text")) else {}
    return "bert8" in text or "l-8_h-256" in text or "l_8_h_256" in text or text_cfg.get("num_layers") == 8


def _trial_kind(trial: Mapping[str, Any]) -> str | None:
    """Assign only the explicitly requested ledger groups."""

    text = _trial_text(trial)
    config = trial.get("config") if _is_mapping(trial.get("config")) else {}
    text_cfg = config.get("text") if _is_mapping(config.get("text")) else {}
    train_cfg = config.get("train") if _is_mapping(config.get("train")) else {}
    checkpoint = _lower(trial.get("checkpoint"))
    if _is_bert8(trial):
        if "no_corruption" in text or train_cfg.get("no_corruption") is True:
            return "BERT8no_corruption"
        if "no_cls" in text or "nocls" in text or text_cfg.get("cls_context") is False:
            return "BERT8no_cls"
        pooling = _lower(text_cfg.get("pooling") or text_cfg.get("representation") or train_cfg.get("pooling"))
        if "mean_pool" in text or "meanpool" in text or pooling in {"mean", "mean_pool", "meanpool"}:
            return "BERT8mean_pool"
        if "bert8_all" in text or "bert8all" in text or "attn_bert8_all" in text:
            return "BERT8all"
        # A plain BERT-8 exploratory trial (for example
        # ``attn_bert8_v1``) is not the final BERT8all group.  Require an
        # explicit all/final marker so it cannot dilute the requested
        # three-seed ledger summary.
        return "BERT8all" if any(marker in text for marker in ("bert8_all", "bert8all", "attn_bert8_all")) else None
    if "runs_baseline_20260924" in checkpoint:
        for variant, label in (("full", "Historical full"),
                               ("late_clean", "Historical late_clean"),
                               ("late_aug", "Historical late_aug")):
            if f"/{variant}/" in checkpoint:
                return label
    return None


def _test_evaluation(trial: Mapping[str, Any]) -> Mapping[str, Any] | None:
    evaluations = trial.get("evaluations")
    if not isinstance(evaluations, list):
        return None
    candidates = [row for row in evaluations if isinstance(row, Mapping) and row.get("split") == "test"]
    if not candidates:
        return None
    # Prefer a final-control/final evaluation when a checkpoint was read more
    # than once.  This is a path preference, never a score-based selection.
    candidates.sort(key=lambda row: ("final" not in _lower(row.get("source")), _lower(row.get("source"))))
    return candidates[0]


def _ledger_metric(record: Mapping[str, Any], split: str, key: str) -> Any:
    payload: Mapping[str, Any] | None
    if split == "valid":
        payload = record.get("valid") if _is_mapping(record.get("valid")) else None
    else:
        payload = _test_evaluation(record)
    if payload is None:
        return None
    aliases = (key, key.replace("_", "-"))
    for alias in aliases:
        if alias in payload:
            return payload[alias]
    return None


LEDGER_METRIC_KEYS = (
    "missing_accuracy", "missing_macro_f1", "missing_weighted_f1", "missing_mae",
    "missing_pearson", "clean_accuracy", "clean_macro_f1", "clean_weighted_f1",
    "clean_mae", "clean_pearson",
)


def _ledger_table(ledger: Any, requested_kinds: Sequence[str]) -> list[list[str]]:
    trials = ledger.get("trials", []) if _is_mapping(ledger) else []
    by_kind: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    seen: set[str] = set()
    for trial in trials:
        if not isinstance(trial, Mapping):
            continue
        kind = _trial_kind(trial)
        checkpoint = str(trial.get("checkpoint", ""))
        if kind in requested_kinds and checkpoint not in seen:
            by_kind[kind].append(trial)
            seen.add(checkpoint)
    result: list[list[str]] = []
    for kind in requested_kinds:
        group = by_kind.get(kind, [])
        for split in ("valid", "test"):
            values = {key: [_ledger_metric(trial, split, key) for trial in group] for key in LEDGER_METRIC_KEYS}
            seeds = sorted({str(trial.get("seed")) for trial in group if trial.get("seed") is not None})
            evidence_count = sum(any(_number(value) is not None for value in values[key]) for key in LEDGER_METRIC_KEYS)
            result.append([kind, split, str(len(group)), ", ".join(seeds) if seeds else MISSING,
                           *[_fmt_mean_sd(values[key]) for key in LEDGER_METRIC_KEYS]])
    return result


def _selection_metadata(root: Path, notes: list[str]) -> Mapping[str, Any] | None:
    for path in (root / "results/final/selection.json", root / "configs/final_selection.json"):
        payload = _read_json(path, notes)
        if _is_mapping(payload):
            return payload
    return None


def _plot(root: Path, data: Mapping[str, Sequence[Mapping[str, Any]]], notes: list[str]) -> list[Path]:
    """Write one English-labelled valid/test × factor figure in both formats."""

    available = {split: rows for split, rows in data.items() if rows}
    if not available:
        return []
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        notes.append(f"未生成图形（matplotlib 不可用：{exc}）。")
        return []

    figure_dir = root / "results/final/figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    factors = (("modality", MAIN_PATTERNS), ("position", MAIN_POSITIONS), ("rho", MAIN_RHOS))
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), squeeze=False, sharey=True)
    for row_index, split in enumerate(("valid", "test")):
        rows = data.get(split, [])
        _, main = _clean_and_main(rows)
        for col_index, (factor, labels) in enumerate(factors):
            axis = axes[row_index][col_index]
            values = []
            for label in labels:
                selected = []
                for metric_row in main:
                    parts = _scenario_parts(_scenario_name(metric_row))
                    if parts is None:
                        continue
                    selected.append(metric_row if ((factor == "modality" and parts[0] == label)
                                                   or (factor == "position" and parts[1] == label)
                                                   or (factor == "rho" and parts[2] == label)) else None)
                selected = [item for item in selected if item is not None]
                values.append(_mean(_value(item, "macro_f1") for item in selected))
            x = list(range(len(labels)))
            if any(value is not None for value in values):
                axis.plot(x, [float(value) if value is not None else math.nan for value in values],
                          marker="o", linewidth=1.8, label="Macro-F1")
            axis.set_xticks(x, labels)
            axis.set_xlabel(factor.capitalize())
            axis.set_title(f"{split.capitalize()} — {factor.capitalize()}")
            axis.grid(alpha=0.3)
            if col_index == 0:
                axis.set_ylabel("Macro-F1")
            if any(value is not None for value in values):
                axis.set_ylim(bottom=max(0.0, min(value for value in values if value is not None) - 0.05),
                              top=min(1.0, max(value for value in values if value is not None) + 0.05))
    fig.suptitle("Q2 performance across local corruption scenarios", fontsize=14)
    fig.tight_layout()
    paths = [figure_dir / "valid_test_modal_position_rho.png", figure_dir / "valid_test_modal_position_rho.svg"]
    fig.savefig(paths[0], dpi=180)
    fig.savefig(paths[1])
    plt.close(fig)
    return paths


def _source_line(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def build_report(root: Path) -> tuple[str, list[Path]]:
    notes: list[str] = []
    root = Path(root)
    final = root / "results/final"
    split_rows: dict[str, list[dict[str, Any]]] = {}
    csv_data: dict[str, dict[str, list[dict[str, str]]]] = {}
    for split in ("valid", "test"):
        metrics_path = final / split / "metrics.json"
        payload = _read_json(metrics_path, notes)
        split_rows[split] = _metrics_rows(payload, metrics_path, notes)
        csv_data[split] = {
            name: _read_csv(final / split / f"{name}.csv", notes)
            for name in ("paired_conditions", "deletion_rates", "matched_deletion_rates", "dual_both_damaged")
        }

    ledger_path = root / "results/experiment_ledger.json"
    ledger = _read_json(ledger_path, notes)
    selection = _selection_metadata(root, notes)
    all_empty_path = final / "test/all_empty.json"
    all_empty = _read_json(all_empty_path, notes)
    special_path = _special_path(root)
    special = _read_csv(special_path, notes) if special_path else []
    if special_path is None:
        notes.append("未找到附件3专项 30 行预测 CSV。")
    elif len(special) != 30:
        notes.append(f"专项 CSV 当前为 {len(special)} 行，预期 30 行；按实际行数展示。")

    lines = [
        "# Q2 最终结果表格",
        "",
        "本文件由 `scripts/build_final_tables.py` 从已保存的最终评估证据生成。脚本不重跑模型、不填入预设数字；缺失文件或未定义指标显示为 `—`。",
        "",
        "## 读取范围与报告边界",
        "",
        "最终模型使用单个 BERT8all、seed=1111、无 class bias；具体成员和选择说明以 `results/final/selection.json` 或 `configs/final_selection.json` 为准。若选择元数据记录了先观察 test 后采用结构，本报告沿用该选择并披露其偏离题目要求的 valid-only 结构选择条款。",
        "",
        "valid/test 的 72 场景均为场景级等权均值，PCC 也是先逐场景计算再取均值；clean 单独列出。8 个压力场景、全空输入和附件3专项不并入 72 场景均值。",
        "",
    ]
    if selection:
        lines += ["选择元数据：", "", "```json", json.dumps(selection, ensure_ascii=False, indent=2), "```", ""]
    else:
        lines += ["选择元数据未提供，当前报告不对最终部署成员作数字推断。", ""]

    for split in ("valid", "test"):
        rows = split_rows[split]
        deletion = csv_data[split]["deletion_rates"]
        lines += [f"## {split} clean 与 72 场景均值", ""]
        lines += _table(["条件", "场景定义", "场景数", "平均 N", "Acc", "Macro-F1", "Weighted-F1", "MAE", "PCC"],
                        _summary_rows(rows))
        lines += ["", "PCC 为未定义时保留 `—`，不把未定义相关系数当作 0。", ""]
        lines += [f"### {split} 每类指标", ""]
        lines += _table(["条件", "类别", "Precision", "Recall", "F1", "Support"], _per_class_rows(rows))
        lines += ["", f"### {split} 模态、位置与 rho 均值", ""]
        lines += _table(["分组", "条件", "场景数", "平均 N", "Acc", "Macro-F1", "Weighted-F1", "MAE", "PCC", "实际删除率"],
                        _grouped_rows(rows, deletion))
        lines += [""]
        paired = csv_data[split]["paired_conditions"]
        if paired:
            lines += [f"### {split} 配对共同样本", "", "仅在各比较条件共同产生新增删除的样本上计算；表内按 factor/condition 汇总。", ""]
            lines += _table(["Factor", "Condition", "有效组数", "平均共同 N", "Macro-F1", "MAE", "实际删除率"],
                            _paired_summary(paired))
            lines += [""]
        else:
            lines += [f"### {split} 配对共同样本", "", "配对 CSV 未提供。", ""]
        matched = csv_data[split]["matched_deletion_rates"]
        if matched:
            lines += [f"### {split} 同实际删除率分层", ""]
            lines += _table(["比较", "实际率箱", "模态", "有效组数", "平均共同 N", "Macro-F1", "MAE"],
                            _matched_summary(matched))
            lines += [""]
        dual = csv_data[split]["dual_both_damaged"]
        if dual:
            lines += [f"### {split} 双模态均实际受损", ""]
            lines += _table(["模态组合", "有效场景数", "平均共同 N", "Macro-F1", "MAE"], _dual_summary(dual))
            lines += [""]
        stress_names = {_scenario_name(row) for row in _scope_rows(rows)
                        if _scenario_name(row) not in {"clean"} and not _is_main_scenario(_scenario_name(row))}
        lines += [f"### {split} 8 个压力场景", ""]
        if stress_names:
            lines += _table(["场景", "N", "Acc", "Macro-F1", "Weighted-F1", "MAE", "PCC", "ΔMacro-F1", "ΔMAE", "受损 N", "实际删除率"],
                            _scenario_rows(rows, deletion, stress_names))
        else:
            lines += ["压力场景数据未提供。"]
        lines += [""]

    # The complete 72-row test table is intentionally separate from the
    # compact factor means so that no scenario is hidden by aggregation.
    lines += ["## test 72 个主场景完整表", ""]
    test_main_names = {_scenario_name(row) for row in _scope_rows(split_rows["test"])
                       if _is_main_scenario(_scenario_name(row))}
    if test_main_names:
        lines += _table(["场景", "N", "Acc", "Macro-F1", "Weighted-F1", "MAE", "PCC", "ΔMacro-F1", "ΔMAE", "受损 N", "实际删除率"],
                        _scenario_rows(split_rows["test"], csv_data["test"]["deletion_rates"], test_main_names))
    else:
        lines += ["test 主场景 metrics.json 未提供。"]
    lines += [""]

    lines += ["## test 全空输入独立结果", ""]
    if _is_mapping(all_empty):
        lines += [f"来源：`{_source_line(all_empty_path, root)}`。这是单独的全 T/A/V 空输入压力检查，不并入 clean 或 72 场景均值。", ""]
        lines += _table(["N", "Acc", "Macro-F1", "Weighted-F1", "MAE", "PCC"],
                        [[_fmt_count(_value(all_empty, "n")), *_payload_metric_row(all_empty)]])
        lines += ["", "### 全空输入每类指标", ""]
        lines += _table([["类别", "Precision", "Recall", "F1", "Support"][i] for i in range(5)],
                        [[class_name, _fmt(_class_value(all_empty, class_name, "precision")),
                          _fmt(_class_value(all_empty, class_name, "recall")),
                          _fmt(_class_value(all_empty, class_name, "f1")),
                          _fmt_count(_class_value(all_empty, class_name, "support"))]
                         for class_name in CLASS_NAMES])
    else:
        lines += ["`results/final/test/all_empty.json` 未提供。"]
    lines += [""]

    lines += ["## 附件3专项 30 行预测", ""]
    if special:
        lines += [f"来源：`{_source_line(special_path, root)}`；当前行数：{len(special)}。专项没有真实标签，不计算准确率。", ""]
        lines += _table(["sample_id", "pred_label", "pred_score", "P(Negative)", "P(Neutral)", "P(Positive)"],
                        _special_rows(special))
    else:
        lines += ["专项预测 CSV 未提供。"]
    lines += [""]

    lines += ["## ledger 三种子汇总与消融", "", "下表只读取 `results/experiment_ledger.json` 中已有的 valid 与 test 记录；均值 ± SD 使用样本标准差。没有足够记录时显示 `—`，不从别的实验推算。", ""]
    requested_kinds = ("BERT8all", "Historical full", "Historical late_clean", "Historical late_aug",
                       "BERT8no_corruption", "BERT8no_cls", "BERT8mean_pool")
    lines += _table(["组别", "split", "run 数", "seeds",
                     "Missing Acc", "Missing Macro-F1", "Missing Weighted-F1", "Missing MAE", "Missing PCC",
                     "Clean Acc", "Clean Macro-F1", "Clean Weighted-F1", "Clean MAE", "Clean PCC"],
                    _ledger_table(ledger, requested_kinds) if ledger is not None else [])
    if ledger is None:
        lines += ["ledger 未提供。"]
    lines += ["", "BERT8no_corruption、BERT8no_cls、BERT8mean_pool 仅在 ledger 能由 checkpoint/config 明确识别时归入对应消融；未识别的 trial 不会被强行归类。", ""]

    # Figures are deliberately generated after all source parsing.  They use
    # only the same main rows shown in the tables above.
    figure_paths = _plot(root, split_rows, notes)
    lines += ["## 图形", ""]
    if figure_paths:
        lines += ["English-labelled valid/test × modality/position/rho Macro-F1 figure:", ""]
        lines += [f"- `{_source_line(path, root)}`" for path in figure_paths]
    else:
        lines += ["最终 metrics 尚未提供或 matplotlib 不可用，暂未生成图形。"]
    lines += ["", "## 证据文件", ""]
    lines += ["- `results/final/valid/metrics.json`、`results/final/test/metrics.json`：evaluate_model 原始行列表。",
              "- `paired_conditions.csv`、`deletion_rates.csv`、`matched_deletion_rates.csv`、`dual_both_damaged.csv`：同样本、实际删除率和双模态条件分析。",
              "- `results/experiment_ledger.json`：历史 run、seed 和已记录评估；test 是否独立以 ledger/选择元数据为准。", ""]
    if notes:
        lines += ["## 数据状态说明", ""]
        lines += [f"- {note}" for note in notes]
        lines += [""]
    return "\n".join(lines).rstrip() + "\n", figure_paths


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="Q2 project root")
    parser.add_argument("--output", type=Path, default=None, help="Markdown output path")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    output = args.output.resolve() if args.output else root / "docs/最终结果表格.md"
    report, figures = build_report(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    print(json.dumps({"report": str(output), "figures": [str(path) for path in figures]},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
