from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from .alignment import align_token_arrays_detailed, ctc_forward, greedy_diagnostic, load_ctc_model
from .config import load_config, save_config
from .manifest import Sample, build_manifest, load_manifest
from .quality import audit_pairing, load_reviews, word_error_counts
from .report import report_run
from .storage import append_event, load_npz, read_json, read_jsonl, write_csv, write_json, write_jsonl, write_npz


class PipelineError(RuntimeError):
    pass


def _is_cuda_runtime_failure(exc: BaseException) -> bool:
    text = str(exc).lower()
    return any(token in text for token in (
        "cuda", "cudnn", "out of memory", "no kernel image", "driver version",
        "device-side", "invalid device function",
    ))


def _code_state(project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root)
    try:
        head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain", "--", str(root)], check=True, capture_output=True, text=True).stdout.splitlines()
        return {"git_head": head, "dirty": bool(dirty), "dirty_paths": dirty}
    except Exception as exc:
        revision_path = root / "SOURCE_REVISION"
        revision = revision_path.read_text(encoding="ascii").strip() if revision_path.is_file() else ""
        if re.fullmatch(r"[0-9a-fA-F]{40}", revision):
            return {
                "git_head": revision.lower(), "dirty": False,
                "source": "git-archive-export", "git_probe_error": repr(exc),
            }
        return {"git_head": None, "dirty": True, "error": repr(exc)}


def _environment_snapshot(cfg: dict[str, Any]) -> dict[str, Any]:
    from importlib import metadata
    packages = {}
    for name in ("torch", "torchaudio", "numpy", "transformers", "espnet", "ctc-segmentation", "opensmile", "av", "openpyxl", "PyYAML"):
        try: packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError: packages[name] = None
    def command(args: list[str]) -> str | None:
        try:
            completed = subprocess.run(args, check=True, capture_output=True, text=True, timeout=20)
            return (completed.stdout or completed.stderr).strip()
        except Exception:
            return None
    root = Path(cfg["project_root"])
    result: dict[str, Any] = {
        "python": sys.version, "platform": platform.platform(), "machine": platform.machine(),
        "packages": packages, "ffmpeg": command(["ffmpeg", "-version"]),
        "nvidia_smi": command(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]),
    }
    for name, path in {"mmsa": root / "third_party" / "MMSA-FET", "openface": root / "third_party" / "OpenFace"}.items():
        result[f"{name}_revision"] = command(["git", "-C", str(path), "rev-parse", "HEAD"]) if path.is_dir() else None
    result["model_download_record"] = read_json(root / "models" / "download-record.json", None)
    return result


def _ensure_resume_allowed(cfg: dict[str, Any], run_dir: Path) -> None:
    recorded = read_json(run_dir / "environment.json", {}).get("code_state")
    current = _code_state(cfg["project_root"])
    if not recorded or recorded.get("dirty") or current.get("dirty") or recorded.get("git_head") != current.get("git_head"):
        raise PipelineError("--resume requires the same clean Git revision recorded by prepare")


def seed_everything(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def choose_device(requested: str, allow_cpu: bool = True) -> tuple[str, dict[str, Any]]:
    import torch
    info: dict[str, Any] = {"requested": requested}
    if requested.startswith("cuda") and torch.cuda.is_available():
        try:
            x = torch.ones((64, 64), dtype=torch.float32, device=requested)
            y = x @ x
            torch.cuda.synchronize()
            info.update({"actual": requested, "gpu_name": torch.cuda.get_device_name(torch.device(requested)), "probe": float(y[0, 0])})
            return requested, info
        except Exception as exc:
            info["cuda_error"] = repr(exc)
    if allow_cpu:
        info["actual"] = "cpu"
        return "cpu", info
    raise PipelineError(f"requested device unavailable: {requested}")


def _sample_dir(run_dir: Path, sample: Sample) -> Path:
    return run_dir / "samples" / f"{sample.sample_index:06d}"


def _status(path: Path, sample: Sample) -> dict[str, Any]:
    return read_json(path, {
        "sample_id": sample.sample_id, "sample_index": sample.sample_index,
        "stages": {}, "processing_status": "not_processed", "pairing_status": "unverifiable",
        "paired_use": False, "devices": {}, "errors": [],
    })


def _update_stage(run_dir: Path, sample: Sample, stage: str, state: str, **extra: Any) -> None:
    path = _sample_dir(run_dir, sample) / "status.json"
    value = _status(path, sample)
    value["stages"][stage] = {"status": state, **extra}
    write_json(path, value)


def _event(run_dir: Path, sample: Sample | None, stage: str, status: str, **extra: Any) -> None:
    append_event(run_dir / "logs" / "events.jsonl", {
        "timestamp": time.time(), "sample_id": sample.sample_id if sample else None,
        "sample_index": sample.sample_index if sample else None, "stage": stage, "status": status, **extra,
    })


def _selected(manifest: list[Sample], ids_file: str | Path | None) -> tuple[list[Sample], set[str] | None]:
    if ids_file is None:
        return manifest, None
    requested = {line.strip() for line in Path(ids_file).read_text(encoding="utf-8").splitlines() if line.strip()}
    known = {sample.sample_id for sample in manifest}
    unknown = requested - known
    if unknown:
        raise PipelineError("unknown sample IDs: " + ", ".join(sorted(unknown)))
    return [sample for sample in manifest if sample.sample_id in requested], requested


def prepare_run(config_path: str | Path, run_dir: str | Path, data_root: str | Path, labels_xlsx: str | Path | None = None) -> list[Sample]:
    run = Path(run_dir).resolve()
    if (run / "config.yaml").exists():
        raise PipelineError(f"run already exists: {run}; choose a new run name")
    cfg = load_config(config_path, data_root=data_root)
    seed_everything(int(cfg["seed"]))
    run.mkdir(parents=True, exist_ok=True)
    for sub in ("samples", "features", "reports/timelines", "logs"):
        (run / sub).mkdir(parents=True, exist_ok=True)
    save_config(cfg, run / "config.yaml")
    manifest = build_manifest(cfg["paths"]["data_root"], run, labels_xlsx=labels_xlsx)
    for sample in manifest:
        sample_dir = _sample_dir(run, sample)
        sample_dir.mkdir(parents=True, exist_ok=True)
        write_json(sample_dir / "status.json", _status(sample_dir / "status.json", sample))
    env = _environment_snapshot(cfg)
    env.update({"seed": cfg["seed"], "argv": sys.argv, "code_state": _code_state(cfg["project_root"])})
    write_json(run / "environment.json", env)
    _event(run, None, "prepare", "ok", samples=len(manifest))
    return manifest


def load_run(config_path: str | Path, run_dir: str | Path) -> tuple[dict[str, Any], Path, list[Sample]]:
    run = Path(run_dir).resolve()
    if not (run / "manifest.jsonl").is_file():
        raise PipelineError(f"run is not prepared: {run}")
    cfg = load_config(config_path, run_config=run / "config.yaml")
    saved = load_config(run / "config.yaml")
    # project_root reflects the file location and is not a semantic setting.
    lhs, rhs = dict(cfg), dict(saved)
    lhs.pop("project_root", None); rhs.pop("project_root", None)
    if lhs != rhs:
        raise PipelineError("current configuration differs from saved run configuration; create a new run or use explicit redo")
    return cfg, run, load_manifest(run / "manifest.jsonl")


def _run_per_sample(
    cfg: dict[str, Any], run: Path, samples: list[Sample], stage: str,
    action: Callable[[Sample, Path], dict[str, Any] | None], *, resume: bool,
) -> None:
    if resume:
        _ensure_resume_allowed(cfg, run)
    for sample in samples:
        sample_dir = _sample_dir(run, sample)
        current = _status(sample_dir / "status.json", sample)
        if resume and current.get("stages", {}).get(stage, {}).get("status") == "ok":
            _event(run, sample, stage, "resume_skip")
            continue
        started = time.monotonic()
        peak_before = None
        try:
            import resource
            peak_before = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        except (ImportError, AttributeError):
            pass
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except ImportError:
            torch = None
        try:
            result = action(sample, sample_dir) or {}
            elapsed = time.monotonic() - started
            resources: dict[str, Any] = {}
            try:
                import resource
                resources["process_peak_rss_kb"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
                resources["process_peak_rss_before_kb"] = peak_before
            except (ImportError, AttributeError):
                pass
            try:
                import torch as _torch
                if _torch.cuda.is_available(): resources["cuda_peak_allocated_bytes"] = int(_torch.cuda.max_memory_allocated())
            except ImportError:
                pass
            _update_stage(run, sample, stage, "ok", elapsed_s=elapsed, resources=resources, **result)
            _event(run, sample, stage, "ok", elapsed_s=elapsed)
        except Exception as exc:
            elapsed = time.monotonic() - started
            detail = {"elapsed_s": elapsed, "error": repr(exc), "traceback": traceback.format_exc()}
            _update_stage(run, sample, stage, "failed", **detail)
            _event(run, sample, stage, "failed", error=repr(exc), elapsed_s=elapsed)


def stage_media(cfg: dict[str, Any], run: Path, samples: list[Sample], *, resume: bool) -> None:
    from .media import probe_and_decode
    data_root = Path(cfg["paths"]["data_root"])
    def action(sample: Sample, sample_dir: Path) -> dict[str, Any]:
        if sample.input_status not in {"ok", "integer_equivalent"}:
            raise PipelineError(f"input status is {sample.input_status}")
        media = probe_and_decode(sample, cfg, sample_dir, data_root=data_root)
        return {"duration": float(media["duration"]), "audio_rows": int(media.get("audio_frame_count", 0)), "video_rows": int(media.get("video_frame_count", 0))}
    _run_per_sample(cfg, run, samples, "media", action, resume=resume)
    valid: list[tuple[float, str]] = []
    for sample in samples:
        media = read_json(_sample_dir(run, sample) / "media.json", {})
        duration = float(media.get("duration", 0.0) or 0.0)
        if duration > 0:
            valid.append((duration, sample.sample_id))
    valid.sort()
    if len(valid) >= 3:
        positions = sorted({0, (len(valid) - 1) // 2, len(valid) - 1})
        (run / "reports" / "review_samples.txt").write_text(
            "".join(valid[position][1] + "\n" for position in positions), encoding="utf-8"
        )


def stage_text(cfg: dict[str, Any], run: Path, samples: list[Sample], *, resume: bool) -> None:
    from .extractors.bert import BertWordEncoder
    from .text_map import build_words
    device, device_info = choose_device(cfg["runtime"]["device"], cfg["runtime"].get("allow_same_model_cpu_fallback", True))
    allow_cpu = bool(cfg["runtime"].get("allow_same_model_cpu_fallback", True))
    try:
        encoder = BertWordEncoder(cfg["paths"]["bert_dir"], device)
    except Exception as exc:
        if not (allow_cpu and device.startswith("cuda") and _is_cuda_runtime_failure(exc)):
            raise
        device_info["model_load_cuda_error"] = repr(exc)
        device = "cpu"
        device_info["actual"] = "cpu"
        encoder = BertWordEncoder(cfg["paths"]["bert_dir"], device)
    def action(sample: Sample, sample_dir: Path) -> dict[str, Any]:
        nonlocal device, encoder
        words = build_words(sample.raw_text)
        try:
            encoded = encoder.encode(
                words, content_tokens_per_chunk=int(cfg["text"]["content_tokens_per_chunk"]),
                chunk_stride=int(cfg["text"]["chunk_stride"]),
            )
        except Exception as exc:
            if not (allow_cpu and device.startswith("cuda") and _is_cuda_runtime_failure(exc)):
                raise
            device_info["forward_cuda_error"] = repr(exc)
            encoder.close()
            device = "cpu"
            device_info["actual"] = "cpu"
            encoder = BertWordEncoder(cfg["paths"]["bert_dir"], device)
            encoded = encoder.encode(
                words, content_tokens_per_chunk=int(cfg["text"]["content_tokens_per_chunk"]),
                chunk_stride=int(cfg["text"]["chunk_stride"]),
            )
        features, tokens, chunks, statuses = encoded
        word_rows = [asdict(w) if is_dataclass(w) else dict(w) for w in words]
        for row, info in zip(word_rows, statuses):
            row.update(info)
            row.setdefault("alignment_reasons", [])
            row.setdefault("candidate_interval", None)
            row.setdefault("accepted_interval", None)
        write_jsonl(sample_dir / "words.jsonl", word_rows)
        write_jsonl(sample_dir / "bert_tokens.jsonl", tokens)
        write_jsonl(sample_dir / "bert_chunks.jsonl", chunks)
        write_npz(sample_dir / "bert_words.npz", word_features=features, word_ids=np.arange(len(words), dtype=np.int64))
        return {"device": device, "word_count": len(words), "token_count": len(tokens)}
    try:
        _run_per_sample(cfg, run, samples, "text", action, resume=resume)
    finally:
        encoder.close()
    env = read_json(run / "environment.json", {})
    env["bert_device"] = device_info
    write_json(run / "environment.json", env)


def stage_align(cfg: dict[str, Any], run: Path, samples: list[Sample], *, resume: bool) -> None:
    import sentencepiece as spm
    import soundfile as sf
    from .alignment import find_ctc_files
    from .text_map import normalize_ctc_text
    device, device_info = choose_device(cfg["runtime"]["device"], cfg["runtime"].get("allow_same_model_cpu_fallback", True))
    allow_cpu = bool(cfg["runtime"].get("allow_same_model_cpu_fallback", True))
    try:
        model, _args = load_ctc_model(cfg["paths"]["ctc_dir"], device)
    except Exception as exc:
        if not (allow_cpu and device.startswith("cuda") and _is_cuda_runtime_failure(exc)):
            raise
        device_info["model_load_cuda_error"] = repr(exc)
        device = "cpu"
        device_info["actual"] = "cpu"
        model, _args = load_ctc_model(cfg["paths"]["ctc_dir"], device)
    _config, _weight, bpe_path = find_ctc_files(cfg["paths"]["ctc_dir"])
    sp = spm.SentencePieceProcessor(model_file=str(bpe_path))
    token_to_id = {token: i for i, token in enumerate(model.token_list)}
    unk_id = token_to_id.get("<unk>")
    def action(sample: Sample, sample_dir: Path) -> dict[str, Any]:
        nonlocal device, model
        words = read_jsonl(sample_dir / "words.jsonl")
        media = read_json(sample_dir / "media.json", {})
        if media.get("audio_discontinuous"):
            raise PipelineError("audio_discontinuous")
        wav, sr = sf.read(sample_dir / "audio_16k.wav", dtype="float32", always_2d=False)
        if sr != 16000 or np.asarray(wav).ndim != 1:
            raise PipelineError("derived WAV is not 16k mono")
        try:
            lpz, index_duration = ctc_forward(model, np.asarray(wav, np.float32), device)
        except Exception as exc:
            if not (allow_cpu and device.startswith("cuda") and _is_cuda_runtime_failure(exc)):
                raise
            device_info["forward_cuda_error"] = repr(exc)
            try:
                import torch
                model.to("cpu")
                torch.cuda.empty_cache()
            except Exception:
                pass
            del model
            device = "cpu"
            device_info["actual"] = "cpu"
            model, _ = load_ctc_model(cfg["paths"]["ctc_dir"], device)
            lpz, index_duration = ctc_forward(model, np.asarray(wav, np.float32), device)
        ctc_texts: list[str] = []
        token_arrays: list[np.ndarray] = []
        row_to_word: list[int] = []
        for word in words:
            normalized = normalize_ctc_text(word["raw_word"])
            ctc_text = normalized.ctc_text if hasattr(normalized, "ctc_text") else normalized["ctc_text"]
            word["ctc_text"] = ctc_text
            word["normalization_assumed"] = bool(getattr(normalized, "normalization_assumed", False))
            word["unsupported_text"] = bool(getattr(normalized, "unsupported_text", False))
            if not ctc_text:
                word["alignment_reasons"] = ["no_spoken_content"]
                continue
            pieces = sp.encode(ctc_text, out_type=str)
            contains_unk = any(piece not in token_to_id or piece == "<unk>" for piece in pieces)
            ids = [token_to_id.get(piece, unk_id) for piece in pieces]
            if any(value is None for value in ids):
                raise PipelineError("CTC vocabulary has no <unk> token")
            word["ctc_pieces"] = pieces
            word["contains_unk"] = contains_unk
            ctc_texts.append(ctc_text)
            token_arrays.append(np.asarray(ids, dtype=np.int64))
            row_to_word.append(int(word["word_id"]))
        segmentation = align_token_arrays_detailed(
            lpz, token_arrays, ctc_texts, list(model.token_list), index_duration=index_duration,
            min_window_size=int(cfg["alignment"]["min_window_size"]),
            max_window_size=int(cfg["alignment"]["max_window_size"]),
            score_min_mean_over_L=int(cfg["alignment"]["score_min_mean_over_L"]),
        ) if token_arrays else {"segments": [], "timings": np.empty(0, np.float64), "char_probs": np.empty(0, np.float64), "state_list": []}
        segments = segmentation["segments"]
        a0 = float(media.get("audio_offset", 0.0))
        duration = float(media["duration"])
        previous: tuple[float, float, int] | None = None
        accepted = 0
        for (start, end, score), word_id in zip(segments, row_to_word):
            word = words[word_id]
            reasons = list(word.get("alignment_reasons", []))
            local_duration = len(wav) / 16000.0
            frame = index_duration
            if start < -frame or end > local_duration + frame or not np.isfinite([start, end, score]).all() or end <= start:
                reasons.append("invalid_interval")
            start_c, end_c = max(0.0, start), min(local_duration, end)
            if start_c != start or end_c != end:
                reasons.append("boundary_clipped")
            if score < float(cfg["alignment"]["min_log_score"]): reasons.append("low_ctc_score")
            if word.get("contains_unk"): reasons.append("contains_unk")
            if word.get("unsupported_text"): reasons.append("unsupported_text")
            if previous is not None:
                ps, pe, pid = previous
                if start_c < ps or end_c < pe:
                    reasons.append("nonmonotonic_alignment")
                    words[pid].setdefault("alignment_reasons", []).append("nonmonotonic_alignment")
                if start_c < pe - frame:
                    reasons.append("overlap_alignment")
                    words[pid].setdefault("alignment_reasons", []).append("overlap_alignment")
            previous = (start_c, end_c, word_id)
            candidate = [a0 + start_c, a0 + end_c]
            word["candidate_interval"] = candidate
            word["ctc_log_score"] = score
            fatal = {"invalid_interval", "low_ctc_score", "contains_unk", "unsupported_text", "nonmonotonic_alignment", "overlap_alignment"}
            if not fatal.intersection(reasons) and candidate[0] >= -frame and candidate[1] <= duration + frame:
                word["accepted_interval"] = [max(0.0, candidate[0]), min(duration, candidate[1])]
            word["alignment_reasons"] = sorted(set(reasons))
        fatal_reasons = {"invalid_interval", "low_ctc_score", "contains_unk", "unsupported_text", "nonmonotonic_alignment", "overlap_alignment"}
        accepted = 0
        for word in words:
            if fatal_reasons.intersection(word.get("alignment_reasons", [])):
                word["accepted_interval"] = None
            elif word.get("accepted_interval"):
                accepted += 1
        diagnostic = greedy_diagnostic(lpz, list(model.token_list))
        reference = " ".join(row["ctc_text"] for row in words if row.get("ctc_text"))
        s, d, i, wer = word_error_counts(reference, diagnostic)
        diagnostic_wer = wer if reference.split() else None
        write_jsonl(sample_dir / "words.jsonl", words)
        write_npz(sample_dir / "ctc_segmentation.npz", timings=segmentation["timings"], char_probs=segmentation["char_probs"])
        write_json(sample_dir / "ctc_state.json", {"state_list": segmentation["state_list"]})
        write_json(sample_dir / "diagnostic.json", {
            "diagnostic_transcript": diagnostic, "normalized_reference": reference,
            "substitutions": s, "deletions": d, "insertions": i, "diagnostic_wer": diagnostic_wer,
            "frame_count": int(lpz.shape[0]), "index_duration": index_duration,
            "ctc_row_to_word_id": row_to_word, "initially_accepted_words": accepted,
        })
        return {"device": device, "ctc_frames": int(lpz.shape[0]), "accepted_words": accepted}
    _run_per_sample(cfg, run, samples, "align", action, resume=resume)
    env = read_json(run / "environment.json", {})
    env["ctc_device"] = device_info
    write_json(run / "environment.json", env)
    del model


def stage_audio(cfg: dict[str, Any], run: Path, samples: list[Sample], *, resume: bool) -> None:
    import soundfile as sf
    from .extractors.audio import create_smile, extract_audio
    smile = create_smile()
    feature_names: list[str] | None = None
    def action(sample: Sample, sample_dir: Path) -> dict[str, Any]:
        nonlocal feature_names
        media = read_json(sample_dir / "media.json", {})
        if media.get("audio_discontinuous"):
            raise PipelineError("audio_discontinuous")
        wav, sr = sf.read(sample_dir / "audio_16k.wav", dtype="float32", always_2d=False)
        if sr != 16000:
            raise PipelineError("derived WAV sample rate mismatch")
        series, names, rows = extract_audio(np.asarray(wav), float(media.get("audio_offset", 0.0)), smile)
        if feature_names is None: feature_names = names
        if names != feature_names: raise PipelineError("audio feature column order changed")
        write_npz(sample_dir / "audio_native.npz", values=series.values, intervals=series.intervals, eligible=series.eligible.astype(np.uint8), source_ids=series.source_ids)
        write_jsonl(sample_dir / "audio_rows.jsonl", rows)
        return {"native_rows": len(rows), "eligible_rows": int(series.eligible.sum())}
    _run_per_sample(cfg, run, samples, "audio", action, resume=resume)
    if feature_names is not None:
        write_json(run / "features" / "audio_feature_names.json", feature_names)


def stage_vision(cfg: dict[str, Any], run: Path, samples: list[Sample], *, resume: bool, reuse_raw_csv: bool = False) -> None:
    from .extractors.vision import VISION_COLUMNS, read_openface_csv, run_openface, select_vision_rows
    if not reuse_raw_csv:
        openface_bin = Path(cfg["paths"]["openface_bin"])
        openface_env = Path(cfg["paths"]["openface_env"])
        if not openface_bin.is_file():
            raise PipelineError(f"OpenFace executable is missing: {openface_bin}")
        if not openface_env.is_dir():
            raise PipelineError(f"OpenFace environment is missing: {openface_env}")
        if shutil.which("micromamba") is None:
            raise PipelineError("micromamba is not available for the OpenFace runtime")
    review_all = load_reviews(cfg["paths"]["face_review"])
    def action(sample: Sample, sample_dir: Path) -> dict[str, Any]:
        media = read_json(sample_dir / "media.json", {})
        frames = read_jsonl(sample_dir / "video_frames.jsonl")
        if not frames:
            raise PipelineError("no selected video frames")
        csv_path = sample_dir / "openface" / "features.csv"
        if not reuse_raw_csv or not csv_path.is_file():
            process = run_openface(
                cfg["paths"]["openface_bin"], cfg["paths"]["openface_env"], sample_dir / "frames",
                sample_dir / "openface", int(media["display_width"]), int(media["display_height"]),
                int(cfg["runtime"]["external_command_timeout_s"]),
            )
            write_json(sample_dir / "openface" / "process.json", {
                "args": [str(value) for value in process.args], "returncode": process.returncode,
                "stdout": process.stdout, "stderr": process.stderr,
            })
        candidates = read_openface_csv(csv_path)
        reviews = [r for r in review_all if r.get("sample_id") == sample.sample_id]
        series, rows, diagnostics = select_vision_rows(
            frames, candidates, confidence_min=float(cfg["vision"]["confidence_min"]), reviews=reviews,
            frames_dir=sample_dir / "frames", trajectory_gap_s=float(cfg["vision"]["trajectory_gap_s"]),
            bbox_iou_split=float(cfg["vision"]["bbox_iou_split"]),
            bbox_center_jump_fraction=float(cfg["vision"]["bbox_center_jump_fraction"]),
            scene_hist_l1_split=float(cfg["vision"]["scene_hist_l1_split"]),
        )
        width, height = int(media["display_width"]), int(media["display_height"])
        diagnostics["camera_intrinsics"] = {
            "basis": "estimated", "fx": 500.0 * width / 640.0,
            "fy": 500.0 * height / 480.0, "cx": width / 2.0, "cy": height / 2.0,
        }
        write_npz(sample_dir / "vision_native.npz", values=series.values, intervals=series.intervals, eligible=series.eligible.astype(np.uint8), source_ids=series.source_ids)
        write_jsonl(sample_dir / "vision_rows.jsonl", rows)
        write_json(sample_dir / "vision_diagnostic.json", diagnostics)
        return {"native_rows": len(rows), "eligible_rows": int(series.eligible.sum()), **diagnostics}
    _run_per_sample(cfg, run, samples, "vision", action, resume=resume and not reuse_raw_csv)
    write_json(run / "features" / "vision_feature_names.json", VISION_COLUMNS)


def stage_audit(cfg: dict[str, Any], run: Path, samples: list[Sample]) -> None:
    issue_rows: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []
    review_all = load_reviews(cfg["paths"]["alignment_review"])
    for sample in samples:
        sample_dir = _sample_dir(run, sample)
        diagnostic = read_json(sample_dir / "diagnostic.json", {})
        words = read_jsonl(sample_dir / "words.jsonl")
        spoken = [w for w in words if w.get("ctc_text")]
        accepted = [w for w in spoken if w.get("accepted_interval")]
        low_score = [w for w in spoken if "low_ctc_score" in w.get("alignment_reasons", [])]
        unlocated = [
            {"word_id": w.get("word_id"), "raw_word": w.get("raw_word"), "reasons": w.get("alignment_reasons", [])}
            for w in spoken if not w.get("accepted_interval")
        ]
        fraction = len(accepted) / max(1, len(spoken)) if spoken else 1.0
        media = read_json(sample_dir / "media.json", {})
        vision_diagnostic = read_json(sample_dir / "vision_diagnostic.json", {})
        automatic: list[dict[str, Any]] = []
        pairing_issue_types = {"audio_discontinuous", "audio_duration_mismatch", "sample_rate_change", "channel_layout_change", "non_finite_pcm", "gap", "overlap"}
        for issue in media.get("issues", []):
            if issue.get("type") in pairing_issue_types:
                automatic.append({"sample_id": sample.sample_id, "issue_type": issue.get("type", "media_issue"), **issue})
        current_status = _status(sample_dir / "status.json", sample)
        if float(media.get("duration", 0.0) or 0.0) <= 0:
            automatic.append({"sample_id": sample.sample_id, "issue_type": "media_structure_failed", "evidence": "no reliable positive common duration"})
        if current_status.get("stages", {}).get("align", {}).get("status") != "ok":
            automatic.append({"sample_id": sample.sample_id, "issue_type": "alignment_failed", "evidence": current_status.get("stages", {}).get("align", {}).get("error", "alignment stage not completed")})
        if int(vision_diagnostic.get("episode_count", 0) or 0) > 1 and not bool(vision_diagnostic.get("automatic_single")):
            automatic.append({"sample_id": sample.sample_id, "issue_type": "identity_unknown", "evidence": f"episode_count={vision_diagnostic.get('episode_count')} requires target review"})
        result = audit_pairing(
            diagnostic_wer=diagnostic.get("diagnostic_wer"), aligned_word_fraction=fraction,
            automatic_issues=automatic,
            reviews=[r for r in review_all if r.get("sample_id") == sample.sample_id],
            wer_warn=float(cfg["quality"]["diagnostic_wer_warn"]),
            aligned_fraction_warn=float(cfg["quality"]["aligned_word_fraction_warn"]),
        )
        status_path = sample_dir / "status.json"
        status = _status(status_path, sample)
        status.update({
            "pairing_status": result["pairing_status"], "paired_use": result["paired_use"],
            "review_scope": result["review_scope"], "text_time_policy": result["text_time_policy"],
            "text_quarantine_intervals": result["text_quarantine_intervals"],
        })
        status["stages"]["audit"] = {"status": "ok"}
        write_json(status_path, status)
        for issue in result["issues"]:
            issue_rows.append({"sample_id": sample.sample_id, "file": sample.video_relpath, "start": issue.get("start", ""), "end": issue.get("end", ""), "issue_type": issue.get("issue_type", "unknown"), "status": result["pairing_status"], "evidence": issue.get("evidence", json.dumps(issue, ensure_ascii=False)), "action": "quarantine_pairing" if not result["paired_use"] else "none"})
        accepted_by_time = sorted(accepted, key=lambda w: float(w["accepted_interval"][0]))
        first_accepted = float(accepted_by_time[0]["accepted_interval"][0]) if accepted_by_time else ""
        last_accepted = float(accepted_by_time[-1]["accepted_interval"][1]) if accepted_by_time else ""
        vision_rows = read_jsonl(sample_dir / "vision_rows.jsonl")
        stage_states = current_status.get("stages", {})
        required_audits_attempted = all(name in stage_states for name in ("media", "align", "vision"))
        quality_rows.append({
            "sample_id": sample.sample_id, "input_status": sample.input_status,
            "duration": media.get("duration", ""), "audio_offset": media.get("audio_offset", ""),
            "video_stream_index": media.get("video_stream_index", ""), "audio_stream_index": media.get("audio_stream_index", ""),
            "first_pts_difference_s": media.get("first_pts_difference_s", ""),
            "terminal_difference_s": media.get("terminal_difference_s", ""),
            "audio_discontinuous": media.get("audio_discontinuous", ""),
            "audio_frame_count": media.get("audio_frame_count", ""), "video_frame_count": media.get("video_frame_count", ""),
            "bad_audio_frame_count": media.get("bad_audio_frame_count", ""),
            "bad_video_frame_count": media.get("bad_video_frame_count", ""),
            "resample_duration_difference_s": media.get("resample_duration_difference_s", ""),
            "selected_video_frames": media.get("selected_video_frames", ""),
            "media_issues": json.dumps(media.get("issues", []), ensure_ascii=False),
            "spoken_words": len(spoken), "accepted_words": len(accepted), "aligned_word_fraction": fraction,
            "low_score_words": len(low_score), "low_score_word_fraction": len(low_score) / max(1, len(spoken)),
            "unlocated_words": json.dumps(unlocated, ensure_ascii=False),
            "first_accepted_word_start_s": first_accepted, "last_accepted_word_end_s": last_accepted,
            "first_word_boundary_distance_s": first_accepted if first_accepted != "" else "",
            "last_word_boundary_distance_s": float(media.get("duration", 0.0)) - last_accepted if last_accepted != "" else "",
            "diagnostic_wer": diagnostic.get("diagnostic_wer", ""), "pairing_status": result["pairing_status"],
            "diagnostic_transcript": diagnostic.get("diagnostic_transcript", ""),
            "normalized_reference": diagnostic.get("normalized_reference", ""),
            "paired_use": int(result["paired_use"]),
            "vision_successful_candidates": vision_diagnostic.get("successful_candidate_count", ""),
            "vision_episode_count": vision_diagnostic.get("episode_count", ""),
            "vision_scene_break_count": vision_diagnostic.get("scene_break_count", ""),
            "vision_single_visible_assumption": vision_diagnostic.get("automatic_single", ""),
            "vision_eligible_rows": sum(bool(row.get("eligible")) for row in vision_rows),
            "vision_eligible_fraction": sum(bool(row.get("eligible")) for row in vision_rows) / max(1, len(vision_rows)),
            "automatic_check_status": "completed" if required_audits_attempted else "incomplete",
            "automatic_issue_count": len(result["issues"]),
            "review_scope": json.dumps(result["review_scope"], ensure_ascii=False),
        })
    write_csv(run / "reports" / "quality.csv", quality_rows, list(quality_rows[0]) if quality_rows else ["sample_id"])
    write_csv(run / "reports" / "alignment_issues.csv", issue_rows, ["sample_id", "file", "start", "end", "issue_type", "status", "evidence", "action"])
    audit_lines = ["# Alignment audit", "", "Automatic checks cover every processed sample. Human audiovisual review is recorded only by explicit review-table rows.", "", "| sample_id | issue | status | evidence |", "|---|---|---|---|"]
    audit_lines.extend(f"| {r['sample_id']} | {r['issue_type']} | {r['status']} | {r['evidence']} |" for r in issue_rows)
    (run / "reports" / "alignment_audit.md").write_text("\n".join(audit_lines) + "\n", encoding="utf-8")


def _empty_series(dim: int):
    from .pooling import NativeSeries
    return NativeSeries(np.empty((0, dim), np.float32), np.empty((0, 2), np.float64), np.empty(0, bool), np.empty(0, np.int64))


def stage_pool(cfg: dict[str, Any], run: Path, samples: list[Sample]) -> None:
    from .pooling import NativeSeries, pool50
    angle_min = float(cfg["pool"]["angle_resultant_min"])
    for sample in samples:
        sample_dir = _sample_dir(run, sample)
        media = read_json(sample_dir / "media.json", {})
        duration = float(media.get("duration", 0.0) or 0.0)
        status = _status(sample_dir / "status.json", sample)
        words = read_jsonl(sample_dir / "words.jsonl")
        bert = load_npz(sample_dir / "bert_words.npz") if (sample_dir / "bert_words.npz").is_file() else {"word_features": np.empty((0, 768), np.float32)}
        text_intervals, text_values, text_eligible, text_ids = [], [], [], []
        for word in words:
            wid = int(word["word_id"])
            interval = word.get("accepted_interval") or [0.0, 0.0]
            text_intervals.append(interval); text_ids.append(wid)
            text_values.append(bert["word_features"][wid] if wid < len(bert["word_features"]) else np.zeros(768, np.float32))
            accepted_interval = word.get("accepted_interval")
            text_policy = status.get("text_time_policy", "accept" if status.get("paired_use") else "quarantine_all")
            allowed = bool(accepted_interval) and text_policy != "quarantine_all"
            if allowed and text_policy == "quarantine_intervals":
                start, end = map(float, accepted_interval)
                allowed = not any(min(end, float(qe)) > max(start, float(qs)) for qs, qe in status.get("text_quarantine_intervals", []))
            text_eligible.append(allowed)
        text_series = NativeSeries(np.asarray(text_values, np.float32).reshape((-1, 768)), np.asarray(text_intervals, np.float64).reshape((-1, 2)), np.asarray(text_eligible, bool), np.asarray(text_ids, np.int64)) if words else _empty_series(768)
        def native(name: str, dim: int):
            path = sample_dir / f"{name}_native.npz"
            if not path.is_file(): return _empty_series(dim)
            data = load_npz(path)
            return NativeSeries(data["values"].astype(np.float32), data["intervals"].astype(np.float64), data["eligible"].astype(bool), data["source_ids"].astype(np.int64))
        audio_series, vision_series = native("audio", 25), native("vision", 22)
        series_list = [text_series, audio_series, vision_series]
        kinds = ["text", "audio", "vision"]
        pooled = []
        for series, kind in zip(series_list, kinds):
            if duration > 0:
                pooled.append(pool50(series, duration, kind, n_bins=int(cfg["pool"]["bins"]), angle_r_threshold=angle_min))
            else:
                from types import SimpleNamespace
                pooled.append(SimpleNamespace(
                    features=np.zeros((50, series.d), np.float32), observed=np.zeros(50, np.uint8),
                    coverage=np.zeros(50, np.float32), indptr=np.zeros(51, np.int64),
                    source_ids=np.empty(0, np.int64), overlap_s=np.empty(0, np.float64),
                    weights=np.empty(0, np.float64),
                ))
        text, audio, vision = [p.features for p in pooled]
        valid = np.ones(50, np.uint8) if duration > 0 else np.zeros(50, np.uint8)
        observed = np.stack([p.observed for p in pooled], axis=1).astype(np.uint8)
        coverage = np.stack([p.coverage for p in pooled], axis=1).astype(np.float32)
        perturb = np.zeros((50, 3), np.uint8)
        edges = np.linspace(0.0, duration, 51, dtype=np.float64) if duration > 0 else np.zeros(51, np.float64)
        intervals = np.stack([edges[:-1], edges[1:]], axis=1)
        write_npz(sample_dir / "native.npz",
            word_features=text_series.values, word_intervals=text_series.intervals, word_eligible=text_series.eligible.astype(np.uint8), word_source_ids=text_series.source_ids,
            audio_features=audio_series.values, audio_intervals=audio_series.intervals, audio_eligible=audio_series.eligible.astype(np.uint8), audio_source_ids=audio_series.source_ids,
            vision_features=vision_series.values, vision_intervals=vision_series.intervals, vision_eligible=vision_series.eligible.astype(np.uint8), vision_source_ids=vision_series.source_ids)
        for name, result in zip(kinds, pooled):
            write_npz(sample_dir / f"map_{name}.npz", indptr=result.indptr, source_ids=result.source_ids, overlap_s=result.overlap_s, weights=result.weights)
        write_npz(sample_dir / "compact50.npz", text=text, audio=audio, vision=vision, valid_mask=valid,
            observed_mask=observed, perturb_mask=perturb, coverage=coverage, time_intervals=intervals,
            duration=np.asarray(duration, np.float64), valid_length=np.asarray(int(valid.sum()), np.int64),
            sample_index=np.asarray(sample.sample_index, np.int64), paired_use=np.asarray(int(status.get("paired_use", False)), np.uint8))
        bin_rows = []
        for k in range(50):
            row: dict[str, Any] = {"bin_index": k, "start": float(intervals[k, 0]), "end": float(intervals[k, 1])}
            for m, name in enumerate(kinds):
                reasons = [] if observed[k, m] else (["no_word"] if name == "text" else ["outside_audio"] if name == "audio" else ["unsampled_interval"])
                if name == "text" and not observed[k, m] and status.get("text_time_policy") == "quarantine_all": reasons = ["pairing_mismatch" if status.get("pairing_status") == "confirmed_mismatch" else "pairing_suspected"]
                row[name] = {"observed": int(observed[k, m]), "coverage": float(coverage[k, m]), "source_count": int(pooled[m].indptr[k + 1] - pooled[m].indptr[k]), "reasons": reasons}
            bin_rows.append(row)
        write_jsonl(sample_dir / "bin_quality.jsonl", bin_rows)
        status["stages"]["pool"] = {"status": "ok"}
        required = [status.get("stages", {}).get(x, {}).get("status") for x in ("media", "text", "align", "audio", "vision")]
        status["processing_status"] = "ok" if all(x == "ok" for x in required) and observed[:, 0].any() and observed[:, 1].any() and observed[:, 2].any() else "partial" if duration > 0 else "failed"
        write_json(sample_dir / "status.json", status)


def collect_run(run: Path, manifest: list[Sample]) -> Path:
    shapes = {"text": (50, 768), "audio": (50, 25), "vision": (50, 22), "valid_mask": (50,), "observed_mask": (50, 3), "perturb_mask": (50, 3), "coverage": (50, 3), "time_intervals": (50, 2)}
    dtypes = {"text": np.float32, "audio": np.float32, "vision": np.float32, "valid_mask": np.uint8, "observed_mask": np.uint8, "perturb_mask": np.uint8, "coverage": np.float32, "time_intervals": np.float64}
    output = {k: np.zeros((len(manifest),) + shape, dtype=dtypes[k]) for k, shape in shapes.items()}
    output.update({"duration": np.zeros(len(manifest), np.float64), "valid_length": np.zeros(len(manifest), np.int64), "sample_index": np.arange(len(manifest), dtype=np.int64), "paired_use": np.zeros(len(manifest), np.uint8)})
    for sample in manifest:
        path = _sample_dir(run, sample) / "compact50.npz"
        if not path.is_file():
            continue
        data = load_npz(path)
        for key in output:
            if key in data:
                output[key][sample.sample_index] = data[key]
    target = run / "features" / "q1_compact50.npz"
    write_npz(target, **output)
    write_json(run / "features" / "feature_names.json", {"modalities": ["text", "audio", "vision"], "text_dim": 768, "audio_dim": 25, "vision_dim": 22, "vision_angles": [17, 18, 19, 20, 21]})
    return target


def validate_run(run: Path, manifest: list[Sample], selected_ids: set[str] | None = None) -> list[str]:
    from .pooling import interval_union_length
    errors: list[str] = []
    recomputed_windows: list[dict[str, Any]] = []
    path = run / "features" / "q1_compact50.npz"
    if not path.is_file():
        return ["missing features/q1_compact50.npz"]
    data = load_npz(path)
    expected = {"text": ((len(manifest), 50, 768), np.float32), "audio": ((len(manifest), 50, 25), np.float32), "vision": ((len(manifest), 50, 22), np.float32), "valid_mask": ((len(manifest), 50), np.uint8), "observed_mask": ((len(manifest), 50, 3), np.uint8), "perturb_mask": ((len(manifest), 50, 3), np.uint8), "coverage": ((len(manifest), 50, 3), np.float32), "time_intervals": ((len(manifest), 50, 2), np.float64)}
    expected.update({
        "duration": ((len(manifest),), np.float64), "valid_length": ((len(manifest),), np.int64),
        "sample_index": ((len(manifest),), np.int64), "paired_use": ((len(manifest),), np.uint8),
    })
    structural_errors: list[str] = []
    for key, (shape, dtype) in expected.items():
        if key not in data:
            structural_errors.append(f"missing array {key}")
            continue
        if data[key].shape != shape:
            structural_errors.append(f"{key} shape {data[key].shape} != {shape}")
        if data[key].dtype != dtype: errors.append(f"{key} dtype {data[key].dtype} != {dtype}")
        if not np.isfinite(data[key]).all(): errors.append(f"{key} contains non-finite values")
    errors.extend(structural_errors)
    if structural_errors:
        write_json(run / "reports" / "validation.json", {
            "ok": False, "errors": errors, "selected_only": selected_ids is not None,
            "recomputed_observed_window_count": 0, "seed_2026_recomputed_examples": [],
        })
        return errors
    if "coverage" in data and (np.any(data["coverage"] < -1e-7) or np.any(data["coverage"] > 1 + 1e-7)): errors.append("coverage outside [0,1]")
    if "observed_mask" in data:
        if not np.isin(data["observed_mask"], [0, 1]).all(): errors.append("observed_mask is not binary")
        for m, key in enumerate(("text", "audio", "vision")):
            missing = data["observed_mask"][:, :, m] == 0
            if np.any(data[key][missing] != 0): errors.append(f"{key} has nonzero values where observed=0")
            if np.any(data["coverage"][:, :, m][missing] != 0): errors.append(f"{key} has nonzero coverage where observed=0")
    if "sample_index" in data and not np.array_equal(data["sample_index"], np.arange(len(manifest), dtype=np.int64)): errors.append("sample_index array is not manifest order")
    if not np.isin(data["valid_mask"], [0, 1]).all(): errors.append("valid_mask is not binary")
    if "valid_length" in data and "valid_mask" in data and not np.array_equal(data["valid_length"], data["valid_mask"].sum(axis=1, dtype=np.int64)): errors.append("valid_length != sum(valid_mask)")
    if "perturb_mask" in data and np.any(data["perturb_mask"] != 0): errors.append("Q1 perturb_mask must be all zero")
    if "paired_use" in data and not np.isin(data["paired_use"], [0, 1]).all(): errors.append("paired_use is not binary")
    if any(key.lower() in {"label", "labels", "annotation"} for key in data): errors.append("labels/annotations must not be stored in feature NPZ")
    if all(key in data for key in ("duration", "valid_mask", "time_intervals")):
        for i, duration in enumerate(data["duration"]):
            if duration > 0:
                if not np.all(data["valid_mask"][i] == 1): errors.append(f"row {i}: positive duration but V is not all one")
                intervals = data["time_intervals"][i]
                if not (np.isclose(intervals[0, 0], 0.0) and np.isclose(intervals[-1, 1], duration) and np.allclose(intervals[:-1, 1], intervals[1:, 0])):
                    errors.append(f"row {i}: time intervals do not cover [0,D] continuously")
            elif not np.all(data["valid_mask"][i] == 0):
                errors.append(f"row {i}: failed duration but V is not all zero")
    for sample in manifest:
        if selected_ids is not None and sample.sample_id not in selected_ids:
            continue
        sample_dir = _sample_dir(run, sample)
        sample_status = _status(sample_dir / "status.json", sample)
        if selected_ids is None and sample_status.get("processing_status") == "not_processed":
            errors.append(f"{sample.sample_id}: not_processed in full validation")
        if sample_status.get("pairing_status") in {"suspected", "confirmed_mismatch", "unverifiable"} and bool(data["paired_use"][sample.sample_index]):
            errors.append(f"{sample.sample_id}: quarantined pairing marked paired_use=true")
        if not (sample_dir / "compact50.npz").is_file(): errors.append(f"{sample.sample_id}: selected sample not pooled"); continue
        native_path = sample_dir / "native.npz"
        compact_path = sample_dir / "compact50.npz"
        if not native_path.is_file(): errors.append(f"{sample.sample_id}: missing native.npz"); continue
        native_data, sample_compact = load_npz(native_path), load_npz(compact_path)
        compact_required = {
            "text", "audio", "vision", "valid_mask", "observed_mask", "perturb_mask",
            "coverage", "time_intervals", "duration", "valid_length", "sample_index", "paired_use",
        }
        missing_compact = sorted(compact_required - set(sample_compact))
        if missing_compact:
            errors.append(f"{sample.sample_id}: compact50 missing arrays {missing_compact}")
            continue
        duration = float(sample_compact["duration"])
        native_prefix = {"text": "word", "audio": "audio", "vision": "vision"}
        modality_index = {"text": 0, "audio": 1, "vision": 2}
        for name in ("text", "audio", "vision"):
            map_path = sample_dir / f"map_{name}.npz"
            if not map_path.is_file(): errors.append(f"{sample.sample_id}: missing map_{name}.npz"); continue
            mapping = load_npz(map_path)
            prefix = native_prefix[name]
            native_keys = {f"{prefix}_features", f"{prefix}_intervals", f"{prefix}_eligible", f"{prefix}_source_ids"}
            missing_native = sorted(native_keys - set(native_data))
            if missing_native:
                errors.append(f"{sample.sample_id}: native missing {name} arrays {missing_native}")
                continue
            mapping_keys = {"indptr", "source_ids", "overlap_s", "weights"}
            missing_mapping = sorted(mapping_keys - set(mapping))
            if missing_mapping:
                errors.append(f"{sample.sample_id}: map_{name} missing arrays {missing_mapping}")
                continue
            if mapping["indptr"].shape != (51,):
                errors.append(f"{sample.sample_id}: {name} indptr shape")
                continue
            if mapping["indptr"].dtype != np.int64:
                errors.append(f"{sample.sample_id}: {name} indptr dtype")
            if mapping["source_ids"].dtype != np.int64:
                errors.append(f"{sample.sample_id}: {name} source_ids dtype")
            if mapping["overlap_s"].dtype != np.float64 or mapping["weights"].dtype != np.float64:
                errors.append(f"{sample.sample_id}: {name} overlap/weights dtype")
            payload_length = len(mapping["source_ids"])
            if mapping["overlap_s"].shape != (payload_length,) or mapping["weights"].shape != (payload_length,):
                errors.append(f"{sample.sample_id}: {name} CSR payload shape mismatch")
                continue
            if mapping["indptr"][0] != 0 or np.any(np.diff(mapping["indptr"]) < 0) or mapping["indptr"][-1] != payload_length:
                errors.append(f"{sample.sample_id}: {name} invalid CSR indptr")
                continue
            if not np.isfinite(mapping["overlap_s"]).all() or not np.isfinite(mapping["weights"]).all():
                errors.append(f"{sample.sample_id}: {name} non-finite CSR payload")
                continue
            if np.any(mapping["overlap_s"] <= 0) or np.any(mapping["weights"] <= 0):
                errors.append(f"{sample.sample_id}: {name} non-positive CSR payload")
            values = native_data[f"{prefix}_features"]
            intervals = native_data[f"{prefix}_intervals"]
            eligible = native_data[f"{prefix}_eligible"].astype(bool)
            source_ids = native_data[f"{prefix}_source_ids"].astype(np.int64)
            expected_dim = {"text": 768, "audio": 25, "vision": 22}[name]
            native_length = len(source_ids)
            if values.shape != (native_length, expected_dim) or intervals.shape != (native_length, 2) or eligible.shape != (native_length,):
                errors.append(f"{sample.sample_id}: invalid {name} native shapes")
                continue
            if not np.isfinite(values).all() or not np.isfinite(intervals).all():
                errors.append(f"{sample.sample_id}: non-finite {name} native values")
                continue
            if len(np.unique(source_ids)) != len(source_ids): errors.append(f"{sample.sample_id}: duplicate {name} source_ids")
            row_by_source = {int(source_id): row for row, source_id in enumerate(source_ids)}
            for row in np.flatnonzero(eligible):
                start, end = map(float, intervals[row])
                if not (np.isfinite([start, end]).all() and end > start and start >= -1e-8 and end <= duration + 1e-8):
                    errors.append(f"{sample.sample_id}: invalid eligible {name} interval row {row}")
            mapped_sources = set(map(int, mapping["source_ids"]))
            expected_sources = {int(source_ids[row]) for row in np.flatnonzero(eligible) if intervals[row, 1] > 0 and intervals[row, 0] < duration}
            if not expected_sources.issubset(mapped_sources):
                errors.append(f"{sample.sample_id}: {name} eligible sources missing from CSR: {sorted(expected_sources - mapped_sources)}")
            for k in range(50):
                left, right = int(mapping["indptr"][k]), int(mapping["indptr"][k + 1])
                ids = mapping["source_ids"][left:right]
                overlap = mapping["overlap_s"][left:right]
                weights = mapping["weights"][left:right]
                if len(ids) and (np.any(np.diff(ids) <= 0)):
                    errors.append(f"{sample.sample_id}: {name} bin {k} source_ids not strictly increasing")
                if len(weights) and not np.isclose(weights.sum(), 1.0, atol=1e-6): errors.append(f"{sample.sample_id}: {name} bin {k} weights sum")
                observed = bool(sample_compact["observed_mask"][k, modality_index[name]])
                if observed != bool(len(weights)):
                    errors.append(f"{sample.sample_id}: {name} bin {k} observed/CSR mismatch")
                if not observed:
                    continue
                rows = []
                missing_source = False
                for source_id in ids:
                    if int(source_id) not in row_by_source:
                        errors.append(f"{sample.sample_id}: {name} unknown source_id {int(source_id)}")
                        missing_source = True
                    else:
                        rows.append(row_by_source[int(source_id)])
                if missing_source:
                    continue
                rows_array = np.asarray(rows, dtype=np.int64)
                recomputed = np.sum(values[rows_array].astype(np.float64) * weights[:, None], axis=0)
                if name == "vision":
                    angles = values[rows_array, 17:22].astype(np.float64)
                    sine = np.sum(np.sin(angles) * weights[:, None], axis=0)
                    cosine = np.sum(np.cos(angles) * weights[:, None], axis=0)
                    recomputed[17:22] = np.arctan2(sine, cosine)
                    actual = sample_compact[name][k].astype(np.float64)
                    ordinary_ok = np.allclose(actual[:17], recomputed[:17], rtol=1e-5, atol=1e-6)
                    circular_delta = np.abs((actual[17:22] - recomputed[17:22] + np.pi) % (2 * np.pi) - np.pi)
                    vector_ok = ordinary_ok and bool(np.all(circular_delta <= 1e-6))
                else:
                    vector_ok = bool(np.allclose(sample_compact[name][k], recomputed, rtol=1e-5, atol=1e-6))
                if not vector_ok: errors.append(f"{sample.sample_id}: {name} bin {k} native/CSR recomputation mismatch")
                bin_start, bin_end = map(float, sample_compact["time_intervals"][k])
                clipped = np.column_stack((np.maximum(intervals[rows_array, 0], bin_start), np.minimum(intervals[rows_array, 1], bin_end)))
                expected_coverage = interval_union_length(clipped) / (bin_end - bin_start)
                if not np.isclose(float(sample_compact["coverage"][k, modality_index[name]]), expected_coverage, rtol=1e-5, atol=1e-6):
                    errors.append(f"{sample.sample_id}: {name} bin {k} coverage union mismatch")
                recomputed_windows.append({"sample_id": sample.sample_id, "modality": name, "bin_index": k, "source_ids": ids.astype(int).tolist(), "overlap_s": overlap.astype(float).tolist()})
    rng = np.random.default_rng(2026)
    if len(recomputed_windows) > 10:
        positions = sorted(rng.choice(len(recomputed_windows), size=10, replace=False).tolist())
        checked_examples = [recomputed_windows[position] for position in positions]
    else:
        checked_examples = recomputed_windows
    write_json(run / "reports" / "validation.json", {
        "ok": not errors, "errors": errors, "selected_only": selected_ids is not None,
        "recomputed_observed_window_count": len(recomputed_windows),
        "seed_2026_recomputed_examples": checked_examples,
    })
    return errors


def check_models(cfg: dict[str, Any], run: Path, samples: list[Sample]) -> None:
    if not samples:
        raise PipelineError("check-models needs at least one selected sample")
    # Real checks are deliberately composed from the same production stages.
    stage_text(cfg, run, samples[:1], resume=False)
    stage_align(cfg, run, samples[:1], resume=False)
    stage_audio(cfg, run, samples[:1], resume=False)
    stage_vision(cfg, run, samples[:1], resume=False)
    failed = []
    for sample in samples[:1]:
        state = _status(_sample_dir(run, sample) / "status.json", sample)
        for stage in ("text", "align", "audio", "vision"):
            if state["stages"].get(stage, {}).get("status") != "ok": failed.append(stage)
    if failed: raise PipelineError("real model checks failed: " + ", ".join(failed))
    env = read_json(run / "environment.json", {})
    env.update(_environment_snapshot(cfg))
    write_json(run / "environment.json", env)


def run_pipeline(cfg: dict[str, Any], run: Path, manifest: list[Sample], samples: list[Sample], selected_ids: set[str] | None, *, resume: bool, redo_stages: set[str] | None = None) -> list[str]:
    redo = redo_stages or set()
    dependencies = {
        "media": {"media", "text", "align", "audio", "vision"},
        "text": {"text"}, "align": {"align"}, "audio": {"audio"}, "vision": {"vision"},
    }
    forced = set().union(*(dependencies[name] for name in redo)) if redo else set()
    stage_media(cfg, run, samples, resume=resume and "media" not in forced)
    stage_text(cfg, run, samples, resume=resume and "text" not in forced)
    stage_align(cfg, run, samples, resume=resume and "align" not in forced)
    stage_audio(cfg, run, samples, resume=resume and "audio" not in forced)
    stage_vision(cfg, run, samples, resume=resume and "vision" not in forced)
    stage_audit(cfg, run, samples)
    stage_pool(cfg, run, samples)
    collect_run(run, manifest)
    errors = validate_run(run, manifest, selected_ids)
    report_run(run)
    return errors
