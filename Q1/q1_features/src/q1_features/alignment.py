from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any

import numpy as np
import yaml


class AlignmentError(RuntimeError):
    pass


def find_ctc_files(ctc_dir: str | Path) -> tuple[Path, Path, Path]:
    root = Path(ctc_dir)
    configs = sorted(root.glob("exp/asr_train_*/config.yaml"))
    weights = sorted(root.glob("exp/asr_train_*/*.pth"))
    bpe = root / "data/token_list/bpe_unigram5000/bpe.model"
    if len(configs) != 1 or not weights or not bpe.is_file():
        raise AlignmentError("CTC model snapshot is incomplete or ambiguous")
    preferred = [p for p in weights if p.name == "valid.acc.ave_10best.pth"]
    if len(preferred) != 1:
        raise AlignmentError("valid.acc.ave_10best.pth not found uniquely")
    return configs[0], preferred[0], bpe


def prepare_ctc_config(ctc_dir: str | Path) -> tuple[Path, Path]:
    root = Path(ctc_dir).resolve()
    config_file, weight_file, bpe_file = find_ctc_files(root)
    cfg = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    if not isinstance(cfg.get("token_list"), list):
        raise AlignmentError("model token_list is not embedded as a list")
    cfg["bpemodel"] = str(bpe_file.resolve())
    normalize = cfg.get("normalize_conf") or {}
    stats = normalize.get("stats_file")
    if stats:
        stats_path = Path(stats)
        if not stats_path.is_absolute():
            candidates = [root / stats_path, root / "exp" / stats_path]
            stats_path = next((p for p in candidates if p.exists()), candidates[0])
        normalize["stats_file"] = str(stats_path.resolve())
    cfg["normalize_conf"] = normalize
    runtime = root / "runtime-config.yaml"
    runtime.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return runtime, weight_file.resolve()


def load_ctc_model(ctc_dir: str | Path, device: str):
    from espnet2.tasks.asr import ASRTask
    runtime, weights = prepare_ctc_config(ctc_dir)
    model, train_args = ASRTask.build_model_from_file(str(runtime), str(weights), device=device)
    model.eval().requires_grad_(False)
    if not isinstance(model.token_list, list):
        raise AlignmentError("runtime ASR model token_list is not a list")
    return model, train_args


def ctc_forward(model: Any, wav_float32: np.ndarray, device: str) -> tuple[np.ndarray, float]:
    import torch
    wav = np.asarray(wav_float32, dtype=np.float32)
    with torch.inference_mode():
        speech = torch.from_numpy(wav).unsqueeze(0).to(device)
        lengths = torch.tensor([speech.shape[1]], dtype=torch.long, device=device)
        encoded, encoded_lengths = model.encode(speech=speech, speech_lengths=lengths)
        frame_count = int(encoded_lengths[0].item())
        lpz = model.ctc.log_softmax(encoded)[0, :frame_count].cpu().numpy()
    if lpz.ndim != 2 or lpz.shape[1] != len(model.token_list):
        raise AlignmentError("CTC output vocabulary dimension mismatch")
    return lpz, (len(wav) / 16000.0) / frame_count


def greedy_diagnostic(lpz: np.ndarray, token_list: list[str]) -> str:
    ids = np.asarray(lpz).argmax(axis=1).tolist()
    collapsed = [idx for i, idx in enumerate(ids) if i == 0 or idx != ids[i - 1]]
    pieces = [token_list[idx] for idx in collapsed if idx != 0 and token_list[idx] not in {"<blank>", "<sos/eos>", "<sos>", "<eos>", "<pad>"}]
    text = "".join(pieces).replace("▁", " ").strip()
    return re.sub(r"\s+", " ", text)


def align_token_arrays(
    lpz: np.ndarray, token_arrays: list[np.ndarray], ctc_texts: list[str], token_list: list[str],
    *, index_duration: float, min_window_size: int = 8000, max_window_size: int = 100000,
    score_min_mean_over_L: int = 30,
) -> list[tuple[float, float, float]]:
    result = align_token_arrays_detailed(
        lpz, token_arrays, ctc_texts, token_list, index_duration=index_duration,
        min_window_size=min_window_size, max_window_size=max_window_size,
        score_min_mean_over_L=score_min_mean_over_L,
    )
    return result["segments"]


def align_token_arrays_detailed(
    lpz: np.ndarray, token_arrays: list[np.ndarray], ctc_texts: list[str], token_list: list[str],
    *, index_duration: float, min_window_size: int = 8000, max_window_size: int = 100000,
    score_min_mean_over_L: int = 30,
) -> dict[str, Any]:
    from ctc_segmentation import CtcSegmentationParameters, ctc_segmentation, determine_utterance_segments, prepare_token_list
    params = CtcSegmentationParameters(
        char_list=token_list, blank=0, index_duration=index_duration,
        min_window_size=min_window_size, max_window_size=max_window_size,
        score_min_mean_over_L=score_min_mean_over_L,
    )
    ground_truth, begin = prepare_token_list(params, token_arrays)
    timings, char_probs, state_list = ctc_segmentation(params, lpz, ground_truth)
    return {
        "segments": [(float(s), float(e), float(score)) for s, e, score in determine_utterance_segments(params, begin, char_probs, timings, ctc_texts)],
        "timings": np.asarray(timings, dtype=np.float64),
        "char_probs": np.asarray(char_probs, dtype=np.float64),
        "state_list": [str(value) for value in state_list],
    }
