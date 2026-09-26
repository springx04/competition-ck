from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from q1_features.pooling import NativeSeries


class AudioExtractionError(RuntimeError):
    pass


def create_smile():
    import opensmile
    return opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
        sampling_rate=16000,
        resample=False,
        num_workers=1,
    )


def extract_audio(
    wav_float32: np.ndarray,
    audio_offset: float,
    smile: Any,
    duration: float | None = None,
) -> tuple[NativeSeries, list[str], list[dict[str, Any]]]:
    wav = np.asarray(wav_float32, dtype=np.float32)
    if wav.ndim != 1:
        raise AudioExtractionError("expected mono waveform")
    frame = smile.process_signal(wav, 16000)
    names = [str(name) for name in frame.columns]
    if len(names) != 25:
        raise AudioExtractionError(f"expected 25 eGeMAPSv02 LLD columns, found {len(names)}")
    values = frame.to_numpy(dtype=np.float32, copy=True)
    intervals = np.zeros((len(frame), 2), dtype=np.float64)
    eligible = np.ones(len(frame), dtype=bool)
    rows: list[dict[str, Any]] = []
    wav_end = len(wav) / 16000.0
    if duration is not None:
        # Resampling can round the derived WAV up by a fraction of a sample.
        # Keep every native interval inside the authoritative media timeline.
        wav_end = min(wav_end, max(0.0, float(duration) - float(audio_offset)))
    for row_id, idx in enumerate(frame.index):
        if not isinstance(idx, tuple) or len(idx) < 2:
            raise AudioExtractionError("openSMILE LLD index must provide start/end")
        local_start = float(idx[-2].total_seconds())
        local_end = float(idx[-1].total_seconds())
        reasons: list[str] = []
        if not np.isfinite(local_end):
            local_end = wav_end
            reasons.append("end_replaced_with_wav_end")
        local_start = max(0.0, local_start)
        local_end = min(wav_end, local_end)
        finite = bool(np.isfinite(values[row_id]).all())
        valid_interval = local_end > local_start
        eligible[row_id] = finite and valid_interval
        if not finite:
            reasons.append("invalid_output")
        if not valid_interval:
            reasons.append("invalid_interval")
        if not eligible[row_id]:
            values[row_id] = 0.0
        intervals[row_id] = (audio_offset + local_start, audio_offset + local_end)
        rows.append({
            "audio_row_id": row_id, "start": intervals[row_id, 0], "end": intervals[row_id, 1],
            "local_start": local_start, "local_end": local_end,
            "wav_sample_start": max(0, int(np.floor(local_start * 16000))),
            "wav_sample_end": min(len(wav), int(np.ceil(local_end * 16000))),
            "eligible": bool(eligible[row_id]), "reasons": reasons,
            "index_text": str(idx),
        })
    return NativeSeries(values, intervals, eligible, np.arange(len(frame), dtype=np.int64)), names, rows
