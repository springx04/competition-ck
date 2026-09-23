"""Media clocks and source-frame sampling for Q1.

The functions in this module deliberately operate on small, serialisable frame
records.  Decoding code can adapt PyAV/ffprobe objects to :class:`AudioFrame`
and :class:`VideoFrame`; the clock and sampling logic then remains deterministic
and testable without either PyAV or ffmpeg being installed.

Important invariants from the implementation plan:

* a timestamp is ``pts * time_base`` (or an explicitly supplied
  ``best_effort_timestamp_time`` when PTS is unavailable);
* ``t0`` is the first valid start across the selected audio/video streams;
* audio ends are ``start + nb_samples / sample_rate``;
* video display intervals use source duration where possible and conservative
  PTS-based estimates otherwise;
* a failed source frame is never removed from the time axis and is never
  replaced by a copied neighbouring frame;
* the 25 Hz grid is a set of query times, not a new frame-rate clock.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
from dataclasses import dataclass, field, replace
from fractions import Fraction
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Iterator, Mapping, Optional, Sequence, Union

import numpy as np


Number = Union[int, float]
TimeBaseLike = Union["TimeBase", Fraction, tuple[int, int], list[int], str, Any]


@dataclass(frozen=True, slots=True)
class TimeBase:
    """A positive rational media time base.

    ``TimeBase(1, 25)`` means one PTS tick is 1/25 seconds.  PyAV's
    ``Fraction``-like time-base objects and ``(numerator, denominator)`` pairs
    are accepted by the public functions as well.
    """

    numerator: int = 1
    denominator: int = 1

    def __post_init__(self) -> None:
        numerator = int(self.numerator)
        denominator = int(self.denominator)
        if denominator == 0:
            raise ValueError("time_base denominator must not be zero")
        if denominator < 0:
            numerator = -numerator
            denominator = -denominator
        if numerator <= 0:
            raise ValueError("time_base numerator must be positive")
        object.__setattr__(self, "numerator", numerator)
        object.__setattr__(self, "denominator", denominator)

    @property
    def num(self) -> int:
        """Short alias used by JSON/source mapping writers."""

        return self.numerator

    @property
    def den(self) -> int:
        """Short alias used by JSON/source mapping writers."""

        return self.denominator

    def seconds(self, pts: Number) -> float:
        """Convert an integer/floating PTS to float64 seconds."""

        return float(pts) * (self.numerator / self.denominator)


def _coerce_time_base(
    value: TimeBaseLike = TimeBase(),
    numerator: Optional[int] = None,
    denominator: Optional[int] = None,
) -> TimeBase:
    """Convert common PyAV/test representations to :class:`TimeBase`."""

    if numerator is not None or denominator is not None:
        if numerator is None or denominator is None:
            raise ValueError("time_base_num and time_base_den must be supplied together")
        return TimeBase(int(numerator), int(denominator))
    if isinstance(value, TimeBase):
        return value
    if isinstance(value, Fraction):
        return TimeBase(value.numerator, value.denominator)
    if isinstance(value, str):
        if "/" not in value:
            raise ValueError(f"invalid rational time base: {value!r}")
        num, den = value.split("/", 1)
        return TimeBase(int(num), int(den))
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return TimeBase(int(value[0]), int(value[1]))
    # PyAV time_base and Fraction-like values expose numerator/denominator.
    if hasattr(value, "numerator") and hasattr(value, "denominator"):
        return TimeBase(int(value.numerator), int(value.denominator))
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        fraction = Fraction(float(value)).limit_denominator(1_000_000)
        return TimeBase(fraction.numerator, fraction.denominator)
    raise TypeError(f"unsupported time_base value: {value!r}")


@dataclass(frozen=True, slots=True)
class VideoFrame:
    """A decoded video source-frame record.

    ``duration`` is in ticks of ``time_base``, matching PyAV's frame duration
    convention.  Use ``duration_seconds`` when an adapter already has a
    duration in seconds.  ``decode_status`` may be ``ok``/``success`` or a
    failure status such as ``failed``/``error``.
    """

    source_frame_index: int = 0
    pts: Optional[Number] = None
    time_base: TimeBaseLike = field(default_factory=TimeBase)
    duration: Optional[Number] = None
    duration_seconds: Optional[float] = None
    best_effort_timestamp_time: Optional[float] = None
    decode_status: str = "ok"
    time_base_num: Optional[int] = None
    time_base_den: Optional[int] = None
    duration_time_base: Optional[TimeBaseLike] = None
    valid: Optional[bool] = None
    reason: Optional[str] = None


@dataclass(frozen=True, slots=True)
class AudioFrame:
    """A decoded audio source-frame record.

    Audio ``duration`` is intentionally not used to determine the end: the
    source sample count and sample rate are authoritative per the plan.
    """

    source_frame_index: int = 0
    pts: Optional[Number] = None
    time_base: TimeBaseLike = field(default_factory=TimeBase)
    sample_rate: int = 0
    nb_samples: int = 0
    channels: Optional[int] = None
    channel_layout: Optional[str] = None
    pcm: Optional[Any] = None
    best_effort_timestamp_time: Optional[float] = None
    decode_status: str = "ok"
    time_base_num: Optional[int] = None
    time_base_den: Optional[int] = None
    valid: Optional[bool] = None
    reason: Optional[str] = None


# Names used by a few adapters/readers; keeping them aliases avoids making the
# source-record schema depend on the decoder library in use.
VideoFrameRecord = VideoFrame
AudioFrameRecord = AudioFrame
RationalTimeBase = TimeBase


def _frame_time_base(frame: Any) -> TimeBase:
    return _coerce_time_base(
        getattr(frame, "time_base", TimeBase()),
        getattr(frame, "time_base_num", None),
        getattr(frame, "time_base_den", None),
    )


def _timestamp(frame: Any) -> tuple[Optional[float], str, Optional[str]]:
    """Return seconds, timestamp basis, and an error reason if unavailable."""

    pts = getattr(frame, "pts", None)
    if pts is not None:
        try:
            value = _frame_time_base(frame).seconds(pts)
        except (TypeError, ValueError, OverflowError):
            value = float("nan")
        if math.isfinite(value):
            return value, "pts", None

    best_effort = getattr(frame, "best_effort_timestamp_time", None)
    if best_effort is not None:
        try:
            value = float(best_effort)
        except (TypeError, ValueError, OverflowError):
            value = float("nan")
        if math.isfinite(value):
            return value, "best_effort", None

    return None, "unavailable", "missing_timestamp"


def pts_to_seconds(pts: Number, time_base: TimeBaseLike = TimeBase()) -> float:
    """Convert a PTS using its rational time base without frame-index math."""

    return _coerce_time_base(time_base).seconds(pts)


def _is_successful(frame: Any) -> bool:
    if getattr(frame, "valid", None) is False:
        return False
    status = str(getattr(frame, "decode_status", "ok") or "ok").strip().lower()
    return status not in {
        "failed",
        "failure",
        "error",
        "decode_failed",
        "invalid",
        "unavailable",
    }


def _rate_to_float(rate: Any) -> Optional[float]:
    if rate is None:
        return None
    try:
        if isinstance(rate, str) and "/" in rate:
            num, den = rate.split("/", 1)
            value = float(num) / float(den)
        else:
            value = float(rate)
    except (TypeError, ValueError, ZeroDivisionError, OverflowError):
        return None
    return value if math.isfinite(value) and value > 0 else None


def _duration_seconds(frame: Any) -> tuple[Optional[float], Optional[str]]:
    """Return explicit duration seconds and a diagnostic for bad duration."""

    duration_seconds = getattr(frame, "duration_seconds", None)
    if duration_seconds is not None:
        try:
            value = float(duration_seconds)
        except (TypeError, ValueError, OverflowError):
            return None, "invalid_duration"
        if not math.isfinite(value):
            return None, "invalid_duration"
        if value > 0:
            return value, None
        return None, "negative_duration" if value < 0 else "non_positive_duration"

    duration = getattr(frame, "duration", None)
    if duration is None:
        return None, "missing_duration"
    try:
        base = getattr(frame, "duration_time_base", None)
        if base is None:
            base = _frame_time_base(frame)
        value = _coerce_time_base(base).seconds(duration)
    except (TypeError, ValueError, OverflowError):
        return None, "invalid_duration"
    if not math.isfinite(value):
        return None, "invalid_duration"
    if value > 0:
        return value, None
    return None, "negative_duration" if value < 0 else "non_positive_duration"


def _join_reasons(*parts: Optional[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(part for part in parts if part))


@dataclass(frozen=True, slots=True)
class VideoDisplayInterval:
    """One source-frame display interval.

    ``absolute_start``/``absolute_end`` are in the stream/container clock;
    ``start``/``end`` are shifted by the ``origin`` passed to the estimator.
    The default origin is zero, while :func:`build_media_clock` sets it to the
    common ``t0``.  ``candidate`` is false for failed frames and duplicate-PTS
    frames which are retained for provenance but cannot be selected.
    """

    source_frame_index: int
    source_pts: Optional[Number]
    time_base_num: int
    time_base_den: int
    absolute_start: Optional[float]
    absolute_end: Optional[float]
    start: Optional[float]
    end: Optional[float]
    duration: Optional[float]
    duration_basis: str
    timestamp_basis: str
    decode_status: str
    candidate: bool
    reasons: tuple[str, ...] = ()

    @property
    def eligible(self) -> bool:
        return bool(
            self.candidate
            and self.start is not None
            and self.end is not None
            and math.isfinite(self.start)
            and math.isfinite(self.end)
            and self.end > self.start
        )

    @property
    def is_candidate(self) -> bool:
        return self.candidate

    @property
    def display_start(self) -> Optional[float]:
        return self.start

    @property
    def display_end(self) -> Optional[float]:
        return self.end

    @property
    def reason(self) -> Optional[str]:
        return ";".join(self.reasons) if self.reasons else None


def estimate_video_display_intervals(
    frames: Iterable[VideoFrame],
    *,
    origin: float = 0.0,
    t0: Optional[float] = None,
    average_frame_rate: Any = None,
    stream_average_frame_rate: Any = None,
) -> list[VideoDisplayInterval]:
    """Estimate ``[start, end)`` display intervals from source frame records.

    The fallback duration follows the specified conservative rule: use the
    next different PTS interval, capped by the median positive adjacent PTS
    interval; use that median for the last frame; use the declared average
    frame rate only when there is a single valid timestamp.  Timestamped
    failed frames remain as boundaries, so a later successful frame cannot be
    pulled forward across a known failure.
    """

    if t0 is not None:
        origin = float(t0)
    origin = float(origin)
    frame_list = list(frames)
    raw: list[dict[str, Any]] = []
    for decode_order, frame in enumerate(frame_list):
        timestamp, timestamp_basis, timestamp_reason = _timestamp(frame)
        duration, duration_reason = _duration_seconds(frame)
        try:
            time_base = _frame_time_base(frame)
        except (TypeError, ValueError):
            time_base = TimeBase()
            timestamp = None
            timestamp_basis = "unavailable"
            timestamp_reason = "invalid_time_base"
        raw.append(
            {
                "frame": frame,
                "decode_order": decode_order,
                "timestamp": timestamp,
                "timestamp_basis": timestamp_basis,
                "timestamp_reason": timestamp_reason,
                "duration": duration,
                "duration_reason": duration_reason,
                "time_base": time_base,
                "successful": _is_successful(frame),
            }
        )

    timestamped = [item for item in raw if item["timestamp"] is not None]
    unique_timestamps = sorted({float(item["timestamp"]) for item in timestamped})
    positive_diffs = [
        right - left
        for left, right in zip(unique_timestamps, unique_timestamps[1:])
        if right - left > 0 and math.isfinite(right - left)
    ]
    median_delta = float(median(positive_diffs)) if positive_diffs else None
    rate = _rate_to_float(
        average_frame_rate if average_frame_rate is not None else stream_average_frame_rate
    )

    # The first successful frame in decode order is the only candidate for a
    # repeated PTS, even though output records are later sorted by display PTS.
    first_success_by_timestamp: dict[float, int] = {}
    for item_index, item in enumerate(raw):
        timestamp = item["timestamp"]
        if timestamp is not None and item["successful"]:
            first_success_by_timestamp.setdefault(float(timestamp), item_index)

    next_different: dict[float, Optional[float]] = {}
    for position, timestamp in enumerate(unique_timestamps):
        next_different[timestamp] = (
            unique_timestamps[position + 1]
            if position + 1 < len(unique_timestamps)
            else None
        )

    previous_decode_timestamp: Optional[float] = None
    for item in raw:
        timestamp = item["timestamp"]
        if timestamp is not None and previous_decode_timestamp is not None:
            if timestamp < previous_decode_timestamp:
                item["order_reason"] = "out_of_order_pts"
            else:
                item["order_reason"] = None
        else:
            item["order_reason"] = None
        if timestamp is not None:
            previous_decode_timestamp = float(timestamp)

    output: list[VideoDisplayInterval] = []
    for item_index, item in enumerate(raw):
        frame = item["frame"]
        timestamp = item["timestamp"]
        reasons = list(
            _join_reasons(item["timestamp_reason"], item["duration_reason"], item["order_reason"])
        )
        if not item["successful"]:
            reasons.append("decode_failed")
        if timestamp is None:
            output.append(
                VideoDisplayInterval(
                    source_frame_index=int(getattr(frame, "source_frame_index", item_index)),
                    source_pts=getattr(frame, "pts", None),
                    time_base_num=item["time_base"].numerator,
                    time_base_den=item["time_base"].denominator,
                    absolute_start=None,
                    absolute_end=None,
                    start=None,
                    end=None,
                    duration=None,
                    duration_basis="unavailable",
                    timestamp_basis=item["timestamp_basis"],
                    decode_status=str(getattr(frame, "decode_status", "ok") or "ok"),
                    candidate=False,
                    reasons=tuple(dict.fromkeys(reasons)),
                )
            )
            continue

        timestamp = float(timestamp)
        explicit_duration = item["duration"]
        duration_basis = "frame_duration" if explicit_duration is not None else "estimated"
        duration = explicit_duration
        if duration is None:
            next_timestamp = next_different.get(timestamp)
            next_delta = (
                float(next_timestamp - timestamp)
                if next_timestamp is not None and next_timestamp > timestamp
                else None
            )
            if next_delta is not None:
                duration = next_delta
                if median_delta is not None:
                    duration = min(duration, median_delta)
            elif median_delta is not None:
                duration = median_delta
            elif rate is not None:
                duration = 1.0 / rate
            else:
                duration = None
            if duration is None or duration <= 0 or not math.isfinite(duration):
                duration = None
                duration_basis = "unavailable"
                reasons.append("display_duration_unavailable")

        absolute_end = timestamp + duration if duration is not None else None
        if duration is not None and (not math.isfinite(absolute_end) or absolute_end <= timestamp):
            absolute_end = None
            duration = None
            duration_basis = "unavailable"
            reasons.append("invalid_display_interval")

        candidate = (
            item["successful"]
            and first_success_by_timestamp.get(timestamp) == item_index
        )
        if not candidate:
            if item["successful"] and timestamp in first_success_by_timestamp:
                reasons.append("duplicate_pts")
            elif not item["successful"]:
                # decode_failed is already present; this makes the reason
                # unambiguous for a failed frame sharing a successful PTS.
                reasons.append("not_a_candidate")

        output.append(
            VideoDisplayInterval(
                source_frame_index=int(getattr(frame, "source_frame_index", item_index)),
                source_pts=getattr(frame, "pts", None),
                time_base_num=item["time_base"].numerator,
                time_base_den=item["time_base"].denominator,
                absolute_start=timestamp,
                absolute_end=absolute_end,
                start=timestamp - origin,
                end=absolute_end - origin if absolute_end is not None else None,
                duration=duration,
                duration_basis=duration_basis,
                timestamp_basis=item["timestamp_basis"],
                decode_status=str(getattr(frame, "decode_status", "ok") or "ok"),
                candidate=candidate,
                reasons=tuple(dict.fromkeys(reasons)),
            )
        )

    # PTS order is the display order.  Python's sort is stable, preserving
    # decode order among duplicate timestamps for provenance.
    output.sort(
        key=lambda item: (
            float("inf") if item.absolute_start is None else item.absolute_start,
            item.source_frame_index,
        )
    )
    return output


# More explicit/short aliases used by callers.
build_video_display_intervals = estimate_video_display_intervals
estimate_video_intervals = estimate_video_display_intervals


def _audio_rate(frame: Any, fallback: Optional[int]) -> Optional[int]:
    value = getattr(frame, "sample_rate", None)
    if value in (None, 0) and fallback is not None:
        value = fallback
    try:
        value = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return value if value > 0 else None


@dataclass(frozen=True, slots=True)
class AudioFrameInterval:
    """One source audio frame mapped to a half-open common-time interval."""

    source_frame_index: int
    source_pts: Optional[Number]
    time_base_num: int
    time_base_den: int
    absolute_start: Optional[float]
    absolute_end: Optional[float]
    start: Optional[float]
    end: Optional[float]
    sample_rate: Optional[int]
    nb_samples: int
    timestamp_basis: str
    decode_status: str
    channels: Optional[int] = None
    channel_layout: Optional[str] = None
    reasons: tuple[str, ...] = ()

    @property
    def eligible(self) -> bool:
        return bool(
            self.start is not None
            and self.end is not None
            and self.sample_rate is not None
            and self.nb_samples >= 0
            and self.end > self.start
            and _is_successful_status(self.decode_status)
            and "decode_failed" not in self.reasons
        )

    @property
    def reason(self) -> Optional[str]:
        return ";".join(self.reasons) if self.reasons else None


def _is_successful_status(status: str) -> bool:
    return str(status or "ok").strip().lower() not in {
        "failed",
        "failure",
        "error",
        "decode_failed",
        "invalid",
        "unavailable",
    }


def build_audio_intervals(
    frames: Iterable[AudioFrame],
    *,
    origin: float = 0.0,
    t0: Optional[float] = None,
    sample_rate: Optional[int] = None,
) -> list[AudioFrameInterval]:
    """Build audio intervals from PTS and ``nb_samples / sample_rate``."""

    if t0 is not None:
        origin = float(t0)
    origin = float(origin)
    output: list[AudioFrameInterval] = []
    for position, frame in enumerate(list(frames)):
        try:
            time_base = _frame_time_base(frame)
        except (TypeError, ValueError):
            time_base = TimeBase()
        timestamp, timestamp_basis, timestamp_reason = _timestamp(frame)
        rate = _audio_rate(frame, sample_rate)
        try:
            nb_samples = int(getattr(frame, "nb_samples", 0))
        except (TypeError, ValueError, OverflowError):
            nb_samples = -1
        reasons = list(_join_reasons(timestamp_reason))
        if rate is None:
            reasons.append("invalid_sample_rate")
        if nb_samples < 0:
            reasons.append("invalid_nb_samples")
        if not _is_successful(frame):
            reasons.append("decode_failed")
        start = end = None
        if timestamp is not None and rate is not None and nb_samples >= 0:
            start = float(timestamp)
            end = start + nb_samples / rate
            if not math.isfinite(end) or end < start:
                start = end = None
                reasons.append("invalid_audio_interval")
        output.append(
            AudioFrameInterval(
                source_frame_index=int(getattr(frame, "source_frame_index", position)),
                source_pts=getattr(frame, "pts", None),
                time_base_num=time_base.numerator,
                time_base_den=time_base.denominator,
                absolute_start=start,
                absolute_end=end,
                start=start - origin if start is not None else None,
                end=end - origin if end is not None else None,
                sample_rate=rate,
                nb_samples=nb_samples,
                timestamp_basis=timestamp_basis,
                decode_status=str(getattr(frame, "decode_status", "ok") or "ok"),
                channels=getattr(frame, "channels", None),
                channel_layout=getattr(frame, "channel_layout", None),
                reasons=tuple(dict.fromkeys(reasons)),
            )
        )
    return output


@dataclass(frozen=True, slots=True)
class AudioContinuityIssue:
    """A concrete audio continuity/layout/PCM problem."""

    kind: str
    source_frame_index: Optional[int]
    next_source_frame_index: Optional[int]
    start: Optional[float]
    end: Optional[float]
    delta_seconds: Optional[float]
    message: str


@dataclass(frozen=True, slots=True)
class AudioContinuityReport:
    """Result of checking decoded audio frames in decode order."""

    continuous: bool
    tolerance_samples: int
    tolerance_seconds: float
    intervals: tuple[AudioFrameInterval, ...]
    issues: tuple[AudioContinuityIssue, ...]

    @property
    def audio_discontinuous(self) -> bool:
        return not self.continuous

    @property
    def is_continuous(self) -> bool:
        return self.continuous


def check_audio_continuity(
    frames: Iterable[AudioFrame],
    *,
    tolerance_samples: int = 2,
    sample_rate: Optional[int] = None,
    source_sr: Optional[int] = None,
) -> AudioContinuityReport:
    """Check adjacent decoded audio boundaries with a two-sample tolerance.

    The first frame is not compared with video or with zero: an entire audio
    track starting later than the video is a legitimate stream offset.  Only
    internal adjacent frame boundaries, sample-rate/layout changes, and
    non-finite decoded PCM are reported.
    """

    if source_sr is not None:
        if sample_rate is not None and int(sample_rate) != int(source_sr):
            raise ValueError("sample_rate and source_sr disagree")
        sample_rate = source_sr
    tolerance_samples = int(tolerance_samples)
    if tolerance_samples < 0:
        raise ValueError("tolerance_samples must be non-negative")

    frame_list = list(frames)
    intervals = build_audio_intervals(frame_list, sample_rate=sample_rate)
    default_rate = next((item.sample_rate for item in intervals if item.sample_rate), None)
    tolerance_seconds = (
        tolerance_samples / default_rate if default_rate is not None else 0.0
    )
    issues: list[AudioContinuityIssue] = []

    for frame, interval in zip(frame_list, intervals):
        if not _is_successful(frame):
            issues.append(
                AudioContinuityIssue(
                    kind="decode_failed",
                    source_frame_index=interval.source_frame_index,
                    next_source_frame_index=None,
                    start=interval.absolute_start,
                    end=interval.absolute_end,
                    delta_seconds=None,
                    message="audio frame decode failed; continuity is not established through it",
                )
            )
        pcm = getattr(frame, "pcm", None)
        if pcm is not None:
            try:
                finite = bool(np.isfinite(np.asarray(pcm, dtype=np.float64)).all())
            except (TypeError, ValueError, OverflowError):
                finite = False
            if not finite:
                issues.append(
                    AudioContinuityIssue(
                        kind="non_finite_pcm",
                        source_frame_index=interval.source_frame_index,
                        next_source_frame_index=None,
                        start=interval.absolute_start,
                        end=interval.absolute_end,
                        delta_seconds=None,
                        message="decoded PCM contains non-finite values",
                    )
                )
        if interval.sample_rate is None:
            issues.append(
                AudioContinuityIssue(
                    kind="invalid_sample_rate",
                    source_frame_index=interval.source_frame_index,
                    next_source_frame_index=None,
                    start=interval.absolute_start,
                    end=interval.absolute_end,
                    delta_seconds=None,
                    message="audio frame has no positive sample rate",
                )
            )
        if interval.nb_samples < 0:
            issues.append(
                AudioContinuityIssue(
                    kind="invalid_nb_samples",
                    source_frame_index=interval.source_frame_index,
                    next_source_frame_index=None,
                    start=interval.absolute_start,
                    end=interval.absolute_end,
                    delta_seconds=None,
                    message="audio frame has a negative sample count",
                )
            )

    for previous_frame, current_frame, previous, current in zip(
        frame_list,
        frame_list[1:],
        intervals,
        intervals[1:],
    ):
        if previous.sample_rate != current.sample_rate:
            issues.append(
                AudioContinuityIssue(
                    kind="sample_rate_change",
                    source_frame_index=previous.source_frame_index,
                    next_source_frame_index=current.source_frame_index,
                    start=previous.absolute_end,
                    end=current.absolute_start,
                    delta_seconds=None,
                    message=(
                        f"sample rate changed from {previous.sample_rate} to "
                        f"{current.sample_rate} between adjacent audio frames"
                    ),
                )
            )
        previous_layout = (previous.channels, previous.channel_layout)
        current_layout = (current.channels, current.channel_layout)
        if (
            previous_layout != (None, None)
            or current_layout != (None, None)
        ):
            if previous_layout != current_layout:
                issues.append(
                    AudioContinuityIssue(
                        kind="channel_layout_change",
                        source_frame_index=previous.source_frame_index,
                        next_source_frame_index=current.source_frame_index,
                        start=previous.absolute_end,
                        end=current.absolute_start,
                        delta_seconds=None,
                        message="channel layout changed between adjacent audio frames",
                    )
                )

        if previous.absolute_end is None or current.absolute_start is None:
            continue
        rate = current.sample_rate or previous.sample_rate or default_rate
        tolerance = tolerance_samples / rate if rate else 0.0
        delta = current.absolute_start - previous.absolute_end
        # PTS/time-base products are float64 values.  The mathematical
        # two-sample boundary must remain inclusive despite a representation
        # such as 0.0020000000000000018.
        boundary_epsilon = max(1e-12, abs(tolerance) * 1e-12)
        if abs(delta) > tolerance + boundary_epsilon:
            if delta > 0:
                kind = "gap"
                message = (
                    f"audio gap of {delta:.9f}s exceeds {tolerance_samples} samples "
                    f"at sample rate {rate}"
                )
            else:
                kind = "overlap"
                message = (
                    f"audio overlap of {-delta:.9f}s exceeds {tolerance_samples} samples "
                    f"at sample rate {rate}"
                )
            issues.append(
                AudioContinuityIssue(
                    kind=kind,
                    source_frame_index=previous.source_frame_index,
                    next_source_frame_index=current.source_frame_index,
                    start=previous.absolute_end,
                    end=current.absolute_start,
                    delta_seconds=delta,
                    message=message,
                )
            )

    return AudioContinuityReport(
        continuous=not issues,
        tolerance_samples=tolerance_samples,
        tolerance_seconds=tolerance_seconds,
        intervals=tuple(intervals),
        issues=tuple(issues),
    )


validate_audio_continuity = check_audio_continuity


@dataclass(frozen=True, slots=True)
class MediaClock:
    """A common media clock and its source interval tables."""

    t0: Optional[float]
    duration: float
    bin_edges: np.ndarray
    video_frame_intervals: tuple[VideoDisplayInterval, ...]
    audio_frame_intervals: tuple[AudioFrameInterval, ...]
    audio_continuity: AudioContinuityReport
    status: str
    failure_reason: Optional[str] = None
    container_start_time: Optional[float] = None

    @property
    def D(self) -> float:
        """The plan's common duration symbol."""

        return self.duration

    @property
    def duration_seconds(self) -> float:
        return self.duration

    @property
    def valid(self) -> bool:
        return self.status != "failed" and self.duration > 0

    @property
    def video_intervals(self) -> tuple[VideoDisplayInterval, ...]:
        return self.video_frame_intervals

    @property
    def audio_intervals(self) -> tuple[AudioFrameInterval, ...]:
        return self.audio_frame_intervals

    @property
    def video_candidates(self) -> tuple[VideoDisplayInterval, ...]:
        return tuple(item for item in self.video_frame_intervals if item.eligible)

    @property
    def audio_valid_intervals(self) -> tuple[AudioFrameInterval, ...]:
        return tuple(item for item in self.audio_frame_intervals if item.eligible)


def _shift_video(interval: VideoDisplayInterval, origin: float) -> VideoDisplayInterval:
    return replace(
        interval,
        start=interval.absolute_start - origin
        if interval.absolute_start is not None
        else None,
        end=interval.absolute_end - origin
        if interval.absolute_end is not None
        else None,
    )


def _shift_audio(interval: AudioFrameInterval, origin: float) -> AudioFrameInterval:
    return replace(
        interval,
        start=interval.absolute_start - origin
        if interval.absolute_start is not None
        else None,
        end=interval.absolute_end - origin
        if interval.absolute_end is not None
        else None,
    )


def build_media_clock(
    video_frames: Iterable[VideoFrame],
    audio_frames: Iterable[AudioFrame],
    *,
    average_frame_rate: Any = None,
    stream_average_frame_rate: Any = None,
    audio_sample_rate: Optional[int] = None,
    bins: int = 50,
    container_start_time: Optional[float] = None,
) -> MediaClock:
    """Build the common ``t0``, ``D`` and 50-bin edges from source records.

    ``container_start_time`` is retained as metadata only.  It never replaces
    the decoded PTS-derived duration.  The returned source tables include
    failed/duplicate records for provenance; only ``video_candidates`` and
    ``audio_valid_intervals`` contribute trusted ends to ``D``.
    """

    if bins <= 0:
        raise ValueError("bins must be positive")
    video_list = list(video_frames)
    audio_list = list(audio_frames)
    video_absolute = estimate_video_display_intervals(
        video_list,
        origin=0.0,
        average_frame_rate=(
            average_frame_rate
            if average_frame_rate is not None
            else stream_average_frame_rate
        ),
    )
    audio_absolute = build_audio_intervals(audio_list, origin=0.0, sample_rate=audio_sample_rate)
    continuity = check_audio_continuity(
        audio_list,
        tolerance_samples=2,
        sample_rate=audio_sample_rate,
    )

    # A valid display PTS establishes the stream start even if its display
    # duration is unusable.  Duration validity is required for D, not for t0.
    video_starts = [
        item.absolute_start
        for item in video_absolute
        if item.candidate and item.absolute_start is not None
    ]
    audio_starts = [
        item.absolute_start
        for item in audio_absolute
        if item.eligible and item.absolute_start is not None
    ]
    starts = [float(value) for value in (*video_starts, *audio_starts) if math.isfinite(value)]
    t0 = min(starts) if starts else None

    trusted_ends = [
        item.absolute_end
        for item in video_absolute
        if item.eligible and item.absolute_end is not None
    ]
    trusted_ends.extend(
        item.absolute_end
        for item in audio_absolute
        if item.eligible and item.absolute_end is not None
    )
    finite_ends = [float(value) for value in trusted_ends if math.isfinite(value)]

    failure_reason: Optional[str] = None
    if t0 is None:
        duration = 0.0
        status = "failed"
        failure_reason = "no_valid_stream_timestamp"
        clock_t0 = None
    elif not finite_ends:
        duration = 0.0
        status = "failed"
        failure_reason = "no_trusted_media_end"
        clock_t0 = t0
    else:
        end = max(finite_ends)
        duration = end - t0
        if not math.isfinite(duration) or duration <= 0:
            duration = 0.0
            status = "failed"
            failure_reason = "non_positive_media_duration"
            clock_t0 = t0
        else:
            status = "ok"
            clock_t0 = t0

    if status == "failed":
        bin_edges = np.zeros(bins + 1, dtype=np.float64)
    else:
        bin_edges = np.linspace(0.0, duration, bins + 1, dtype=np.float64)

    if clock_t0 is None:
        video_common = tuple(video_absolute)
        audio_common = tuple(audio_absolute)
    else:
        video_common = tuple(_shift_video(item, clock_t0) for item in video_absolute)
        audio_common = tuple(_shift_audio(item, clock_t0) for item in audio_absolute)

    return MediaClock(
        t0=clock_t0,
        duration=float(duration),
        bin_edges=bin_edges,
        video_frame_intervals=video_common,
        audio_frame_intervals=audio_common,
        audio_continuity=continuity,
        status=status,
        failure_reason=failure_reason,
        container_start_time=(
            float(container_start_time) if container_start_time is not None else None
        ),
    )


build_common_media_clock = build_media_clock
build_common_clock = build_media_clock


@dataclass(frozen=True, slots=True)
class SelectedVideoFrame:
    """One exported source frame and the 25 Hz queries it serves."""

    image_index: int
    filename: str
    source_frame_index: int
    source_pts: Optional[Number]
    time_base_num: int
    time_base_den: int
    start: float
    end: float
    duration: float
    duration_basis: str
    requested_grid_indices: list[int]
    timestamp_basis: str = "pts"
    decode_status: str = "ok"

    def as_mapping(self) -> dict[str, Any]:
        """Return the source mapping schema used by the PNG table."""

        return {
            "image_index": self.image_index,
            "filename": self.filename,
            "source_frame_index": self.source_frame_index,
            "source_pts": self.source_pts,
            "time_base_num": self.time_base_num,
            "time_base_den": self.time_base_den,
            "start": self.start,
            "end": self.end,
            "duration_basis": self.duration_basis,
            "requested_grid_indices": list(self.requested_grid_indices),
            "decode_status": self.decode_status,
        }


@dataclass(frozen=True, slots=True)
class VideoSelection:
    """25 Hz selection result, including intentionally missing queries."""

    duration: float
    target_hz: float
    grid_times: tuple[float, ...]
    selected_frames: tuple[SelectedVideoFrame, ...]
    missing_grid_indices: tuple[int, ...]
    grid_indices: tuple[int, ...] = ()

    @property
    def frames(self) -> tuple[SelectedVideoFrame, ...]:
        return self.selected_frames

    @property
    def missing_grid_times(self) -> tuple[float, ...]:
        index_to_time = dict(zip(self.grid_indices, self.grid_times))
        if not index_to_time:
            index_to_time = dict(enumerate(self.grid_times))
        return tuple(index_to_time[index] for index in self.missing_grid_indices)

    def __iter__(self) -> Iterator[SelectedVideoFrame]:
        return iter(self.selected_frames)

    def __len__(self) -> int:
        return len(self.selected_frames)

    def __getitem__(self, item: int) -> SelectedVideoFrame:
        return self.selected_frames[item]


def _as_video_intervals(
    frames_or_intervals: Iterable[Union[VideoFrame, VideoDisplayInterval]],
    *,
    origin: float,
    average_frame_rate: Any = None,
) -> list[VideoDisplayInterval]:
    values = list(frames_or_intervals)
    if all(isinstance(value, VideoDisplayInterval) for value in values):
        return values  # type: ignore[return-value]
    return estimate_video_display_intervals(
        values, origin=origin, average_frame_rate=average_frame_rate
    )  # type: ignore[arg-type]


def select_video_frames_25hz(
    frames_or_clock: Union[
        MediaClock,
        Iterable[Union[VideoFrame, VideoDisplayInterval]],
    ],
    duration: Optional[float] = None,
    *,
    D: Optional[float] = None,
    target_hz: float = 25.0,
    origin: float = 0.0,
    t0: Optional[float] = None,
    average_frame_rate: Any = None,
) -> VideoSelection:
    """Select source frames for ``q_r = r / target_hz``.

    A query is accepted only when ``start <= q < end`` for an eligible source
    frame.  No nearest-frame fallback or gap filling is performed.  A source
    frame selected by several queries is emitted once and receives all of its
    ``requested_grid_indices``.
    """

    if D is not None:
        if duration is not None and not math.isclose(float(duration), float(D)):
            raise ValueError("duration and D disagree")
        duration = D
    if isinstance(frames_or_clock, MediaClock):
        clock = frames_or_clock
        if duration is None:
            duration = clock.duration
        intervals = list(clock.video_frame_intervals)
    else:
        intervals = _as_video_intervals(
            frames_or_clock,
            origin=float(t0 if t0 is not None else origin),
            average_frame_rate=average_frame_rate,
        )
    if duration is None:
        ends = [item.end for item in intervals if item.eligible and item.end is not None]
        duration = max(ends, default=0.0)
    duration = float(duration)
    target_hz = float(target_hz)
    if not math.isfinite(duration) or duration < 0:
        raise ValueError("duration must be a finite non-negative number")
    if not math.isfinite(target_hz) or target_hz <= 0:
        raise ValueError("target_hz must be a positive finite number")

    grid_count = int(math.ceil(duration * target_hz))
    grid_pairs = tuple(
        (index, float(index / target_hz))
        for index in range(grid_count)
        if float(index / target_hz) < duration
    )
    grid_indices = tuple(index for index, _ in grid_pairs)
    grid_times = tuple(query_time for _, query_time in grid_pairs)
    requested_by_source: dict[int, list[int]] = {}
    interval_by_source: dict[int, VideoDisplayInterval] = {}

    candidates = sorted(
        (item for item in intervals if item.eligible),
        key=lambda item: (float(item.start), float(item.end), item.source_frame_index),
    )
    missing: list[int] = []
    for grid_index, query_time in grid_pairs:
        chosen: Optional[VideoDisplayInterval] = None
        for interval in candidates:
            # Half-open intervals ensure a query at an exact boundary goes to
            # the next source display interval, when one exists.
            # The small lower-bound epsilon only compensates for a PTS product
            # such as 6 * 0.1 becoming 0.6000000000000001.  The upper bound
            # remains strict (with epsilon subtracted), so a query at the
            # mathematical end is never assigned to the previous frame.
            boundary_epsilon = max(1e-12, abs(query_time) * 1e-12)
            if (
                query_time + boundary_epsilon >= interval.start
                and query_time < interval.end - boundary_epsilon
            ):
                chosen = interval
                break
        if chosen is None:
            missing.append(grid_index)
            continue
        source_id = chosen.source_frame_index
        requested_by_source.setdefault(source_id, []).append(grid_index)
        interval_by_source.setdefault(source_id, chosen)

    selected: list[SelectedVideoFrame] = []
    for image_index, source_id in enumerate(
        sorted(
            requested_by_source,
            key=lambda key: (
                float(interval_by_source[key].start),
                interval_by_source[key].source_frame_index,
            ),
        ),
        start=1,
    ):
        interval = interval_by_source[source_id]
        assert interval.start is not None and interval.end is not None
        selected.append(
            SelectedVideoFrame(
                image_index=image_index,
                filename=f"{image_index:06d}.png",
                source_frame_index=interval.source_frame_index,
                source_pts=interval.source_pts,
                time_base_num=interval.time_base_num,
                time_base_den=interval.time_base_den,
                start=float(interval.start),
                end=float(interval.end),
                duration=float(interval.end - interval.start),
                duration_basis=interval.duration_basis,
                requested_grid_indices=list(requested_by_source[source_id]),
                timestamp_basis=interval.timestamp_basis,
                decode_status=interval.decode_status,
            )
        )

    return VideoSelection(
        duration=duration,
        target_hz=target_hz,
        grid_times=grid_times,
        selected_frames=tuple(selected),
        missing_grid_indices=tuple(missing),
        grid_indices=grid_indices,
    )


select_frames_25hz = select_video_frames_25hz
select_video_frames = select_video_frames_25hz


def run_external_command(
    args: Sequence[Union[str, os.PathLike[str]]],
    *,
    timeout_s: float = 1800.0,
    cwd: Optional[Union[str, os.PathLike[str]]] = None,
) -> subprocess.CompletedProcess[str]:
    """Run an external media command with list arguments and no shell."""

    if isinstance(args, (str, bytes)) or not args:
        raise TypeError("args must be a non-empty sequence, not a shell command string")
    argv = [os.fspath(value) for value in args]
    return subprocess.run(
        argv,
        check=True,
        capture_output=True,
        text=True,
        shell=False,
        timeout=timeout_s,
        cwd=os.fspath(cwd) if cwd is not None else None,
    )


def ffprobe_json(
    media_path: Union[str, os.PathLike[str]],
    *,
    ffprobe_bin: Union[str, os.PathLike[str]] = "ffprobe",
    timeout_s: float = 1800.0,
) -> Mapping[str, Any]:
    """Safely invoke ffprobe and decode its raw JSON output."""

    result = run_external_command(
        [
            os.fspath(ffprobe_bin),
            "-v",
            "error",
            "-show_format",
            "-show_streams",
            "-of",
            "json",
            os.fspath(media_path),
        ],
        timeout_s=timeout_s,
    )
    value = json.loads(result.stdout)
    if not isinstance(value, Mapping):
        raise ValueError("ffprobe JSON root must be an object")
    return value


def ffmpeg_extract_audio(
    media_path: Union[str, os.PathLike[str]],
    output_path: Union[str, os.PathLike[str]],
    *,
    ffmpeg_bin: Union[str, os.PathLike[str]] = "ffmpeg",
    timeout_s: float = 1800.0,
) -> subprocess.CompletedProcess[str]:
    """Safely derive the mandated 16 kHz mono float32 WAV.

    The destination must not already exist, preventing an accidental overwrite
    of a source or a previous run.  No trimming, loudness normalisation, or VAD
    is added here.
    """

    destination = Path(output_path)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing audio: {destination}")
    return run_external_command(
        [
            os.fspath(ffmpeg_bin),
            "-nostdin",
            "-v",
            "error",
            "-i",
            os.fspath(media_path),
            "-map",
            "0:a:0",
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_f32le",
            os.fspath(destination),
        ],
        timeout_s=timeout_s,
    )


__all__ = [
    "AudioContinuityIssue",
    "AudioContinuityReport",
    "AudioFrame",
    "AudioFrameInterval",
    "AudioFrameRecord",
    "MediaClock",
    "RationalTimeBase",
    "SelectedVideoFrame",
    "TimeBase",
    "VideoDisplayInterval",
    "VideoFrame",
    "VideoFrameRecord",
    "VideoSelection",
    "build_audio_intervals",
    "build_common_clock",
    "build_common_media_clock",
    "build_media_clock",
    "build_video_display_intervals",
    "check_audio_continuity",
    "estimate_video_display_intervals",
    "estimate_video_intervals",
    "ffmpeg_extract_audio",
    "ffprobe_json",
    "pts_to_seconds",
    "run_external_command",
    "select_frames_25hz",
    "select_video_frames",
    "select_video_frames_25hz",
    "validate_audio_continuity",
    "probe_and_decode",
]


def _attached_picture(stream: Any) -> bool:
    disposition = getattr(stream, "disposition", None)
    value = getattr(disposition, "attached_pic", 0) if disposition is not None else 0
    try:
        return bool(int(value))
    except (TypeError, ValueError):
        return False


def _rotation_from_stream(stream: Any) -> int:
    raw = (getattr(stream, "metadata", {}) or {}).get("rotate", 0)
    try:
        value = int(round(float(raw))) % 360
    except (TypeError, ValueError):
        value = 0
    return value


def _serialisable_dataclass(value: Any) -> dict[str, Any]:
    from dataclasses import asdict
    row = asdict(value)
    for key, item in list(row.items()):
        if isinstance(item, tuple):
            row[key] = list(item)
    return row


def probe_and_decode(
    sample: Any,
    cfg: Mapping[str, Any],
    work_dir: Union[str, os.PathLike[str]],
    *,
    data_root: Union[str, os.PathLike[str]],
) -> dict[str, Any]:
    """Decode one real MP4, preserve source clocks, and export mandated media.

    This adapter is intentionally thin: all clock and selection decisions are
    delegated to the tested pure functions above.  It writes the raw ffprobe
    response, source frame tables, one WAV, the unique selected PNGs, and a
    compact ``media.json`` record consumed by later stages.
    """
    import av
    import soundfile as sf
    from PIL import Image

    from .storage import write_json, write_jsonl

    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    media_path = Path(data_root) / str(sample.video_relpath)
    if not media_path.is_file():
        raise FileNotFoundError(media_path)
    timeout_s = float(cfg["runtime"]["external_command_timeout_s"])
    probe = ffprobe_json(media_path, timeout_s=timeout_s)
    write_json(work / "ffprobe.json", probe)

    with av.open(str(media_path)) as container:
        video_streams = [s for s in container.streams if s.type == "video" and not _attached_picture(s)]
        audio_streams = [s for s in container.streams if s.type == "audio"]
        video_stream = video_streams[0] if video_streams else None
        audio_stream = audio_streams[0] if audio_streams else None
        video_stream_index = video_stream.index if video_stream is not None else None
        audio_stream_index = audio_stream.index if audio_stream is not None else None
        average_rate = getattr(video_stream, "average_rate", None) if video_stream is not None else None
        rotation = _rotation_from_stream(video_stream) if video_stream is not None else 0
        source_width = int(getattr(video_stream.codec_context, "width", 0)) if video_stream is not None else 0
        source_height = int(getattr(video_stream.codec_context, "height", 0)) if video_stream is not None else 0

    video_records: list[VideoFrame] = []
    if video_stream_index is not None:
        with av.open(str(media_path)) as container:
            stream = container.streams[video_stream_index]
            for index, frame in enumerate(container.decode(stream)):
                duration = getattr(frame, "duration", None)
                if duration is None:
                    duration = getattr(frame, "pkt_duration", None)
                video_records.append(VideoFrame(
                    source_frame_index=index, pts=frame.pts,
                    time_base=frame.time_base or stream.time_base,
                    duration=duration, decode_status="ok",
                ))

    audio_records: list[AudioFrame] = []
    if audio_stream_index is not None:
        with av.open(str(media_path)) as container:
            stream = container.streams[audio_stream_index]
            for index, frame in enumerate(container.decode(stream)):
                pcm = frame.to_ndarray()
                audio_records.append(AudioFrame(
                    source_frame_index=index, pts=frame.pts,
                    time_base=frame.time_base or stream.time_base,
                    sample_rate=int(frame.sample_rate or stream.codec_context.sample_rate),
                    nb_samples=int(frame.samples),
                    channels=len(frame.layout.channels) if frame.layout is not None else None,
                    channel_layout=frame.layout.name if frame.layout is not None else None,
                    pcm=pcm, decode_status="ok",
                ))

    clock = build_media_clock(
        video_records, audio_records, average_frame_rate=average_rate,
        audio_sample_rate=audio_records[0].sample_rate if audio_records else None,
        bins=int(cfg["pool"]["bins"]),
        container_start_time=float((probe.get("format") or {}).get("start_time", 0.0) or 0.0),
    )
    if not clock.valid:
        raise ValueError(f"media clock failed: {clock.failure_reason}")

    audio_rows = [_serialisable_dataclass(row) for row in clock.audio_frame_intervals]
    video_source_rows = [_serialisable_dataclass(row) for row in clock.video_frame_intervals]
    write_jsonl(work / "audio_frames.jsonl", audio_rows)
    write_jsonl(work / "video_source_frames.jsonl", video_source_rows)

    display_width, display_height = source_width, source_height
    if rotation in {90, 270}:
        display_width, display_height = source_height, source_width
    selection = select_video_frames_25hz(clock, target_hz=float(cfg["media"]["target_video_hz"]))
    selected_rows = [frame.as_mapping() for frame in selection.selected_frames]
    for row in selected_rows:
        row["display_width"] = display_width
        row["display_height"] = display_height
    frames_dir = work / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    wanted = {frame.source_frame_index: frame for frame in selection.selected_frames}
    if wanted:
        with av.open(str(media_path)) as container:
            stream = container.streams[video_stream_index]
            for index, frame in enumerate(container.decode(stream)):
                selected = wanted.get(index)
                if selected is None:
                    continue
                image = frame.to_image()
                if rotation == 90:
                    image = image.transpose(Image.Transpose.ROTATE_270)
                elif rotation == 180:
                    image = image.transpose(Image.Transpose.ROTATE_180)
                elif rotation == 270:
                    image = image.transpose(Image.Transpose.ROTATE_90)
                image.save(frames_dir / selected.filename, format="PNG")
    write_jsonl(work / "video_frames.jsonl", selected_rows)

    issues = [
        {"type": issue.kind, "start": issue.start, "end": issue.end, "evidence": issue.message}
        for issue in clock.audio_continuity.issues
    ]
    if selection.missing_grid_indices:
        issues.append({
            "type": "unsampled_video_grid", "start": None, "end": None,
            "evidence": f"{len(selection.missing_grid_indices)} target-grid positions had no source display interval",
        })
    if rotation not in {0, 90, 180, 270}:
        issues.append({"type": "unsupported_rotation", "evidence": str(rotation)})

    audio_path = work / "audio_16k.wav"
    audio_offset = 0.0
    derived_samples = 0
    if audio_records and clock.audio_continuity.continuous:
        if audio_path.exists():
            audio_path.unlink()
        ffmpeg_extract_audio(media_path, audio_path, timeout_s=timeout_s)
        wav, rate = sf.read(audio_path, dtype="float32", always_2d=False)
        if int(rate) != int(cfg["media"]["sample_rate"]) or np.asarray(wav).ndim != 1:
            raise ValueError("derived WAV is not 16kHz mono")
        if not np.isfinite(wav).all():
            raise ValueError("derived WAV contains non-finite PCM")
        audio_offset = float(clock.audio_frame_intervals[0].start or 0.0)
        derived_samples = int(len(wav))
        source_duration = sum(row.nb_samples / row.sample_rate for row in clock.audio_frame_intervals if row.sample_rate)
        if abs(derived_samples / int(rate) - source_duration) > float(cfg["media"]["audio_duration_tolerance_s"]):
            issues.append({
                "type": "audio_duration_mismatch", "start": audio_offset,
                "end": audio_offset + derived_samples / int(rate),
                "evidence": f"derived={derived_samples / int(rate):.6f}s source={source_duration:.6f}s",
            })
    elif audio_records:
        issues.append({"type": "audio_discontinuous", "evidence": "WAV/CTC/openSMILE intentionally disabled"})
    else:
        issues.append({"type": "audio_missing", "evidence": "no selected audio stream"})

    media = {
        "sample_id": sample.sample_id, "source_path": str(media_path),
        "video_stream_index": video_stream_index, "audio_stream_index": audio_stream_index,
        "other_stream_count": max(0, len(probe.get("streams", [])) - int(video_stream_index is not None) - int(audio_stream_index is not None)),
        "t0": clock.t0, "duration": clock.duration, "bin_edges": clock.bin_edges.tolist(),
        "audio_offset": audio_offset, "audio_discontinuous": not clock.audio_continuity.continuous,
        "audio_frame_count": len(audio_records), "video_frame_count": len(video_records),
        "derived_audio_samples": derived_samples,
        "selected_video_frames": len(selection.selected_frames),
        "missing_grid_indices": list(selection.missing_grid_indices),
        "source_width": source_width, "source_height": source_height,
        "display_width": display_width, "display_height": display_height,
        "rotation": rotation, "issues": issues,
    }
    write_json(work / "media.json", media)
    return media
