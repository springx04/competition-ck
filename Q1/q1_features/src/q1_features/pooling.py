"""Pure 50-window pooling utilities for Q1 native feature sequences.

The implementation in this module deliberately knows nothing about the feature
extractors.  Extractors provide a :class:`NativeSeries` whose rows already
carry their public ``eligible`` decision and their source-row id.  Pooling then
does only the time arithmetic specified in sections 13, 15 and 18 of the Q1
implementation note.

Times are seconds on the common media axis and all intervals are half-open
``[start, end)`` intervals.  Internal accumulation uses ``float64``; the
feature and coverage arrays returned to consumers are ``float32``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np


N_BINS = 50
MODALITIES: Tuple[str, str, str] = ("text", "audio", "vision")
VISION_ANGLE_SLICE = slice(17, 22)  # OpenFace dimensions 17--21, zero-based.
VISION_DIM = 22
ANGLE_RESULTANT_EPS = 1e-6


@dataclass
class NativeSeries:
    """A variable-length native sequence ready for temporal pooling.

    The four fields intentionally mirror the on-disk native representation:

    * ``values``: ``float32[n, d]``;
    * ``intervals``: ``float64[n, 2]`` with half-open ``[start, end)`` rows;
    * ``eligible``: boolean row-level usability decided by the extractor/audit;
    * ``source_ids``: stable ``int64`` row ids used by the reverse mapping.

    Invalid or non-finite rows are retained by this container so that an
    extractor can preserve its diagnostic row table.  ``pool50`` simply does
    not accept such rows as observations; it never turns them into a valid
    zero-valued observation.
    """

    values: np.ndarray
    intervals: np.ndarray
    eligible: np.ndarray
    source_ids: np.ndarray

    def __post_init__(self) -> None:
        values = np.ascontiguousarray(np.asarray(self.values, dtype=np.float32))
        intervals = np.ascontiguousarray(
            np.asarray(self.intervals, dtype=np.float64)
        )
        eligible = np.ascontiguousarray(np.asarray(self.eligible, dtype=np.bool_))
        source_ids = np.ascontiguousarray(
            np.asarray(self.source_ids, dtype=np.int64)
        )

        if values.ndim != 2:
            raise ValueError("values must have shape (n, d)")
        n = values.shape[0]
        if intervals.shape != (n, 2):
            raise ValueError("intervals must have shape (n, 2)")
        if eligible.shape != (n,):
            raise ValueError("eligible must have shape (n,)")
        if source_ids.shape != (n,):
            raise ValueError("source_ids must have shape (n,)")

        self.values = values
        self.intervals = intervals
        self.eligible = eligible
        self.source_ids = source_ids

    @property
    def n(self) -> int:
        """Number of native rows."""

        return int(self.values.shape[0])

    @property
    def d(self) -> int:
        """Feature dimension, including for an empty sequence."""

        return int(self.values.shape[1])

    @classmethod
    def empty(cls, dimension: int) -> "NativeSeries":
        """Create a correctly shaped empty native sequence."""

        if isinstance(dimension, bool) or int(dimension) != dimension or dimension < 0:
            raise ValueError("dimension must be a non-negative integer")
        dimension = int(dimension)
        return cls(
            values=np.empty((0, dimension), dtype=np.float32),
            intervals=np.empty((0, 2), dtype=np.float64),
            eligible=np.empty((0,), dtype=np.bool_),
            source_ids=np.empty((0,), dtype=np.int64),
        )


@dataclass
class CSRMapping:
    """CSR source mapping for one pooled modality.

    For bin ``k``, accepted contributions occupy
    ``[indptr[k], indptr[k + 1])``.  ``weights`` are normalized by the sum of
    accepted overlap seconds in that bin, while ``overlap_s`` retains the
    unnormalized seconds needed for audit/recalculation.
    """

    indptr: np.ndarray
    source_ids: np.ndarray
    overlap_s: np.ndarray
    weights: np.ndarray

    def __post_init__(self) -> None:
        indptr = np.ascontiguousarray(np.asarray(self.indptr, dtype=np.int64))
        source_ids = np.ascontiguousarray(
            np.asarray(self.source_ids, dtype=np.int64)
        )
        overlap_s = np.ascontiguousarray(
            np.asarray(self.overlap_s, dtype=np.float64)
        )
        weights = np.ascontiguousarray(np.asarray(self.weights, dtype=np.float64))

        if indptr.ndim != 1 or indptr.size == 0:
            raise ValueError("indptr must be a non-empty one-dimensional array")
        if source_ids.ndim != 1 or overlap_s.ndim != 1 or weights.ndim != 1:
            raise ValueError("CSR payload arrays must be one-dimensional")
        if not (source_ids.size == overlap_s.size == weights.size):
            raise ValueError("CSR payload arrays must have equal lengths")
        if int(indptr[0]) != 0 or np.any(indptr[1:] < indptr[:-1]):
            raise ValueError("indptr must be monotone and start at zero")
        if int(indptr[-1]) != source_ids.size:
            raise ValueError("indptr[-1] must equal the number of contributions")
        if not np.isfinite(overlap_s).all() or not np.isfinite(weights).all():
            raise ValueError("CSR overlap and weight arrays must be finite")
        if np.any(overlap_s <= 0) or np.any(weights <= 0):
            raise ValueError("CSR contributions must have positive overlap/weight")

        self.indptr = indptr
        self.source_ids = source_ids
        self.overlap_s = overlap_s
        self.weights = weights

    @property
    def n_bins(self) -> int:
        return int(self.indptr.size - 1)

    @property
    def indices(self) -> np.ndarray:
        """Alias useful to consumers that call CSR columns ``indices``."""

        return self.source_ids

    def __iter__(self) -> Iterator[np.ndarray]:
        yield self.indptr
        yield self.source_ids
        yield self.overlap_s
        yield self.weights

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, str):
            aliases = {
                "indptr": self.indptr,
                "source_ids": self.source_ids,
                "indices": self.source_ids,
                "overlap_s": self.overlap_s,
                "weights": self.weights,
            }
            try:
                return aliases[key]
            except KeyError as exc:
                raise KeyError(key) from exc
        return tuple(self)[key]

    def as_dict(self) -> Dict[str, np.ndarray]:
        return {
            "indptr": self.indptr,
            "source_ids": self.source_ids,
            "overlap_s": self.overlap_s,
            "weights": self.weights,
        }


@dataclass
class Pool50Result:
    """Result of pooling one modality.

    ``mapping`` contains only accepted contributions.  For the visual
    circular-mean degeneracy rule, the candidate fields retain the rejected
    contribution set and its candidate coverage for audit, while the primary
    feature/observed/coverage/mapping fields remain the Q1 output.
    """

    features: np.ndarray
    observed_mask: np.ndarray
    coverage: np.ndarray
    mapping: CSRMapping
    candidate_coverage: Optional[np.ndarray] = None
    candidate_mapping: Optional[CSRMapping] = None
    duration: float = 0.0
    kind: str = ""
    valid_mask: Optional[np.ndarray] = None
    perturb_mask: Optional[np.ndarray] = None
    usable_mask: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        features = np.ascontiguousarray(np.asarray(self.features, dtype=np.float32))
        observed = np.ascontiguousarray(
            _coerce_binary_mask(self.observed_mask, name="observed_mask")
        )
        coverage = np.ascontiguousarray(
            np.asarray(self.coverage, dtype=np.float32)
        )
        if features.ndim != 2:
            raise ValueError("features must have shape (n_bins, d)")
        n_bins = features.shape[0]
        if observed.shape != (n_bins,):
            raise ValueError("observed_mask must have shape (n_bins,)")
        if coverage.shape != (n_bins,):
            raise ValueError("coverage must have shape (n_bins,)")
        if not np.isfinite(features).all() or not np.isfinite(coverage).all():
            raise ValueError("pooled features and coverage must be finite")
        if np.any(coverage < 0) or np.any(coverage > 1):
            raise ValueError("coverage must be in [0, 1]")
        if self.mapping.n_bins != n_bins:
            raise ValueError("mapping and feature arrays must have the same bins")

        candidate_coverage = (
            coverage.copy()
            if self.candidate_coverage is None
            else np.ascontiguousarray(
                np.asarray(self.candidate_coverage, dtype=np.float32)
            )
        )
        candidate_mapping = (
            self.mapping if self.candidate_mapping is None else self.candidate_mapping
        )
        if candidate_coverage.shape != (n_bins,):
            raise ValueError("candidate_coverage must have shape (n_bins,)")
        if not np.isfinite(candidate_coverage).all() or np.any(
            (candidate_coverage < 0) | (candidate_coverage > 1)
        ):
            raise ValueError("candidate_coverage must be in [0, 1]")
        if candidate_mapping.n_bins != n_bins:
            raise ValueError("candidate mapping and feature arrays must match")

        duration = float(self.duration)
        if not np.isfinite(duration) or duration < 0:
            raise ValueError("duration must be finite and non-negative")

        valid = (
            valid_mask_for_duration(duration, n_bins=n_bins)
            if self.valid_mask is None
            else _coerce_binary_mask(self.valid_mask, name="valid_mask")
        )
        if valid.shape != (n_bins,):
            raise ValueError("valid_mask must have shape (n_bins,)")

        if self.perturb_mask is None:
            perturb = np.zeros((n_bins, 1), dtype=np.uint8)
        else:
            perturb = _coerce_binary_mask(self.perturb_mask, name="perturb_mask")
            if perturb.ndim == 1:
                if perturb.shape != (n_bins,):
                    raise ValueError("perturb_mask must have n_bins rows")
                perturb = perturb[:, None]
            if perturb.shape[0] != n_bins:
                raise ValueError("perturb_mask must have n_bins rows")

        usable = (
            compute_usable_mask(valid, observed, perturb)
            if self.usable_mask is None
            else _coerce_binary_mask(self.usable_mask, name="usable_mask")
        )

        self.features = features
        self.observed_mask = observed
        self.coverage = coverage
        self.candidate_coverage = candidate_coverage
        self.candidate_mapping = candidate_mapping
        self.duration = duration
        self.valid_mask = valid
        self.perturb_mask = perturb
        self.usable_mask = usable

    @property
    def values(self) -> np.ndarray:
        return self.features

    @property
    def n_bins(self) -> int:
        return int(self.features.shape[0])

    @property
    def valid_length(self) -> np.int64:
        return valid_length(self.valid_mask)

    @property
    def time_intervals(self) -> np.ndarray:
        """The common half-open window intervals used by this result."""

        return make_time_bins(self.duration, n_bins=self.n_bins)

    @property
    def intervals(self) -> np.ndarray:
        """Alias for :attr:`time_intervals`."""

        return self.time_intervals

    @property
    def pooled(self) -> np.ndarray:
        return self.features

    @property
    def observed(self) -> np.ndarray:
        return self.observed_mask

    @property
    def V(self) -> np.ndarray:
        return self.valid_mask

    @property
    def O(self) -> np.ndarray:
        return self.observed_mask

    @property
    def P(self) -> np.ndarray:
        return self.perturb_mask

    @property
    def U(self) -> np.ndarray:
        return self.usable_mask

    @property
    def csr(self) -> CSRMapping:
        return self.mapping

    @property
    def indptr(self) -> np.ndarray:
        return self.mapping.indptr

    @property
    def source_ids(self) -> np.ndarray:
        return self.mapping.source_ids

    @property
    def overlap_s(self) -> np.ndarray:
        return self.mapping.overlap_s

    @property
    def weights(self) -> np.ndarray:
        return self.mapping.weights

    def as_dict(self) -> Dict[str, Any]:
        return {
            "features": self.features,
            "observed_mask": self.observed_mask,
            "observed": self.observed_mask,
            "coverage": self.coverage,
            "time_intervals": self.time_intervals,
            "valid_length": self.valid_length,
            "mapping": self.mapping,
            "csr": self.mapping,
            "indptr": self.indptr,
            "source_ids": self.source_ids,
            "overlap_s": self.overlap_s,
            "weights": self.weights,
            "candidate_coverage": self.candidate_coverage,
            "candidate_mapping": self.candidate_mapping,
            "valid_mask": self.valid_mask,
            "V": self.valid_mask,
            "O": self.observed_mask,
            "P": self.perturb_mask,
            "U": self.usable_mask,
            "perturb_mask": self.perturb_mask,
            "usable_mask": self.usable_mask,
        }

    def __iter__(self) -> Iterator[Any]:
        """Allow the compact ``features, O, C, mapping = pool50(...)`` form."""

        yield self.features
        yield self.observed_mask
        yield self.coverage
        yield self.mapping

    def __len__(self) -> int:
        return 4

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, str):
            try:
                return self.as_dict()[key]
            except KeyError as exc:
                raise KeyError(key) from exc
        return tuple(self)[key]

    def keys(self) -> Tuple[str, ...]:
        return tuple(self.as_dict().keys())


def _coerce_binary_mask(mask: Any, *, name: str) -> np.ndarray:
    arr = np.asarray(mask)
    if arr.ndim == 0:
        raise ValueError(f"{name} must be an array, not a scalar")
    if np.any((arr != 0) & (arr != 1)):
        raise ValueError(f"{name} must contain only 0/1 values")
    return np.asarray(arr, dtype=np.uint8)


def _validate_bin_count(n_bins: int) -> int:
    if isinstance(n_bins, bool) or int(n_bins) != n_bins or int(n_bins) <= 0:
        raise ValueError("n_bins must be a positive integer")
    return int(n_bins)


def _positive_duration(duration: float) -> float:
    try:
        value = float(duration)
    except (TypeError, ValueError) as exc:
        raise ValueError("duration must be a finite positive number") from exc
    if not np.isfinite(value) or value <= 0:
        raise ValueError("duration must be a finite positive number")
    return value


def make_time_bins(duration: float, n_bins: int = N_BINS) -> np.ndarray:
    """Return ``[start, end)`` bins covering exactly ``[0, duration)``."""

    duration = _positive_duration(duration)
    n_bins = _validate_bin_count(n_bins)
    edges = np.linspace(0.0, duration, n_bins + 1, dtype=np.float64)
    edges[-1] = duration
    return np.column_stack((edges[:-1], edges[1:])).astype(np.float64, copy=False)


def valid_mask_for_duration(
    duration: float, n_bins: int = N_BINS
) -> np.ndarray:
    """Implement V semantics for a time structure.

    A positive finite duration establishes the 50 real time positions, so V is
    all ones.  A failed/unknown duration produces an all-zero V mask without
    inventing a time axis.
    """

    n_bins = _validate_bin_count(n_bins)
    try:
        duration_value = float(duration)
    except (TypeError, ValueError):
        duration_value = np.nan
    value = np.isfinite(duration_value) and duration_value > 0
    return np.full(n_bins, 1 if value else 0, dtype=np.uint8)


def q1_perturb_mask(
    n_bins: int = N_BINS, n_modalities: int = 3
) -> np.ndarray:
    """Return Q1's P mask: no artificial masking is applied in Q1."""

    n_bins = _validate_bin_count(n_bins)
    if isinstance(n_modalities, bool) or int(n_modalities) != n_modalities:
        raise ValueError("n_modalities must be an integer")
    n_modalities = int(n_modalities)
    if n_modalities <= 0:
        raise ValueError("n_modalities must be positive")
    return np.zeros((n_bins, n_modalities), dtype=np.uint8)


def compute_usable_mask(
    valid_mask: Any,
    observed_mask: Any,
    perturb_mask: Optional[Any] = None,
) -> np.ndarray:
    """Compute ``U = V[:, None] * O * (1 - P)`` with uint8 output.

    ``observed_mask`` may be one-dimensional for a single modality or
    two-dimensional for the required ``(50, 3)`` multimodal layout.  P is
    optional and defaults to Q1's all-zero mask.  This function never changes
    O: an artificially hidden observation remains an observed fact.
    """

    valid = _coerce_binary_mask(valid_mask, name="valid_mask")
    observed = _coerce_binary_mask(observed_mask, name="observed_mask")
    if valid.ndim != 1:
        raise ValueError("valid_mask must be one-dimensional")
    if observed.shape[0] != valid.shape[0]:
        raise ValueError("valid_mask and observed_mask must have the same rows")

    if perturb_mask is None:
        perturb = np.zeros_like(observed, dtype=np.uint8)
    else:
        perturb = _coerce_binary_mask(perturb_mask, name="perturb_mask")
        if perturb.ndim == 1 and observed.ndim == 2:
            if perturb.shape != (valid.size,):
                raise ValueError("perturb_mask must have one value per bin")
            perturb = np.broadcast_to(perturb[:, None], observed.shape)
        elif perturb.ndim == 2 and observed.ndim == 1:
            if perturb.shape != (valid.size, 1):
                raise ValueError("single-modality perturb_mask must have shape (n, 1)")
            perturb = perturb[:, 0]
        elif perturb.shape != observed.shape:
            raise ValueError("perturb_mask and observed_mask must have equal shape")

    valid_bool = valid.astype(bool)
    if observed.ndim == 2:
        valid_bool = valid_bool[:, None]
    return (
        (valid_bool & observed.astype(bool) & ~np.asarray(perturb, dtype=bool))
    ).astype(np.uint8)


def usable_mask(
    valid_mask: Any,
    observed_mask: Any,
    perturb_mask: Optional[Any] = None,
) -> np.ndarray:
    """Readable alias for :func:`compute_usable_mask`."""

    return compute_usable_mask(valid_mask, observed_mask, perturb_mask)


def build_q1_masks(
    duration: float,
    observed_mask: Any,
    perturb_mask: Optional[Any] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build the V/O/P/U arrays for a three-modality Q1 sample."""

    observed = _coerce_binary_mask(observed_mask, name="observed_mask")
    if observed.ndim != 2 or observed.shape[1] != 3:
        raise ValueError("Q1 observed_mask must have shape (n_bins, 3)")
    valid = valid_mask_for_duration(duration, n_bins=observed.shape[0])
    if perturb_mask is None:
        perturb = q1_perturb_mask(observed.shape[0], observed.shape[1])
    else:
        perturb = _coerce_binary_mask(perturb_mask, name="perturb_mask")
        if perturb.shape != observed.shape:
            raise ValueError("Q1 perturb_mask must have shape (n_bins, 3)")
    usable = compute_usable_mask(valid, observed, perturb)
    return valid, observed, perturb, usable


def valid_length(valid_mask: Any) -> np.int64:
    """Return the required ``int64`` count of real time windows."""

    valid = _coerce_binary_mask(valid_mask, name="valid_mask")
    if valid.ndim != 1:
        raise ValueError("valid_mask must be one-dimensional")
    return np.int64(valid.sum(dtype=np.int64))


def interval_union_length(intervals: Any) -> float:
    """Return the union length of finite, positive half-open intervals."""

    arr = np.asarray(intervals, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("intervals must have shape (n, 2)")
    finite = np.isfinite(arr).all(axis=1)
    positive = arr[:, 1] > arr[:, 0]
    arr = arr[finite & positive]
    if arr.size == 0:
        return 0.0
    order = np.lexsort((arr[:, 1], arr[:, 0]))
    ordered = arr[order]
    start = float(ordered[0, 0])
    end = float(ordered[0, 1])
    total = 0.0
    for next_start, next_end in ordered[1:]:
        next_start = float(next_start)
        next_end = float(next_end)
        if next_start <= end:
            if next_end > end:
                end = next_end
        else:
            total += end - start
            start, end = next_start, next_end
    return float(total + end - start)


def coverage_from_intervals(
    intervals: Any, window_start: float, window_end: float
) -> float:
    """Compute interval-union coverage inside one target window."""

    start = float(window_start)
    end = float(window_end)
    if not np.isfinite(start) or not np.isfinite(end) or end <= start:
        raise ValueError("window must be a finite positive interval")
    arr = np.asarray(intervals, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("intervals must have shape (n, 2)")
    clipped_start = np.maximum(arr[:, 0], start)
    clipped_end = np.minimum(arr[:, 1], end)
    clipped = np.column_stack((clipped_start, clipped_end))
    return float(np.clip(interval_union_length(clipped) / (end - start), 0.0, 1.0))


def _candidate_rows(
    series: NativeSeries, window_start: float, window_end: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    starts = series.intervals[:, 0]
    ends = series.intervals[:, 1]
    overlap = np.minimum(ends, window_end) - np.maximum(starts, window_start)
    valid_values = np.isfinite(series.values).all(axis=1)
    valid_intervals = np.isfinite(series.intervals).all(axis=1) & (ends > starts)
    mask = series.eligible & valid_values & valid_intervals & (overlap > 0.0)
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        return indices, np.empty((0,), dtype=np.float64), np.empty((0, 2), dtype=np.float64)
    overlap = np.asarray(overlap[indices], dtype=np.float64)
    clipped = np.column_stack(
        (
            np.maximum(starts[indices], window_start),
            np.minimum(ends[indices], window_end),
        )
    ).astype(np.float64, copy=False)
    return indices.astype(np.int64, copy=False), overlap, clipped


def _mapping_from_bins(
    contributions: Sequence[Sequence[Tuple[int, float, float]]],
) -> CSRMapping:
    indptr: List[int] = [0]
    source_ids: List[int] = []
    overlap_s: List[float] = []
    weights: List[float] = []

    for bin_contributions in contributions:
        # A valid native table has one row per source id.  Grouping here also
        # guarantees the documented no-duplicate-source invariant if a caller
        # supplies a malformed table with repeated ids.
        grouped: Dict[int, List[float]] = {}
        for source_id, overlap, weight in bin_contributions:
            key = int(source_id)
            if key not in grouped:
                grouped[key] = [0.0, 0.0]
            grouped[key][0] += float(overlap)
            grouped[key][1] += float(weight)
        for source_id in sorted(grouped):
            overlap, weight = grouped[source_id]
            if overlap <= 0.0 or weight <= 0.0:
                continue
            source_ids.append(source_id)
            overlap_s.append(overlap)
            weights.append(weight)
        indptr.append(len(source_ids))

    return CSRMapping(
        indptr=np.asarray(indptr, dtype=np.int64),
        source_ids=np.asarray(source_ids, dtype=np.int64),
        overlap_s=np.asarray(overlap_s, dtype=np.float64),
        weights=np.asarray(weights, dtype=np.float64),
    )


def _normalise_kind(kind: str) -> str:
    if not isinstance(kind, str):
        raise ValueError("kind must be one of 'text', 'audio', or 'vision'")
    normalized = kind.strip().lower()
    if normalized not in MODALITIES:
        raise ValueError("kind must be one of 'text', 'audio', or 'vision'")
    return normalized


def pool50(
    series: NativeSeries,
    duration: float,
    kind: str,
    *,
    n_bins: int = N_BINS,
    angle_r_threshold: float = ANGLE_RESULTANT_EPS,
) -> Pool50Result:
    """Pool a native sequence into relative-time windows.

    Ordinary dimensions use overlap-seconds weights.  For ``kind='vision'``,
    dimensions 0--16 (AUs and any ordinary leading dimensions) use the same
    weighted mean, while dimensions 17--21 use weighted circular means in
    radians.  If any circular resultant has ``R < angle_r_threshold``, the
    entire visual bin is made unavailable: primary features, O and accepted
    CSR contributions are all zero/empty.  Candidate coverage and candidate
    CSR remain available on the result for diagnosis.
    """

    if not isinstance(series, NativeSeries):
        raise TypeError("series must be a NativeSeries")
    duration = _positive_duration(duration)
    n_bins = _validate_bin_count(n_bins)
    kind = _normalise_kind(kind)
    if kind == "vision" and series.d < VISION_DIM:
        raise ValueError(
            "vision NativeSeries must have at least 22 dimensions; "
            "angles are dimensions 17--21"
        )
    try:
        angle_r_threshold = float(angle_r_threshold)
    except (TypeError, ValueError) as exc:
        raise ValueError("angle_r_threshold must be a finite non-negative number") from exc
    if not np.isfinite(angle_r_threshold) or angle_r_threshold < 0:
        raise ValueError("angle_r_threshold must be a finite non-negative number")

    bins = make_time_bins(duration, n_bins=n_bins)
    dimension = series.d
    features = np.zeros((n_bins, dimension), dtype=np.float32)
    observed = np.zeros((n_bins,), dtype=np.uint8)
    coverage = np.zeros((n_bins,), dtype=np.float32)
    candidate_coverage = np.zeros((n_bins,), dtype=np.float32)
    accepted_contributions: List[List[Tuple[int, float, float]]] = [
        [] for _ in range(n_bins)
    ]
    candidate_contributions: List[List[Tuple[int, float, float]]] = [
        [] for _ in range(n_bins)
    ]

    for bin_index, (window_start, window_end) in enumerate(bins):
        indices, overlap, clipped = _candidate_rows(
            series, float(window_start), float(window_end)
        )
        if indices.size == 0:
            continue

        z = float(np.sum(overlap, dtype=np.float64))
        if not np.isfinite(z) or z <= 0.0:
            continue

        weights = overlap / z
        union = interval_union_length(clipped)
        candidate_coverage[bin_index] = np.float32(
            np.clip(union / (float(window_end) - float(window_start)), 0.0, 1.0)
        )
        candidate_contributions[bin_index] = [
            (
                int(series.source_ids[row_index]),
                float(row_overlap),
                float(weight),
            )
            for row_index, row_overlap, weight in zip(indices, overlap, weights)
        ]

        values = series.values[indices].astype(np.float64, copy=False)
        pooled = np.sum(values * weights[:, None], axis=0, dtype=np.float64)

        if kind == "vision":
            angles = values[:, VISION_ANGLE_SLICE]
            sine_sum = np.sum(np.sin(angles) * weights[:, None], axis=0, dtype=np.float64)
            cosine_sum = np.sum(np.cos(angles) * weights[:, None], axis=0, dtype=np.float64)
            resultant = np.hypot(sine_sum, cosine_sum)
            if np.any(resultant < angle_r_threshold):
                # The candidate data above is intentionally retained for audit,
                # but it is not an accepted observation or accepted CSR entry.
                continue
            pooled[VISION_ANGLE_SLICE] = np.arctan2(sine_sum, cosine_sum)

        features[bin_index] = pooled.astype(np.float32)
        observed[bin_index] = np.uint8(1)
        coverage[bin_index] = candidate_coverage[bin_index]
        accepted_contributions[bin_index] = candidate_contributions[bin_index]

    mapping = _mapping_from_bins(accepted_contributions)
    candidate_mapping = _mapping_from_bins(candidate_contributions)
    valid = valid_mask_for_duration(duration, n_bins=n_bins)
    perturb = np.zeros((n_bins, 1), dtype=np.uint8)
    single_usable = compute_usable_mask(valid, observed, perturb)
    return Pool50Result(
        features=features,
        observed_mask=observed,
        coverage=coverage,
        mapping=mapping,
        candidate_coverage=candidate_coverage,
        candidate_mapping=candidate_mapping,
        duration=duration,
        kind=kind,
        valid_mask=valid,
        perturb_mask=perturb,
        usable_mask=single_usable,
    )


__all__ = [
    "ANGLE_RESULTANT_EPS",
    "CSRMapping",
    "MODALITIES",
    "N_BINS",
    "NativeSeries",
    "Pool50Result",
    "VISION_ANGLE_SLICE",
    "VISION_DIM",
    "build_q1_masks",
    "compute_usable_mask",
    "coverage_from_intervals",
    "interval_union_length",
    "make_time_bins",
    "pool50",
    "q1_perturb_mask",
    "usable_mask",
    "valid_length",
    "valid_mask_for_duration",
]
