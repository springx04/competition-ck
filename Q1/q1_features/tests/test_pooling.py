"""Small, model-free tests for the Q1 50-window pooling contract."""

from pathlib import Path
import sys

import numpy as np


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from q1_features.pooling import (  # noqa: E402
    NativeSeries,
    build_q1_masks,
    compute_usable_mask,
    pool50,
    q1_perturb_mask,
    valid_length,
)


def test_native_series_and_pool_output_have_spec_dtypes_and_shapes():
    series = NativeSeries(
        values=[[1.0, 2.0]],
        intervals=[[0.0, 1.0]],
        eligible=[True],
        source_ids=[9],
    )

    result = pool50(series, duration=50.0, kind="text")

    assert series.values.dtype == np.float32
    assert series.intervals.dtype == np.float64
    assert series.eligible.dtype == np.bool_
    assert series.source_ids.dtype == np.int64
    assert result.features.shape == (50, 2)
    assert result.features.dtype == np.float32
    assert result.observed_mask.shape == (50,)
    assert result.observed_mask.dtype == np.uint8
    assert result.coverage.shape == (50,)
    assert result.coverage.dtype == np.float32
    assert result.indptr.shape == (51,)
    assert result.indptr.dtype == np.int64
    assert result.source_ids.dtype == np.int64
    assert result.overlap_s.dtype == np.float64
    assert result.weights.dtype == np.float64
    assert result.valid_mask.dtype == np.uint8
    assert result.valid_length == np.int64(50)
    assert result.time_intervals.shape == (50, 2)
    assert result.perturb_mask.shape == (50, 1)
    assert np.all(result.perturb_mask == 0)


def test_empty_native_series_keeps_dimension_and_writes_zero_observation_state():
    result = pool50(NativeSeries.empty(3), duration=50.0, kind="text")

    assert result.features.shape == (50, 3)
    assert np.all(result.features == 0.0)
    assert np.all(result.observed_mask == 0)
    assert np.all(result.coverage == 0.0)
    assert np.array_equal(result.indptr, np.zeros(51, dtype=np.int64))


def test_overlapping_intervals_use_overlap_weights_but_union_coverage():
    # The first 50-window is [0, 1).  The two rows overlap in [0.2, 0.4].
    series = NativeSeries(
        values=[[2.0], [6.0]],
        intervals=[[0.0, 0.4], [0.2, 0.6]],
        eligible=[True, True],
        source_ids=[20, 10],
    )

    result = pool50(series, duration=50.0, kind="audio")

    # Both rows contribute 0.4 s, so the ordinary mean is 4.  The union is
    # [0, 0.6), not the sum of row lengths (0.8 s).
    assert np.isclose(result.features[0, 0], 4.0)
    assert result.observed_mask[0] == 1
    assert np.isclose(result.coverage[0], 0.6)
    assert result.coverage[0] <= 1.0
    assert np.array_equal(result.source_ids, np.array([10, 20], dtype=np.int64))
    assert np.allclose(result.overlap_s, [0.4, 0.4])
    assert np.allclose(result.weights, [0.5, 0.5])
    assert result.indptr[0] == 0 and result.indptr[1] == 2


def test_subword_averaging_is_source_level_before_equal_duration_pooling():
    # The first value is already the parent-word average of three subwords.
    # pool50 receives one row per word, hence equal-duration words get equal
    # source-level weights rather than one word receiving three subword votes.
    first_word = np.mean([1.0, 2.0, 3.0])
    series = NativeSeries(
        values=[[first_word], [6.0]],
        intervals=[[0.0, 1.0], [0.0, 1.0]],
        eligible=[True, True],
        source_ids=[100, 101],
    )

    result = pool50(series, duration=50.0, kind="text")

    assert np.isclose(result.features[0, 0], 4.0)
    assert np.allclose(result.weights, [0.5, 0.5])


def test_legal_all_zero_observation_remains_observed():
    series = NativeSeries(
        values=np.zeros((1, 25), dtype=np.float32),
        intervals=[[1.0, 2.0]],
        eligible=[True],
        source_ids=[4],
    )

    result = pool50(series, duration=50.0, kind="audio")

    assert result.observed_mask[1] == 1
    assert np.all(result.features[1] == 0.0)
    assert np.isclose(result.coverage[1], 1.0)
    assert result.indptr[2] - result.indptr[1] == 1
    assert result.observed_mask[0] == 0


def _vision_series(angle_values):
    values = np.zeros((len(angle_values), 22), dtype=np.float32)
    values[:, 17:22] = np.asarray(angle_values, dtype=np.float32)[:, None]
    return NativeSeries(
        values=values,
        intervals=np.tile(np.array([[0.0, 1.0]], dtype=np.float64), (len(angle_values), 1)),
        eligible=np.ones(len(angle_values), dtype=np.bool_),
        source_ids=np.arange(len(angle_values), dtype=np.int64),
    )


def _circular_distance(a, b):
    return abs((float(a) - float(b) + np.pi) % (2.0 * np.pi) - np.pi)


def test_visual_angles_wrap_at_plus_or_minus_179_degrees():
    series = _vision_series(np.deg2rad([179.0, -179.0]))

    result = pool50(series, duration=50.0, kind="vision")

    assert result.observed_mask[0] == 1
    assert np.isclose(result.coverage[0], 1.0)
    assert all(
        _circular_distance(result.features[0, dimension], np.pi) < 1e-5
        for dimension in range(17, 22)
    )


def test_opposite_visual_angles_make_the_whole_bin_unobserved():
    series = _vision_series([0.0, np.pi])

    result = pool50(series, duration=50.0, kind="vision")

    assert result.observed_mask[0] == 0
    assert np.all(result.features[0] == 0.0)
    assert result.coverage[0] == 0.0
    assert result.indptr[0] == result.indptr[1] == 0
    # Candidate evidence is retained separately, but is not accepted CSR.
    assert np.isclose(result.candidate_coverage[0], 1.0)
    assert np.allclose(result.candidate_mapping.weights, [0.5, 0.5])


def test_v_o_p_semantics_keep_observation_fact_when_manually_masked():
    observed = np.zeros((50, 3), dtype=np.uint8)
    observed[7] = [1, 1, 0]
    observed_before = observed.copy()
    valid, returned_observed, perturb, usable = build_q1_masks(10.0, observed)

    assert np.array_equal(valid, np.ones(50, dtype=np.uint8))
    assert np.array_equal(returned_observed, observed_before)
    assert np.array_equal(perturb, q1_perturb_mask())
    assert np.array_equal(usable, observed_before)
    assert valid_length(valid) == np.int64(50)

    manually_hidden = perturb.copy()
    manually_hidden[7, 1] = 1
    usable_after = compute_usable_mask(valid, observed, manually_hidden)
    assert observed[7, 1] == 1  # O is the original observation fact.
    assert usable_after[7, 1] == 0  # P hides it only for downstream use.
    assert usable_after[7, 0] == 1
    assert usable_after[7, 2] == 0
