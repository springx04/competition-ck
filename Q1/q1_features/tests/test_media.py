"""Substantive, decoder-free tests for the Q1 media clock rules."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from q1_features.media import (  # noqa: E402
    AudioFrame,
    TimeBase,
    VideoFrame,
    build_media_clock,
    check_audio_continuity,
    estimate_video_display_intervals,
    select_video_frames_25hz,
)


def test_common_clock_uses_pts_and_keeps_audio_late_by_0_4_seconds() -> None:
    """A late audio stream is an offset, not an internal audio discontinuity."""

    video = [
        VideoFrame(source_frame_index=0, pts=0, time_base=(1, 1), duration=1),
    ]
    audio = [
        AudioFrame(
            source_frame_index=0,
            pts=400,
            time_base=(1, 1000),
            sample_rate=1000,
            nb_samples=600,
            pcm=np.zeros(600, dtype=np.float32),
        )
    ]

    clock = build_media_clock(video, audio)

    assert clock.status == "ok"
    assert clock.t0 == 0.0
    assert clock.D == 1.0
    assert clock.audio_intervals[0].start == 0.4
    assert clock.audio_intervals[0].end == 1.0
    assert clock.audio_continuity.continuous
    np.testing.assert_allclose(clock.bin_edges, np.linspace(0.0, 1.0, 51))


def test_audio_continuity_has_exact_two_sample_tolerance() -> None:
    base = dict(time_base=(1, 1000), sample_rate=1000, nb_samples=100)
    within_tolerance = [
        AudioFrame(source_frame_index=0, pts=0, **base),
        AudioFrame(source_frame_index=1, pts=102, **base),
    ]
    outside_tolerance = [
        AudioFrame(source_frame_index=0, pts=0, **base),
        AudioFrame(source_frame_index=1, pts=103, **base),
    ]

    assert check_audio_continuity(within_tolerance).continuous
    report = check_audio_continuity(outside_tolerance)
    assert not report.continuous
    assert any(issue.kind == "gap" for issue in report.issues)
    assert report.tolerance_seconds == 0.002


def test_video_intervals_use_pts_not_source_frame_index() -> None:
    frames = [
        VideoFrame(source_frame_index=10, pts=100, time_base=(1, 100), duration=2),
        VideoFrame(source_frame_index=11, pts=105, time_base=(1, 100), duration=2),
        VideoFrame(source_frame_index=12, pts=110, time_base=(1, 100), duration=2),
    ]

    intervals = estimate_video_display_intervals(frames)

    assert [item.source_frame_index for item in intervals] == [10, 11, 12]
    np.testing.assert_allclose([item.start for item in intervals], [1.0, 1.05, 1.10])
    # The source IDs are deliberately non-contiguous: no index/FPS clock can
    # produce these starts.
    np.testing.assert_allclose([item.end for item in intervals], [1.02, 1.07, 1.12])


def test_ten_fps_selection_exports_each_source_once_and_groups_queries() -> None:
    frames = [
        VideoFrame(
            source_frame_index=index,
            pts=index,
            time_base=(1, 10),
            duration=1,
        )
        for index in range(10)
    ]

    selection = select_video_frames_25hz(frames, D=1.0)

    assert len(selection.grid_times) == 25
    assert len(selection) == 10
    assert [item.source_frame_index for item in selection] == list(range(10))
    assert len({item.source_frame_index for item in selection}) == 10
    assert sum(len(item.requested_grid_indices) for item in selection) == 25
    assert selection[0].requested_grid_indices == [0, 1, 2]
    assert selection[9].requested_grid_indices == [23, 24]
    assert selection.missing_grid_indices == ()


def test_middle_failed_frame_leaves_a_real_gap_and_does_not_shift_later_pts() -> None:
    frames = [
        VideoFrame(source_frame_index=0, pts=0, time_base=(1, 25), duration=1),
        VideoFrame(
            source_frame_index=1,
            pts=1,
            time_base=(1, 25),
            duration=1,
            decode_status="failed",
        ),
        VideoFrame(source_frame_index=2, pts=2, time_base=(1, 25), duration=1),
        VideoFrame(source_frame_index=3, pts=3, time_base=(1, 25), duration=1),
    ]

    intervals = estimate_video_display_intervals(frames)
    by_source = {item.source_frame_index: item for item in intervals}
    assert by_source[2].start == 2 / 25
    assert by_source[2].start != 1 / 25
    assert not by_source[1].eligible
    assert "decode_failed" in by_source[1].reasons

    selection = select_video_frames_25hz(frames, D=4 / 25)
    assert [item.source_frame_index for item in selection] == [0, 2, 3]
    assert selection.missing_grid_indices == (1,)
    assert selection[1].requested_grid_indices == [2]


def test_best_effort_timestamp_is_explicit_fallback_and_invalid_clock_has_zero_edges() -> None:
    frame = VideoFrame(
        source_frame_index=7,
        pts=None,
        time_base=TimeBase(1, 25),
        best_effort_timestamp_time=0.4,
        duration_seconds=0.1,
    )
    intervals = estimate_video_display_intervals([frame])
    assert intervals[0].timestamp_basis == "best_effort"
    assert intervals[0].start == 0.4

    failed_clock = build_media_clock(
        [VideoFrame(source_frame_index=0, pts=None, decode_status="failed")],
        [],
    )
    assert failed_clock.status == "failed"
    assert failed_clock.D == 0.0
    assert failed_clock.bin_edges.shape == (51,)
    assert np.all(failed_clock.bin_edges == 0.0)
