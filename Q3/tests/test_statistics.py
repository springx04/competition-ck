import numpy as np
from q3.statistics import cluster_bootstrap, summarize_faithfulness


def test_cluster_bootstrap_keeps_video_clusters():
    samples = cluster_bootstrap([1, 3, 10], ["a", "a", "b"], repeats=20, seed=3)
    assert samples.shape == (20,)
    assert summarize_faithfulness([{"target": 2, "random": 1}])["mean_G"] == 1


def test_cluster_bootstrap_weights_resampled_clusters_by_sample_count():
    """A cluster draw carries all of that video's samples into the mean."""
    samples = cluster_bootstrap(
        [1, 3, 10], ["a", "a", "b"], repeats=5, seed=3
    )

    np.testing.assert_allclose(samples, [14 / 3, 2.0, 14 / 3, 10.0, 2.0])


def test_summarize_faithfulness_keeps_total_and_eligible_denominators():
    rows = [
        {"target": 4.0, "random": 1.0},
        {"target": None, "random": 2.0},
        {"target": 2.0, "random": None},
    ]

    summary = summarize_faithfulness(rows)

    assert summary["n_total"] == 3
    assert summary["n_eligible"] == 1
    assert summary["mean_G"] == 3.0
