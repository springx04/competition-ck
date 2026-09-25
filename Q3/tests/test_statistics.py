import numpy as np
from q3.statistics import cluster_bootstrap, summarize_faithfulness

def test_cluster_bootstrap_keeps_video_clusters():
    samples=cluster_bootstrap([1,3,10],["a","a","b"],repeats=20,seed=3)
    assert samples.shape==(20,)
    assert summarize_faithfulness([{"target":2,"random":1}])["mean_G"]==1
