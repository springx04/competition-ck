import numpy as np

def summarize_faithfulness(rows):
    values = np.asarray([row["target"] - row["random"] for row in rows if row.get("target") is not None and row.get("random") is not None], dtype=float)
    if values.size == 0: return {"n_total": len(rows), "n_eligible": 0}
    return {"n_total": len(rows), "n_eligible": int(values.size), "mean_G": float(values.mean()), "median_G": float(np.median(values)), "q25_G": float(np.quantile(values, .25)), "q75_G": float(np.quantile(values, .75)), "positive_fraction": float((values > 0).mean())}

def cluster_bootstrap(values, groups, repeats=1000, seed=0):
    values, groups = np.asarray(values, float), np.asarray(groups)
    unique, inverse = np.unique(groups, return_inverse=True)
    counts = np.bincount(inverse)
    sums = np.bincount(inverse, weights=values)
    rng = np.random.default_rng(seed)
    picked = rng.integers(0, len(unique), size=(repeats,len(unique)))
    return sums[picked].sum(axis=1) / counts[picked].sum(axis=1)
