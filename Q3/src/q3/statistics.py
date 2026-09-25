import numpy as np

def summarize_faithfulness(rows):
    values = np.asarray([row["target"] - row["random"] for row in rows if row.get("target") is not None and row.get("random") is not None], dtype=float)
    if values.size == 0: return {"n_total": len(rows), "n_eligible": 0}
    return {"n_total": len(rows), "n_eligible": int(values.size), "mean_G": float(values.mean()), "median_G": float(np.median(values)), "q25_G": float(np.quantile(values, .25)), "q75_G": float(np.quantile(values, .75)), "positive_fraction": float((values > 0).mean())}

def cluster_bootstrap(values, groups, repeats=1000, seed=0):
    values, groups = np.asarray(values, float), np.asarray(groups)
    unique = np.unique(groups); rng = np.random.default_rng(seed); result = []
    for _ in range(repeats):
        picked = rng.choice(unique, len(unique), replace=True)
        result.append(np.mean([values[groups == group].mean() for group in picked]))
    return np.asarray(result)
