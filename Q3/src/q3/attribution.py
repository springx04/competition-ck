from dataclasses import dataclass
import numpy as np

@dataclass
class View:
    delete_j: tuple
    logits: np.ndarray
    probs: np.ndarray
    score: float
    predicted_class: int
    delta_class: float
    delta_score_raw: float
    delta_score: float
    D: float
    TV: float
    class_changed: bool

def view_result(delete_j, original, perturbed, c_star):
    p0 = np.asarray(original.probs).reshape(-1)
    p = np.asarray(perturbed.probs).reshape(-1)
    s0 = float(np.asarray(original.score).reshape(-1)[0])
    score = float(np.asarray(perturbed.score).reshape(-1)[0])
    delta_class = float(p0[c_star] - p[c_star])
    delta_score_raw = s0 - score
    tv = 0.5 * float(np.abs(p0 - p).sum())
    return View(tuple(sorted(delete_j)), np.asarray(perturbed.logits).reshape(-1), p, score,
                int(np.argmax(p)), delta_class, delta_score_raw, delta_score_raw / 6.0,
                0.5 * abs(delta_class) + 0.5 * abs(delta_score_raw / 6.0), tv,
                int(np.argmax(p)) != c_star)

def shapley(subset_probs, subset_scores, c_star):
    probs = np.asarray(subset_probs, dtype=np.float64)
    scores = np.asarray(subset_scores, dtype=np.float64).reshape(8)
    values = np.column_stack((probs[:, c_star], scores / 6.0))
    phi = np.zeros((3, 2), dtype=np.float64)
    for modality in range(3):
        bit = 1 << modality
        for subset in range(8):
            if subset & bit:
                continue
            phi[modality] += {0: 1 / 3, 1: 1 / 6, 2: 1 / 3}[subset.bit_count()] * (values[subset | bit] - values[subset])
    return phi, phi.sum(axis=0) - (values[7] - values[0])

def modality_summary(views, observed, eps=1e-6):
    rows = []
    for modality in range(3):
        atoms = tuple(3 * t + modality for t in range(50) if observed[t, modality])
        rows.append(views.get(atoms))
    d = np.asarray([row.D if row else 0.0 for row in rows])
    total = float(d.sum())
    weights = (d / total).tolist() if total > eps else None
    primary = [] if weights is None else ["TAV"[i] for i, value in enumerate(d) if observed[:, i].any() and value >= d.max() - eps]
    return {"D": d.tolist(), "D_sum": total, "weights": weights, "primary_modalities": primary, "views": rows}
