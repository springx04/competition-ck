"""Continuous local corruption and fixed evaluation descriptors."""
from dataclasses import dataclass
from itertools import product
import math

import numpy as np
import torch


MODALITIES = {"T": 0, "A": 1, "V": 2}


@dataclass
class Perturbation:
    P: torch.Tensor  # B,T,3, true only for newly removed observations
    descriptors: list[dict]


def _span(j: torch.Tensor, rho: float, position: str, rng=None) -> tuple[int, int]:
    indices = torch.where(j)[0]
    if rho <= 0 or indices.numel() == 0:
        return 0, 0
    a, b = int(indices[0]), int(indices[-1]) + 1
    length = b - a
    width = min(length, max(1, math.floor(rho * length + 0.5)))
    if position == "front":
        start = a
    elif position == "middle":
        start = a + (length - width) // 2
    elif position == "rear":
        start = b - width
    elif position == "random":
        if rng is None:
            raise ValueError("random position requires rng")
        start = int(rng.integers(a, b - width + 1))
    else:
        raise ValueError(f"unknown position: {position}")
    return start, start + width


def make_span_mask(raw_batch: dict, state0, descriptors: list[dict]) -> Perturbation:
    U, J = state0.U, state0.J
    if len(descriptors) != U.shape[0]:
        raise ValueError("one descriptor required per sample")
    P = torch.zeros_like(U)
    resolved = []
    for i, descriptor in enumerate(descriptors):
        pattern = descriptor.get("pattern", "")
        rho = float(descriptor.get("rho", 0))
        positions = descriptor.get("positions")
        if positions is None:
            positions = {c: descriptor.get("position", "middle") for c in pattern}
        intervals = {}
        synchronized = None
        for c in pattern:
            if c not in MODALITIES:
                raise ValueError(f"unknown modality: {c}")
            if "intervals" in descriptor and c in descriptor["intervals"]:
                start, end = descriptor["intervals"][c]
            elif "positions" not in descriptor and synchronized is not None:
                start, end = synchronized
            else:
                start, end = _span(J[i], rho, positions[c], descriptor.get("rng"))
                if "positions" not in descriptor:
                    synchronized = start, end
            intervals[c] = [int(start), int(end)]
            if end > start:
                P[i, start:end, MODALITIES[c]] = U[i, start:end, MODALITIES[c]]
        out = {k: v for k, v in descriptor.items() if k != "rng"}
        out["intervals"] = intervals
        out["new_count"] = {c: int(P[i, :, MODALITIES[c]].sum()) for c in pattern}
        out["no_new_damage"] = not bool(P[i].any())
        selected = [MODALITIES[c] for c in pattern]
        observed_selected = U[i, :, selected].sum() if selected else 0
        out["actual_rate"] = (float(P[i, :, selected].sum() / observed_selected)
                              if selected and observed_selected else 0.0)
        resolved.append(out)
    return Perturbation(P=P, descriptors=resolved)


def apply_span(raw_batch: dict, perturbation: Perturbation) -> dict:
    result = {k: v.clone() if isinstance(v, torch.Tensor) else list(v) if isinstance(v, list) else v
              for k, v in raw_batch.items()}
    P = perturbation.P
    result["input_ids"][P[:, :, 0]] = 103
    result["audio"][P[:, :, 1]] = 0
    result["vision"][P[:, :, 2]] = 0
    return result


def sample_train_descriptor(seed: int, epoch: int, sample_index: int, uniform_spans: bool = False,
                            total_epochs: int = 60, warmup_epochs: int = 5,
                            stress_text: bool = False) -> dict:
    rng = np.random.default_rng(np.random.SeedSequence([seed, epoch, sample_index]))
    if epoch <= warmup_epochs:
        return {"pattern": "", "rho": 0, "position": "middle"}
    progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
    if stress_text:
        pattern = str(rng.choice(["T", "T", "T", "TA", "TV", "AV", "A", "V"]))
        rho = float(rng.uniform(.35, .85))
        return {"pattern": pattern, "rho": rho, "position": "random", "rng": rng}
    phase = int(rng.choice([1, 2, 3], p=[10/55, 15/55, 30/55])) if uniform_spans else (1 if progress <= 10/55 else 2 if progress <= 25/55 else 3)
    if rng.random() < 0.2:
        return {"pattern": "", "rho": 0, "position": "middle"}
    if phase == 1:
        pattern = str(rng.choice(list("TAV")))
        rho = float(rng.uniform(.05, .20))
    else:
        dual = rng.random() < (.2 if phase == 2 else .4)
        pattern = str(rng.choice(["TA", "TV", "AV"] if dual else list("TAV")))
        rho = float(rng.uniform(.10, .50 if phase == 2 else .80))
    if len(pattern) == 2 and phase == 3 and rng.random() < .1:
        return {"pattern": pattern, "rho": rho, "positions": {c: "random" for c in pattern}, "rng": rng}
    return {"pattern": pattern, "rho": rho, "position": "random", "rng": rng}


def evaluation_grid(stress: bool = False) -> list[dict]:
    if not stress:
        return [{"pattern": p, "position": pos, "rho": rho,
                 "name": f"{p}_{pos}_{rho:.1f}"}
                for p, pos, rho in product(["T", "A", "V", "TA", "TV", "AV"],
                                           ["front", "middle", "rear"], [.2, .4, .6, .8])]
    grid = [{"pattern": c, "position": "middle", "rho": 1.0, "name": f"{c}_whole"} for c in "TAV"]
    grid += [{"pattern": "TAV", "position": "middle", "rho": r, "name": f"TAV_middle_{r:.1f}"} for r in (.4, .8)]
    grid += [{"pattern": p, "positions": {p[0]: "front", p[1]: "rear"}, "rho": .4,
              "name": f"{p}_staggered_0.4"} for p in ("TA", "TV", "AV")]
    return grid
