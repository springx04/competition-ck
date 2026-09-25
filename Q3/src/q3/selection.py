from dataclasses import dataclass

@dataclass
class Selection:
    units: list
    target_budget: int
    actual_cost: int
    intervals: list
    proxy_value: float
    status: str = "complete"
    reason: str | None = None
    @property
    def atoms(self): return tuple(sorted(atom for unit in self.units for atom in unit.atoms))
    @property
    def counts_by_modality(self): return [sum(atom % 3 == m for atom in self.atoms) for m in range(3)]

def budget_count(n, pct):
    return 0 if n == 0 else min(n, max(1, (pct * n + 50) // 100))

def select_units(units, budget, max_intervals=3, objective="max", exact_cost=False):
    units = list(units)
    states = {(0, 0, 0): (0.0, [], [])}
    for unit in units:
        updated = {}
        for (cost, intervals_opened, last), (value, chosen, intervals) in states.items():
            skip_key = (cost, intervals_opened, 0)
            previous = updated.get(skip_key)
            candidate_key = (value, len(intervals), sum(b-a for a,b in intervals), tuple(a for x in chosen for a in x.atoms))
            previous_key = None if previous is None else (previous[0], len(previous[2]), sum(b-a for a,b in previous[2]), tuple(a for x in previous[1] for a in x.atoms))
            if previous is None or _prefer(candidate_key, previous_key, objective):
                updated[skip_key] = (value, chosen, intervals)
            new_cost = cost + unit.cost
            if new_cost > budget:
                continue
            new_count = intervals_opened + (0 if last else 1)
            if new_count > max_intervals:
                continue
            new_intervals = list(intervals)
            if not last:
                new_intervals.append((unit.lo, unit.hi))
            else:
                new_intervals[-1] = (new_intervals[-1][0], unit.hi)
            key = (new_cost, new_count, 1)
            candidate = (value + unit.proxy, chosen + [unit], new_intervals)
            previous = updated.get(key)
            candidate_key = (candidate[0], len(candidate[2]), sum(b-a for a,b in candidate[2]), tuple(a for x in candidate[1] for a in x.atoms))
            previous_key = None if previous is None else (previous[0], len(previous[2]), sum(b-a for a,b in previous[2]), tuple(a for x in previous[1] for a in x.atoms))
            if previous is None or _prefer(candidate_key, previous_key, objective):
                updated[key] = candidate
        states = updated
    candidates = [v for (cost, count, last), v in states.items() if (cost == budget if exact_cost else 0 < cost <= budget)]
    if not candidates:
        return Selection([], budget, 0, [], 0.0, "not_applicable", "budget_unreachable")
    best = candidates[0]
    for candidate in candidates[1:]:
        ck = (candidate[0], len(candidate[2]), sum(b-a for a,b in candidate[2]), tuple(a for x in candidate[1] for a in x.atoms))
        bk = (best[0], len(best[2]), sum(b-a for a,b in best[2]), tuple(a for x in best[1] for a in x.atoms))
        if _prefer(ck, bk, objective):
            best = candidate
    return Selection(best[1], budget, sum(unit.cost for unit in best[1]), best[2], best[0])

def _prefer(candidate, previous, objective):
    if previous is None:
        return True
    if candidate[0] != previous[0]:
        return candidate[0] > previous[0] if objective == "max" else candidate[0] < previous[0]
    return candidate[1:] < previous[1:]
