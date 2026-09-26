from q3.grouping import Unit
from q3.selection import select_units


def test_budget_and_intervals():
    units = [
        Unit("a", (0,), 0, 1, 1, 1),
        Unit("b", (1,), 1, 2, 1, 3),
        Unit("c", (2,), 2, 3, 1, 2),
    ]
    result = select_units(units, 2, objective="max", exact_cost=True)
    assert result.atoms == (1, 2)
    assert result.actual_cost == 2


def test_non_exact_selection_uses_largest_reachable_cost_first():
    """A high proxy at a smaller cost must not beat the filled budget."""
    units = [
        Unit("a", (0,), 0, 1, 2, 10.0),
        Unit("b", (1,), 1, 2, 3, 1.0),
    ]

    result = select_units(units, 3, objective="max", exact_cost=False)

    assert result.actual_cost == 3
    assert result.atoms == (1,)
    assert result.proxy_value == 1.0


def test_direction_selection_breaks_equal_proxy_by_minimum_cost():
    """The directional pass may use less than budget, but ties use less cost."""
    units = [
        Unit("expensive", (0,), 0, 1, 3, 2.0),
        Unit("cheap", (1,), 1, 2, 2, 2.0),
    ]

    result = select_units(
        units, 3, objective="max", exact_cost=False, fill_budget=False
    )

    assert result.actual_cost == 2
    assert result.atoms == (1,)
    assert result.proxy_value == 2.0


def _intervals_for_mask(units, mask):
    intervals = []
    start = None
    for index, selected in enumerate(mask + (False,)):
        if selected and start is None:
            start = index
        elif not selected and start is not None:
            intervals.append((units[start].lo, units[index - 1].hi))
            start = None
    return intervals


def _feasible_subsets(units, budget, max_intervals):
    for bits in range(1, 1 << len(units)):
        mask = tuple(bool(bits & (1 << index)) for index in range(len(units)))
        cost = sum(unit.cost for unit, selected in zip(units, mask) if selected)
        if cost > budget:
            continue
        intervals = _intervals_for_mask(units, mask)
        if len(intervals) <= max_intervals:
            yield mask, cost, sum(
                unit.proxy for unit, selected in zip(units, mask) if selected
            )


def test_small_exhaustive_budget_and_objective_invariants():
    """DP choices agree with exhaustive feasible subsets on a tiny instance."""
    units = [
        Unit("u0", (0,), 0, 1, 2, 1.5),
        Unit("u1", (1,), 1, 2, 1, -4.0),
        Unit("u2", (2,), 2, 3, 2, 2.0),
        Unit("u3", (3,), 3, 4, 3, -1.0),
    ]
    budget = 5
    feasible = list(_feasible_subsets(units, budget, max_intervals=3))

    for objective in ("max", "min"):
        for fill_budget in (True, False):
            candidates = feasible
            if fill_budget:
                largest_cost = max(cost for _, cost, _ in candidates)
                candidates = [
                    candidate for candidate in candidates if candidate[1] == largest_cost
                ]
            if objective == "max":
                best_proxy = max(proxy for _, _, proxy in candidates)
            else:
                best_proxy = min(proxy for _, _, proxy in candidates)
            candidates = [
                candidate for candidate in candidates if candidate[2] == best_proxy
            ]
            expected_cost = min(cost for _, cost, _ in candidates)

            result = select_units(
                units,
                budget,
                max_intervals=3,
                objective=objective,
                exact_cost=False,
                fill_budget=fill_budget,
            )

            assert result.actual_cost == expected_cost
            assert result.proxy_value == best_proxy
