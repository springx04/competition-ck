from collections import Counter
from q3.controls import ControlSpace, MatchedControlSpace
from q3.grouping import Unit


def test_rank_roundtrip():
    units = [
        Unit("a", (0,), 0, 1, 1),
        Unit("b", (1,), 1, 2, 1),
        Unit("c", (2,), 2, 3, 1),
    ]
    space = ControlSpace(units, (0,), 1)
    assert space.count() == 2
    assert space.unrank(space.rank([units[1]]))[0].unit_id == "b"


def _matched_bruteforce(units, target_atoms, intervals):
    target = tuple(sorted(target_atoms))
    target_lengths = Counter(b - a for a, b in intervals)
    target_costs = tuple(sum(atom % 3 == modality for atom in target) for modality in range(3))
    expected = set()

    for bits in range(1 << len(units)):
        chosen = [index for index in range(len(units)) if bits & (1 << index)]
        if not chosen:
            continue
        atoms = tuple(sorted(atom for index in chosen for atom in units[index].atoms))
        if atoms == target:
            continue

        runs = []
        start = previous = chosen[0]
        for index in chosen[1:]:
            if index != previous + 1:
                runs.append((start, previous))
                start = index
            previous = index
        runs.append((start, previous))
        lengths = Counter(
            units[end].hi - units[start].lo for start, end in runs
        )
        costs = tuple(
            sum(atom % 3 == modality for atom in atoms) for modality in range(3)
        )
        if lengths == target_lengths and costs == target_costs:
            expected.add(atoms)
    return expected


def test_matched_control_space_matches_small_exhaustive_space():
    """Only complete ordered runs with matching spans and modality costs survive."""
    units = [
        Unit("u0", (0,), 0, 1, 1),
        Unit("u1", (4,), 1, 2, 1),
        Unit("u2", (8,), 2, 3, 1),
        Unit("u3", (9,), 3, 4, 1),
        Unit("u4", (13,), 4, 5, 1),
        Unit("u5", (17,), 5, 6, 1),
        Unit("u6", (18,), 6, 7, 1),
        Unit("u7", (22,), 7, 8, 1),
    ]
    target_indices = (1, 2, 4)
    target_atoms = tuple(atom for index in target_indices for atom in units[index].atoms)
    intervals = [(1, 3), (4, 5)]
    expected = _matched_bruteforce(units, target_atoms, intervals)
    space = MatchedControlSpace(units, target_atoms, intervals)

    enumerated = [space.unrank(rank) for rank in range(space.total)]
    assert space.total == len(expected) + 1
    assert space.available == len(expected)
    assert len(set(enumerated)) == space.total
    assert set(enumerated) == expected | {tuple(sorted(target_atoms))}
    assert tuple(sorted(target_atoms)) not in space.sample(space.available, seed=11)
    assert space.target_rank is not None


def test_matched_control_space_sampling_is_bounded_deterministic_and_excludes_target():
    units = [
        Unit(str(index), (3 * index + index % 3,), index, index + 1, 1)
        for index in range(8)
    ]
    target_indices = (1, 2, 4)
    target_atoms = tuple(atom for index in target_indices for atom in units[index].atoms)
    space = MatchedControlSpace(units, target_atoms, [(1, 3), (4, 5)])

    all_controls = {
        space.unrank(rank) for rank in range(space.total)
    } - {tuple(sorted(target_atoms))}
    sampled = space.sample(100, seed=17)
    sampled_again = space.sample(100, seed=17)

    assert len(sampled) == len(all_controls)
    assert len(set(sampled)) == len(sampled)
    assert set(sampled) == all_controls
    assert sampled_again == sampled


def test_matched_control_space_uses_official_span_not_atom_count():
    """A run's geometry is its [lo, hi) span even when units have sparse atoms."""
    units = [
        Unit("u0", (0,), 0, 2, 1),
        Unit("u1", (3,), 2, 5, 1),
        Unit("u2", (6,), 5, 7, 1),
        Unit("u3", (9,), 7, 9, 1),
    ]
    target_atoms = units[1].atoms + units[2].atoms
    space = MatchedControlSpace(units, target_atoms, [(2, 7)])

    # u0+u1 is also a length-five run; u2+u3 spans only four units.  Matching
    # the official span, rather than the atom count, therefore yields one
    # control alongside the target.
    assert space.total == 2
    assert space.target_rank == 1
    assert space.sample(10, seed=0) == [(0, 3)]
