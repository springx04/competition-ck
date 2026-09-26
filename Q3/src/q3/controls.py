from dataclasses import dataclass
import itertools
from functools import lru_cache
from collections import Counter
import random

@dataclass
class ControlSpace:
    units: list
    target_atoms: tuple
    target_cost: int
    def all(self):
        candidates = []
        for bits in itertools.product((0, 1), repeat=len(self.units)):
            chosen = [unit for unit, bit in zip(self.units, bits) if bit]
            if sum(unit.cost for unit in chosen) != self.target_cost: continue
            atoms = tuple(sorted(atom for unit in chosen for atom in unit.atoms))
            if atoms != tuple(sorted(self.target_atoms)): candidates.append(chosen)
        for chosen in sorted(candidates, key=lambda value: tuple(sorted(unit.unit_id for unit in value))):
            yield chosen
    def count(self): return sum(1 for _ in self.all())
    def rank(self, selection):
        key = tuple(sorted(unit.unit_id for unit in selection))
        return sum(tuple(sorted(unit.unit_id for unit in candidate)) < key for candidate in self.all())
    def unrank(self, rank):
        for candidate in self.all():
            if rank == 0: return candidate
            rank -= 1
        raise IndexError(rank)

def sample_controls(space, target_rank, n, rng):
    total = space.count(); ranks = list(range(total));
    if target_rank in ranks: ranks.remove(target_rank)
    rng.shuffle(ranks)
    return [space.unrank(rank) for rank in ranks[:min(n, len(ranks))]]


class MatchedControlSpace:
    """Exact count/unrank over canonical runs of complete ordered units.

    Runs have the target's official-span multiset and modality costs. At least
    one unselected unit separates runs. The target mask is excluded once.
    """
    def __init__(self, units, target_atoms, intervals):
        self.units = list(units)
        self.target = tuple(sorted(target_atoms))
        self.costs = tuple(sum(a % 3 == m for a in self.target) for m in range(3))
        self.lengths = tuple(sorted(Counter(b-a for a, b in intervals).items()))
        self.segments = []
        for i in range(len(self.units)):
            row = []
            atoms = []
            for j in range(i, len(self.units)):
                atoms.extend(self.units[j].atoms)
                length = self.units[j].hi - self.units[i].lo
                if length in dict(self.lengths):
                    costs = tuple(sum(a % 3 == m for a in atoms) for m in range(3))
                    row.append((j, length, costs, tuple(sorted(atoms))))
            self.segments.append(row)
        self._count = lru_cache(None)(self._count_impl)
        self.total = self._count(0, self.lengths, self.costs)
        self.target_rank = self._rank_target()
        self.available = self.total - int(self.target_rank is not None)

    def _branches(self, next_i, lengths, costs):
        for i in range(next_i, len(self.units)):
            for j, length, used, atoms in self.segments[i]:
                remaining = dict(lengths)
                if not remaining.get(length) or any(a > b for a, b in zip(used, costs)):
                    continue
                remaining[length] -= 1
                remaining = tuple((k, v) for k, v in sorted(remaining.items()) if v)
                cost_left = tuple(b-a for a, b in zip(used, costs))
                yield (j+2, remaining, cost_left), atoms

    def _count_impl(self, next_i, lengths, costs):
        if not lengths:
            return int(not any(costs))
        return sum(self._count(*state) for state, _ in self._branches(next_i, lengths, costs))

    def _rank_target(self):
        state = (0, self.lengths, self.costs)
        target = set(self.target)
        runs = []
        opened = False
        for unit in self.units:
            overlap = target.intersection(unit.atoms)
            if overlap and overlap != set(unit.atoms):
                return None
            if overlap:
                if not opened:
                    runs.append([])
                runs[-1].extend(unit.atoms)
                opened = True
            else:
                opened = False
        rank = 0
        for run in runs:
            for child, atoms in self._branches(*state):
                count = self._count(*child)
                if count and atoms == tuple(sorted(run)):
                    state = child
                    break
                rank += count
            else:
                return None
        return rank if not state[1] and not any(state[2]) else None

    def unrank(self, rank):
        if not 0 <= rank < self.total:
            raise IndexError(rank)
        state = (0, self.lengths, self.costs)
        atoms = []
        while state[1]:
            for child, picked in self._branches(*state):
                count = self._count(*child)
                if rank < count:
                    atoms.extend(picked)
                    state = child
                    break
                rank -= count
        return tuple(sorted(atoms))

    def sample(self, n, seed):
        rng = random.Random(int(seed))
        # Floyd's algorithm uses O(n) storage even when the space is enormous.
        ranks = set()
        for j in range(self.available - min(n, self.available), self.available):
            candidate = rng.randrange(j+1)
            ranks.add(j if candidate in ranks else candidate)
        result = []
        ordered_ranks = sorted(ranks)
        # A trial index must be an exchangeable draw, not the kth smallest
        # position. Independent order is also required when combining modalities.
        rng.shuffle(ordered_ranks)
        for rank in ordered_ranks:
            full_rank = rank + int(self.target_rank is not None and rank >= self.target_rank)
            result.append(self.unrank(full_rank))
        return result
