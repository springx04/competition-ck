from dataclasses import dataclass
import itertools

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
