from dataclasses import dataclass
import re

@dataclass(frozen=True)
class Unit:
    unit_id: str
    atoms: tuple
    lo: int
    hi: int
    cost: int
    proxy: float = 0.0
    modality: str | None = None

def singleton_units(observed, modality):
    m = "TAV".index(modality)
    return [Unit(f"{modality}.t{t}", (3 * t + m,), t, t + 1, 1, 0.0, modality) for t in range(50) if observed[t, m]]

def block3_units(observed, modality):
    m = "TAV".index(modality); result = []
    for start in range(0, 50, 3):
        ts = [t for t in range(start, min(start + 3, 50)) if observed[t, m]]
        if ts:
            result.append(Unit(f"{modality}.b{start:02d}", tuple(3 * t + m for t in ts), start, max(ts) + 1, len(ts), 0.0, modality))
    return result

def center3_atoms(observed, modality, center):
    m = "TAV".index(modality)
    return tuple(3 * t + m for t in range(max(0, center - 1), min(50, center + 2)) if observed[t, m])

def text_word_units(raw_text, offsets, observed):
    words = [(match.start(), match.end()) for match in re.finditer(r"\S+", raw_text or "")]
    result = []
    for word_id, (start, end) in enumerate(words):
        ts = [t for t in range(50) if observed[t, 0] and t < len(offsets) and max(start, offsets[t][0]) < min(end, offsets[t][1])]
        if ts:
            result.append(Unit(f"T.word{word_id}", tuple(3 * t for t in ts), min(ts), max(ts) + 1, len(ts), 0.0, "T"))
    return result
