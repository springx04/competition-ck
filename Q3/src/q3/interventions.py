import numpy as np

def atom_index(t, m):
    return 3 * t + m

def mask_from_atoms(atoms, shape=(50, 3)):
    mask = np.zeros(shape, dtype=bool)
    for j in atoms:
        mask[j // 3, j % 3] = True
    return mask

def apply_delete(raw, delete_mask, original_u):
    out = {k: (v.clone() if hasattr(v, "clone") else np.array(v, copy=True)) for k, v in raw.items()}
    mask = np.asarray(delete_mask, dtype=bool)
    observed = np.asarray(original_u, dtype=bool)
    if mask.shape != observed.shape or np.any(mask & ~observed):
        raise ValueError("delete mask must be a subset of original observations")
    if mask.ndim != 2:
        raise ValueError("apply_delete expects one sample mask with shape (50,3)")
    for modality, key in enumerate(("input_ids", "audio", "vision")):
        indices = np.where(mask[:, modality])[0]
        if len(indices):
            out[key][indices] = 103 if modality == 0 else 0
    return out

def view_atoms(mask):
    mask = np.asarray(mask)
    return tuple(3 * t + m for t in range(mask.shape[-2]) for m in range(3) if mask[t, m])
