"""Audit attached Q2 arrays without changing them or using labels for training."""
from collections import defaultdict
import json
from pathlib import Path
import numpy as np

root = Path(__file__).resolve().parents[1]
result = {}
for split in ("train", "valid", "test"):
    directory = root / "data/processed" / split
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    audio = np.load(directory / "audio.npy", mmap_mode="r")
    vision = np.load(directory / "vision.npy", mmap_mode="r")
    score = np.load(directory / "score.npy", mmap_mode="r")
    labels = np.load(directory / "class_id.npy", mmap_mode="r")
    rows = {}
    for name, values in (("audio", audio), ("vision", vision)):
        observed = np.any(values != 0, axis=-1)
        nonzero = np.asarray(values)[observed]
        rows[name] = {
            "zero_rows": int((~observed).sum()), "observed_rows": int(observed.sum()),
            "nan": int(np.isnan(nonzero).sum()), "inf": int(np.isinf(nonzero).sum()),
            "min": float(nonzero.min()), "max": float(nonzero.max()),
        }
    groups = defaultdict(list)
    for index, text in enumerate(metadata.get("raw_text", [])):
        groups[text].append(index)
    duplicate_groups = [indices for indices in groups.values() if len(indices) > 1]
    rows.update({"samples": len(labels), "class_counts": np.bincount(labels, minlength=3).tolist(),
                 "score_min": float(score.min()), "score_max": float(score.max()),
                 "duplicate_text_groups": len(duplicate_groups),
                 "duplicate_text_rows": int(sum(map(len, duplicate_groups)))})
    result[split] = rows
(root / "reports").mkdir(exist_ok=True)
(root / "reports/data_quality_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2))
