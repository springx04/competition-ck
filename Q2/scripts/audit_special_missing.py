"""Describe attachment 3 input availability without reading labels or fitting models."""
import argparse
import json
from pathlib import Path
import pickle

import numpy as np


def audit(root):
    rows = []
    for path in sorted(Path(root).rglob("附件3_*.pkl")):
        if path.parent.name != "对齐版本":
            continue
        sample = pickle.load(path.open("rb"))["test"]
        ids, attention = sample["text_bert"][0, :2]
        content = (attention == 1) & ~np.isin(ids, [0, 101, 102])
        text = content & (ids != 103)
        av = [np.any(sample[m][0] != 0, axis=-1) for m in ("audio", "vision")]
        row = {"file": path.name, "content_positions": int(content.sum()),
               "text_mask_positions": int((content & ~text).sum()),
               "av_availability_equal": bool(np.array_equal(*av))}
        for name, observed in zip(("audio", "vision"), av):
            missing = content & ~observed
            edges = np.diff(np.r_[False, missing, False].astype(int))
            widths = np.where(edges == -1)[0] - np.where(edges == 1)[0]
            row[name] = {"unavailable_positions": int(missing.sum()),
                         "unavailable_fraction": float(missing.sum() / max(1, content.sum())),
                         "gap_count": len(widths), "longest_gap": int(widths.max()) if len(widths) else 0}
        rows.append(row)
    return {"source": str(Path(root)), "scope": "aligned attachment 3 inputs only; zero rows are unavailable, not proven artificial deletion",
            "samples": len(rows), "samples_with_text_mask": sum(r["text_mask_positions"] > 0 for r in rows),
            "samples_with_equal_av_availability": sum(r["av_availability_equal"] for r in rows), "rows": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.data_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, ensure_ascii=False))
