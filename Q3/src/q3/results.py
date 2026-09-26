from pathlib import Path
import json
import os

def write_json(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)

def append_jsonl(path, row):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")

def view_id(delete_j):
    values = tuple(sorted(delete_j))
    return "v000000" if not values else "v" + "_".join(f"{j:03d}" for j in values)
