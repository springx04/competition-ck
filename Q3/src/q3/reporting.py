from pathlib import Path
from .results import write_json

def write_status_report(root, split, status, details=None):
    root = Path(root); report_dir = root / "reports"; report_dir.mkdir(parents=True, exist_ok=True)
    payload = {"split": split, "status": status, "details": details or {}}
    write_json(report_dir / f"{split}_status.json", payload)
    return payload
