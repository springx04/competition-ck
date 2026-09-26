from pathlib import Path
import yaml

def load_config(path):
    path = Path(path).resolve()
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    root = Path(data.get("project", {}).get("root", path.parent))
    if not root.is_absolute():
        root = (path.parent.parent / root).resolve()
    data.setdefault("project", {})["root"] = str(root)
    for section in ("project", "output", "alignment"):
        for key, value in list(data.get(section, {}).items()):
            if isinstance(value, str) and not Path(value).is_absolute() and (section == "output" or key.endswith(("_dir", "_root", "_file", "_path")) or key in {"root", "bundle_dir", "valid_dir", "review_file", "av_provenance_file"}):
                data[section][key] = str((root / value).resolve())
    return data
