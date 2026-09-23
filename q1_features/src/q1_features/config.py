from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    pass


def _project_root(config_path: Path) -> Path:
    config_path = config_path.resolve()
    if config_path.parent.name == "configs":
        return config_path.parent.parent
    return config_path.parent


def _resolve_path(value: str | None, root: Path) -> str | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    return str((root / path).resolve() if not path.is_absolute() else path.resolve())


def load_config(
    config_path: str | Path,
    *,
    data_root: str | Path | None = None,
    run_config: str | Path | None = None,
) -> dict[str, Any]:
    """Load and resolve one effective configuration.

    A saved run configuration supplies ``data_root`` for later stages.  The
    command-line value only overrides it when explicitly provided.
    """
    source = Path(config_path)
    if not source.is_file():
        raise ConfigError(f"configuration does not exist: {source}")
    cfg = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict) or cfg.get("schema_version") != 1:
        raise ConfigError("expected schema_version: 1")
    cfg = copy.deepcopy(cfg)
    if run_config is not None and Path(run_config).is_file():
        saved = yaml.safe_load(Path(run_config).read_text(encoding="utf-8")) or {}
        saved_root = (saved.get("paths") or {}).get("data_root")
        if saved_root and data_root is None:
            cfg.setdefault("paths", {})["data_root"] = saved_root
    if data_root is not None:
        cfg.setdefault("paths", {})["data_root"] = str(Path(data_root).expanduser().resolve())

    root = _project_root(source)
    cfg["project_root"] = str(root)
    for key in ("data_root", "bert_dir", "ctc_dir", "openface_bin", "openface_env", "face_review", "alignment_review"):
        if key in cfg.get("paths", {}):
            cfg["paths"][key] = _resolve_path(cfg["paths"][key], root)
    return cfg


def save_config(cfg: dict[str, Any], destination: str | Path) -> None:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    tmp.replace(path)


def comparable_config(cfg: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(cfg)
    result.pop("project_root", None)
    return result

