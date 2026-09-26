"""Download the specified Q1 models and prepare the CTC runtime config.

Run this file from the q1_features project root after the pinned Python
environment has been installed.  The current Hub revision is resolved once,
passed explicitly to the snapshot download, and persisted for offline reuse.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from huggingface_hub import HfApi, snapshot_download


BERT_REPO = "google-bert/bert-base-uncased"
BERT_REVISION = "86b5e0934494bd15c9632b12f734a8a67f723594"
CTC_REPO = (
    "espnet/kamo-naoyuki_librispeech_asr_train_asr_conformer5_raw_bpe5000_"
    "schedule-truncated-c8e5f9"
)
CTC_REVISION = "e6a0f274799b5a4c157d1d5bc20de4c569e25f7e"

ASR_DIR = (
    "exp/asr_train_asr_conformer5_raw_bpe5000_scheduler_confwarmup_steps25000_"
    "batch_bins140000000_optim_conflr0.0015_initnone_accum_grad2_sp"
)

MODEL_JOBS = (
    {
        "repo_id": BERT_REPO,
        "revision": BERT_REVISION,
        "name": "bert",
        "patterns": [
            "config.json",
            "model.safetensors",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.txt",
            "README.md",
        ],
    },
    {
        "repo_id": CTC_REPO,
        "revision": CTC_REVISION,
        "name": "ctc",
        "patterns": [
            "meta.yaml",
            "README.md",
            "data/token_list/**",
            "exp/asr_train_*/config.yaml",
            "exp/asr_train_*/*.pth",
            "exp/asr_stats_*/train/feats_stats.npz",
        ],
    },
)


def _absolute_model_path(value: Any, ctc_dir: Path, field_name: str) -> str:
    """Resolve one documented CTC path field without changing its meaning."""

    if not isinstance(value, str) or not value:
        raise ValueError(f"CTC config field {field_name!r} must be a non-empty string")
    path = Path(value)
    if not path.is_absolute():
        path = ctc_dir / path
    return str(path.resolve())


def prepare_ctc_config(root: Path) -> Path:
    """Write the CTC runtime config required by the execution document.

    Only bpemodel and normalize_conf.stats_file are path-resolved. Every
    other model parameter is retained as loaded. The downloaded ASR YAML is
    never overwritten. A non-list inline token_list is an explicit
    model-version mismatch, not a reason to infer or substitute a token list.
    """

    ctc_dir = root / "models" / "ctc"
    source = ctc_dir / ASR_DIR / "config.yaml"
    destination = ctc_dir / "runtime-config.yaml"

    if not source.is_file():
        raise FileNotFoundError(f"missing downloaded CTC config: {source}")

    with source.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("CTC config is not a mapping; model version mismatch")
    if not isinstance(config.get("token_list"), list):
        raise ValueError(
            "CTC token_list is not an inline list; model version mismatch"
        )

    normalize_conf = config.get("normalize_conf")
    if not isinstance(normalize_conf, dict):
        raise ValueError("CTC normalize_conf is missing; model version mismatch")
    if "bpemodel" not in config:
        raise ValueError("CTC bpemodel is missing; model version mismatch")
    if "stats_file" not in normalize_conf:
        raise ValueError(
            "CTC normalize_conf.stats_file is missing; model version mismatch"
        )

    config["bpemodel"] = _absolute_model_path(
        config["bpemodel"], ctc_dir, "bpemodel"
    )
    normalize_conf["stats_file"] = _absolute_model_path(
        normalize_conf["stats_file"], ctc_dir, "normalize_conf.stats_file"
    )

    with destination.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=False)
    return destination


def download_models(root: Path) -> list[dict[str, str]]:
    """Resolve, download, and record one immutable revision per model."""

    records: list[dict[str, str]] = []
    api = HfApi()
    for job in MODEL_JOBS:
        destination = root / "models" / job["name"]
        revision = api.model_info(job["repo_id"]).sha
        snapshot_download(
            repo_id=job["repo_id"],
            revision=revision,
            local_dir=destination,
            local_dir_use_symlinks=False,
            allow_patterns=job["patterns"],
        )
        records.append(
            {
                "repo_id": job["repo_id"],
                "revision": revision,
                "local_dir": str(destination.relative_to(root)),
            }
        )

    record_path = root / "models" / "download-record.json"
    record_path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    prepare_ctc_config(root)
    return records


def main() -> int:
    root = Path.cwd()
    records = download_models(root)
    for record in records:
        print(f"downloaded {record['repo_id']} at revision {record['revision']}")
    print(f"prepared {root / 'models' / 'ctc' / 'runtime-config.yaml'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
