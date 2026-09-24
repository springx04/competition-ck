"""Official aligned pickle adapters and train-only preprocessing."""
from dataclasses import dataclass
import json
from pathlib import Path
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset


FIELDS = ("input_ids", "stored_attention", "token_type_ids", "audio", "vision")
SPLITS = ("train", "valid", "test")


def load_pickle(path: Path):
    with Path(path).open("rb") as handle:
        try:
            return pickle.load(handle)
        except ModuleNotFoundError as exc:
            if not (exc.name or "").startswith("numpy._core"):
                raise
            handle.seek(0)
            class NumpyCoreCompatUnpickler(pickle.Unpickler):
                def find_class(self, module, name):
                    if module.startswith("numpy._core"):
                        module = module.replace("numpy._core", "numpy.core", 1)
                    return super().find_class(module, name)
            return NumpyCoreCompatUnpickler(handle).load()


def unpack_record(data: dict, level: str) -> dict:
    """Return arrays with a sample axis for the three documented pickle layouts."""
    if level == "attachment2":
        record = data
    elif level == "attachment3":
        record = data["test"]
    elif level == "attachment4":
        record = data
    else:
        raise ValueError(level)
    text = np.asarray(record["text_bert"])
    if level == "attachment4":
        text = text[None, ...]
    if text.ndim != 3 or text.shape[1:] != (3, 50):
        raise ValueError(f"text_bert must be N,3,50, got {text.shape}")
    if not np.isfinite(text).all() or not np.equal(text, np.floor(text)).all():
        raise ValueError("text_bert contains non-integer or non-finite values")
    text = text.astype(np.int64)
    audio = np.asarray(record["audio"], dtype=np.float32)
    vision = np.asarray(record["vision"], dtype=np.float32)
    if level == "attachment4":
        audio, vision = audio[None, ...], vision[None, ...]
    n = text.shape[0]
    if audio.shape != (n, 50, 74) or vision.shape != (n, 50, 35):
        raise ValueError(f"audio/vision shapes: {audio.shape}, {vision.shape}")
    if not np.isfinite(audio).all() or not np.isfinite(vision).all():
        raise ValueError("non-finite audio or vision values")
    if np.any((text[:, 0] < 0) | (text[:, 0] >= 30522)):
        raise ValueError("token id outside specified BERT vocabulary")
    if not np.isin(text[:, 1], [0, 1]).all() or not np.isin(text[:, 2], [0, 1]).all():
        raise ValueError("unexpected attention or token type values")
    result = dict(zip(FIELDS, (text[:, 0], text[:, 1], text[:, 2], audio, vision)))
    if "classification_labels" in record:
        cls = np.asarray(record["classification_labels"]).reshape(n)
        if not np.isfinite(cls).all() or not np.isin(cls, [0, 1, 2]).all():
            raise ValueError("classification_labels must be 0,1,2")
        result["class_id"] = cls.astype(np.int64)
    if "regression_labels" in record:
        score = np.asarray(record["regression_labels"], dtype=np.float32).reshape(n)
        if not np.isfinite(score).all():
            raise ValueError("non-finite regression label")
        result["score"] = score
    for key in ("id", "raw_text"):
        if key in record:
            value = record[key]
            if level == "attachment4" and isinstance(value, str):
                value = [value]
            result[key] = [str(item) for item in value]
    return result


def find_inputs(data_root: Path) -> dict:
    root = Path(data_root)
    main = root / "附件2-数据集特征文件" / "aligned_50.pkl"
    special = root / "附件3-模态缺失特征样本" / "对齐版本"
    attachment4 = root / "附件4-可解释专项视频样本与特征文件" / "附件4-可解释专项视频样本与特征文件" / "对齐版本"
    if not main.is_file():
        raise FileNotFoundError(main)
    files = sorted(special.glob("附件3_*.pkl"))
    if len(files) != 30:
        raise ValueError(f"expected 30 aligned attachment3 files at {special}; found {len(files)}")
    return {"main": main, "special": files, "attachment4": attachment4}


@dataclass
class Normalizer:
    audio_mean: np.ndarray
    audio_std: np.ndarray
    vision_mean: np.ndarray
    vision_std: np.ndarray
    audio_count: int
    vision_count: int
    class_prior: np.ndarray
    score_prior: float

    @classmethod
    def fit(cls, train: dict):
        stats = {}
        for key, dim in (("audio", 74), ("vision", 35)):
            values = train[key]
            rows = values[np.any(values != 0, axis=-1)].astype(np.float64)
            if len(rows):
                mean = rows.mean(axis=0)
                std = rows.std(axis=0, ddof=0)
                std[std < 1e-6] = 1
            else:
                mean, std = np.zeros(dim), np.ones(dim)
            stats[key] = (mean.astype(np.float32), std.astype(np.float32), len(rows))
        counts = np.bincount(train["class_id"], minlength=3).astype(np.float64)
        prior = (counts / counts.sum()).astype(np.float32)
        return cls(stats["audio"][0], stats["audio"][1],
                   stats["vision"][0], stats["vision"][1],
                   stats["audio"][2], stats["vision"][2],
                   prior, float(np.median(train["score"])))

    def save(self, path: Path):
        np.savez(path, **vars(self))

    @classmethod
    def load(cls, path: Path):
        with np.load(path) as source:
            return cls(*(source[key].item() if key in ("audio_count", "vision_count", "score_prior") else source[key]
                         for key in cls.__dataclass_fields__))

    def transform(self, raw: dict, key: str) -> torch.Tensor:
        x = raw[key]
        mean = torch.as_tensor(getattr(self, key + "_mean"), device=x.device)
        std = torch.as_tensor(getattr(self, key + "_std"), device=x.device)
        available = (x != 0).any(dim=-1, keepdim=True)
        return torch.where(available, (x - mean) / std, torch.zeros_like(x))


def prepare_data(data_root: Path, output_dir: Path) -> dict:
    paths = find_inputs(data_root)
    source = load_pickle(paths["main"])
    processed = {}
    for split in SPLITS:
        record = unpack_record(source[split], "attachment2")
        dest = Path(output_dir) / split
        dest.mkdir(parents=True, exist_ok=True)
        for key in (*FIELDS, "class_id", "score"):
            np.save(dest / f"{key}.npy", record[key])
        metadata = {key: record.get(key, []) for key in ("id", "raw_text")}
        metadata["video_id"] = [item.split("$_$")[0] for item in metadata["id"]]
        (dest / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")
        processed[split] = record
    normalizer = Normalizer.fit(processed["train"])
    normalizer.save(Path(output_dir) / "normalizer.npz")
    return {split: len(processed[split]["input_ids"]) for split in SPLITS}


class AlignedDataset(Dataset):
    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self.arrays = {key: np.load(self.directory / f"{key}.npy", mmap_mode="r")
                       for key in (*FIELDS, "class_id", "score")}
        self.metadata = json.loads((self.directory / "metadata.json").read_text(encoding="utf-8"))

    def __len__(self):
        return len(self.arrays["input_ids"])

    def __getitem__(self, index: int) -> dict:
        result = {key: torch.from_numpy(np.array(value[index], copy=True)) for key, value in self.arrays.items()}
        result["sample_id"] = self.metadata["id"][index]
        result["sample_index"] = index
        return result


def collate_raw(samples: list[dict]) -> dict:
    return {key: torch.stack([sample[key] for sample in samples]) if isinstance(samples[0][key], torch.Tensor)
            else torch.tensor([sample[key] for sample in samples], dtype=torch.long) if isinstance(samples[0][key], int)
            else [sample[key] for sample in samples] for key in samples[0]}
