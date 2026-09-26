from dataclasses import dataclass
from pathlib import Path
import json
import pickle
import numpy as np
import torch

FIELDS = ("input_ids", "stored_attention", "token_type_ids", "audio", "vision")

def load_pickle(path):
    with Path(path).open("rb") as handle:
        try:
            return pickle.load(handle)
        except ModuleNotFoundError as exc:
            if not (exc.name or "").startswith("numpy._core"):
                raise
            handle.seek(0)
            class Compat(pickle.Unpickler):
                def find_class(self, module, name):
                    return super().find_class(module.replace("numpy._core", "numpy.core", 1), name)
            return Compat(handle).load()

def unpack_record(data, level):
    record = data["test"] if level == "attachment3" else data
    text = np.asarray(record["text_bert"])
    audio = np.asarray(record["audio"], dtype=np.float32)
    vision = np.asarray(record["vision"], dtype=np.float32)
    if level == "attachment4":
        text, audio, vision = text[None, ...], audio[None, ...], vision[None, ...]
    if text.ndim != 3 or text.shape[1:] != (3, 50) or audio.shape != (len(text), 50, 74) or vision.shape != (len(text), 50, 35):
        raise ValueError("invalid aligned_50 shapes")
    if not np.isfinite(text).all() or not np.isfinite(audio).all() or not np.isfinite(vision).all():
        raise ValueError("non-finite input")
    result = dict(zip(FIELDS, (text[:, 0].astype(np.int64), text[:, 1].astype(np.int64), text[:, 2].astype(np.int64), audio, vision)))
    for key in ("id", "raw_text"):
        if key in record:
            value = [record[key]] if level == "attachment4" and isinstance(record[key], str) else record[key]
            result[key] = [str(item) for item in value]
    if "classification_labels" in record:
        result["class_id"] = np.asarray(record["classification_labels"]).reshape(len(text)).astype(np.int64)
    if "regression_labels" in record:
        result["score"] = np.asarray(record["regression_labels"], dtype=np.float32).reshape(len(text))
    return result

@dataclass
class Sample:
    split: str
    sample_index: int
    sample_id: str
    source_id: str | None
    video_id: str | None
    raw_text: str | None
    raw: dict
    label: int | None = None
    score_label: float | None = None
    error: str | None = None

def _sample(split, index, arrays, metadata):
    raw = {key: torch.from_numpy(np.array(arrays[key][index], copy=True)) for key in FIELDS}
    sample_id = str(metadata.get("id", [index])[index])
    video_id = str(metadata['video_id'][index]) if 'video_id' in metadata else sample_id.split('$_$')[0]
    text = metadata.get("raw_text", [None] * len(arrays["input_ids"]))[index]
    return Sample(split, index, sample_id, sample_id, video_id, text, raw,
                  int(arrays["class_id"][index]) if "class_id" in arrays else None,
                  float(arrays["score"][index]) if "score" in arrays else None)

def load_samples(config, split):
    data_root = Path(config["project"]["data_root"])
    if split == "valid":
        processed = Path(config["project"]["valid_dir"])
        if (processed / "input_ids.npy").is_file():
            arrays = {key: np.load(processed / f"{key}.npy", mmap_mode="r") for key in FIELDS + ("class_id", "score") if (processed / f"{key}.npy").is_file()}
            metadata = json.loads((processed / "metadata.json").read_text(encoding="utf-8"))
            return [_sample(split, i, arrays, metadata) for i in range(len(arrays["input_ids"]))]
        source = load_pickle(data_root / "附件2-数据集特征文件" / "aligned_50.pkl")["valid"]
        arrays = unpack_record(source, "attachment2")
        metadata = {"id": arrays.get("id", [str(i) for i in range(len(arrays["input_ids"]))]), "raw_text": arrays.get("raw_text", [None] * len(arrays["input_ids"]))}
        metadata["video_id"] = [item.split("$_$")[0] for item in metadata["id"]]
        return [_sample(split, i, arrays, metadata) for i in range(len(arrays["input_ids"]))]
    if split == "special":
        special_dir = data_root / "附件4-可解释专项视频样本与特征文件" / "附件4-可解释专项视频样本与特征文件" / "对齐版本"
        samples = []
        for number in range(1, 21):
            try:
                arrays = unpack_record(load_pickle(special_dir / f"{number:02d}.pkl"), "attachment4")
                metadata = {"id": [f"{number:02d}"], "raw_text": arrays.get("raw_text", [None]), "video_id": [f"{number:02d}"]}
                sample = _sample(split, 0, arrays, metadata)
                sample.sample_index = number - 1
                samples.append(sample)
            except Exception as exc:
                samples.append(Sample(split, number - 1, f"{number:02d}", None, None, None, {}, error=str(exc)))
        return samples
    raise ValueError("split must be valid or special")
