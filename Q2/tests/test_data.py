from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from q2.data import (  # noqa: E402
    AlignedDataset,
    FIELDS,
    Normalizer,
    collate_raw,
    unpack_record,
)


def _canonical_record() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    text = np.zeros((1, 3, 50), dtype=np.int64)
    text[0, 0, :4] = [101, 200, 201, 102]
    text[0, 1, :4] = [1, 1, 1, 1]
    audio = np.zeros((1, 50, 74), dtype=np.float32)
    vision = np.zeros((1, 50, 35), dtype=np.float32)
    audio[0, 1, 0] = 1.5
    vision[0, 1, 0] = -2.0
    return text, audio, vision


def _as_single_sample(record: dict, index: int = 0) -> dict:
    sample = {
        key: torch.from_numpy(np.array(record[key][index], copy=True))
        for key in FIELDS
    }
    sample["sample_id"] = record.get("id", [f"sample-{index}"])[index]
    sample["sample_index"] = index
    return sample


def test_attachment2_3_4_adapters_produce_the_same_single_raw_batch() -> None:
    text, audio, vision = _canonical_record()
    attachment2 = {
        "text_bert": text.copy(),
        "audio": audio.copy(),
        "vision": vision.copy(),
        "id": ["sample-01"],
        "raw_text": ["same content"],
    }
    attachment3 = {
        "test": {
            "text_bert": text.astype(np.float32),
            "audio": audio.copy(),
            "vision": vision.copy(),
            "id": ["sample-01"],
            "raw_text": ["same content"],
        }
    }
    attachment4 = {
        "text_bert": text[0].copy(),
        "audio": audio[0].copy(),
        "vision": vision[0].copy(),
        "id": "sample-01",
        "raw_text": "same content",
    }

    records = [
        unpack_record(attachment2, "attachment2"),
        unpack_record(attachment3, "attachment3"),
        unpack_record(attachment4, "attachment4"),
    ]
    for record in records:
        assert record["input_ids"].shape == (1, 50)
        assert record["stored_attention"].shape == (1, 50)
        assert record["token_type_ids"].shape == (1, 50)
        assert record["audio"].shape == (1, 50, 74)
        assert record["vision"].shape == (1, 50, 35)
    for key in FIELDS:
        for record in records[1:]:
            np.testing.assert_array_equal(record[key], records[0][key])

    batches = [collate_raw([_as_single_sample(record)]) for record in records]
    for key in FIELDS:
        for batch in batches[1:]:
            assert torch.equal(batch[key], batches[0][key])
    assert [batch["sample_id"] for batch in batches] == [["sample-01"]] * 3


def _train_statistics_fixture() -> dict[str, np.ndarray]:
    audio = np.zeros((3, 50, 74), dtype=np.float32)
    vision = np.zeros((3, 50, 35), dtype=np.float32)
    audio[0, 0] = 1.0
    audio[1, 0] = 3.0
    audio[2, 0] = 5.0
    vision[0, 0] = 2.0
    vision[1, 0] = 4.0
    vision[2, 0] = 6.0
    return {
        "audio": audio,
        "vision": vision,
        "class_id": np.array([0, 1, 2], dtype=np.int64),
        "score": np.array([-1.0, 0.0, 1.0], dtype=np.float32),
    }


def test_normalizer_uses_only_train_available_rows_and_keeps_zero_placeholders() -> None:
    train = _train_statistics_fixture()
    normalizer = Normalizer.fit(train)

    assert normalizer.audio_count == 3
    assert normalizer.vision_count == 3
    np.testing.assert_allclose(normalizer.audio_mean, np.full(74, 3.0, dtype=np.float32))
    np.testing.assert_allclose(normalizer.vision_mean, np.full(35, 4.0, dtype=np.float32))
    np.testing.assert_allclose(normalizer.class_prior, np.full(3, 1 / 3, dtype=np.float32))
    assert normalizer.score_prior == pytest.approx(0.0)

    valid = _train_statistics_fixture()
    valid["audio"][0, 0] = 10_000.0
    valid["vision"][0, 0] = -10_000.0
    repeat = Normalizer.fit(train)
    np.testing.assert_array_equal(repeat.audio_mean, normalizer.audio_mean)
    np.testing.assert_array_equal(repeat.vision_mean, normalizer.vision_mean)

    raw = {
        "audio": torch.zeros((2, 50, 74), dtype=torch.float32),
        "vision": torch.zeros((2, 50, 35), dtype=torch.float32),
    }
    raw["audio"][1, 0] = 5.0
    raw["vision"][1, 0] = 6.0
    audio_norm = normalizer.transform(raw, "audio")
    vision_norm = normalizer.transform(raw, "vision")
    assert torch.count_nonzero(audio_norm[0]) == 0
    assert torch.count_nonzero(vision_norm[0]) == 0
    assert torch.allclose(audio_norm[1, 0], torch.full((74,), 2.0 / np.sqrt(8 / 3)))
    assert torch.allclose(vision_norm[1, 0], torch.full((35,), 2.0 / np.sqrt(8 / 3)))


def test_aligned_dataset_returns_copies_and_does_not_pollute_mmap_arrays(tmp_path: Path) -> None:
    directory = tmp_path / "train"
    directory.mkdir()
    text, audio, vision = _canonical_record()
    arrays = {
        "input_ids": text[:, 0],
        "stored_attention": text[:, 1],
        "token_type_ids": text[:, 2],
        "audio": audio,
        "vision": vision,
        "class_id": np.array([1], dtype=np.int64),
        "score": np.array([0.25], dtype=np.float32),
    }
    for key, value in arrays.items():
        np.save(directory / f"{key}.npy", value)
    (directory / "metadata.json").write_text(
        json.dumps({"id": ["sample-01"], "raw_text": ["same content"]}),
        encoding="utf-8",
    )

    dataset = AlignedDataset(directory)
    first = dataset[0]
    first["input_ids"][1] = 999
    first["audio"][1, 0] = 999.0
    first["vision"][1, 0] = 999.0
    first["class_id"] = torch.tensor(2, dtype=torch.long)
    second = dataset[0]

    assert int(second["input_ids"][1]) == 200
    assert float(second["audio"][1, 0]) == pytest.approx(1.5)
    assert float(second["vision"][1, 0]) == pytest.approx(-2.0)
    assert int(second["class_id"]) == 1
