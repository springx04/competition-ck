"""Train-only soft targets from a separately fitted clean-input teacher."""
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import torch


@dataclass
class TrainTeacherTargets:
    logits: torch.Tensor
    score: torch.Tensor

    @classmethod
    def load(cls, path, sample_ids):
        with np.load(path, allow_pickle=False) as data:
            if data["source_split"].item() != "train":
                raise ValueError("offline teacher targets must come from train")
            if data["sample_id"].tolist() != list(sample_ids):
                raise ValueError("offline teacher sample IDs/order do not match train")
            logits = np.asarray(data["logits"], dtype=np.float32)
            score = np.asarray(data["score"], dtype=np.float32)
        if logits.shape != (len(sample_ids), 3) or score.shape != (len(sample_ids),):
            raise ValueError("offline teacher target shapes do not match train")
        if not np.isfinite(logits).all() or not np.isfinite(score).all():
            raise ValueError("offline teacher targets must be finite")
        return cls(torch.from_numpy(logits), torch.from_numpy(score))

    def batch(self, indices, device):
        indices = torch.as_tensor(indices, dtype=torch.long, device="cpu")
        return SimpleNamespace(logits=self.logits[indices].to(device),
                               score=self.score[indices].to(device), F=None)
