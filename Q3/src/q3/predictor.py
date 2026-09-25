from dataclasses import dataclass
from pathlib import Path
import importlib
import sys
import numpy as np
import torch

@dataclass
class Prediction:
    logits: np.ndarray
    probs: np.ndarray
    score: np.ndarray
    U: np.ndarray
    J: np.ndarray

def ensure_q2_import_root(bundle_dir):
    source = str(Path(bundle_dir) / "src")
    if source not in sys.path:
        sys.path.insert(0, source)

class Predictor:
    def __init__(self, bundle_dir, device="cpu"):
        ensure_q2_import_root(bundle_dir)
        self.device = device
        self.bundle = None
        self.load_error = None
        try:
            self.bundle = importlib.import_module("q2.ensemble").load_ensemble(Path(bundle_dir), device=device)
        except Exception as exc:
            self.load_error = exc

    def predict(self, raw):
        if self.bundle is None:
            raise RuntimeError(f"Q2 predictor unavailable: {self.load_error}")
        tensors = {k: (v if isinstance(v, torch.Tensor) else torch.as_tensor(v)) for k, v in raw.items() if k in ("input_ids", "stored_attention", "token_type_ids", "audio", "vision")}
        with torch.no_grad():
            output = self.bundle.predict_raw(tensors, return_details=True)
        logits = output.logits.detach().cpu().numpy().astype(float)
        probs = torch.softmax(output.logits.float(), dim=-1).detach().cpu().numpy().astype(float)
        score = output.score.detach().cpu().numpy().astype(float).reshape(-1)
        return Prediction(logits, probs, score, output.U.detach().cpu().numpy().astype(bool), output.J.detach().cpu().numpy().astype(bool))
