"""Reevaluate a saved tuned checkpoint without touching historical metrics."""
import argparse
import json
from pathlib import Path
import torch
from q2.data import AlignedDataset, Normalizer
from q2.evaluate import evaluate_model, selection_key
from q2.model.network import Student
from q2.text import load_text_encoder

parser = argparse.ArgumentParser()
parser.add_argument("checkpoint", type=Path)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
assert checkpoint.get("text_encoder") is not None, "checkpoint has no tuned text weights"
normalizer = Normalizer.load(root / "data/processed/normalizer.npz")
student = Student(checkpoint["variant"], normalizer.class_prior, normalizer.score_prior).cuda()
student.load_state_dict(checkpoint["student"])
encoder = load_text_encoder(root / "models/text_encoder", "cuda")
encoder.load_state_dict(checkpoint["text_encoder"])
rows, _ = evaluate_model(student, encoder, normalizer,
    AlignedDataset(root / "data/processed/valid"), "cuda",
    root / "data/masks/valid", None, output_dir=args.output)
result = {"checkpoint": str(args.checkpoint), "epoch": checkpoint["epoch"],
          "split": "valid", "text_features": "current_checkpoint_live",
          "selection_key": selection_key(rows, checkpoint["epoch"])}
(args.output / "summary.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result), flush=True)
