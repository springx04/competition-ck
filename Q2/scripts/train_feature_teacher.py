"""Train a clean-input teacher from attachment2 features; never a missing-input predictor.

Only train and valid fields are used. The resulting train-only soft targets may
supervise a token-reencoding student; official contextual text features must
never be passed to the student at validation, test, or deployment time.
"""
import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from q2.config import load_config
from q2.data import FIELDS, Normalizer, find_inputs, load_pickle, unpack_record
from q2.losses import task_loss
from q2.metrics import compute_metrics
from q2.model.network import ModelInput, Student
from q2.trainer import learning_rate, set_seed


def make_inputs(record, official_text, normalizer, device):
    raw = {key: torch.as_tensor(record[key], device=device) for key in FIELDS}
    text = torch.as_tensor(official_text, dtype=torch.float32, device=device)
    # Pool each feature sequence independently. A 50-row matrix does not prove
    # that official text features share the WordPiece axis of text_bert, or
    # that the first row is CLS. Do not use token IDs to mask these vectors.
    # All supplied text rows are nonzero in the audited attachment. Keep all
    # 50 vectors as clean teacher features, without calling them valid tokens.
    observed = torch.stack((torch.ones(text.shape[:2], dtype=torch.bool, device=device), (raw["audio"] != 0).any(dim=-1),
                            (raw["vision"] != 0).any(dim=-1)), dim=-1)
    return ModelInput(text, normalizer.transform(raw, "audio"), normalizer.transform(raw, "vision"),
                      observed, observed.any(dim=-1), text.new_zeros(*observed.shape, 5))


def subset(inputs, indices):
    return ModelInput(*(getattr(inputs, key)[indices] for key in
                        ("text_features", "audio_norm", "vision_norm", "U", "J", "q")))


@torch.no_grad()
def predict(model, inputs, batch_size=128):
    model.eval()
    logits, scores = [], []
    for indices in torch.arange(len(inputs.U), device=inputs.U.device).split(batch_size):
        output = model(subset(inputs, indices))
        logits.append(output.logits.cpu())
        scores.append(output.score.cpu())
    return torch.cat(logits).numpy(), torch.cat(scores).numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--config", default="configs/quick_tune.yaml")
    parser.add_argument("--seed", type=int, default=1111)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / args.config)
    output_dir = root / "experiments" / args.name
    output_dir.mkdir(parents=True, exist_ok=False)
    source_path = find_inputs(Path(config["project"]["data_root"]))["main"]
    source = load_pickle(source_path)
    normalizer = Normalizer.load(root / "data/processed/normalizer.npz")
    device = torch.device("cuda:0")
    set_seed(args.seed)
    records, inputs = {}, {}
    for split in ("train", "valid"):
        record = unpack_record(source[split], "attachment2")
        metadata = json.loads((root / "data/processed" / split / "metadata.json").read_text())
        if record["id"] != metadata["id"]:
            raise ValueError(f"official feature IDs do not match processed {split} order")
        for label in ("class_id", "score"):
            if not np.array_equal(record[label], np.load(root / "data/processed" / split / f"{label}.npy")):
                raise ValueError(f"official {split} {label} does not match processed data")
        text = np.asarray(source[split]["text"], dtype=np.float32)
        if text.shape != (len(record["id"]), 50, 768) or not np.isfinite(text).all():
            raise ValueError(f"unexpected official text features for {split}: {text.shape}")
        records[split] = record
        inputs[split] = make_inputs(record, text, normalizer, device)
    del source

    model = Student("late_attn_tune", normalizer.class_prior, normalizer.score_prior)
    # This teacher is separate from the fixed 256-dimensional student frontend.
    model.encoders[0][0] = nn.Linear(768, 128)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    labels = {key: torch.as_tensor(records["train"][key], device=device) for key in ("class_id", "score")}
    weight = torch.as_tensor(normalizer.class_prior, device=device).pow(-.5)
    weight = weight / weight.mean()
    generator = torch.Generator().manual_seed(args.seed)
    settings = {**vars(args), "source": str(source_path), "type": "clean_feature_teacher",
                "fit_split": "train", "selection_split": "valid", "text_dimension": 768,
                "feature_pooling": "all 50 supplied text rows; independent nonzero AV rows; no WordPiece alignment assumption",
                "class_weight_power": .5, "regression_weight": 1,
                "train_batch_size": 32, "exported_target_split": "train"}
    (output_dir / "settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    history, best = [], None
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        rate = learning_rate(epoch, args.epochs, 2, args.learning_rate / 10, args.learning_rate)
        for group in optimizer.param_groups:
            group["lr"] = rate
        total, count = 0., 0
        for indices in torch.randperm(len(inputs["train"].U), generator=generator).split(32):
            indices = indices.to(device)
            output = model(subset(inputs["train"], indices))
            loss = task_loss(output, labels["class_id"][indices], labels["score"][indices], weight, 1.)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            total += float(loss.detach()) * len(indices)
            count += len(indices)
        logits, scores = predict(model, inputs["valid"])
        metric = compute_metrics(records["valid"]["class_id"], logits, records["valid"]["score"], scores)
        row = {"epoch": epoch, "train_loss": total / count, "valid_clean_macro_f1": metric["macro_f1"],
               "valid_clean_mae": metric["mae"]}
        history.append(row)
        print(json.dumps(row), flush=True)
        key = (metric["macro_f1"], -metric["mae"])
        if best is None or key > best:
            best = key
            torch.save({"teacher": model.state_dict(), "settings": settings, "epoch": epoch,
                        "valid_metrics": metric}, output_dir / "best.pt")
    with (output_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    checkpoint = torch.load(output_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["teacher"])
    logits, scores = predict(model, inputs["train"])
    np.savez(output_dir / "train_targets.npz", sample_id=np.asarray(records["train"]["id"]),
             logits=logits, score=scores, source_split=np.asarray("train"))
    result = {"name": args.name, "selected_epoch": checkpoint["epoch"],
              "valid_clean_macro_f1": best[0], "valid_clean_mae": -best[1],
              "seconds": time.time() - start, "target_count": len(logits),
              "deployment_candidate": False, "test_evaluated": False,
              "limitation": "Clean teacher only; official full-text vectors are not valid missing-input features."}
    (output_dir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
