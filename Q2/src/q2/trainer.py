"""Fixed 60-epoch training with EMA and resumable run checkpoints."""
import copy
import csv
import json
import math
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import AlignedDataset, Normalizer, collate_raw
from .evaluate import evaluate_model, better_key, selection_key
from .losses import compute_losses
from .masking import apply_span, make_span_mask, sample_train_descriptor
from .model.network import Student, encode_view, TUNED_VARIANTS, CLEAN_VARIANTS
from .state import infer_state


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def learning_rate(epoch, epochs=60, warmup_epochs=5, min_lr=3e-5, max_lr=3e-4):
    if epoch <= warmup_epochs:
        return max_lr * epoch / warmup_epochs
    decay_span = max(1, epochs - warmup_epochs)
    return min_lr + (max_lr - min_lr) * (1 + math.cos(math.pi * (epoch - warmup_epochs) / decay_span)) / 2


def optimizer_for(model, text_encoder=None, text_lr=1e-5, student_lr=3e-4):
    weighted, unweighted = [], []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            (weighted if parameter.ndim >= 2 and not name.endswith("bias") else unweighted).append(parameter)
    groups = [{"params": weighted, "weight_decay": 1e-4, "lr": student_lr, "lr_scale": 1.0},
              {"params": unweighted, "weight_decay": 0, "lr": student_lr, "lr_scale": 1.0}]
    if text_encoder is not None:
        groups.append({"params": [p for p in text_encoder.parameters() if p.requires_grad],
                       "weight_decay": 1e-4, "lr": text_lr, "lr_scale": text_lr / student_lr})
    return torch.optim.AdamW(groups,
                             lr=3e-4, betas=(.9, .999), eps=1e-8)


def set_learning_rate(optimizer, rate, text_scale=1.0):
    for index, group in enumerate(optimizer.param_groups):
        scale = group.get("lr_scale", 1.0 if index < 2 else text_scale)
        group["lr"] = rate * scale


@torch.no_grad()
def update_ema(teacher, student, decay=.99):
    for old, new in zip(teacher.parameters(), student.parameters()):
        old.mul_(decay).add_(new, alpha=1-decay)
    for old, new in zip(teacher.buffers(), student.buffers()):
        old.copy_(new)


def _rng_state(generator):
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "shuffle_generator": generator.get_state()}


def _restore_rng(state, generator):
    random.setstate(state["python"]); np.random.set_state(state["numpy"])
    def _byte_tensor(value):
        # RNG state APIs require CPU ByteTensors even when a checkpoint was
        # loaded onto a CUDA device.
        return torch.as_tensor(value, dtype=torch.uint8, device="cpu")

    torch.set_rng_state(_byte_tensor(state["torch_cpu"]))
    if state["torch_cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([_byte_tensor(value) for value in state["torch_cuda"]])
    generator.set_state(_byte_tensor(state["shuffle_generator"]))


def _save_checkpoint(path, student, teacher, optimizer, epoch, best_key, generator, config, variant, seed,
                     text_encoder=None):
    torch.save({"student": student.state_dict(), "teacher": teacher.state_dict() if teacher else None,
                "optimizer": optimizer.state_dict(), "epoch": epoch, "best_key": best_key,
                "rng": _rng_state(generator), "config": config, "variant": variant, "seed": seed,
                "text_encoder": text_encoder.state_dict() if text_encoder is not None else None}, path)


def _gpu_batch(batch, device):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def train_one(root: Path, config: dict, variant: str, seed: int, resume=False, device=None):
    from .text import load_text_encoder
    root = Path(root)
    device = torch.device(device or "cuda:0")
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal training requires the specified CUDA GPU")
    set_seed(seed)
    run_dir = Path(config["project"]["output_root"]) / "runs" / variant / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    train_set = AlignedDataset(root / "data/processed/train")
    valid_set = AlignedDataset(root / "data/processed/valid")
    normalizer = Normalizer.load(root / "data/processed/normalizer.npz")
    frozen = load_text_encoder(root / config["text"]["model_dir"], device)
    tune_text = variant in TUNED_VARIANTS
    if bool(config["text"]["frozen"]) == tune_text:
        raise ValueError("text.frozen must match the selected frozen/tuned variant")
    if tune_text:
        frozen.requires_grad_(True).train()
    clean_cache = None if tune_text else np.load(root / ".cache/text/train/features.npy", mmap_mode="r")
    valid_cache = None if tune_text else np.load(root / ".cache/text/valid/features.npy", mmap_mode="r")
    student = Student(variant, normalizer.class_prior, normalizer.score_prior).to(device)
    class_weight = None
    class_weight_power = config["loss"].get("class_weight_power",
        .5 if variant in ("late_balanced", "late_tune_balanced", "late_classifier") else 0.0)
    if class_weight_power:
        # This is a train-only prior correction, evaluated as a separate
        # hyperparameter rather than assumed to improve minority-class F1.
        prior = torch.as_tensor(normalizer.class_prior, device=device, dtype=torch.float32)
        class_weight = prior.pow(-class_weight_power)
        class_weight = class_weight / class_weight.mean()
    optimizer = optimizer_for(student, frozen if tune_text else None,
                              text_lr=config["text"].get("learning_rate", 1e-5),
                              student_lr=config["train"]["learning_rate"])
    generator = torch.Generator().manual_seed(seed)
    teacher = None
    best_key = None
    start_epoch = 1
    last = run_dir / "last.pt"
    if resume and last.exists():
        checkpoint = torch.load(last, map_location="cpu", weights_only=False)
        if checkpoint["variant"] != variant or checkpoint["seed"] != seed or checkpoint["config"] != config:
            raise ValueError("resume checkpoint does not match this run")
        student.load_state_dict(checkpoint["student"])
        if tune_text:
            if checkpoint.get("text_encoder") is None:
                raise ValueError("tuned checkpoint is missing text_encoder weights")
            frozen.load_state_dict(checkpoint["text_encoder"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if checkpoint["teacher"] is not None:
            teacher = copy.deepcopy(student).requires_grad_(False).eval()
            teacher.load_state_dict(checkpoint["teacher"])
        best_key = checkpoint["best_key"]
        start_epoch = checkpoint["epoch"] + 1
        _restore_rng(checkpoint["rng"], generator)
    elif last.exists():
        raise FileExistsError(f"existing run at {last}; use --resume")
    (run_dir / "resolved_config.yaml").write_text(__import__("yaml").safe_dump(config, allow_unicode=True), encoding="utf-8")
    history_path = run_dir / "history.csv"
    masks_path = run_dir / "train_masks.jsonl"
    start_time = time.time()
    torch.cuda.reset_peak_memory_stats(device)
    epochs = int(config["train"]["epochs"])
    warmup_epochs = int(config["train"]["warmup_epochs"])
    min_lr = float(config["train"]["min_learning_rate"])
    max_lr = float(config["train"]["learning_rate"])
    for epoch in range(start_epoch, epochs + 1):
        student.train()
        set_learning_rate(optimizer, learning_rate(epoch, epochs, warmup_epochs, min_lr, max_lr),
                          config["text"].get("learning_rate", 1e-5) / max_lr)
        if tune_text:
            frozen.train()
        loader = DataLoader(train_set, batch_size=config["data"]["train_batch_size"], shuffle=True, generator=generator,
                            num_workers=config["data"]["num_workers"], pin_memory=True,
                            drop_last=False, collate_fn=collate_raw)
        totals = {key: 0.0 for key in ("total", "task_clean", "task_corrupt", "msd", "span", "calibration", "consistency")}
        omega_count = 0
        descriptors_for_epoch = []
        batches = 0
        for batch in loader:
            original_indices = batch["sample_index"]
            raw = _gpu_batch(batch, device)
            state0 = infer_state(raw)
            descriptors = ([{"pattern": "", "rho": 0, "position": "middle"} for _ in original_indices]
                           if variant in CLEAN_VARIANTS else
                           [sample_train_descriptor(seed, epoch, int(index), variant == "uniform_spans",
                                                    total_epochs=epochs, warmup_epochs=warmup_epochs)
                            for index in original_indices])
            perturbation = make_span_mask(raw, state0, descriptors)
            descriptors_for_epoch.extend({"sample_index": int(index), "epoch": epoch, **desc}
                                         for index, desc in zip(original_indices, perturbation.descriptors))
            corrupted = apply_span(raw, perturbation)
            if variant in CLEAN_VARIANTS:
                corrupted = raw
            cached = None if tune_text else torch.from_numpy(np.asarray(clean_cache[original_indices.numpy()]).copy()).to(device)
            clean_input = encode_view(raw, frozen, normalizer, cached, text_grad=tune_text)
            if variant in CLEAN_VARIANTS:
                corrupt_input = clean_input
            else:
                corrupt_input = encode_view(corrupted, frozen, normalizer, cached,
                                            perturbation.P[:, :, 0].any(dim=1),
                                            text_grad=tune_text)
            optimizer.zero_grad(set_to_none=True)
            clean_output = student(clean_input, return_details=True, return_attention=False)
            corrupt_output = (student(corrupt_input, return_details=True, return_attention=False)
                              if epoch > warmup_epochs else None)
            with torch.no_grad():
                teacher_output = (teacher(clean_input, return_details=True, return_attention=False)
                                  if teacher is not None else None)
            labels = {"class_id": raw["class_id"], "score": raw["score"]}
            losses = compute_losses(clean_output, corrupt_output, teacher_output, labels, perturbation,
                                    state0, epoch, student, class_weight=class_weight,
                                    regression_weight=config["loss"]["regression_weight"],
                                    warmup_epochs=warmup_epochs,
                                    ramp_epochs=config["loss"]["ramp_epochs"])
            if not torch.isfinite(losses.total):
                raise FloatingPointError(f"non-finite loss in {variant}/seed_{seed}/epoch_{epoch}")
            losses.total.backward()
            parameters = [parameter for group in optimizer.param_groups for parameter in group["params"]]
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            if teacher is not None:
                update_ema(teacher, student)
            batches += 1
            omega_count += losses.omega_count
            totals["total"] += float(losses.total.detach())
            for key, value in losses.terms.items():
                totals[key] += float(value.detach())
        if epoch == warmup_epochs and not student.late and variant != "no_teacher":
            teacher = copy.deepcopy(student).requires_grad_(False).eval()
        with masks_path.open("a", encoding="utf-8") as handle:
            for descriptor in sorted(descriptors_for_epoch, key=lambda item: item["sample_index"]):
                handle.write(json.dumps(descriptor, ensure_ascii=False) + "\n")
        row = {"epoch": epoch, "lr": learning_rate(epoch, epochs, warmup_epochs, min_lr, max_lr), "omega_count": omega_count,
               **{key: val / batches for key, val in totals.items()},
               "missing_macro_f1": "", "missing_mae": "", "clean_macro_f1": "", "clean_mae": ""}
        if epoch in config["train"]["eval_epochs"]:
            validation_dir = run_dir / "validation" / f"epoch_{epoch}"
            rows, _ = evaluate_model(student, frozen, normalizer, valid_set, device,
                                     root / "data/masks/valid", None if tune_text else valid_cache,
                                     batch_size=config["data"]["eval_batch_size"], output_dir=validation_dir)
            candidate = selection_key(rows, epoch)
            row.update({"missing_macro_f1": candidate[0], "missing_mae": candidate[1],
                        "clean_macro_f1": candidate[2], "clean_mae": candidate[3]})
            if better_key(candidate, best_key):
                best_key = candidate
                shutil.copytree(validation_dir, run_dir / "best_validation", dirs_exist_ok=True)
                _save_checkpoint(run_dir / "best.pt", student, teacher, optimizer, epoch,
                                 best_key, generator, config, variant, seed,
                                 frozen if tune_text else None)
        with history_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            if handle.tell() == 0:
                writer.writeheader()
            writer.writerow(row)
        _save_checkpoint(last, student, teacher, optimizer, epoch, best_key,
                         generator, config, variant, seed,
                         frozen if tune_text else None)
    usage = {"seconds": time.time()-start_time, "peak_memory_bytes": torch.cuda.max_memory_allocated(device),
             "parameter_count": sum(p.numel() for p in student.parameters()), "completed_epoch": epochs}
    (run_dir / "resource_usage.json").write_text(json.dumps(usage, indent=2), encoding="utf-8")
    return usage
