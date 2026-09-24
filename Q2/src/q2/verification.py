"""Real-input semantic and learnability checks before formal training."""
import copy
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .data import AlignedDataset, Normalizer, collate_raw
from .losses import compute_losses, task_loss
from .masking import apply_span, make_span_mask
from .model.network import Student, encode_view
from .state import infer_state
from .text import load_text_encoder
from .trainer import set_seed


def verify_real_batch(root: Path, config: dict):
    if not torch.cuda.is_available():
        raise RuntimeError("real Q2 verification requires CUDA; no CPU result is reported as formal validation")
    set_seed(1111)
    device = torch.device("cuda:0")
    root = Path(root)
    dataset = AlignedDataset(root / "data/processed/train")
    normalizer = Normalizer.load(root / "data/processed/normalizer.npz")
    cls_context = bool(config["text"].get("cls_context", False))
    encoder = load_text_encoder(root / "models/text_encoder", device)
    selected = []
    for i in range(len(dataset)):
        sample = dataset[i]
        if infer_state(collate_raw([sample])).J.any():
            selected.append(sample)
        if len(selected) == 32:
            break
    if len(selected) != 32:
        raise RuntimeError("fewer than 32 content-bearing train samples")
    raw_cpu = collate_raw(selected)
    raw = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in raw_cpu.items()}
    state0 = infer_state(raw)
    with torch.no_grad():
        clean_input = encode_view(raw, encoder, normalizer, cls_context=cls_context)
    labels = {"class_id": raw["class_id"], "score": raw["score"]}

    debug = Student("no_msd", normalizer.class_prior, normalizer.score_prior).to(device)
    for module in debug.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.0
    optimizer = torch.optim.AdamW(debug.parameters(), lr=.001)
    debug.train()
    first = None
    for step in range(300):
        optimizer.zero_grad(set_to_none=True)
        output = debug(clean_input)
        loss = task_loss(output, labels["class_id"], labels["score"])
        if first is None:
            first = float(loss.detach())
        loss.backward()
        optimizer.step()
    last = float(loss.detach())
    if last >= first * .8:
        raise AssertionError(f"32-sample learnability failed: {first} -> {last}")

    torch.cuda.reset_peak_memory_stats(device)
    student = Student("full", normalizer.class_prior, normalizer.score_prior).to(device)
    teacher = copy.deepcopy(student).requires_grad_(False).eval()
    descriptor = [{"pattern": "TA", "position": "middle", "rho": .4} for _ in range(32)]
    perturbation = make_span_mask(raw, state0, descriptor)
    corrupt = apply_span(raw, perturbation)
    with torch.no_grad():
        corrupt_input = encode_view(corrupt, encoder, normalizer, cls_context=cls_context)
    student.train()
    clean_output = student(clean_input, return_details=True)
    corrupt_output = student(corrupt_input, return_details=True)
    with torch.no_grad():
        teacher_output = teacher(clean_input, return_details=True)
    losses = compute_losses(clean_output, corrupt_output, teacher_output, labels,
                            perturbation, state0, 6, student)
    if not torch.isfinite(losses.total):
        raise AssertionError("full batch loss is non-finite")
    losses.total.backward()
    if any(parameter.grad is not None for parameter in teacher.parameters()):
        raise AssertionError("teacher received a gradient")
    if any(parameter.grad is not None for parameter in encoder.parameters()):
        raise AssertionError("frozen BERT received a gradient")
    return {"debug_initial_loss": first, "debug_final_loss": last,
            "full_loss": float(losses.total.detach()),
            "terms": {key: float(value.detach()) for key, value in losses.terms.items()},
            "omega_count": losses.omega_count,
            "peak_memory_bytes": torch.cuda.max_memory_allocated(device)}
