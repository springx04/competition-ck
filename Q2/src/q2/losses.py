"""The fixed supervised, MSD, teacher and span losses."""
from dataclasses import dataclass
from itertools import combinations

import torch
from torch.nn import functional as F

from .model.encoders import masked_mean


def task_loss(output, class_id, score, class_weight=None, regression_weight=1.0):
    return (F.cross_entropy(output.logits, class_id, weight=class_weight, reduction="none")
            + regression_weight * F.huber_loss(output.score, score, delta=1.0, reduction="none")).mean()


def msd_loss(clean, student, class_id, score):
    zero = clean.logits.sum() * 0
    if not student.use_msd:
        return {"shared": zero, "specific": zero, "unimodal": zero, "total": zero}
    U = clean.U
    pooled_c, pooled_s, has = [], [], []
    for m in range(3):
        c, valid = masked_mean(clean.C[:, :, m], U[:, :, m])
        s, _ = masked_mean(clean.S[:, :, m], U[:, :, m])
        pooled_c.append(c)
        pooled_s.append(s)
        has.append(valid)
    shared, specific = [], []
    for m, n in combinations(range(3), 2):
        valid = has[m] & has[n]
        if valid.any():
            specific.append(F.cosine_similarity(pooled_s[m][valid], pooled_s[n][valid], dim=-1).square().mean())
        if int(valid.sum()) >= 2:
            left = F.normalize(pooled_c[m][valid], dim=-1, eps=1e-8)
            right = F.normalize(pooled_c[n][valid], dim=-1, eps=1e-8)
            similarity = left @ right.T / .1
            labels = torch.arange(len(left), device=left.device)
            shared.append(.5 * (F.cross_entropy(similarity, labels) + F.cross_entropy(similarity.T, labels)))
    unimodal = []
    aux = student.decomposition
    for m in range(3):
        valid = has[m]
        if valid.any():
            logits = aux.aux_class[m](pooled_s[m][valid])
            regression = 3 * torch.tanh(aux.aux_score[m](pooled_s[m][valid]).squeeze(-1))
            unimodal.append(F.cross_entropy(logits, class_id[valid])
                            + F.huber_loss(regression, score[valid], delta=1.0))
    a = torch.stack(shared).mean() if shared else zero
    b = torch.stack(specific).mean() if specific else zero
    c = torch.stack(unimodal).mean() if unimodal else zero
    return {"shared": a, "specific": b, "unimodal": c, "total": a + .1 * b + .5 * c}


def masked_sample_modality_mean(values: torch.Tensor, omega: torch.Tensor):
    counts = omega.sum(dim=1)
    valid = counts > 0
    if not valid.any():
        return values.sum() * 0
    per_pair = (values * omega).sum(dim=1) / counts.clamp(min=1)
    return per_pair[valid].mean()


@dataclass
class LossOutput:
    total: torch.Tensor
    terms: dict
    omega_count: int


def unimodal_classification_loss(output, class_id, class_weight=None):
    """Equal mean over available modalities, each with its own observed samples."""
    logits = getattr(output, "unimodal_logits", None)
    if logits is None:
        return output.logits.sum() * 0
    observed = output.U.any(dim=1)
    terms = []
    for m in range(3):
        valid = observed[:, m]
        if valid.any():
            terms.append(F.cross_entropy(logits[valid, m], class_id[valid],
                                         weight=class_weight, reduction="none").mean())
    return torch.stack(terms).mean() if terms else output.logits.sum() * 0


def compute_losses(clean_output, corrupt_output, teacher_output, labels, perturbation,
                   state0, epoch: int, student, class_weight=None,
                   regression_weight=1.0, warmup_epochs=5, ramp_epochs=10,
                   unimodal_weight=.2, consistency_weight=.2) -> LossOutput:
    cls, score = labels["class_id"], labels["score"]
    clean_task = task_loss(clean_output, cls, score, class_weight, regression_weight)
    msd = msd_loss(clean_output, student, cls, score)["total"]
    zero = clean_task * 0
    terms = {"task_clean": clean_task, "task_corrupt": zero, "msd": msd,
             "span": zero, "calibration": zero, "consistency": zero}
    total = clean_task + .05 * msd
    if getattr(clean_output, "unimodal_logits", None) is not None:
        terms["unimodal"] = unimodal_classification_loss(clean_output, cls, class_weight)
        total = total + unimodal_weight * terms["unimodal"]
    if epoch <= warmup_epochs:
        return LossOutput(total, terms, 0)
    corrupt_task = task_loss(corrupt_output, cls, score, class_weight, regression_weight)
    terms["task_corrupt"] = corrupt_task
    total = total + corrupt_task
    if "unimodal" in terms:
        corrupt_aux = unimodal_classification_loss(corrupt_output, cls, class_weight)
        terms["unimodal"] = terms["unimodal"] + corrupt_aux
        total = total + unimodal_weight * corrupt_aux
    omega = perturbation.P & state0.U & corrupt_output.B_comp
    if teacher_output is not None and corrupt_output.F_hat is not None and teacher_output.F is not None:
        target = F.layer_norm(teacher_output.F.detach().float(), (128,), eps=1e-5)
        estimate = F.layer_norm(corrupt_output.F_hat.float(), (128,), eps=1e-5)
        residual = (estimate - target).square().mean(dim=-1)
        if student.variant not in ("no_span", "no_comp", "no_teacher"):
            terms["span"] = masked_sample_modality_mean(residual, omega)
        if corrupt_output.e_hat is not None:
            calibration = F.smooth_l1_loss(corrupt_output.e_hat, residual.detach(), beta=1.0,
                                           reduction="none")
            terms["calibration"] = masked_sample_modality_mean(calibration, omega)
    if teacher_output is not None and student.variant not in ("no_cons", "no_teacher"):
        teacher_prob = torch.softmax(teacher_output.logits.detach(), dim=-1)
        weight = (teacher_prob.gather(1, cls[:, None]).squeeze(-1)
                  * torch.exp(-(teacher_output.score.detach() - score).abs() / 6))
        kl = F.kl_div(F.log_softmax(corrupt_output.logits, dim=-1), teacher_prob,
                      reduction="none").sum(dim=-1)
        huber = F.huber_loss(corrupt_output.score, teacher_output.score.detach(),
                             delta=1.0, reduction="none")
        terms["consistency"] = (weight * (kl + huber)).mean()
    ramp = min(1, (epoch - warmup_epochs) / max(1, ramp_epochs))
    total = total + ramp * (.2 * terms["span"] + .1 * terms["calibration"] + consistency_weight * terms["consistency"])
    return LossOutput(total, terms, int(omega.sum()))
