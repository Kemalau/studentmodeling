"""Loss utilities for M3/M4 recurrent-state behavioral distillation."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def topk_teacher_cross_entropy(
    student_logits: torch.Tensor,
    student_labels: torch.Tensor,
    target_token_ids: torch.Tensor,
    teacher_topk_ids: torch.Tensor,
    teacher_topk_logprobs: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Distill the teacher distribution conditioned on its cached top-k."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    valid = student_labels[:, 1:].ne(-100)
    labels = student_labels[:, 1:][valid]
    if not torch.equal(labels.detach().cpu(), target_token_ids.detach().cpu()):
        raise ValueError("Cached target tokens do not align with student labels")
    student = student_logits[:, :-1].float()[valid]
    ids = teacher_topk_ids.to(student.device, dtype=torch.long)
    teacher_logp = teacher_topk_logprobs.to(student.device, dtype=torch.float32)
    if ids.shape != teacher_logp.shape or ids.shape[0] != student.shape[0]:
        raise ValueError("Cached top-k shape does not align with student targets")
    student_topk_logp = F.log_softmax(student.gather(-1, ids) / temperature, dim=-1)
    teacher_topk_prob = F.softmax(teacher_logp / temperature, dim=-1)
    return -(teacher_topk_prob * student_topk_logp).sum(dim=-1).mean() * (
        temperature**2
    )


def supervised_token_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    student_labels: torch.Tensor,
    teacher_labels: torch.Tensor | None = None,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Teacher-to-state KL aligned by target tokens, not prompt positions."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if student_logits.shape[0] != teacher_logits.shape[0]:
        raise ValueError("Student/teacher batch sizes differ")
    if student_logits.shape[-1] != teacher_logits.shape[-1]:
        raise ValueError("Student/teacher vocabulary sizes differ")
    if student_labels.shape != student_logits.shape[:2]:
        raise ValueError(
            f"Student labels shape {tuple(student_labels.shape)} does not match logits "
            f"{tuple(student_logits.shape[:2])}"
        )
    if teacher_labels is None:
        teacher_labels = student_labels
    if teacher_labels.shape != teacher_logits.shape[:2]:
        raise ValueError(
            f"Teacher labels shape {tuple(teacher_labels.shape)} does not match logits "
            f"{tuple(teacher_logits.shape[:2])}"
        )
    student_valid = student_labels[:, 1:].ne(-100)
    teacher_valid = teacher_labels[:, 1:].ne(-100)
    if not student_valid.any() or not teacher_valid.any():
        raise ValueError("No supervised positions for behavioral distillation")
    student_targets = student_labels[:, 1:][student_valid]
    teacher_targets = teacher_labels[:, 1:][teacher_valid]
    if student_targets.shape != teacher_targets.shape or not torch.equal(
        student_targets, teacher_targets
    ):
        raise ValueError("Student/teacher supervised target tokens are not aligned")
    student = student_logits[:, :-1].float()[student_valid] / temperature
    teacher = teacher_logits[:, :-1].detach().float()[teacher_valid] / temperature
    return F.kl_div(
        F.log_softmax(student, dim=-1),
        F.softmax(teacher, dim=-1),
        reduction="batchmean",
    ) * (temperature**2)
