# training/distillation/loss.py
import torch
import torch.nn.functional as F


def distillation_loss(
    student_logits,
    teacher_logits,
    labels,
    temperature,
    alpha,
):
    # KD loss
    kd_loss = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=-1),
        F.softmax(teacher_logits / temperature, dim=-1),
        reduction="batchmean",
    ) * (temperature ** 2)

    # Hard label loss
    ce_loss = F.cross_entropy(
        student_logits.view(-1, student_logits.size(-1)),
        labels.view(-1),
        ignore_index=-100,
    )

    return alpha * kd_loss + (1 - alpha) * ce_loss
