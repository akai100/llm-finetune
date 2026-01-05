# training/distillation/distill_runner.py
import torch
from src.training.distillation.loss import distillation_loss


class DistillStepRunner:
    def __init__(self, student, teacher, optimizer, cfg):
        self.student = student
        self.teacher = teacher
        self.optimizer = optimizer
        self.cfg = cfg

    def run_step(self, batch):
        student_out = self.student(**batch)
        student_logits = student_out.logits

        with torch.no_grad():
            teacher_logits = self.teacher.forward(**batch)

        loss = distillation_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            labels=batch["labels"],
            temperature=self.cfg.distillation.temperature,
            alpha=self.cfg.distillation.alpha,
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.student.parameters(),
            self.cfg.stability.max_grad_norm,
        )

        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

        return loss.item()
