# training/distillation/teacher_loader.py
import torch
from transformers import AutoModelForCausalLM


class TeacherModel:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.teacher.model_name,
            torch_dtype=torch.float16 if cfg.teacher.fp16 else torch.float32,
            device_map="auto",
        )

        if cfg.teacher.eval_mode:
            self.model.eval()
            for p in self.model.parameters():
                p.requires_grad = False

    @torch.no_grad()
    def forward(self, **inputs):
        return self.model(**inputs).logits
