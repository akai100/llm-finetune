# train.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from configs.config import load_config
from src.training.controller import TrainingController
from training.data import build_dataloader
from training.optimizer import build_optimizer


def main():
    # =========================
    # Load Config
    # =========================
    cfg = load_config(
        train_yaml="configs/train.yaml",
        lora_yaml="configs/lora.yaml",
    )

    # =========================
    # Seed
    # =========================
    torch.manual_seed(cfg.training.seed)

    # =========================
    # Model & Tokenizer
    # =========================
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.model_name,
        use_fast=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # =========================
    # LoRA Inject
    # =========================
    if cfg.lora.enabled:
        from peft import LoraConfig, get_peft_model

        peft_cfg = LoraConfig(
            r=cfg.lora.r,
            lora_alpha=cfg.lora.lora_alpha,
            lora_dropout=cfg.lora.lora_dropout,
            target_modules=cfg.lora.target_modules,
            bias=cfg.lora.bias,
            task_type=cfg.lora.task_type,
        )
        model = get_peft_model(model, peft_cfg)

    # =========================
    # Optimizer
    # =========================
    optimizer = build_optimizer(model, cfg)

    # =========================
    # Data
    # =========================
    dataloader = build_dataloader(tokenizer, cfg)

    # =========================
    # Training Controller
    # =========================
    controller = TrainingController(
        model=model,
        optimizer=optimizer,
        dataloader=dataloader,
        tokenizer=tokenizer,
        cfg=cfg,
    )

    # =========================
    # Train
    # =========================
    controller.train()


if __name__ == "__main__":
    main()
