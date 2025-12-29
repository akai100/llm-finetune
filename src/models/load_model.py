from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

def load_model(cfg):
  tokenizer = AutoTokenizer.from_pretrained(
    cfg.model_name,
    use_fast=True,
    trust_remote_code=True
  )

  model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=cfg.dtype,
        device_map=None   # accelerate 接管
  )

  if cfg.use_lora:
        lora_cfg = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            target_modules=cfg.target_modules,
            lora_dropout=cfg.lora_dropout,
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, lora_cfg)
  return model, tokenizer
