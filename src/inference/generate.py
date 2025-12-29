import torch
from transformers import GenerationConfig

def batch_generate(model, tokenizer, texts, gen_cfg):
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True
    ).to(model.device)

    outputs = model.generate(
        **inputs,
        generation_config=gen_cfg
    )
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)
