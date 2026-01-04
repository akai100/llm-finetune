# training/quantization/post_quant.py
import torch
from transformers import BitsAndBytesConfig


def load_quantized(model_name, bits):
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=(bits == 8),
        load_in_4bit=(bits == 4),
        bnb_4bit_compute_dtype=torch.float16,
    )

    return AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )
