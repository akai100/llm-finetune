import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class ModelService:
    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16
        ).cuda()
        self.model.eval()

    def generate(self, prompt, gen_cfg):
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = self.model.generate(**inputs, **gen_cfg)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
