# training/post_train_eval.py
import torch


def sanity_check(model, tokenizer, prompts):
    model.eval()
    results = []

    for p in prompts:
        inputs = tokenizer(p, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=64)
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        results.append(text)

    return results
