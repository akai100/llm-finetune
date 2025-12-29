import json
import random
from collections import Counter
from data.data_auditor import DataAuditor

def preprocess_dataset(path):
    auditor = DataAuditor()
    raw = load_jsonl(path)
    processed = []

    for s in raw:
        s["text"] = build_prompt(s)
        if auditor.audit_sample(s):
            processed.append(s)

    auditor.report()
    return processed

def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]

def build_prompt(sample):
    instruction = sample.get("instruction", "")
    input_text = sample.get("input", "")
    output = sample.get("output", "")
    prompt = f"""### Instruction:
{instruction}

### Input:
{input_text}

### Response:
{output}"""
    return prompt

def filter_sample(sample, max_len=4096):
    if not sample.get("instruction") or not sample.get("output"):
        return False
    return True

def preprocess_dataset(path):
    raw = load_jsonl(path)
    processed = []
    for s in raw:
        if not filter_sample(s):
            continue
        s["text"] = build_prompt(s)
        processed.append(s)
    return processed

def train_eval_split(data, ratio=0.95, seed=42):
    random.seed(seed)
    random.shuffle(data)
    split = int(len(data) * ratio)
    return data[:split], data[split:]


