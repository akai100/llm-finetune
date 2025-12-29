from data.data_auditor import DataAuditor

def build_prompt(sample):
    ...

def filter_sample(sample):
    ...

def split_dataset(data, ratio):
    ...

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

