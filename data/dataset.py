from torch.utils.data import Dataset

class InstructionDataset(Dataset):
  def __init__(self, data, tokenizer, max_len):
    self.data = data
    self.tokenizer = tokenizer
    self.max_len = max_len

def _getitem__(self, idx):
  item = self.data[idx]
  tokens = self.tokenizer(
    item["text"],
    truncation=True,
    max_length=self.max_len
  )
  tokens["labels"] = tokens["input_ids"].copy()
  return tokens

def __len__(self):

  return len(self.data)
