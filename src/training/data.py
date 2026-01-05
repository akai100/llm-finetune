# src/training/data.py
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from datasets import load_dataset


def build_dataloader(tokenizer: AutoTokenizer, cfg) -> DataLoader:
    """
    构建 DataLoader 用于训练
    Args:
        tokenizer (AutoTokenizer): 预训练的分词器
        cfg (Config): 配置对象，其中包含 batch_size, dataset 等信息
    Returns:
        DataLoader: PyTorch DataLoader 对象
    """

    # =========================
    # 加载数据集
    # =========================
    # 使用 Hugging Face `datasets` 加载数据集（可以换成自定义数据集）
    raw_datasets = load_dataset(cfg.dataset.name, split='train')  # 假设你有一个训练数据集

    # =========================
    # 数据预处理（tokenization）
    # =========================
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding=True, truncation=True, max_length=cfg.training.max_seq_length)

    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

    # =========================
    # 自定义 Dataset（如果需要）
    # =========================
    class CustomDataset(Dataset):
        def __init__(self, tokenized_datasets):
            self.tokenized_datasets = tokenized_datasets

        def __len__(self):
            return len(self.tokenized_datasets)

        def __getitem__(self, idx):
            return {key: torch.tensor(value[idx]) for key, value in self.tokenized_datasets.items()}

    dataset = CustomDataset(tokenized_datasets)

    # =========================
    # 创建 DataLoader
    # =========================
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,  # 是否随机打乱数据
        num_workers=cfg.training.num_workers,  # 使用多少个 CPU 核心来加载数据
        pin_memory=True,  # 是否将数据加载到 GPU
    )

    return dataloader
