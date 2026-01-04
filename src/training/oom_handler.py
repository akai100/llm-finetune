# src/training/oom_handler.py

import torch
import gc
from transformers import Trainer
from time import sleep


class OOMHandler:
    def __init__(self, max_retries=3, initial_batch_size=None):
        """
        初始化 OOMHandler
        Args:
            max_retries (int): 最大重试次数，在 OOM 发生后重试多少次
            initial_batch_size (int): 初始的 batch size
        """
        self.max_retries = max_retries
        self.initial_batch_size = initial_batch_size
        self.retries = 0

    def check_oom(self, loss):
        """
        检查是否出现 OOM 错误
        这里的 loss 仅用于表示当前训练是否完成，也可以是其他关键指标
        """
        if torch.cuda.is_available():
            # 当 GPU 显存满了时，会抛出 RuntimeError: CUDA out of memory 错误
            if not torch.is_available():
                raise RuntimeError("CUDA is not available")
            return torch.cuda.memory_allocated() >= torch.cuda.memory_reserved()
        return False

    def handle_oom(self, model, optimizer, dataloader, loss, batch_size):
        """
        处理 OOM 错误并采取相应的措施，如减小 batch_size 或释放缓存
        """
        if self.retries >= self.max_retries:
            print("Maximum retry attempts reached. Stopping training.")
            raise RuntimeError("OOM Error: Training aborted after maximum retries.")

        print(f"OOM detected. Attempting to recover... Retry {self.retries + 1}/{self.max_retries}")

        # 释放显存
        self.clear_cuda_cache()

        # 动态减少 batch_size（可以根据需求调整）
        new_batch_size = batch_size // 2
        if new_batch_size <= 0:
            print("Batch size too small to continue. Stopping training.")
            raise RuntimeError("OOM Error: Batch size too small to continue.")

        print(f"Reducing batch size to {new_batch_size}.")
        batch_size = new_batch_size

        # 调整 DataLoader 和其他参数
        dataloader.batch_size = batch_size

        # 重试训练
        self.retries += 1
        return model, optimizer, dataloader, batch_size

    def clear_cuda_cache(self):
        """
        清理 CUDA 缓存以释放显存
        """
        torch.cuda.empty_cache()
        gc.collect()
        sleep(2)  # 确保缓存完全清理

    def train_step_with_oom_handler(self, model, optimizer, dataloader, batch_size, step_fn):
        """
        训练步骤中集成 OOM 处理器
        Args:
            model (nn.Module): 训练的模型
            optimizer (Optimizer): 优化器
            dataloader (DataLoader): 数据加载器
            batch_size (int): 当前的 batch size
            step_fn (callable): 训练步骤函数，返回训练的 loss
        """
        while True:
            try:
                # 执行训练步骤
                loss = step_fn(model, optimizer, dataloader, batch_size)

                # 如果损失计算成功且没有 OOM 错误
                if not self.check_oom(loss):
                    return loss

            except RuntimeError as e:
                # 如果是 OOM 错误
                if "out of memory" in str(e):
                    # 调用 OOM 处理函数
                    model, optimizer, dataloader, batch_size = self.handle_oom(model, optimizer, dataloader, loss,
                                                                               batch_size)
                    continue
                else:
                    raise e  # 其他非 OOM 错误，继续抛出
