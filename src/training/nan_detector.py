"""
1. 实时检测 NaN / Inf

2. 自动跳过 batch

3. 记录异常样本 index

4. 防止污染模型
"""

import torch
import logging

logger = logging.getLogger(__name__)

def has_nan(tensor):
    return torch.isnan(tensor).any() or torch.isinf(tensor).any()

def check_loss(loss, step):
    if has_nan(loss):
        logger.error(f"[NaN] Loss became NaN at step {step}")
        return True
    return False
