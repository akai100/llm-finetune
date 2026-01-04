"""
1. 捕获 OOM

2. 自动清理 CUDA cache

3. 安全退出 or 跳过当前 step

4. 保留 checkpoint，支持 resume
"""

import torch
import logging
import gc

logger = logging.getLogger(__name__)

def handle_oom(e, step=None):
    if "out of memory" not in str(e):
        raise e

    logger.error(f"[OOM] CUDA out of memory at step {step}")

    # 清理显存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    gc.collect()

    logger.warning("CUDA cache cleared, exiting safely.")
    return True  # 表示已处理
