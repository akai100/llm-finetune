数据 → 训练 → 稳定性 → 上线 → GPU → 事故

## 数据层

“在训练之前，我们先做数据审计，因为线上事故里，数据问题远多于模型问题。”

关键词一定要做到：

+ 重复样本（hash）

+ 格式校验

+ 数据分布漂移

+ 可回溯

## 训练平台

“训练不是一次 Trainer.train()，而是一个多阶段 Pipeline。”

重点强调：

+ SFT / Domain / Safety 分阶段

+ 每一阶段都是 可中断、可恢复

+ Checkpoint 有 完整性校验

## 训练稳定性

“我们遇到过 loss 不 NaN，但梯度已经爆炸，模型不可用的问题。”

于是你有：

+ 梯度范数监控

+ NaN / Inf

+ OOM 捕获

+ Training Watchdog（防 NCCL 假死）

## 推理入口

不要急着讲 GPU）

“推理服务是 FastAPI，但真正的复杂度不在 API，而在 GPU 管理。”

## GPU Router

这里是你最值钱的地方之一：

“我们不是简单 round-robin，而是显存感知调度。”

必须点名：

+ torch.cuda.mem_get_info

+ free / total ratio

+ unhealthy GPU 自动下线

再补一句：

“调度时还要考虑 batch 放大显存。”

## Dynamic Batch + Queue

“GPU Worker 会在 queue 上做动态 batch，平衡吞吐和延迟。”

## KV Cache + Session Sticky

“多轮对话我们做了 session sticky routing，KV Cache 不跨 GPU。”

强调风险控制：

+ TTL

+ context 截断

+ OOM 清 cache

## Inference Watchdog

“CUDA hang 时不会报错，所以我们有 inference watchdog，超时直接 kill 进程。”
