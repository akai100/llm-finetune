# training/controller.py
import time
import torch

from training.step_runner import StepRunner
from training.oom_handler import OOMHandler
from training.watchdog import TrainingWatchdog
from training.checkpoint_guard import CheckpointGuard

from training.distillation.distill_runner import DistillStepRunner
from training.distillation.teacher_loader import TeacherModel

from training.pruning.pruner import StructuredPruner
from training.pruning.schedule import PruneScheduler

from training.quantization.post_quant import export_quantized_model


class TrainingController:
    """
    工业级训练总控：
    - SFT / LoRA
    - Distillation
    - Structured Pruning
    - Quantization Export
    - OOM / Watchdog / Checkpoint Safety
    """

    def __init__(self, model, optimizer, dataloader, tokenizer, cfg):
        self.model = model
        self.optimizer = optimizer
        self.dataloader = dataloader
        self.tokenizer = tokenizer
        self.cfg = cfg

        # ===== Core Runners =====
        self.step_runner = StepRunner(model, optimizer, cfg)
        self.oom_handler = OOMHandler(cfg)
        self.watchdog = TrainingWatchdog(cfg)
        self.ckpt_guard = CheckpointGuard(cfg)

        # ===== Distillation =====
        self.distill_runner = None
        if cfg.compression.distillation.enabled:
            self.teacher = TeacherModel(cfg.compression.distillation)
            self.distill_runner = DistillStepRunner(
                student=model,
                teacher=self.teacher,
                optimizer=optimizer,
                cfg=cfg,
            )

        # ===== Pruning =====
        self.pruner = None
        self.prune_scheduler = None
        if cfg.compression.pruning.enabled:
            self.pruner = StructuredPruner(model, cfg.compression)
            self.prune_scheduler = PruneScheduler(cfg.compression)

        self.global_step = 0

    # =========================================================
    # Training Loop
    # =========================================================
    def train(self):
        self.model.train()
        total_epochs = self.cfg.control.num_train_epochs

        for epoch in range(total_epochs):
            print(f"\n[Training] Epoch {epoch + 1}/{total_epochs}")

            for batch in self.dataloader:
                self.global_step += 1
                step_start = time.time()

                try:
                    loss = self._run_step(batch)
                    self.watchdog.step_ok(step_start)

                except RuntimeError as e:
                    if self.oom_handler.is_oom(e):
                        if self.oom_handler.handle(e):
                            continue
                    raise e

                # ===== Pruning Schedule =====
                if self.pruner and self.prune_scheduler.should_prune(self.global_step):
                    self.pruner.apply()

                # ===== Checkpoint =====
                if self.global_step % self.cfg.control.save_steps == 0:
                    self.ckpt_guard.save_and_validate(
                        self.model, self.optimizer, self.global_step
                    )

        # ===== Training Finished =====
        self._finalize_training()

    # =========================================================
    # Single Step
    # =========================================================
    def _run_step(self, batch):
        if self.distill_runner:
            return self.distill_runner.run_step(batch)
        else:
            return self.step_runner.run_step(batch)

    # =========================================================
    # Finalization
    # =========================================================
    def _finalize_training(self):
        print("\n[Training] Finished. Finalizing model...")

        # ===== Remove Pruning Mask =====
        if self.pruner and self.pruner.pruned:
            print("[Training] Removing pruning masks")
            self.pruner.remove()

        # ===== Final Checkpoint =====
        self.ckpt_guard.save_and_validate(
            self.model, self.optimizer, self.global_step
        )

        # ===== Quantization Export =====
        if self.cfg.compression.quantization.enabled:
            print("[Training] Exporting quantized model")
            export_quantized_model(
                model=self.model,
                tokenizer=self.tokenizer,
                cfg=self.cfg.compression.quantization,
                output_dir=self.cfg.training.output_dir,
            )

        print("[Training] All done ✔")
