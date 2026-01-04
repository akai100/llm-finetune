# training/controller.py
import time
import torch
from training.step_runner import StepRunner
from training.oom_handler import OOMHandler
from training.watchdog import TrainingWatchdog
from training.checkpoint_guard import CheckpointGuard


class TrainingController:
    def __init__(self, model, optimizer, dataloader, config):
        self.model = model
        self.optimizer = optimizer
        self.dataloader = dataloader
        self.cfg = config

        self.runner = StepRunner(model, optimizer, config)
        self.oom_handler = OOMHandler(config)
        self.watchdog = TrainingWatchdog(config)
        self.ckpt_guard = CheckpointGuard(config)

        if self.cfg.distillation.enabled:
            self.teacher = TeacherModel(self.cfg.distillation)
            self.distill_runner = DistillStepRunner(
                self.model, self.teacher, self.optimizer, self.cfg
            )
        if self.cfg.compression.pruning.enabled:
            self.pruner = StructuredPruner(self.model, self.cfg.compression)


    def train(self):
        global_step = 0
        self.model.train()

        for epoch in range(self.cfg.control.num_train_epochs):
            for batch in self.dataloader:
                global_step += 1
                step_start = time.time()

                try:
                    if self.cfg.distillation.enabled:
                        loss = self.distill_runner.run_step(batch)
                    else:
                        loss = self.runner.run_step(batch)

                    self.watchdog.step_ok(step_start)

                except RuntimeError as e:
                    if self.oom_handler.is_oom(e):
                        handled = self.oom_handler.handle(e)
                        if handled:
                            continue
                    raise e

                if global_step % self.cfg.control.save_steps == 0:
                    self.ckpt_guard.save_and_validate(
                        self.model, self.optimizer, global_step
                    )
