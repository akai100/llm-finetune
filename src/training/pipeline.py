"""
多阶段训练 pipeline
"""

class TrainingStage:
    def __init__(self, name, trainer):
        self.name = name
        self.trainer = trainer

    def run(self):
        print(f"[Stage] Start {self.name}")
        self.trainer.train()
        self.trainer.save_model()
        print(f"[Stage] Finished {self.name}")


class TrainingPipeline:
    def __init__(self, stages):
        self.stages = stages

    def run(self):
        for stage in self.stages:
            stage.run()
