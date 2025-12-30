class TrainingPipeline:
    def __init__(self, stages):
        self.stages = stages

    def run(self):
        for stage in self.stages:
            stage.run()
