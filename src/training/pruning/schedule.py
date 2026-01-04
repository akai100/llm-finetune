# training/pruning/schedule.py
class PruneScheduler:
    def __init__(self, cfg):
        self.start = cfg.pruning.start_step
        self.end = cfg.pruning.end_step

    def should_prune(self, step):
        return self.start <= step <= self.end
