from transformers import TrainerCallback

class WatchdogCallback(TrainerCallback):
    def __init__(self, watchdog):
        self.watchdog = watchdog

    def on_step_end(self, args, state, control, **kwargs):
        self.watchdog.heartbeat()
