from transformers import TrainerCallback
from src.training.nan_detector import check_loss

class NaNLossCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        loss = kwargs.get("loss")
        if loss is not None and check_loss(loss, state.global_step):
            control.should_training_stop = True
            return control
