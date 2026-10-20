"""ARM-BN trainer using batch statistics during evaluation."""

from training.methods.erm import Trainer


class ARMBN(Trainer):
    """ERM trainer that adapts BatchNorm statistics at evaluation time."""

    def prepare_model_eval(self) -> None:
        self.model.train()
