"""TTT-MAE training (Gandelsman et al., 2022).

Two stages, each a ``Trainer`` subclass selected by its own Hydra config:

1. ``MAEPretrainer``: self-supervised masked-image reconstruction of a ViT
   masked autoencoder. Labels are ignored.
2. ``MAEProbe``: trains a classification head on the frozen pretrained
   encoder ("ViT probing"). The checkpoint it writes is self-contained and is
   what ``submissions/submission_mae_ttt.py`` loads; at inference the
   submission adapts the encoder/decoder per test image on the reconstruction
   loss before classifying (see ``model.TTTAdapter``).

Test-time training can be evaluated on a validation subset after probing
(``trainer.ttt.eval_samples > 0``). Setting ``epochs=0`` and pointing
``trainer.output`` at an existing probe checkpoint evaluates TTT alone.
"""
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf

from model import TTTAdapter, create_mae_classifier
from training.methods.erm import Trainer
from training.utils import EarlyStopping


def model_config(trainer: Trainer) -> dict[str, Any]:
    """Constructor arguments needed to rebuild the trained model."""
    config = OmegaConf.to_container(trainer.args.model, resolve=True)
    config.pop("_target_", None)
    return config


# ---------------------------------------------------------------------------
# Stage 1: reconstruction pretraining
# ---------------------------------------------------------------------------


class MAEPretrainer(Trainer):
    """Masked-autoencoder pretraining with validation reconstruction loss.

    Reuses the base epoch loop, optimizer, scheduler, AMP, early stopping and
    checkpointing. Validation masks come from a fixed-seed generator, so the
    validation loss is comparable between epochs. Checkpoints are selected on
    the lowest validation loss (there is no accuracy to maximize).
    """

    def __init__(
        self,
        dataloaders: Any,
        wandb_run: Any | None = None,
        **config: Any,
    ) -> None:
        super().__init__(dataloaders=dataloaders, wandb_run=wandb_run, **config)
        self.early_stopping = EarlyStopping(
            patience=getattr(self.args, "patience", 0),
            mode="min",
        )
        self.best_validation_loss = float("inf")

    def checkpoint_selection_score(
        self,
        validation: dict[str, float],
    ) -> float:
        self.best_validation_loss = min(
            self.best_validation_loss, validation["loss"]
        )
        return validation["loss"]

    def _finish_wandb(self) -> None:
        if self.wandb_run is None:
            return
        self.wandb_run.summary["best_epoch"] = self.best_epoch
        self.wandb_run.summary[
            "best_validation_loss"
        ] = self.best_validation_loss

    def compute_loss(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        images = images.to(self.device, non_blocking=True)
        with torch.autocast(
            device_type=self.device,
            dtype=torch.float16,
            enabled=self.amp_enabled,
        ):
            loss, prediction, _ = self.model(images, generator=generator)
        return prediction, loss, labels

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        self.prepare_model_eval()
        generator = torch.Generator(device=self.device)
        generator.manual_seed(0)
        cumulative_loss = 0.0
        num_samples = 0
        for images, labels, *_ in self.validation_loader:
            _, loss, _ = self.compute_loss(images, labels, generator)
            cumulative_loss += loss.detach().item() * len(images)
            num_samples += len(images)
        if num_samples == 0:
            raise ValueError("No validation batches were processed.")
        return {"loss": cumulative_loss / num_samples}

    def checkpoint_metadata(
        self,
        epoch: int,
        validation: dict[str, float],
    ) -> dict[str, Any]:
        return {
            **super().checkpoint_metadata(epoch, validation),
            "model_name": "mae_vit",
            "model_config": model_config(self),
        }

    def _log_epoch(
        self,
        epoch: int,
        metrics: dict[str, float],
        checkpoint_saved: bool,
    ) -> None:
        print(f"epoch={epoch:03d} val_loss={metrics['loss']:.4f}")
        if checkpoint_saved:
            print(f"Saved new best checkpoint to {self.output_path}")
        if self.wandb_run is not None:
            self.wandb_run.log(
                {
                    "epoch": epoch,
                    "validation/loss": metrics["loss"],
                    "checkpoint/saved": int(checkpoint_saved),
                    "checkpoint/best_validation_loss": (
                        self.best_validation_loss
                    ),
                }
            )


# ---------------------------------------------------------------------------
# Stage 2: classification head on the frozen encoder
# ---------------------------------------------------------------------------


class MAEProbe(Trainer):
    """Train the classification head on top of the frozen MAE encoder.

    The MAE architecture and weights come from ``pretrained_checkpoint``
    (written by ``MAEPretrainer``). Only the head is optimized; the encoder
    is frozen because fine-tuning it makes it specialize for recognition, and
    test-time training would then undo that specialization.
    """

    def __init__(
        self,
        dataloaders: Any,
        wandb_run: Any | None = None,
        **config: Any,
    ) -> None:
        checkpoint = torch.load(
            Path(config["pretrained_checkpoint"]).expanduser(),
            map_location="cpu",
            weights_only=True,
        )
        config["model"] = OmegaConf.merge(
            config["model"], {"mae": checkpoint["model_config"]}
        )
        super().__init__(
            dataloaders=dataloaders,
            wandb_run=wandb_run,
            **config,
        )
        self.model.mae.load_state_dict(checkpoint["model_state_dict"])

    def _trainable_parameters(self) -> Any:
        self.model.mae.requires_grad_(False)
        return self.model.head.parameters()

    def fit(self) -> None:
        super().fit()
        if self.args.ttt.eval_samples > 0:
            self._evaluate_ttt()

    def _ttt_config(self) -> dict[str, Any]:
        return OmegaConf.to_container(self.args.ttt, resolve=True)

    def _evaluate_ttt(self) -> None:
        """Compare accuracy with and without TTT on a validation subset."""
        config = self._ttt_config()
        checkpoint = torch.load(
            self.output_path, map_location=self.device, weights_only=True
        )
        # Rebuild the model from the checkpoint itself, as the submission does:
        # with epochs=0 (TTT-only sweeps) the head settings it was trained
        # with need not match the trainer.model.* defaults.
        self.model = create_mae_classifier(
            num_classes=checkpoint["num_classes"],
            in_chans=checkpoint["num_channels"],
            **checkpoint["model_config"],
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        adapter = TTTAdapter(
            self.model,
            steps=config["steps"],
            lr=config["lr"],
            momentum=config["momentum"],
            weight_decay=config["weight_decay"],
            mask_ratio=config["mask_ratio"],
            batch_size=config["batch_size"],
            amp=config["amp"],
            seed=config["seed"],
        )
        dataset = self.validation_loader.dataset
        count = min(config["eval_samples"], len(dataset))
        indices = np.random.default_rng(config["seed"]).choice(
            len(dataset), count, replace=False
        )

        baseline_correct = 0
        ttt_correct = 0
        for index in indices:
            image, label, *_ = dataset[int(index)]
            image = image.to(self.device).unsqueeze(0)
            with torch.no_grad():
                baseline = self.model(image).argmax(dim=1).item()
            adapted = adapter.predict(image).argmax(dim=1).item()
            baseline_correct += int(baseline == label)
            ttt_correct += int(adapted == label)

        metrics = {
            "validation/ttt_baseline_accuracy": baseline_correct / count,
            "validation/ttt_accuracy": ttt_correct / count,
        }
        metrics["validation/ttt_gain"] = (
            metrics["validation/ttt_accuracy"]
            - metrics["validation/ttt_baseline_accuracy"]
        )
        print(
            f"ttt_samples={count} "
            f"baseline_accuracy={metrics['validation/ttt_baseline_accuracy']:.4f} "
            f"ttt_accuracy={metrics['validation/ttt_accuracy']:.4f} "
            f"gain={metrics['validation/ttt_gain']:+.4f}"
        )
        if self.wandb_run is not None:
            self.wandb_run.log(metrics)
            for name, value in metrics.items():
                self.wandb_run.summary[name] = value

    def checkpoint_metadata(
        self,
        epoch: int,
        validation: dict[str, float],
    ) -> dict[str, Any]:
        config = self._ttt_config()
        config.pop("eval_samples")
        return {
            **super().checkpoint_metadata(epoch, validation),
            "model_name": "mae_vit_ttt",
            "model_config": model_config(self),
            "ttt": config,
        }
