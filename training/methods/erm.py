"""ERM trainer and utilities shared across training methods.

This module contains the concrete plain-ERM ``Trainer`` used by the ResNet18
configuration. DANN and CORAL subclass it, overriding only the model, loss,
optimizer, and loop behavior that differs from ERM.
"""
from math import ceil
from pathlib import Path
from typing import Any

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from sklearn.metrics import accuracy_score
from torch import nn
from torch.utils.data import DataLoader

from model import MODEL_NAME
from training.utils import EarlyStopping


EVALUATION_GROUPS = ("poscon", "rest", "target")


def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    valid = labels != -100
    logits = logits[valid]
    labels = labels[valid]

    predictions = logits.argmax(dim=1)
    return float(accuracy_score(labels.numpy(), predictions.numpy()))


def compute_group_accuracies(
    logits: torch.Tensor,
    labels: torch.Tensor,
    groups: list[str],
) -> dict[str, float]:
    """Compute classification accuracy for each JUMP evaluation group."""
    if len(groups) != len(labels):
        raise ValueError(
            "Groups and labels must contain the same number of rows."
        )

    normalized_groups = [group.lower() for group in groups]
    metrics = {}
    for group in EVALUATION_GROUPS:
        mask = torch.tensor(
            [value == group for value in normalized_groups], dtype=torch.bool
        )
        metrics[f"{group}_accuracy"] = (
            compute_accuracy(logits[mask], labels[mask])
            if mask.any()
            else float("nan")
        )
    return metrics


class Trainer:
    """Small ERM training loop with validation and best-model checkpointing."""

    def __init__(
        self,
        dataloaders: Any,
        wandb_run: Any | None = None,
        **config: Any,
    ) -> None:
        self.args = OmegaConf.create(config)
        self.wandb_run = wandb_run
        self.output_path = Path(self.args.output).expanduser().resolve()
        self.data = dataloaders

        self.train_loader = dataloaders.train_loader
        self.validation_loader = dataloaders.validation_loader
        self.dataset_info = dataloaders.dataset_info

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = instantiate(
            self.args.model,
            num_classes=self.dataset_info.num_classes,
            num_domains=self.dataset_info.num_domains,
            in_chans=self.dataset_info.num_channels,
        ).to(self.device)

        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer_cfg = self.args.optimizer
        self.optimizer = instantiate(
            self.optimizer_cfg,
            params=self._trainable_parameters(),
        )
        self.train_batches = (
            len(self.train_loader)
            if getattr(self.args, "max_train_batches", None) is None
            else min(len(self.train_loader), self.args.max_train_batches)
        )

        accumulation_steps = getattr(
            self.args, "gradient_accumulation_steps", 1
        )
        self.updates_per_epoch = ceil(
            self.train_batches / accumulation_steps
        )
        scheduler_overrides = {}
        if (
            "T_max" in self.args.scheduler
            and self.args.scheduler.T_max is None
        ):
            scheduler_overrides["T_max"] = (
                self.args.epochs * self.updates_per_epoch
            )
        self.scheduler = instantiate(
            self.args.scheduler,
            optimizer=self.optimizer,
            **scheduler_overrides,
        )

        self.amp_enabled = self.device == "cuda" and not getattr(
            self.args, "no_amp", True
        )
        print("amp enabled")
        print(self.amp_enabled)

        self.scaler = torch.amp.GradScaler(
            self.device,
            enabled=self.amp_enabled,
        )
        self.best_validation_accuracy = float("-inf")
        self.early_stopping = EarlyStopping(
            patience=getattr(self.args, "patience", 0),
            mode="max",
        )

    def _trainable_parameters(self) -> Any:
        """Parameters handed to the optimizer (subclasses may freeze some)."""
        return self.model.parameters()

    @property
    def best_epoch(self) -> int:
        return self.early_stopping.best_epoch

    def compute_loss(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (logits, loss, labels) for one batch. This is the one
        method every subclass overrides for its own loss calculation."""
        images = images.to(self.device, non_blocking=True)
        labels = labels.to(self.device, non_blocking=True)
        
        with torch.autocast(
            device_type=self.device,
            dtype=torch.float16,
            enabled=self.amp_enabled,
        ):
            logits = self.model(images)
            loss = self.loss_fn(logits, labels)

        return logits, loss, labels


    def _accumulation_window_size(self, batch: int) -> int:
        steps = getattr(self.args, "gradient_accumulation_steps", 1)
        remaining = self.train_batches % steps
        if remaining and batch > self.train_batches - remaining:
            return remaining
        return steps

    def _should_log_batch(self, batch: int) -> bool:
        interval = getattr(self.args, "log_every_n_batches", 1)
        return interval > 0 and (
            batch % interval == 0 or batch == self.train_batches
        )

    def _on_train_epoch_start(self) -> None:
        pass

    def _training_step(
        self,
        batch: Any,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        images, labels, _ = batch
        logits, loss, labels = self.compute_loss(images, labels)

        return loss, {}

    def train_epoch(self, epoch: int) -> None:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        self._on_train_epoch_start()
        accumulated_loss: torch.Tensor | None = None

        for batch_index, batch in enumerate(self.train_loader, start=1):
            if batch_index > self.train_batches:
                break

            loss, extra_losses = self._training_step(batch)
            window_size = self._accumulation_window_size(batch_index)
            self.scaler.scale(loss / window_size).backward()

            detached_loss = loss.detach()
            accumulated_loss = (
                detached_loss
                if accumulated_loss is None
                else accumulated_loss + detached_loss
            )

            update = (
                batch_index
                % getattr(self.args, "gradient_accumulation_steps", 1)
                == 0
                or batch_index == self.train_batches
            )
            if not update:
                continue

            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)

            if self._should_log_batch(batch_index):
                assert accumulated_loss is not None
                self._log_batch(
                    epoch,
                    batch_index,
                    (accumulated_loss / window_size).item(),
                    extra_losses,
                )
            accumulated_loss = None

    def prepare_model_eval(self) -> None:
        self.model.eval()

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        self.prepare_model_eval()
        cumulative_loss = 0.0
        num_samples = 0
        all_logits = []
        all_labels = []
        all_groups = []

        for images, labels, _, groups in self.validation_loader:
            logits, loss, labels = self.compute_loss(images, labels)
            batch_size = labels.size(0)
            cumulative_loss += loss.detach().item() * batch_size
            num_samples += batch_size
            all_logits.append(logits.detach().cpu())
            all_labels.append(labels.detach().cpu())
            all_groups.extend(groups)

        if num_samples == 0:
            raise ValueError("No validation batches were processed.")
        logits = torch.cat(all_logits)
        labels = torch.cat(all_labels)
        return {
            "loss": cumulative_loss / num_samples,
            "accuracy": compute_accuracy(logits, labels),
            **compute_group_accuracies(logits, labels, all_groups),
        }

    def fit(self) -> None:
        for epoch in range(1, self.args.epochs + 1):
            self.train_epoch(epoch)
            validation = self.evaluate()
            selection_score = self.checkpoint_selection_score(validation)
            saved, should_stop = self.early_stopping.step(
                selection_score,
                epoch,
            )
            if saved:
                self.best_validation_accuracy = selection_score
                self.save_checkpoint(**self.checkpoint_metadata(epoch, validation))
            self._log_epoch(epoch, validation, saved)
            if should_stop:
                print(self.early_stopping.message(epoch))
                break

        self._finish_wandb()

    def checkpoint_selection_score(
        self,
        validation: dict[str, float],
    ) -> float:
        """Return the validation accuracy used for checkpoint selection."""
        return validation["accuracy"]

    def checkpoint_metadata(
        self,
        epoch: int,
        validation: dict[str, float],
    ) -> dict[str, Any]:
        return {
            "epoch": epoch,
            "validation_metrics": validation,
        }

    def _log_batch(
        self,
        epoch: int,
        batch: int,
        loss: float,
        extra_losses: dict[str, torch.Tensor],
    ) -> None:
        del extra_losses
        learning_rate = self.optimizer.param_groups[0]["lr"]
        print(
            f"epoch={epoch:03d} "
            f"batch={batch:04d}/{self.train_batches:04d} "
            f"train_loss={loss:.6f} "
            f"learning_rate={learning_rate:.6g}"
        )
        if self.wandb_run is not None:
            self.wandb_run.log(
                {
                    "epoch": epoch,
                    "train/batch": batch,
                    "train/global_step": (
                        (epoch - 1) * self.train_batches + batch
                    ),
                    "train/loss": loss,
                    "optimization/learning_rate": learning_rate,
                }
            )

    def _log_epoch(
        self,
        epoch: int,
        metrics: dict[str, float],
        checkpoint_saved: bool,
    ) -> None:
        print(
            f"epoch={epoch:03d} "
            f"val_loss={metrics['loss']:.4f} "
            f"val_accuracy={metrics['accuracy']:.4f} "
            f"val_poscon_accuracy={metrics['poscon_accuracy']:.4f} "
            f"val_rest_accuracy={metrics['rest_accuracy']:.4f} "
            f"val_target_accuracy={metrics['target_accuracy']:.4f}"
        )
        if checkpoint_saved:
            print(f"Saved new best checkpoint to {self.output_path}")
        if self.wandb_run is not None:
            self.wandb_run.log(
                {
                    "epoch": epoch,
                    "validation/loss": metrics["loss"],
                    "validation/accuracy": metrics["accuracy"],
                    "validation/poscon_accuracy": metrics["poscon_accuracy"],
                    "validation/rest_accuracy": metrics["rest_accuracy"],
                    "validation/target_accuracy": metrics["target_accuracy"],
                    "checkpoint/saved": int(checkpoint_saved),
                    "checkpoint/best_validation_accuracy": (
                        self.best_validation_accuracy
                    ),
                }
            )

    def save_checkpoint(
        self,
        model_state_dict: dict[str, torch.Tensor] | None = None,
        **metadata: Any,
    ) -> None:
        """Atomically save a checkpoint for any training method."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "model_name": MODEL_NAME,
            "num_classes": self.dataset_info.num_classes,
            "num_channels": self.dataset_info.num_channels,
            "num_domains": self.dataset_info.num_domains,
            "dataset_prefix": self.dataset_info.prefix,
            "domain_column": self.data.domain_column,
            "domain_names": self.dataset_info.domain_names,
            "validation_source": self.data.validation_source,
            **metadata,
            "model_state_dict": (
                self.model.state_dict()
                if model_state_dict is None
                else model_state_dict
            ),
        }
        temporary = self.output_path.with_suffix(".pt.incomplete")
        torch.save(checkpoint, temporary)
        temporary.replace(self.output_path)

    def _finish_wandb(self) -> None:
        if self.wandb_run is None:
            return
        self.wandb_run.summary["best_epoch"] = self.best_epoch
        self.wandb_run.summary[
            "best_validation_accuracy"
        ] = self.best_validation_accuracy
