"""Deep CORAL training.

Aligns source/target feature distributions during training via a
mean/covariance penalty (`coral_penalty` below) on top of a shared
featurizer, alongside the classification loss."""

from typing import Any

import torch
from torch import nn

from training.methods.erm import (
    Trainer as BaseTrainer,
    compute_accuracy,
    compute_group_accuracies,
)

class DeepCoral(BaseTrainer):
    """Deep CORAL training loop with validation and early stopping.

    Subclasses `training.methods.erm.Trainer` and reuses its optimizer,
    scheduler, epoch loop, and metric collection flow. It overrides paired-
    batch preparation, its loss, batch logging, and validation behavior.
    """

    def __init__(
        self,
        dataloaders: Any,
        wandb_run: Any | None = None,
        **config: Any,
    ) -> None:
        super().__init__(
            dataloaders=dataloaders,
            wandb_run=wandb_run,
            **config,
        )
        self.target_loader = dataloaders.target_loader
        if self.args.optimizer.lr <= 0:
            raise ValueError("--learning-rate must be positive.")
        if self.args.penalty_weight < 0:
            raise ValueError("--penalty-weight must be non-negative.")

    @staticmethod
    def _penalty_between(
        first: torch.Tensor,
        second: torch.Tensor,
    ) -> torch.Tensor:
        """Return the mean and covariance discrepancy between two groups."""
        first_mean = first.mean(dim=0, keepdim=True)
        second_mean = second.mean(dim=0, keepdim=True)

        first_centered = first - first_mean
        second_centered = second - second_mean
        first_covariance = (
            first_centered.T @ first_centered / (len(first) - 1)
        )
        second_covariance = (
            second_centered.T @ second_centered / (len(second) - 1)
        )

        mean_difference = (first_mean - second_mean).pow(2).mean()
        covariance_difference = (
            first_covariance - second_covariance
        ).pow(2).mean()

        return mean_difference + covariance_difference

    @classmethod
    def coral_penalty(
        cls,
        features: torch.Tensor,
        groups: torch.Tensor,
    ) -> torch.Tensor:
        """Average the Deep CORAL penalty over every pair of groups."""
        unique_groups = torch.unique(groups)
        penalty = features.new_zeros(())
        pair_count = 0

        for first_index, first_group in enumerate(unique_groups):
            first_features = features[groups == first_group]

            if len(first_features) < 2:
                continue

            for second_group in unique_groups[first_index + 1 :]:
                second_features = features[groups == second_group]

                if len(second_features) < 2:
                    continue
                penalty = penalty + cls._penalty_between(
                    first_features,
                    second_features,
                )
                pair_count += 1

        if pair_count:
            penalty = penalty / pair_count

        return penalty

    def compute_loss(
        self,
        source_inputs: torch.Tensor,
        source_labels: torch.Tensor,
        source_metadata: torch.Tensor,
        target_inputs: torch.Tensor,
        target_metadata: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        source_size = len(source_inputs)
        labels = source_labels.to(self.device, non_blocking=True)

        groups = torch.cat((source_metadata, target_metadata)).to(
            self.device, non_blocking=True
        )

        inputs = torch.cat((source_inputs, target_inputs)).to(
            self.device, non_blocking=True
        )

        features, logits = self.model(inputs)

        class_loss = self.loss_fn(logits[:source_size], labels)
        penalty = self.coral_penalty(features, groups)

        loss = class_loss + self.args.penalty_weight * penalty
        return logits[:source_size], loss, class_loss, penalty, labels

    def _on_train_epoch_start(self) -> None:
        self.target_iterator = iter(self.target_loader)

    def _training_step(
        self,
        batch: Any,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        source_inputs, source_labels, source_metadata = batch
        try:
            target_inputs, _, target_metadata = next(self.target_iterator)
        except StopIteration:
            self.target_iterator = iter(self.target_loader)
            target_inputs, _, target_metadata = next(self.target_iterator)

        logits, loss, class_loss, penalty, labels = self.compute_loss(
            source_inputs,
            source_labels,
            source_metadata,
            target_inputs,
            target_metadata,
        )
        return (
            loss,
            {
                "class_loss": class_loss,
                "coral_penalty": penalty,
            },
        )

    def _log_batch(
        self,
        epoch: int,
        batch: int,
        loss: float,
        extra_losses: dict[str, torch.Tensor],
    ) -> None:
        class_loss = extra_losses["class_loss"].detach().item()
        penalty = extra_losses["coral_penalty"].detach().item()

        print(
            f"epoch={epoch:03d} "
            f"batch={batch:04d}/{self.train_batches:04d} "
            f"train_loss={loss:.6f} "
            f"class_loss={class_loss:.6f} "
            f"coral_penalty={penalty:.6f}"
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
                    "train/class_loss": class_loss,
                    "train/coral_penalty": penalty,
                    "optimization/learning_rate": (
                        self.optimizer.param_groups[0]["lr"]
                    ),
                }
            )

    @torch.inference_mode()
    def evaluate(self) -> dict[str, float]:
        self.model.eval()
        cumulative_loss = 0.0
        num_samples = 0
        all_logits = []
        all_labels = []
        all_groups = []
        for inputs, labels, _, groups in self.validation_loader:
            labels = labels.to(self.device, non_blocking=True)
            inputs = inputs.to(self.device, non_blocking=True)
            _, logits = self.model(inputs)
            loss = self.loss_fn(logits, labels)
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

    def checkpoint_metadata(
        self,
        epoch: int,
        validation: dict[str, float],
    ) -> dict[str, Any]:
        return {
            **super().checkpoint_metadata(epoch, validation),
            "model_name": "resnet18-coral",
            "penalty_weight": self.args.penalty_weight,
        }
