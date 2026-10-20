"""DANN (domain-adversarial) training.

Learns domain-invariant features by training a shared featurizer against a
gradient-reversed domain classifier while a class head is supervised on the
labeled source domain."""

from typing import Any, Iterator

import torch
from torch import nn
from torch.utils.data import DataLoader

from training.methods.erm import (
    Trainer,
    compute_accuracy,
    compute_group_accuracies,
)


def repeat_loader(loader: DataLoader) -> Iterator[Any]:
    """Repeat a loader forever without retaining its yielded batches."""
    while True:
        yield from loader


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class DANN(Trainer):
    """Domain-adversarial training loop with validation and checkpointing.

    Each step forwards a concatenation of a labeled source-domain batch and
    an unlabeled target-domain batch through the shared featurizer. The
    class head is only supervised on the source batch; the domain head is
    supervised on both through a gradient-reversed feature branch, so
    optimizing it end to end pushes the featurizer toward domain-invariant
    features.

    Subclasses ``Trainer`` and shares its epoch loop, scheduler, AMP, and
    gradient accumulation. It overrides paired-batch preparation, metric
    accumulation, loss computation, logging, and validation behavior.
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
        self.class_loss_fn = self.loss_fn
        self.domain_loss_fn = nn.CrossEntropyLoss()

    def compute_loss(
        self,
        source_images: torch.Tensor,
        source_labels: torch.Tensor,
        source_metadata: torch.Tensor,
        target_images: torch.Tensor,
        target_metadata: torch.Tensor,
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor,
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
    ]:
        source_count = source_labels.size(0)
        images = torch.cat((source_images, target_images)).to(
            self.device, non_blocking=True
        )
        labels = source_labels.to(self.device, non_blocking=True)
        domains = torch.cat((source_metadata, target_metadata)).to(
            self.device, non_blocking=True
        )
        with torch.autocast(
            device_type=self.device,
            dtype=torch.float16,
            enabled=self.amp_enabled,
        ):
            class_logits, domain_logits = self.model(images)
            class_logits = class_logits[:source_count]
            class_loss = self.class_loss_fn(class_logits, labels)
            domain_loss = self.domain_loss_fn(domain_logits, domains)
            loss = class_loss + self.args.domain_loss_weight * domain_loss
        return loss, class_loss, domain_loss, class_logits, domain_logits, labels, domains

    def _on_train_epoch_start(self) -> None:
        self.target_iterator = repeat_loader(self.target_loader)

    def _training_step(
        self,
        batch: Any,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        source_images, source_labels, source_metadata = batch
        target_images, _, target_metadata = next(self.target_iterator)
        (
            loss,
            class_loss,
            domain_loss,
            class_logits,
            domain_logits,
            labels,
            domains,
        ) = self.compute_loss(
            source_images,
            source_labels,
            source_metadata,
            target_images,
            target_metadata,
        )
        return (
            loss,
            {
                "class_loss": class_loss,
                "domain_loss": domain_loss,
            },
        )

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        self.model.eval()
        cumulative_loss = 0.0
        cumulative_class_loss = 0.0
        cumulative_domain_loss = 0.0
        num_samples = 0
        all_class_logits = []
        all_labels = []
        all_domain_logits = []
        all_domains = []
        all_groups = []

        for images, labels, domains, groups in self.validation_loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            domains = domains.to(self.device, non_blocking=True)
            with torch.autocast(
                device_type=self.device,
                dtype=torch.float16,
                enabled=self.amp_enabled,
            ):
                class_logits, domain_logits = self.model(images)
                class_loss = self.class_loss_fn(class_logits, labels)
                domain_loss = self.domain_loss_fn(domain_logits, domains)
                loss = class_loss + self.args.domain_loss_weight * domain_loss
            batch_size = labels.size(0)
            cumulative_loss += loss.detach().item() * batch_size
            cumulative_class_loss += (
                class_loss.detach().item() * batch_size
            )
            cumulative_domain_loss += (
                domain_loss.detach().item() * batch_size
            )
            num_samples += batch_size
            all_class_logits.append(class_logits.detach().cpu())
            all_labels.append(labels.detach().cpu())
            all_domain_logits.append(domain_logits.detach().cpu())
            all_domains.append(domains.detach().cpu())
            all_groups.extend(groups)
            
        if num_samples == 0:
            raise ValueError("No validation batches were processed.")
        class_logits = torch.cat(all_class_logits)
        labels = torch.cat(all_labels)
        return {
            "loss": cumulative_loss / num_samples,
            "class_loss": cumulative_class_loss / num_samples,
            "domain_loss": cumulative_domain_loss / num_samples,
            "class_accuracy": compute_accuracy(
                class_logits, labels
            ),
            "domain_accuracy": compute_accuracy(
                torch.cat(all_domain_logits), torch.cat(all_domains)
            ),
            **compute_group_accuracies(class_logits, labels, all_groups),
        }

    def checkpoint_metadata(
        self,
        epoch: int,
        validation: dict[str, float],
    ) -> dict[str, Any]:
        return {
            **super().checkpoint_metadata(epoch, validation),
            "head_hidden_dim": self.args.model.head_hidden_dim,
            "head_hidden_layers": self.args.model.head_hidden_layers,
            "head_dropout": self.args.model.head_dropout,
        }

    def checkpoint_selection_score(
        self,
        validation: dict[str, float],
    ) -> float:
        """Select checkpoints using classification, not domain, accuracy."""
        return validation["class_accuracy"]

    def _log_batch(
        self,
        epoch: int,
        batch: int,
        loss: float,
        extra_losses: dict[str, torch.Tensor],
    ) -> None:
        class_loss = extra_losses["class_loss"].detach().item()
        domain_loss = extra_losses["domain_loss"].detach().item()
        print(
            f"epoch={epoch:03d} "
            f"batch={batch:04d}/{self.train_batches:04d} "
            f"train_loss={loss:.6f} "
            f"class_loss={class_loss:.6f} "
            f"domain_loss={domain_loss:.6f}"
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
                    "train/domain_loss": domain_loss,
                    "optimization/learning_rate": (
                        self.optimizer.param_groups[0]["lr"]
                    ),
                }
            )

    def _log_epoch(
        self,
        epoch: int,
        validation: dict[str, float],
        checkpoint_saved: bool,
    ) -> None:
        print(
            f"epoch={epoch:03d} "
            f"val_loss={validation['loss']:.4f} "
            f"val_class_loss={validation['class_loss']:.4f} "
            f"val_class_accuracy={validation['class_accuracy']:.4f} "
            f"val_domain_accuracy={validation['domain_accuracy']:.4f} "
            f"val_poscon_accuracy={validation['poscon_accuracy']:.4f} "
            f"val_rest_accuracy={validation['rest_accuracy']:.4f} "
            f"val_target_accuracy={validation['target_accuracy']:.4f}"
        )
        if checkpoint_saved:
            print(f"Saved new best checkpoint to {self.output_path}")
        if self.wandb_run is not None:
            self.wandb_run.log(
                {
                    "epoch": epoch,
                    "validation/loss": validation["loss"],
                    "validation/class_loss": validation["class_loss"],
                    "validation/domain_loss": validation["domain_loss"],
                    "validation/class_accuracy": validation["class_accuracy"],
                    "validation/domain_accuracy": validation["domain_accuracy"],
                    "validation/poscon_accuracy": validation[
                        "poscon_accuracy"
                    ],
                    "validation/rest_accuracy": validation["rest_accuracy"],
                    "validation/target_accuracy": validation[
                        "target_accuracy"
                    ],
                    "checkpoint/saved": int(checkpoint_saved),
                    "checkpoint/best_validation_accuracy": (
                        self.best_validation_accuracy
                    ),
                }
            )
