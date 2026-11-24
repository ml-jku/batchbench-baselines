"""Shared training utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch


@dataclass
class EarlyStopping:
    """Track validation improvement and signal when patience is exhausted."""

    patience: int
    mode: Literal["min", "max"] = "min"
    best_score: float | None = None
    best_epoch: int = 0
    epochs_without_improvement: int = 0

    def __post_init__(self) -> None:
        if self.patience < 0:
            raise ValueError("patience must be non-negative.")
        if self.mode not in {"min", "max"}:
            raise ValueError("mode must be either 'min' or 'max'.")

    def step(self, score: float, epoch: int) -> tuple[bool, bool]:
        """Return ``(improved, should_stop)`` after observing a score."""
        improved = self.best_score is None or (
            score > self.best_score
            if self.mode == "max"
            else score < self.best_score
        )
        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1

        should_stop = self.patience > 0 and (
            self.epochs_without_improvement >= self.patience
        )
        return improved, should_stop

    def message(self, epoch: int) -> str:
        """Describe why training stopped."""
        return (
            f"Early stopping after {epoch} epochs; best validation "
            f"score was at epoch {self.best_epoch}."
        )


def warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    T_max: int,
    warmup_fraction: float = 0.05,
    warmup_start_factor: float = 1e-3,
    eta_min: float = 1e-6,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Linear warmup over ``warmup_fraction`` of ``T_max`` optimizer updates,
    then cosine annealing to ``eta_min`` for the remaining updates."""
    warmup_steps = int(round(T_max * warmup_fraction))
    if warmup_steps <= 0:
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=T_max, eta_min=eta_min
        )
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=warmup_start_factor,
        end_factor=1.0,
        total_iters=warmup_steps,
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, T_max - warmup_steps), eta_min=eta_min
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps]
    )
