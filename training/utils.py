"""Shared training utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


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
