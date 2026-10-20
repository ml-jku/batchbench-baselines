from __future__ import annotations

import math
import random
from collections.abc import Iterator, Sequence

import numpy as np
import torch
from torch.utils.data import Sampler


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and Torch for reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_by_groups(
    groups: Sequence[int] | np.ndarray | torch.Tensor,
) -> tuple[dict[int, torch.Tensor], torch.Tensor, torch.Tensor]:
    """Return row indices and counts for each integer group ID."""
    group_tensor = torch.as_tensor(groups)
    if group_tensor.ndim != 1:
        raise ValueError("groups must be a one-dimensional sequence.")

    unique_groups, unique_counts = torch.unique(
        group_tensor,
        sorted=False,
        return_counts=True,
    )
    group_indices = {
        int(group.item()): torch.nonzero(
            group_tensor == group,
            as_tuple=True,
        )[0]
        for group in unique_groups
    }
    return group_indices, unique_groups, unique_counts


class GroupBatchSampler(Sampler[Sequence[int]]):
    """Sample a fixed number of randomly selected groups in each batch.

    Training batches select groups uniformly without replacement, then sample
    rows independently from each selected group. Rows are sampled with
    replacement when a group is no larger than the requested per-group sample
    count. Evaluation instead visits every row once, group by group.

    Negative groups are optional and assume that the corresponding negative
    dataset is concatenated after the positive dataset. A positive group with
    no matching negatives is still sampled, without appended controls.
    """

    def __init__(
        self,
        groups: Sequence[int] | np.ndarray | torch.Tensor,
        batch_size: int,
        groups_per_batch: int,
        eval: bool,
        n_samples_per_group: int | None = None,
        negative_groups: Sequence[int] | np.ndarray | torch.Tensor | None = None,
        neg_batch_size: int | None = None,
        seed: int = 1234,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if groups_per_batch <= 0:
            raise ValueError("groups_per_batch must be positive.")
        if groups_per_batch > batch_size:
            raise ValueError(
                "groups_per_batch cannot be greater than batch_size."
            )

        self.groups = torch.as_tensor(groups)
        self.batch_size = batch_size
        self.neg_batch_size = neg_batch_size
        self.eval = eval
        self.seed = seed
        self._rng = np.random.default_rng(seed)

        (
            self.group_indices,
            self.unique_groups,
            _,
        ) = split_by_groups(self.groups)
        if len(self.unique_groups) == 0:
            raise ValueError("groups must not be empty.")
        self.groups_per_batch = min(groups_per_batch, len(self.unique_groups))

        self.negative_groups = (
            torch.as_tensor(negative_groups)
            if negative_groups is not None
            else None
        )
        if self.negative_groups is not None:
            if neg_batch_size is None or neg_batch_size <= 0:
                raise ValueError(
                    "neg_batch_size must be positive when negative_groups "
                    "are provided."
                )
            (
                self.neg_group_indices,
                self.neg_unique_groups,
                _,
            ) = split_by_groups(self.negative_groups)
        if not self.eval:
            self.num_batches = len(self.groups) // self.batch_size
        else:
            self.num_batches = sum(
                math.ceil(len(group_ids) / self.batch_size)
                for group_ids in self.group_indices.values()
            )

        self.n_samples_per_group = (
            self.batch_size // self.groups_per_batch
            if n_samples_per_group is None
            else n_samples_per_group
        )
        if self.n_samples_per_group <= 0:
            raise ValueError("n_samples_per_group must be positive.")

    def __iter__(self) -> Iterator[Sequence[int]]:
        if self.eval:
            return iter(self._evaluation_batches())
        return iter(self._training_batches())

    def _training_batches(self) -> list[np.ndarray]:
        batch_ids = []
        unique_groups = self.unique_groups.cpu().numpy()
        for _ in range(self.num_batches):
            batch_groups = self._rng.choice(
                unique_groups,
                self.groups_per_batch,
                replace=False,
            )
            sample_ids = [
                self._rng.choice(
                    self.group_indices[int(group)].cpu().numpy(),
                    self.n_samples_per_group,
                    replace=(
                        len(self.group_indices[int(group)])
                        <= self.n_samples_per_group
                    ),
                )
                for group in batch_groups
            ]
            batch = np.concatenate(sample_ids)
            if self.negative_groups is not None:
                negative_ids = [
                    self._rng.choice(
                        self.neg_group_indices[int(group)].cpu().numpy(),
                        self.neg_batch_size,
                        replace=(
                            len(self.neg_group_indices[int(group)])
                            <= self.neg_batch_size
                        ),
                    )
                    + len(self.groups)
                    for group in batch_groups
                    if int(group) in self.neg_group_indices
                ]
                batch = np.concatenate((batch, *negative_ids))
            batch_ids.append(batch)
        return batch_ids

    def _evaluation_batches(self) -> list[np.ndarray]:
        batch_ids = []
        rng = np.random.default_rng(self.seed)
        for group, group_ids_tensor in self.group_indices.items():
            group_ids = group_ids_tensor.cpu().numpy()
            group_ids_permuted = rng.permutation(group_ids)
            for start in range(0, len(group_ids), self.batch_size):
                sample_ids = group_ids_permuted[start : start + self.batch_size]
                if (
                    self.negative_groups is not None
                    and group in self.neg_group_indices
                ):
                    negative_ids = rng.choice(
                        self.neg_group_indices[group].cpu().numpy(),
                        self.neg_batch_size,
                        replace=(
                            len(self.neg_group_indices[group])
                            <= self.neg_batch_size
                        ),
                    ) + len(self.groups)
                    sample_ids = np.concatenate((sample_ids, negative_ids))
                batch_ids.append(sample_ids)
        return batch_ids

    def __len__(self) -> int:
        return self.num_batches
