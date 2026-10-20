from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import numpy as np
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from utils import GroupBatchSampler


_GROUP_COLUMNS = {
    "plate": "Metadata_Plate",
    "batch": "Metadata_Batch",
    "source": "Metadata_Source",
}
_TUPLE_POSITIONS = {
    "source": 0,
    "batch": 1,
    "plate": 2,
}

def metadata_group_values(
    metadata: Sequence[Mapping[str, Any] | tuple[Any, Any, Any]],
    group_by: str,
) -> list[Any]:
    """Return the values of one supported metadata column."""
    normalized_group = group_by.removeprefix("Metadata_").lower()
    if normalized_group not in _GROUP_COLUMNS:
        choices = ", ".join(_GROUP_COLUMNS)
        raise ValueError(
            f"group_by must be one of {choices}; got {group_by!r}."
        )

    values = []
    column = _GROUP_COLUMNS[normalized_group]
    for index, row in enumerate(metadata):
        if isinstance(row, Mapping):
            if column not in row:
                raise KeyError(
                    f"Metadata row {index} has no {column!r} field."
                )
            values.append(row[column])
        elif isinstance(row, tuple):
            if len(row) != 3:
                raise ValueError(
                    "Tuple metadata rows must contain "
                    "(source_id, batch_id, plate_id); "
                    f"row {index} has {len(row)} values."
                )
            values.append(row[_TUPLE_POSITIONS[normalized_group]])
        else:
            raise TypeError(
                "Each metadata row must be a (source_id, batch_id, plate_id) "
                f"tuple or a mapping; row {index} has type "
                f"{type(row).__name__}."
            )

    return values


def metadata_group_ids(
    metadata: Sequence[Mapping[str, Any] | tuple[Any, Any, Any]],
    group_by: str,
) -> np.ndarray:
    """Encode a metadata column as stable, first-appearance-order IDs."""
    values = metadata_group_values(metadata, group_by)
    value_to_id: dict[Any, int] = {}
    try:
        return np.asarray(
            [value_to_id.setdefault(value, len(value_to_id)) for value in values],
            dtype=np.int64,
        )
    except TypeError as error:
        raise TypeError("Metadata group values must be hashable.") from error


# Process-local cache so multiple Dataset instances pointing at the same
# file (e.g. CORAL/DANN's source/target/validation splits) share one
# in-memory array instead of each loading its own full copy.
_IMAGE_CACHE: dict[Path, np.ndarray] = {}


class TransformedImageDataset(Dataset):
    """Return transformed images with their targets and metadata."""

    def __init__(
        self,
        images: str | Path | np.ndarray,
        indices: np.ndarray,
        targets: np.ndarray,
        transform: Any,
        metadata: Sequence[Any],
        groups: Sequence[str] | None = None,
    ) -> None:

        if len(indices) != len(metadata):
            raise ValueError("Dataset indices and metadata must have equal length.")
        # # Accept a path (training) or an already-loaded array/memmap (evaluator).
        self.images_path = (
            Path(images) if isinstance(images, (str, Path)) else None
        )
        self._images = None if self.images_path is not None else images
        self.indices = indices
        self.targets = targets
        self.transform = transform
        self.metadata = metadata
        self.groups = groups

        if groups is not None and len(indices) != len(groups):
            raise ValueError(
                "Dataset indices and groups must have equal length."
            )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int) -> tuple[Any, ...]:
        image = self._open_images()[int(self.indices[position])].copy()
        target = self.targets[position]
        if isinstance(target, np.generic):
            target = target.item()
        sample = (self.transform(image), target, self.metadata[position])
        if self.groups is None:
            return sample
        return (*sample, self.groups[position])

    def _open_images(self) -> np.ndarray:
        if self._images is None:
            self._images = np.load(
                self.images_path
                , mmap_mode="r", allow_pickle=False
            )
        return self._images

    def _read_image(self, index: int) -> np.ndarray:
        """Read one image; one large pread per image instead of page faults.

        Memmap reads become many small requests on network filesystems (NFS),
        which is far slower than a single contiguous read per image.
        """
        if self.images_path is None or not hasattr(os, "pread"):
            return self._open_images()[index].copy()
        if self._raw_layout is None:
            images = self._open_images()
            if not images.flags.c_contiguous:
                return images[index].copy()
            self._raw_layout = (
                images.shape[1:], images.dtype, int(images.offset)
            )
        shape, dtype, offset = self._raw_layout
        if self._fd is None:
            self._fd = os.open(self.images_path, os.O_RDONLY)
        nbytes = int(np.prod(shape)) * dtype.itemsize
        buffer = os.pread(self._fd, nbytes, offset + index * nbytes)
        if len(buffer) != nbytes:
            raise OSError(
                f"Short read for image {index} of {self.images_path}: "
                f"got {len(buffer)} of {nbytes} bytes."
            )
        return np.frombuffer(buffer, dtype=dtype).reshape(shape).copy()

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        if self.images_path is not None:
            state["_images"] = None  # reopen the memmap in each worker
            state["_fd"] = None  # file descriptors are per process
        return state




def make_data_loader(
    dataset: Dataset,
    *,
    batch_size: int,
    shuffle: bool = False,
    workers: int = 0,
    prefetch_factor: int = 2,
    metadata: Sequence[Any] = (),
    group_by: str | None = None,
    groups_per_batch: int | None = None,
    n_samples_per_group: int | None = None,
    negative_dataset: Dataset | None = None,
    negative_metadata: Sequence[Any] = (),
    negative_samples_per_group: int | None = None,
    seed: int = 0,
) -> DataLoader:
    """Build the shared training-validation and submission DataLoader.

    With ``group_by`` set, ``GroupBatchSampler`` is always used. It mixes
    ``groups_per_batch`` groups during training and visits every row once,
    group by group, during evaluation. When omitted, ``groups_per_batch``
    defaults to one. When a negative dataset is supplied, matching controls
    are appended after the regular samples for every selected group.
    """
    options = {
        "num_workers": workers,
        "persistent_workers": False,
    }
    if workers > 0:
        options["prefetch_factor"] = prefetch_factor

    if negative_dataset is not None and group_by is None:
        raise ValueError("Negative controls require group_by to be set.")
    if negative_dataset is None and negative_samples_per_group is not None:
        raise ValueError(
            "negative_samples_per_group requires a negative dataset."
        )
    if negative_dataset is not None:
        if len(negative_dataset) != len(negative_metadata):
            raise ValueError(
                "Negative dataset and metadata must have equal length."
            )
        if not negative_samples_per_group or negative_samples_per_group < 1:
            raise ValueError(
                "negative_samples_per_group must be positive when negative "
                "controls are enabled."
            )

    if group_by is None:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            **options,
        )

    group_values = metadata_group_values(metadata, group_by)
    value_to_id = {
        value: index for index, value in enumerate(dict.fromkeys(group_values))
    }
    groups = np.asarray([value_to_id[value] for value in group_values])

    negative_groups = None
    if negative_dataset is not None:
        negative_values = metadata_group_values(negative_metadata, group_by)
        # Values absent from the positive dataset are never sampled. They share
        # a sentinel ID solely so the sampler can retain dataset-relative IDs.
        negative_groups = np.asarray(
            [value_to_id.get(value, -1) for value in negative_values]
        )
        dataset = ConcatDataset((dataset, negative_dataset))

    sampler = GroupBatchSampler(
        groups,
        batch_size=batch_size,
        groups_per_batch=groups_per_batch or 1,
        eval=not shuffle,
        n_samples_per_group=n_samples_per_group,
        negative_groups=negative_groups,
        neg_batch_size=negative_samples_per_group,
        seed=seed,
    )
    return DataLoader(dataset, batch_sampler=sampler, **options)
