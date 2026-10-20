"""Hydra-instantiated data module for the predefined JUMPCP splits."""
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


from data_loading import (
    TransformedImageDataset,
    make_data_loader,
    metadata_group_ids,
    metadata_group_values,
)
from image_transforms import create_image_transform


@dataclass(frozen=True)
class DatasetInfo:
    prefix: str
    num_classes: int
    num_channels: int
    num_domains: int = 1
    domain_names: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class RawDataset:
    images_path: Path
    labels: np.ndarray
    source_ids: np.ndarray
    batch_ids: np.ndarray
    plate_ids: np.ndarray
    num_classes: int
    num_channels: int
    domain_ids: np.ndarray
    domain_names: list[str]
    groups: np.ndarray


def load_statistics(
    path: str | Path | None,
) -> tuple[tuple[float, ...], tuple[float, ...]] | None:
    if path is None:
        return None
    with Path(path).expanduser().resolve().open(encoding="utf-8") as stream:
        values = json.load(stream)
    return (
        tuple(float(value) for value in values["mean"]),
        tuple(float(value) for value in values["std"]),
    )


def metadata_rows(
    source_ids: np.ndarray,
    batch_ids: np.ndarray,
    plate_ids: np.ndarray,
    indices: np.ndarray,
) -> list[tuple[str, str, str]]:
    return [
        (source_ids[index], batch_ids[index], plate_ids[index])
        for index in indices
    ]


def load_raw_dataset(
    data_dir: Path,
    dataset_prefix: str,
    *,
    domain_column: str | None = None,
) -> RawDataset:
    images_path = data_dir / f"{dataset_prefix}_images.npy"
    labels_path = data_dir / f"{dataset_prefix}_labels.npy"
    metadata_path = data_dir / f"{dataset_prefix}_metadata.npz"
    for path in (images_path, labels_path, metadata_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required training file not found: {path}")

    labels = np.load(labels_path, mmap_mode="r", allow_pickle=False)
    images = np.load(images_path, mmap_mode="r", allow_pickle=False)
    if len(images) != len(labels):
        raise ValueError("Training images and labels have different lengths.")
    num_channels = int(images.shape[-1])
    num_classes = int(labels.max()) + 1
    del images

    with np.load(metadata_path, allow_pickle=False) as metadata:
        source_ids = metadata["source_id"].astype(str)
        batch_ids = metadata["batch_id"].astype(str)
        plate_ids = metadata["plate_id"].astype(str)
        groups = metadata["group"].astype(str)
    if len(source_ids) != len(labels):
        raise ValueError("Training metadata and labels have different lengths.")

    if domain_column is None:
        domain_ids = np.zeros(len(labels), dtype=np.int64)
        domain_names: list[str] = []
    else:
        rows = metadata_rows(
            source_ids,
            batch_ids,
            plate_ids,
            np.arange(len(labels)),
        )
        domain_ids = metadata_group_ids(rows, domain_column)
        normalized = domain_column.removeprefix("Metadata_").lower()
        values = {
            "source": source_ids,
            "batch": batch_ids,
            "plate": plate_ids,
        }[normalized]
        domain_names = sorted(np.unique(values).tolist())
        if len(domain_names) < 2:
            raise ValueError(
                f"domain_column {domain_column!r} must contain at least "
                f"two unique values; found {domain_names}."
            )

    return RawDataset(
        images_path=images_path,
        labels=labels,
        source_ids=source_ids,
        batch_ids=batch_ids,
        plate_ids=plate_ids,
        num_classes=num_classes,
        num_channels=num_channels,
        domain_ids=domain_ids,
        domain_names=domain_names,
        groups=groups,
    )


class JUMPCPDataModule:
    """Own the datasets and dataloaders consumed by every trainer."""

    def __init__(
        self,
        data_dir: str,
        dataset_prefix: str,
        validation_source: str,
        batch_size: int,
        workers: int,
        prefetch_factor: int,
        seed: int,
        statistics: str | None = None,
        domain_column: str | None = None,
        train_group_by: str | None = None,
        validation_group_by: str | None = None,
        groups_per_batch: int | None = None,
        required_source_classes: int | None = None,
        negative_control_dir: str | None = None,
        negative_controls_per_plate: int = 0,
    ) -> None:
        self.data_dir = Path(data_dir).expanduser().resolve()
        self.dataset_prefix = dataset_prefix
        self.validation_source = validation_source
        self.batch_size = batch_size
        self.workers = workers
        self.prefetch_factor = prefetch_factor
        self.seed = seed
        self.statistics = statistics
        self.domain_column = domain_column
        self.train_group_by = train_group_by
        self.validation_group_by = validation_group_by
        self.groups_per_batch = groups_per_batch
        self.required_source_classes = required_source_classes
        self.negative_control_dir = (
            Path(negative_control_dir).expanduser().resolve()
            if negative_control_dir is not None
            else None
        )
        self.negative_controls_per_plate = negative_controls_per_plate

        self._setup()

    def _setup(self) -> None:
        raw = load_raw_dataset(
            self.data_dir,
            self.dataset_prefix,
            domain_column=self.domain_column,
        )
        negative_raw = None
        negative_domain_ids = None
        if self.negative_control_dir is not None:
            if self.negative_controls_per_plate <= 0:
                raise ValueError(
                    "negative_controls_per_plate must be positive when "
                    "negative_control_dir is set."
                )
            if self.train_group_by is None or (
                self.train_group_by.removeprefix("Metadata_").lower()
                != "plate"
            ):
                raise ValueError(
                    "Negative controls require train_group_by=plate."
                )
            negative_raw = load_raw_dataset(
                self.negative_control_dir,
                self.dataset_prefix,
            )
            if negative_raw.num_channels != raw.num_channels:
                raise ValueError(
                    "Training and negative-control images have different "
                    "channel counts."
                )
            if self.domain_column is None:
                negative_domain_ids = np.zeros(
                    len(negative_raw.labels), dtype=np.int64
                )
            else:
                positive_rows = metadata_rows(
                    raw.source_ids,
                    raw.batch_ids,
                    raw.plate_ids,
                    np.arange(len(raw.labels)),
                )
                negative_rows = metadata_rows(
                    negative_raw.source_ids,
                    negative_raw.batch_ids,
                    negative_raw.plate_ids,
                    np.arange(len(negative_raw.labels)),
                )
                positive_values = metadata_group_values(
                    positive_rows, self.domain_column
                )
                value_to_id = dict(
                    zip(positive_values, raw.domain_ids, strict=True)
                )
                negative_domain_ids = np.asarray(
                    [
                        value_to_id.get(value, -1)
                        for value in metadata_group_values(
                            negative_rows, self.domain_column
                        )
                    ],
                    dtype=np.int64,
                )
        validation_indices = np.flatnonzero(
            raw.source_ids == self.validation_source
        )
        training_indices = np.flatnonzero(
            raw.source_ids != self.validation_source
        )
        negative_training_indices = None
        negative_validation_indices = None
        if negative_raw is not None:
            negative_validation_indices = np.flatnonzero(
                negative_raw.source_ids == self.validation_source
            )
            negative_training_indices = np.flatnonzero(
                negative_raw.source_ids != self.validation_source
            )
        if len(training_indices) == 0 or len(validation_indices) == 0:
            raise ValueError(
                "Both training and validation splits must contain data."
            )
        statistics = load_statistics(self.statistics)
        train_transform = create_image_transform(True, statistics)
        evaluation_transform = create_image_transform(False, statistics)

        def rows_for_raw(
            dataset: RawDataset,
            indices: np.ndarray,
        ) -> list[tuple[str, str, str]]:
            return metadata_rows(
                dataset.source_ids,
                dataset.batch_ids,
                dataset.plate_ids,
                indices,
            )

        def loader(
            indices: np.ndarray,
            transform: Any,
            shuffle: bool,
            group_by: str | None,
            negative_indices: np.ndarray | None = None,
            include_groups: bool = False,
        ):
            sampler_metadata = rows_for_raw(raw, indices)
            targets = raw.labels[indices]
            dataset = TransformedImageDataset(
                raw.images_path,
                indices,
                targets,
                transform,
                raw.domain_ids[indices],
                raw.groups[indices] if include_groups else None,
            )
            negative_dataset = None
            negative_metadata = ()
            if negative_indices is not None:
                assert negative_raw is not None
                assert negative_domain_ids is not None
                negative_metadata = rows_for_raw(
                    negative_raw, negative_indices
                )
                negative_dataset = TransformedImageDataset(
                    negative_raw.images_path,
                    negative_indices,
                    np.full(len(negative_indices), -100, dtype=np.int64),
                    transform,
                    negative_domain_ids[negative_indices],
                )
            return make_data_loader(
                dataset,
                batch_size=self.batch_size,
                shuffle=shuffle,
                workers=self.workers,
                prefetch_factor=self.prefetch_factor,
                group_by=group_by,
                metadata=sampler_metadata,
                groups_per_batch=self.groups_per_batch,
                negative_dataset=negative_dataset,
                negative_metadata=negative_metadata,
                negative_samples_per_group=(
                    self.negative_controls_per_plate
                    if negative_dataset is not None
                    else None
                ),
                seed=self.seed,
            )

        if (
            self.groups_per_batch
            and self.groups_per_batch > 1
            and not self.train_group_by
        ):
            raise ValueError(
                "groups_per_batch > 1 requires train_group_by."
            )

        self.train_loader = loader(
            training_indices,
            train_transform,
            True,
            self.train_group_by,
            negative_training_indices,
        )
        self.validation_loader = loader(
            validation_indices,
            evaluation_transform,
            False,
            self.validation_group_by,
            include_groups=True,
        )
        self.target_loader = loader(
            validation_indices,
            train_transform,
            True,
            self.validation_group_by,
            (
                negative_validation_indices
                if self.validation_group_by is not None
                and self.validation_group_by.removeprefix(
                    "Metadata_"
                ).lower() == "plate"
                else None
            ),
        )

        self.dataset_info = DatasetInfo(
            prefix=self.dataset_prefix,
            num_classes=raw.num_classes,
            num_channels=raw.num_channels,
            num_domains=max(len(raw.domain_names), 1),
            domain_names=raw.domain_names,
        )
