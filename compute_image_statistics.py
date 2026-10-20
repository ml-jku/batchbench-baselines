#!/usr/bin/env python3
"""Compute per-channel mean and std for image normalization."""

from __future__ import annotations

import argparse
import json
from argparse import Namespace
from pathlib import Path

import torch

from training.dataset import make_loaders


def compute_statistics(
    images_path: Path,
    output_path: Path,
    excluded_source: str = "source_1",
    training_rows: int | None = None,
    batch_size: int = 8,
    num_workers: int = 0,
) -> None:
    """Compute per-channel statistics for the training partition.
    
    Args:
        images_path: Path to images.npy file (shape: N, H, W, C)
        output_path: Path to output JSON file
        excluded_source: Source held out as validation data
        training_rows: Optional expected number of training rows
    """
    suffix = "_images.npy"
    if not images_path.name.endswith(suffix):
        raise ValueError(
            f"Images filename must end in {suffix!r}: {images_path.name}"
        )
    dataset_prefix = images_path.name[: -len(suffix)]
    loader_args = Namespace(
        dataset_prefix=dataset_prefix,
        validation_source=excluded_source,
        statistics=None,
        batch_size=batch_size,
        workers=num_workers,
        prefetch_factor=2,
        train_group_by=None,
        validation_group_by=None,
        seed=0,
    )
    train_loader, _, dataset_info = make_loaders(
        images_path.parent,
        loader_args,
        torch.device("cpu"),
    )
    actual_training_rows = len(train_loader.dataset)
    if training_rows is not None and training_rows != actual_training_rows:
        raise ValueError(
            f"Expected {training_rows:,} training rows, but the training "
            f"partition contains {actual_training_rows:,}."
        )

    print(
        f"Computing training statistics for {dataset_info.num_channels} "
        f"channels..."
    )

    channel_sums = torch.zeros(dataset_info.num_channels, dtype=torch.float64)
    channel_sumsq = torch.zeros(dataset_info.num_channels, dtype=torch.float64)
    pixels_per_channel = 0

    for images, _, _ in train_loader:
        b, _, h, w = images.shape
        images = images.to(torch.float64)
        channel_sums += images.sum(dim=(0, 2, 3))
        channel_sumsq += images.square().sum(dim=(0, 2, 3))

        pixels_per_channel += b * h * w

    if pixels_per_channel == 0:
        raise ValueError("The training partition contains no images.")

    mean = channel_sums / pixels_per_channel
    variance = channel_sumsq / pixels_per_channel - mean.square()
    std = variance.clamp_min(0).sqrt()

    statistics = {
        "mean": mean.tolist(),
        "std": std.tolist(),
    }
    statistics["excluded_source"] = excluded_source
    statistics["rows"] = actual_training_rows
    statistics["pixels_per_channel"] = pixels_per_channel
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(statistics, f, indent=2)
    
    print(f"\nStatistics computed:")
    for i, (m, s) in enumerate(zip(mean, std)):
        print(f"  Channel {i}: mean={m:.6f}, std={s:.6f}")
    
    print(f"\nStatistics saved to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute per-channel image statistics for normalization."
    )
    parser.add_argument(
        "images",
        type=Path,
        help="Path to images.npy file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default= Path("./image_statistics_rep_.json"),
        help="Output JSON file (default: ./image_statistics.json)",
    )
    parser.add_argument(
        "--validation-source",
        "--excluded-source",
        dest="excluded_source",
        default="source_1",
        help="Source ID held out from the training statistics (default: source_1)",
    )
    parser.add_argument(
        "--training-rows",
        type=int,
        default=None,
        help="Optional number of training rows used",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size used by the DataLoader for statistics accumulation.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of DataLoader workers to use when accumulating statistics.",
    )
    
    args = parser.parse_args()
    
    images_path = args.images.expanduser().resolve()
    if not images_path.is_file():
        raise FileNotFoundError(f"Images file not found: {images_path}")
    
    output_path = args.output

    compute_statistics(
        images_path,
        output_path,
        excluded_source=args.excluded_source,
        training_rows=args.training_rows,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
