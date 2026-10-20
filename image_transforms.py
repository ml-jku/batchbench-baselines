from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torchvision.transforms import v2


STATISTICS_PATH = Path(__file__).resolve().parent / "image_statistics.json"


def load_image_statistics(
    statistics_path: Path = STATISTICS_PATH,
) -> dict[str, object]:
    with statistics_path.open(encoding="utf-8") as stream:
        statistics = json.load(stream)
    if "mean" not in statistics or "std" not in statistics:
        raise ValueError("Image statistics must contain mean and std values.")
    return statistics


def load_channel_statistics(
    statistics_path: Path = STATISTICS_PATH,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    statistics = load_image_statistics(statistics_path)
    mean = tuple(float(value) for value in statistics["mean"])
    std = tuple(float(value) for value in statistics["std"])
    if len(mean) != 3 or len(std) != 3:
        raise ValueError("Image statistics must contain three RGB values.")
    if any(value <= 0 for value in std):
        raise ValueError("Image standard deviations must be positive.")
    return mean, std



def create_image_transform(
    training: bool,
    statistics: tuple[tuple[float, ...], tuple[float, ...]] | None = None,
):
    """Build the shared training and inference image preprocessing pipeline."""
    operations: list[Any] = [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ]
    if training:
        operations.extend(
            [
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomVerticalFlip(p=0.5),
            ]
        )
    if statistics is not None:
        mean, std = statistics
        operations.append(v2.Normalize(mean=mean, std=std))
    return v2.Compose(operations)
