"""Submission entry point for control-sample ARM-BN inference."""

from __future__ import annotations

from pathlib import Path
import sys
from tqdm import tqdm

SUBMISSION_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SUBMISSION_DIR))

import numpy as np
import torch

from data_loading import TransformedImageDataset, make_data_loader
from image_transforms import create_image_transform
from model import NUM_CLASSES, create_resnet


CHECKPOINT_PATH = (
    Path(__file__).resolve().parent
    / "models"
    / "cs_arm_bn.pt"
)

BATCH_SIZE = 256
NEGATIVE_CONTROLS_PER_PLATE = 32
NUM_WORKERS = 8
PREFETCH_FACTOR = 2

# Group treatments by plate and append matching negative controls so every
# BatchNorm adaptation batch represents one coherent experimental domain.
INFERENCE_GROUP_BY: str | None = "plate"


class Submission:
    def __init__(self, group_by: str | None = INFERENCE_GROUP_BY):
        torch.manual_seed(0)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = create_resnet()
        if not CHECKPOINT_PATH.is_file():
            raise FileNotFoundError(
                f"Trained checkpoint not found at {CHECKPOINT_PATH}. "
                "Run the training utilities in training/ before building "
                "the submission image."
            )
        checkpoint = torch.load(
            CHECKPOINT_PATH,
            map_location=self.device,
            weights_only=True,
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)

        self.model.train()

        self.transform = create_image_transform(training=False)
        self.group_by = group_by

    def adapt(
        self,
        test_images,
        test_metadata,
        negcon_images,
        negcon_metadata,
    ) -> None:
        # ARM-BN adapts online during predict() via per-plate batch statistics.
        return None

    def predict(
        self,
        test_images,
        test_metadata,
        negcon_images,
        negcon_metadata,
    ) -> np.ndarray:
        if len(test_images) != len(test_metadata):
            raise ValueError(
                "test_images and test_metadata must have equal length."
            )
        if len(negcon_images) != len(negcon_metadata):
            raise ValueError(
                "negcon_images and negcon_metadata must have equal length."
            )

        if not len(test_images):
            return np.empty((0, NUM_CLASSES), dtype=np.float32)

        row_indices = np.arange(len(test_images), dtype=np.int64)
        dataset = TransformedImageDataset(
            test_images,
            row_indices,
            row_indices,
            self.transform,
            test_metadata,
        )
        negcon_indices = np.arange(len(negcon_images), dtype=np.int64)
        negcon_dataset = TransformedImageDataset(
            negcon_images,
            negcon_indices,
            np.full(len(negcon_images), -1, dtype=np.int64),
            self.transform,
            negcon_metadata,
        )
        loader = make_data_loader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            workers=NUM_WORKERS,
            prefetch_factor=PREFETCH_FACTOR,
            metadata=test_metadata,
            group_by=self.group_by,
            negative_dataset=negcon_dataset,
            negative_metadata=negcon_metadata,
            negative_samples_per_group=NEGATIVE_CONTROLS_PER_PLATE,
        )

        logits = torch.empty(
            (len(dataset), NUM_CLASSES),
            dtype=torch.float32,
        )

        # Train mode makes BatchNorm use each incoming plate batch. Matching
        # controls participate in the statistics, but only treatment logits
        # are copied into the returned predictions.
        self.model.train()

        for inputs, indices, _ in tqdm(loader):
            inputs = inputs.to(self.device, non_blocking=True)

            with torch.inference_mode():
                batch_logits = self.model(inputs)

            treatment_mask = indices >= 0
            logits[indices[treatment_mask]] = batch_logits[
                treatment_mask.to(batch_logits.device)
            ].cpu()

        probabilities = torch.softmax(logits, dim=1)
        return probabilities.numpy().astype(np.float32, copy=False)
