# submission file for AdaBN during inference

from __future__ import annotations

from pathlib import Path
import sys

SUBMISSION_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SUBMISSION_DIR))

import numpy as np
import torch
from tqdm import tqdm
from torch import nn

from data_loading import TransformedImageDataset, make_data_loader
from image_transforms import create_image_transform
from model import NUM_CLASSES, create_resnet


CHECKPOINT_PATH = (
    Path(__file__).resolve().parent
    / "models"
    / "arm_bn.pt"
)

BATCH_SIZE = 256
NUM_WORKERS = 8
PREFETCH_FACTOR = 2

# AdaBN needs each forward batch to be a coherent domain so that its
# BatchNorm statistics are meaningful — group by plate rather than
# using arbitrary shuffled batches.
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

        # Configure the pretrained ERM model for AdaBN.
        self.model.train()

        self.transform = create_image_transform(training=False)
        self.group_by = group_by

    def adapt(self, test_images, test_metadata, *args, **kwargs) -> None:
        # AdaBN adapts online during predict() via batch statistics.
        return None

    def predict(self, test_images, test_metadata, *args, **kwargs) -> np.ndarray:
        if len(test_images) != len(test_metadata):
            raise ValueError(
                "test_images and test_metadata must have equal length."
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
        loader = make_data_loader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            workers=NUM_WORKERS,
            prefetch_factor=PREFETCH_FACTOR,
            metadata=test_metadata,
            group_by=self.group_by,
        )

        logits = torch.empty(
            (len(dataset), NUM_CLASSES),
            dtype=torch.float32,
        )

        # ARMBNBN needs train mode because BatchNorm must use the
        # statistics of each incoming (per-plate) test batch. No
        # gradients are needed since nothing is trained.
        self.model.train()

        for inputs, indices, _ in tqdm(loader):
            inputs = inputs.to(self.device, non_blocking=True)

            with torch.inference_mode():
                batch_logits = self.model(inputs)

            logits[indices] = batch_logits.cpu()

        probabilities = torch.softmax(logits, dim=1)
        return probabilities.numpy().astype(np.float32, copy=False)
