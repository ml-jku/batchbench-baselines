# submission file using tent during inference

from __future__ import annotations

from pathlib import Path
import sys

SUBMISSION_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SUBMISSION_DIR))

import numpy as np
import torch
from torch import nn

from data_loading import (
    TransformedImageDataset,
    make_data_loader,
    metadata_group_values,
)
from image_transforms import create_image_transform
from model import NUM_CLASSES, create_resnet


CHECKPOINT_PATH = (
    Path(__file__).resolve().parent
    / "models"
    / "resnet18.pt"
)

BATCH_SIZE = 256
NUM_WORKERS = 8
PREFETCH_FACTOR = 4
INFERENCE_GROUP_BY: str | None = "plate"

TENT_LR = 1e-3
TENT_STEPS = 10

def configure_model_for_tent(model: nn.Module) -> None:
    """
    Configure model for TENT.

    Freeze all parameters except BatchNorm affine parameters and
    force BatchNorm to use statistics from the current test batch.
    """
    model.train()
    model.requires_grad_(False)

    found_batchnorm = False

    for module in model.modules():
        if isinstance(
            module,
            (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d),
        ):
            found_batchnorm = True

            if module.affine:
                module.weight.requires_grad_(True)
                module.bias.requires_grad_(True)

            # Use statistics from the current test batch instead of
            # the running statistics collected during training.
            module.track_running_stats = False
            module.running_mean = None
            module.running_var = None

    if not found_batchnorm:
        raise RuntimeError(
            "TENT requires BatchNorm layers, but none were found."
        )


def collect_tent_parameters(model: nn.Module) -> list[nn.Parameter]:
    """Collect BatchNorm affine parameters optimized by TENT."""

    parameters = []

    for module in model.modules():
        if isinstance(
            module,
            (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d),
        ):
            if module.affine:
                parameters.append(module.weight)
                parameters.append(module.bias)

    if not parameters:
        raise RuntimeError(
            "No trainable BatchNorm affine parameters found."
        )

    return parameters


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

        # Configure the pretrained ERM model for online TENT.
        configure_model_for_tent(self.model)
        self.optimizer = torch.optim.Adam(
            collect_tent_parameters(self.model), lr=TENT_LR
        )

        self.transform = create_image_transform(training=False)
        self.group_by = group_by
        self.initial_model_state = {
            k: v.detach().clone()
            for k, v in self.model.state_dict().items()
        }
        self.initial_optimizer_state = self.optimizer.state_dict()

    def reset_tent_state(self) -> None:
        # Reset model and optimizer to their initial pretrained state
        self.model.load_state_dict(self.initial_model_state, strict=True)
        self.optimizer.load_state_dict(self.initial_optimizer_state)

    def adapt(self, test_images, test_metadata, *args, **kwargs) -> None:
        # TENT adapts online during predict().
        return None

    def softmax_entropy(self, x: torch.Tensor) -> torch.Tensor:
        """Entropy of softmax distribution from logits."""
        return -(x.softmax(dim=1) * x.log_softmax(dim=1)).sum(dim=1)

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

        # TENT needs train mode because BatchNorm must use the
        # statistics of each incoming test batch.
        self.model.train()
        previous_group = object()  # sentinel: forces a reset on the first batch
        # Group value of every row, using the same column the loader batches by.
        # Without group_by, all rows form a single group (no mid-run reset).
        if self.group_by is None:
            group_values = [None] * len(test_metadata)
        else:
            group_values = metadata_group_values(test_metadata, self.group_by)

        for inputs, indices, _ in loader:
            # reset model if new group is detected
            batch_indices = indices.cpu().numpy()

            group = group_values[batch_indices[0]]

            if group != previous_group:
                self.reset_tent_state()
                previous_group = group

            # actual TENT adaptation step
            inputs = inputs.to(self.device, non_blocking=True)
            for _ in range(TENT_STEPS):
                self.optimizer.zero_grad(set_to_none=True)

                # Forward pass for this test batch.
                batch_logits = self.model(inputs)

                # Store predictions BEFORE adapting on this batch.
                # The resulting parameter update affects subsequent
                # test batches.
                logits[indices] = batch_logits.detach().cpu()

                # TENT objective: minimize prediction entropy.
                loss = self.softmax_entropy(batch_logits).mean()
                loss.backward()

                # Adapt BatchNorm affine parameters.
                self.optimizer.step()

        probabilities = torch.softmax(logits, dim=1)
        return probabilities.numpy().astype(np.float32, copy=False)