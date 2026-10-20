# submission file using DANN (domain adaptation happens at training time)

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
from model import create_dann_model


CHECKPOINT_PATH = (
    Path(__file__).resolve().parent
    / "models"
    / "dann.pt"
)

BATCH_SIZE = 128
NUM_WORKERS = 16
PREFETCH_FACTOR = 4

# DANN does not need test batches to share a domain: its BatchNorm layers use
# the running statistics collected during training, not per-batch statistics.
INFERENCE_GROUP_BY: str | None = None


class Submission:
    def __init__(self, group_by: str | None = INFERENCE_GROUP_BY):
        torch.manual_seed(0)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
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
        self.num_classes = checkpoint["num_classes"]
        self.model = create_dann_model(
            num_classes=self.num_classes,
            num_domains=checkpoint["num_domains"],
            in_chans=checkpoint["num_channels"],
            head_hidden_dim=checkpoint["head_hidden_dim"],
            head_hidden_layers=checkpoint["head_hidden_layers"],
            head_dropout=checkpoint["head_dropout"],
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device).eval()

        self.transform = create_image_transform(training=False)
        self.group_by = group_by

    def adapt(self, test_images, test_metadata,  *args, **kwargs) -> None:
        # DANN adapts to the target domain during training via the
        # gradient-reversed domain classifier; nothing happens at test time.
        return None

    def predict(self, test_images, test_metadata, *args, **kwargs) -> np.ndarray:
        if len(test_images) != len(test_metadata):
            raise ValueError(
                "test_images and test_metadata must have equal length."
            )

        if not len(test_images):
            return np.empty((0, self.num_classes), dtype=np.float32)

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
            (len(dataset), self.num_classes),
            dtype=torch.float32,
        )

        for inputs, indices, _ in tqdm(loader):
            inputs = inputs.to(self.device, non_blocking=True)

            with torch.inference_mode():
                class_logits, _ = self.model(inputs)

            logits[indices] = class_logits.cpu()

        probabilities = torch.softmax(logits, dim=1)
        return probabilities.numpy().astype(np.float32, copy=False)
