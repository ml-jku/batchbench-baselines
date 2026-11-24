# submission file using test-time training with a masked autoencoder (TTT-MAE)

from __future__ import annotations

from pathlib import Path
import sys

SUBMISSION_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SUBMISSION_DIR))

import numpy as np
import torch

from data_loading import TransformedImageDataset, make_data_loader
from image_transforms import create_image_transform
from model import TTTAdapter, create_mae_classifier


CHECKPOINT_PATH = (
    Path(__file__).resolve().parent
    / "models"
    / "mae_ttt"
    / "mae_ttt.pt"
)

# Test-time training adapts every image on its own, so the loader batch size
# only controls how many images are moved to the GPU at once.
BATCH_SIZE = 16
NUM_WORKERS = 8
PREFETCH_FACTOR = 2

# Overrides for the test-time-training settings stored in the checkpoint,
# e.g. {"steps": 10, "batch_size": 8} to trade accuracy for speed. Keys are
# TTTAdapter arguments; empty means use the checkpoint values unchanged.
TTT_OVERRIDES: dict[str, object] = {}

# Every image is adapted independently; test batches need no shared domain.
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
                "Run the MAE pretraining and probing stages in training/ "
                "before building the submission image."
            )
        checkpoint = torch.load(
            CHECKPOINT_PATH,
            map_location=self.device,
            weights_only=True,
        )
        self.num_classes = checkpoint["num_classes"]
        self.model = create_mae_classifier(
            num_classes=self.num_classes,
            in_chans=checkpoint["num_channels"],
            **checkpoint["model_config"],
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device).eval()

        # The test-time-training settings validated during training, unless
        # overridden above.
        unknown = TTT_OVERRIDES.keys() - checkpoint["ttt"].keys()
        if unknown:
            raise ValueError(
                f"Unknown TTT_OVERRIDES keys {sorted(unknown)}; "
                f"valid keys are {sorted(checkpoint['ttt'])}."
            )
        ttt_config = {**checkpoint["ttt"], **TTT_OVERRIDES}
        print(f"TTT config: {ttt_config}")
        self.adapter = TTTAdapter(self.model, **ttt_config)

        self.transform = create_image_transform(training=False)
        self.group_by = group_by

    def adapt(self, test_images, test_metadata, *args, **kwargs) -> None:
        # TTT is performed per image inside predict(); nothing is shared
        # between test images.
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

        for inputs, indices, _ in loader:
            inputs = inputs.to(self.device, non_blocking=True)
            logits[indices] = self.adapter.predict(inputs).cpu()

        probabilities = torch.softmax(logits, dim=1)
        return probabilities.numpy().astype(np.float32, copy=False)
