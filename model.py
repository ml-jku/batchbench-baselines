from __future__ import annotations

from functools import partial
from typing import Any, Optional, Tuple

import numpy as np
import timm
import torch
from timm.layers import PatchEmbed
from timm.models.vision_transformer import Block
from torch import nn
from torch.autograd import Function


MODEL_NAME = "resnet18"
NUM_CLASSES = 2000
NUM_INPUT_CHANNELS = 5
RESNET18_FEATURE_DIM = 512


# ---------------------------------------------------------------------------
# Plain ERM (used by resnet18, adabn, tent)
# ---------------------------------------------------------------------------


def create_resnet(
    num_classes: int = NUM_CLASSES,
    in_chans: int = NUM_INPUT_CHANNELS,
    num_domains: int | None = None,
):
    """Create the competition model with randomly initialized weights."""
    del num_domains
    return timm.create_model(
        MODEL_NAME,
        pretrained=False,
        num_classes=num_classes,
        in_chans=in_chans,
    )


# ---------------------------------------------------------------------------
# CORAL
# ---------------------------------------------------------------------------


class Featurizer(nn.Module):
    """ResNet classifier that exposes features for the CORAL penalty."""

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        in_chans: int = NUM_INPUT_CHANNELS,
        num_domains: int | None = None,
    ) -> None:
        super().__init__()
        del num_domains
        self.featurizer = timm.create_model(
            MODEL_NAME,
            pretrained=False,
            num_classes=0,
            in_chans=in_chans,
        )
        self.classifier = nn.Linear(self.featurizer.num_features, num_classes)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.featurizer(inputs)
        return features, self.classifier(features)


# ---------------------------------------------------------------------------
# DANN
# ---------------------------------------------------------------------------


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, bias=True, batchnorm=False, layernorm=False, dropout=0.5, norm_reduce=False):
        super().__init__()
        self.norm_reduce = norm_reduce

        self.layers = nn.ModuleList()

        batchnorm = nn.BatchNorm1d(hidden_dim) if batchnorm else nn.Identity()
        layernorm = nn.LayerNorm(hidden_dim) if layernorm else nn.Identity()

        for layer in range(n_layers):
            dim = input_dim if layer == 0 else hidden_dim
            self.layers.append(nn.Sequential(
                               nn.Linear(dim, hidden_dim, bias=bias),
                               batchnorm,
                               layernorm,
                               nn.Dropout(p=dropout),
                               nn.ReLU()
            ))


        dim = input_dim if n_layers == 0 else hidden_dim

        self.fc = nn.Linear(dim, output_dim, bias=bias)


    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        x = self.fc(x)

        if self.norm_reduce:
            x = torch.norm(x)

        return x


class GradientReverseFunction(Function):
    """
    Credit: https://github.com/thuml/Transfer-Learning-Library
    """
    @staticmethod
    def forward(
        ctx: Any, input: torch.Tensor, coeff: Optional[float] = 1.0
    ) -> torch.Tensor:
        ctx.coeff = coeff
        output = input * 1.0
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        return grad_output.neg() * ctx.coeff, None


class GradientReverseLayer(nn.Module):
    """
    Credit: https://github.com/thuml/Transfer-Learning-Library
    """
    def __init__(self):
        super().__init__()

    def forward(self, *input):
        return GradientReverseFunction.apply(*input)


class GradRev(nn.Module):
    def __init__(self, classifier, domain_classifier, in_chans=NUM_INPUT_CHANNELS):
        super().__init__()

        self.featurizer = timm.create_model(
            MODEL_NAME,
            pretrained=False,
            num_classes=0,
            in_chans=in_chans,
        )

        self.classifier = MLP(**classifier)
        self.domain_classifier = MLP(**domain_classifier)
        self.gradient_reverse_layer = GradientReverseLayer()

    def forward(self, x):
        features = self.featurizer(x)
        y_pred = self.classifier(features)
        features = self.gradient_reverse_layer(features)
        domains_pred = self.domain_classifier(features)
        return y_pred, domains_pred


def create_dann_model(
    num_classes: int = NUM_CLASSES,
    num_domains: int = 2,
    in_chans: int = NUM_INPUT_CHANNELS,
    head_hidden_dim: int = 64,
    head_hidden_layers: int = 2,
    head_dropout: float = 0.2,
) -> GradRev:
    """Create the competition DANN model with randomly initialized weights."""
    head = {
        "input_dim": RESNET18_FEATURE_DIM,
        "hidden_dim": head_hidden_dim,
        "n_layers": head_hidden_layers,
        "batchnorm": True,
        "dropout": head_dropout,
    }
    return GradRev(
        classifier={**head, "output_dim": num_classes},
        domain_classifier={**head, "output_dim": num_domains},
        in_chans=in_chans,
    )


# ---------------------------------------------------------------------------
# TTT-MAE (masked autoencoder + test-time training)
#
# Following Gandelsman et al., "Test-Time Training with Masked Autoencoders"
# (NeurIPS 2022): a ViT masked autoencoder is pretrained for reconstruction,
# a classification head is trained on the frozen encoder, and at inference
# every test image first adapts the encoder/decoder with a few SGD steps on
# its own reconstruction loss before the head classifies the adapted features.
# ---------------------------------------------------------------------------


def _sincos_1d(embed_dim: int, positions: np.ndarray) -> np.ndarray:
    omega = np.arange(embed_dim // 2, dtype=np.float64) / (embed_dim / 2.0)
    omega = 1.0 / 10000**omega
    angles = np.einsum("m,d->md", positions.reshape(-1), omega)
    return np.concatenate([np.sin(angles), np.cos(angles)], axis=1)


def get_2d_sincos_pos_embed(
    embed_dim: int, grid_size: int, cls_token: bool = False
) -> np.ndarray:
    """Fixed 2D sine-cosine position embeddings, shaped (L[+1], embed_dim)."""
    if embed_dim % 4 != 0:
        raise ValueError("embed_dim must be divisible by 4 for 2D sin-cos.")
    grid_h = np.arange(grid_size, dtype=np.float64)
    grid_w = np.arange(grid_size, dtype=np.float64)
    grid = np.stack(np.meshgrid(grid_w, grid_h), axis=0)
    grid = grid.reshape(2, 1, grid_size, grid_size)
    embedding = np.concatenate(
        [
            _sincos_1d(embed_dim // 2, grid[0]),
            _sincos_1d(embed_dim // 2, grid[1]),
        ],
        axis=1,
    )
    if cls_token:
        embedding = np.concatenate(
            [np.zeros([1, embed_dim]), embedding], axis=0
        )
    return embedding


class MaskedAutoencoderViT(nn.Module):
    """MAE with a ViT encoder and a small ViT decoder (He et al., 2022).

    The encoder only sees the visible patches (plus a class token); the
    decoder sees the encoded tokens plus mask tokens and predicts the pixels
    of every patch. The loss is the mean squared error on masked patches.
    """

    def __init__(
        self,
        img_size: int = 500,
        patch_size: int = 20,
        in_chans: int = NUM_INPUT_CHANNELS,
        embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        decoder_embed_dim: int = 256,
        decoder_depth: int = 4,
        decoder_num_heads: int = 8,
        mlp_ratio: float = 4.0,
        mask_ratio: float = 0.75,
        norm_pix_loss: bool = True,
        fused_attn: bool = True,
    ) -> None:
        super().__init__()
        if img_size % patch_size != 0:
            raise ValueError(
                f"img_size ({img_size}) must be divisible by patch_size "
                f"({patch_size})."
            )
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.mask_ratio = mask_ratio
        self.norm_pix_loss = norm_pix_loss

        # Encoder.
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        grid_size = img_size // patch_size
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.register_buffer(
            "pos_embed",
            torch.from_numpy(
                get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=True)
            ).float().unsqueeze(0),
            persistent=False,
        )
        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                )
                for _ in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

        # Decoder.
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.register_buffer(
            "decoder_pos_embed",
            torch.from_numpy(
                get_2d_sincos_pos_embed(
                    decoder_embed_dim, grid_size, cls_token=True
                )
            ).float().unsqueeze(0),
            persistent=False,
        )
        self.decoder_blocks = nn.ModuleList(
            [
                Block(
                    decoder_embed_dim,
                    decoder_num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                )
                for _ in range(decoder_depth)
            ]
        )
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(
            decoder_embed_dim, patch_size**2 * in_chans
        )

        # Memory-efficient attention backward is not deterministic; allow
        # turning fused attention off when torch deterministic mode is on.
        for block in (*self.blocks, *self.decoder_blocks):
            block.attn.fused_attn = fused_attn

        self.initialize_weights()

    def initialize_weights(self) -> None:
        weight = self.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(weight.view(weight.shape[0], -1))
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)
        self.apply(self._init_module)

    @staticmethod
    def _init_module(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def patchify(self, images: torch.Tensor) -> torch.Tensor:
        """(N, C, H, W) -> (N, L, patch_size**2 * C)."""
        p = self.patch_size
        n, c, height, width = images.shape
        h, w = height // p, width // p
        x = images.reshape(n, c, h, p, w, p)
        x = torch.einsum("nchpwq->nhwpqc", x)
        return x.reshape(n, h * w, p * p * c)

    def unpatchify(self, patches: torch.Tensor) -> torch.Tensor:
        """(N, L, patch_size**2 * C) -> (N, C, H, W)."""
        p = self.patch_size
        h = w = int(patches.shape[1] ** 0.5)
        x = patches.reshape(patches.shape[0], h, w, p, p, self.in_chans)
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(patches.shape[0], self.in_chans, h * p, w * p)

    @staticmethod
    def random_masking(
        x: torch.Tensor,
        mask_ratio: float,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Keep a random subset of tokens independently for each sample."""
        n, length, dim = x.shape
        len_keep = int(length * (1 - mask_ratio))
        noise = torch.rand(n, length, device=x.device, generator=generator)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        x_kept = torch.gather(
            x, 1, ids_keep.unsqueeze(-1).repeat(1, 1, dim)
        )
        mask = torch.ones(n, length, device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, 1, ids_restore)  # 1 = masked, 0 = kept
        return x_kept, mask, ids_restore

    def forward_encoder(
        self,
        images: torch.Tensor,
        mask_ratio: float,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        x = self.patch_embed(images) + self.pos_embed[:, 1:, :]
        if mask_ratio > 0:
            x, mask, ids_restore = self.random_masking(
                x, mask_ratio, generator
            )
        else:
            mask = torch.zeros(x.shape[:2], device=x.device)
            ids_restore = None
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        x = torch.cat((cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        for block in self.blocks:
            x = block(x)
        return self.norm(x), mask, ids_restore

    def forward_decoder(
        self, x: torch.Tensor, ids_restore: torch.Tensor
    ) -> torch.Tensor:
        x = self.decoder_embed(x)
        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1
        )
        x_ = torch.cat((x[:, 1:, :], mask_tokens), dim=1)
        x_ = torch.gather(
            x_, 1, ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2])
        )
        x = torch.cat((x[:, :1, :], x_), dim=1) + self.decoder_pos_embed
        for block in self.decoder_blocks:
            x = block(x)
        x = self.decoder_pred(self.decoder_norm(x))
        return x[:, 1:, :]

    def forward_loss(
        self, images: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        target = self.patchify(images).float()
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1e-6) ** 0.5
        loss = ((pred.float() - target) ** 2).mean(dim=-1)
        return (loss * mask).sum() / mask.sum().clamp(min=1)

    def forward(
        self,
        images: torch.Tensor,
        mask_ratio: float | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (reconstruction loss, predicted patches, mask)."""
        mask_ratio = self.mask_ratio if mask_ratio is None else mask_ratio
        latent, mask, ids_restore = self.forward_encoder(
            images, mask_ratio, generator
        )
        pred = self.forward_decoder(latent, ids_restore)
        return self.forward_loss(images, pred, mask), pred, mask

    def embed(self, images: torch.Tensor, pool: str = "mean") -> torch.Tensor:
        """Encode the full (unmasked) image into one feature vector."""
        latent, _, _ = self.forward_encoder(images, mask_ratio=0.0)
        if pool == "cls":
            return latent[:, 0]
        if pool == "mean":
            return latent[:, 1:].mean(dim=1)
        raise ValueError(f"pool must be 'mean' or 'cls'; got {pool!r}.")


class MAEClassifier(nn.Module):
    """MAE encoder (frozen during head training) plus a classification head.

    The decoder stays inside ``mae`` so test-time training can update it
    together with the encoder.
    """

    def __init__(
        self,
        mae: MaskedAutoencoderViT,
        head: nn.Module,
        pool: str = "mean",
    ) -> None:
        super().__init__()
        self.mae = mae
        self.head = head
        self.pool = pool

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.head(self.mae.embed(images, self.pool))


def create_mae(
    num_classes: int = NUM_CLASSES,
    in_chans: int = NUM_INPUT_CHANNELS,
    num_domains: int | None = None,
    **mae_kwargs: Any,
) -> MaskedAutoencoderViT:
    """Create the randomly initialized MAE used for reconstruction pretraining."""
    del num_classes, num_domains
    return MaskedAutoencoderViT(in_chans=in_chans, **mae_kwargs)


def create_mae_classifier(
    num_classes: int = NUM_CLASSES,
    in_chans: int = NUM_INPUT_CHANNELS,
    num_domains: int | None = None,
    mae: Any = None,
    pool: str = "mean",
    head_hidden_dim: int = 1024,
    head_hidden_layers: int = 1,
    head_dropout: float = 0.2,
) -> MAEClassifier:
    """Create the TTT-MAE classifier with randomly initialized weights.

    ``mae`` holds the ``MaskedAutoencoderViT`` constructor arguments (it is
    filled in from the pretraining checkpoint). ``head_hidden_layers=0``
    gives a linear probe.
    """
    del num_domains
    if mae is None:
        raise ValueError("create_mae_classifier requires the MAE arguments.")
    encoder = MaskedAutoencoderViT(in_chans=in_chans, **dict(mae))
    head = MLP(
        input_dim=encoder.embed_dim,
        hidden_dim=head_hidden_dim,
        output_dim=num_classes,
        n_layers=head_hidden_layers,
        batchnorm=True,
        dropout=head_dropout,
    )
    return MAEClassifier(encoder, head, pool=pool)


class TTTAdapter:
    """Per-image test-time training for an ``MAEClassifier``.

    For each image: reset the MAE (encoder, decoder, class and mask tokens)
    to its pretrained weights, run ``steps`` SGD steps on the masked
    reconstruction loss of that single image (replicated ``batch_size`` times
    with independent random masks), then classify the unmasked image with the
    adapted encoder. The classification head is never updated.
    """

    def __init__(
        self,
        model: MAEClassifier,
        steps: int = 20,
        lr: float = 5e-3,
        momentum: float = 0.9,
        weight_decay: float = 0.2,
        mask_ratio: float = 0.75,
        batch_size: int = 16,
        amp: bool = True,
        seed: int = 0,
    ) -> None:
        self.model = model
        self.steps = steps
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.mask_ratio = mask_ratio
        self.batch_size = batch_size
        self.device = next(model.parameters()).device
        self.amp = amp and self.device.type == "cuda"
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(seed)

        model.mae.requires_grad_(True)
        model.head.requires_grad_(False)
        self.params = list(model.mae.parameters())
        self.initial = [p.detach().clone() for p in self.params]

    def reset(self) -> None:
        with torch.no_grad():
            for param, initial in zip(self.params, self.initial):
                param.copy_(initial)

    def adapt(self, image: torch.Tensor) -> None:
        """Adapt the MAE to one (C, H, W) image."""
        self.reset()
        self.model.mae.train()
        optimizer = torch.optim.SGD(
            self.params,
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )
        batch = image.unsqueeze(0).repeat(self.batch_size, 1, 1, 1)
        with torch.enable_grad():
            for _ in range(self.steps):
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=torch.bfloat16,
                    enabled=self.amp,
                ):
                    loss, _, _ = self.model.mae(
                        batch, self.mask_ratio, self.generator
                    )
                loss.backward()
                optimizer.step()

    def predict(self, images: torch.Tensor) -> torch.Tensor:
        """Return class logits for (N, C, H, W) images, one adaptation each."""
        logits = []
        for image in images:
            self.adapt(image)
            self.model.eval()
            with torch.no_grad():
                logits.append(self.model(image.unsqueeze(0)))
        self.reset()
        return torch.cat(logits)
