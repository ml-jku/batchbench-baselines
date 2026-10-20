"""Hydra entrypoint for all JUMPCP training methods."""
import hydra
import wandb
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from utils import seed_everything


@hydra.main(version_base=None, config_path="../config", config_name="experiment")
def main(cfg: DictConfig) -> None:
    """Instantiate the data module and selected trainer, then fit."""
    seed_everything(cfg.seed)
    torch.use_deterministic_algorithms(True)

    wandb_run = None

    if cfg.wandb_mode != "disabled":
        wandb_run = wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            name=cfg.wandb_run_name,
            mode=cfg.wandb_mode,
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    data = instantiate(cfg.data)
    trainer = instantiate(
        cfg.trainer,
        dataloaders=data,
        wandb_run=wandb_run,
        epochs=cfg.epochs,
        patience=cfg.patience,
        log_every_n_batches=cfg.log_every_n_batches,
        _recursive_=False,
    )

    trainer.fit()
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
