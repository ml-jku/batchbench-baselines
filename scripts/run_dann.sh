#!/usr/bin/env bash

python -m training.main \
  --config-name=experiment_dann \
  ++data.batch_size=32 \
  ++data.domain_column=source \
  ++data.validation_source=source_2 \
  ++patience=10 \
  ++trainer.domain_loss_weight=0.909630225707748 \
  ++trainer.model.head_dropout=0.19109308676248532 \
  ++trainer.model.head_hidden_dim=64 \
  ++trainer.model.head_hidden_layers=2 \
  ++trainer.optimizer.lr=0.0005420100021170347 \
  ++trainer.optimizer.weight_decay=1.8323069661367824e-05 \
  ++epochs=60 \
  ++seed=0 \
  ++trainer.output=/system/user/publicwork/sanchez/gitrepos/jumpcp-baselines/models/dann/dann_0.pt \
  ++wandb_mode=online \
  ++trainer.no_amp=True \
  ++data.workers=8

python -m training.main \
  --config-name=experiment_dann \
  ++data.batch_size=32 \
  ++data.domain_column=source \
  ++data.validation_source=source_2 \
  ++patience=10 \
  ++trainer.domain_loss_weight=0.909630225707748 \
  ++trainer.model.head_dropout=0.19109308676248532 \
  ++trainer.model.head_hidden_dim=64 \
  ++trainer.model.head_hidden_layers=2 \
  ++trainer.optimizer.lr=0.0005420100021170347 \
  ++trainer.optimizer.weight_decay=1.8323069661367824e-05 \
  ++epochs=60 \
  ++seed=42 \
  ++trainer.output=/system/user/publicwork/sanchez/gitrepos/jumpcp-baselines/models/dann/dann_42.pt \
  ++wandb_mode=online \
  ++trainer.no_amp=True \
  ++data.workers=8


python -m training.main \
  --config-name=experiment_dann \
  ++data.batch_size=32 \
  ++data.domain_column=source \
  ++data.validation_source=source_2 \
  ++patience=10 \
  ++trainer.domain_loss_weight=0.909630225707748 \
  ++trainer.model.head_dropout=0.19109308676248532 \
  ++trainer.model.head_hidden_dim=64 \
  ++trainer.model.head_hidden_layers=2 \
  ++trainer.optimizer.lr=0.0005420100021170347 \
  ++trainer.optimizer.weight_decay=1.8323069661367824e-05 \
  ++epochs=60 \
  ++seed=645 \
  ++trainer.output=/system/user/publicwork/sanchez/gitrepos/jumpcp-baselines/models/dann/dann_645.pt \
  ++wandb_mode=online \
  ++trainer.no_amp=True \
  ++data.workers=8

