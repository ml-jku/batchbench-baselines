#!/usr/bin/env bash

python -m training.main \
  --config-name=experiment_coral \
  ++data.batch_size=64 \
  ++data.domain_column=source \
  ++data.validation_source=source_2 \
  ++patience=10 \
  ++trainer.optimizer.lr=0.0008575066313286861 \
  ++trainer.optimizer.weight_decay=2.252661271562055e-05 \
  ++trainer.penalty_weight=8.610150881728 \
  ++epochs=60\
  ++seed=0 \
  ++trainer.output=/system/user/publicwork/sanchez/gitrepos/jumpcp-baselines/models/coral-acc/coral-acc_0.pt \
  ++wandb_mode=online \
  ++trainer.no_amp=True

python -m training.main \
  --config-name=experiment_coral \
  ++data.batch_size=64 \
  ++data.domain_column=source \
  ++data.validation_source=source_2 \
  ++patience=10 \
  ++trainer.optimizer.lr=0.0008575066313286861 \
  ++trainer.optimizer.weight_decay=2.252661271562055e-05 \
  ++trainer.penalty_weight=8.610150881728 \
  ++epochs=60\
  ++seed=42 \
  ++trainer.output=/system/user/publicwork/sanchez/gitrepos/jumpcp-baselines/models/coral-acc/coral-acc_42.pt \
  ++wandb_mode=online \
  ++trainer.no_amp=True

python -m training.main \
  --config-name=experiment_coral \
  ++data.batch_size=64 \
  ++data.domain_column=source \
  ++data.validation_source=source_2 \
  ++patience=10 \
  ++trainer.optimizer.lr=0.0008575066313286861 \
  ++trainer.optimizer.weight_decay=2.252661271562055e-05 \
  ++trainer.penalty_weight=8.610150881728 \
  ++epochs=60\
  ++seed=645 \
  ++trainer.output=/system/user/publicwork/sanchez/gitrepos/jumpcp-baselines/models/coral-acc/coral-acc_645.pt \
  ++wandb_mode=online \
  ++trainer.no_amp=True
