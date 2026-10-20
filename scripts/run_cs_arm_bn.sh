#!/usr/bin/env bash
python -m training.main \
  --config-name=experiment_cs_arm_bn \
  ++trainer.optimizer.lr=5e-4 \
  ++seed=0 \
  ++trainer.output=/system/user/publicwork/sanchez/gitrepos/jumpcp-baselines/models/cs_arm_bn/cs_arm_bn_0.pt \
  ++data.batch_size=64 \
  ++trainer.gradient_accumulation_steps=4 \
  ++data.negative_controls_per_plate=16 \
  ++wandb_mode=online \
  ++trainer.no_amp=True \
  ++data.workers=8 \
  ++epochs=60 \
  ++patience=10

python -m training.main \
  --config-name=experiment_cs_arm_bn \
  ++trainer.optimizer.lr=5e-4 \
  ++seed=42 \
  ++trainer.output=/system/user/publicwork/sanchez/gitrepos/jumpcp-baselines/models/cs_arm_bn/cs_arm_bn_42.pt \
  ++data.batch_size=64 \
  ++trainer.gradient_accumulation_steps=4 \
  ++data.negative_controls_per_plate=16 \
  ++wandb_mode=online \
  ++trainer.no_amp=True \
  ++data.workers=8 \
  ++epochs=60 \
  ++patience=10

python -m training.main \
  --config-name=experiment_cs_arm_bn \
  ++trainer.optimizer.lr=5e-4 \
  ++seed=645 \
  ++trainer.output=/system/user/publicwork/sanchez/gitrepos/jumpcp-baselines/models/cs_arm_bn/cs_arm_bn_645.pt \
  ++data.batch_size=64 \
  ++trainer.gradient_accumulation_steps=4 \
  ++data.negative_controls_per_plate=16 \
  ++wandb_mode=online \
  ++trainer.no_amp=True \
  ++data.workers=8 \
  ++epochs=60 \
  ++patience=10
