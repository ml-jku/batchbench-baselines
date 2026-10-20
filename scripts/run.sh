python -m training.main \
  --config-name=experiment \
  ++data.batch_size=16 \
  ++data.validation_source=source_2 \
  ++trainer.gradient_accumulation_steps=1 \
  ++trainer.optimizer.lr=0.0002704061886417215 \
  ++trainer.optimizer.weight_decay=0.00027372447834720173 \
  ++epochs=60\
  ++patience=10 \
  ++trainer.output=/system/user/publicwork/sanchez/gitrepos/jumpcp-baselines/models/resnet18-acc/resnet18-acc_42.pt \
  ++wandb_mode=online \
  ++trainer.no_amp=True 


python -m training.main \
  --config-name=experiment \
  ++data.batch_size=16 \
  ++data.validation_source=source_2 \
  ++trainer.gradient_accumulation_steps=1 \
  ++trainer.optimizer.lr=0.0002704061886417215 \
  ++trainer.optimizer.weight_decay=0.00027372447834720173 \
  ++epochs=60\
  ++patience=10 \
  ++seed=0\
  ++trainer.output=/system/user/publicwork/sanchez/gitrepos/jumpcp-baselines/models/resnet18-acc/resnet18-acc_0.pt \
  ++wandb_mode=online \
  ++trainer.no_amp=True

python -m training.main \
  --config-name=experiment \
  ++data.batch_size=16 \
  ++data.validation_source=source_2 \
  ++trainer.gradient_accumulation_steps=1 \
  ++trainer.optimizer.lr=0.0002704061886417215 \
  ++trainer.optimizer.weight_decay=0.00027372447834720173 \
  ++epochs=60\
  ++patience=10 \
  ++seed=645\
  ++trainer.output=/system/user/publicwork/sanchez/gitrepos/jumpcp-baselines/models/resnet18-acc/resnet18-acc_645.pt \
  ++wandb_mode=online \
  ++trainer.no_amp=True\
