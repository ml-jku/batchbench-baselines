# BatchBench baselines

This repository provides example baselines for the [BatchBench challenge](https://huggingface.co/spaces/ai-lab-jku/jumpcp-challenge). It includes model architectures, training code, and Docker submission examples for classification and adaptation on JUMP-CP microscopy images. Use these examples to build a submission or add your own baseline.

## Training your own model

Run the following commands from the repository root. First, install the
training dependencies and the [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli):

```bash
python -m pip install -r training/requirements.txt
python -m pip install -U huggingface_hub
```

### Downloading the data

Download the [2k, 500×500 dataset](https://huggingface.co/datasets/ai-lab-jku/jumpcp-challenge-datasets/tree/main/2k/500x500).
Download only the linked dataset folder to a local directory of your choice:

```bash
hf download ai-lab-jku/jumpcp-challenge-datasets \
  --repo-type dataset \
  --include "2k/500x500/*" \
  --local-dir data
```

### Add your own training code

Add a trainer in `training/methods/` and its Hydra configuration in
`config/trainer/`. Reuse the shared data module and `training/main.py`
entrypoint where practical. 

1. If it needs a new architecture, add it to `model.py` alongside the
   existing ones (don't replace anything already there).
2. Add `submissions/submission_<method>.py`, importing whatever
   it needs from the shared `model.py`/`data_loading.py`/
   `image_transforms.py`/`utils.py` — follow an existing submission file's
   `Submission` class shape.
3. Add its checkpoint to `models/`.
4. If it needs its own training procedure, add `training/methods/<method>.py`
   with a named trainer class whose constructor calls the base trainer with
   the shared datamodule. Add its Hydra config under `config/trainer/`; do not
   put dataset construction in the method module or add a method-specific
   entrypoint. Subclass the base `Trainer`, overriding `compute_loss` at
   minimum.
   If the method requires a different data pipeline or training loop, adapt
   this structure to its needs while reusing the shared code where practical.


### Train your model

Then, start training with:

```bash
python -m training.main --config-name=experiment_resnet18 \
  data.data_dir=data/2k/500x500/final \
  epochs=60 \
  trainer.output=models/resnet18.pt \
```

## Preparing your submission code

### Submission examples
The Dockerfile copies the selected submission to `/submission/submission.py`
and bundles its checkpoint. To build a specific method, edit the last two
`COPY` instructions using the files below:

| Method | Submission file | Checkpoint |
| --- | --- | --- |
| resnet18 | `submissions/submission_resnet18.py` | `models/resnet18.pt` |
| adabn | `submissions/submission_adabn.py` | `models/resnet18_adabn_tent.pt` |
| tent | `submissions/submission_tent.py` | `models/resnet18_adabn_tent.pt` |
| arm-bn | `submissions/submission_arm_bn.py` | `models/resnet18_arm_bn.pt` |
| cs-arm-bn | `submissions/submission_cs_arm_bn.py` | `models/resnet18_cs_arm_bn.pt` |
| coral | `submissions/submission_coral.py` | `models/resnet18_coral.pt` |
| dann | `submissions/submission_dann.py` | `models/resnet18_dann.pt` |

```dockerfile
COPY submissions/submission_<method>.py /submission/submission.py
COPY models/<checkpoint>.pt /submission/models/<runtime-checkpoint>.pt
```

The checkpoint paths in the table are suggested source locations. If your
checkpoint is stored elsewhere, use its actual path as the `COPY` source.
The destination filename must match the submission's `CHECKPOINT_PATH`:

| Method | Runtime checkpoint filename |
| --- | --- |
| resnet18, adabn, tent | `resnet18.pt` |
| arm-bn | `arm_bn.pt` |
| cs-arm-bn | `cs_arm_bn.pt` |
| coral | `coral.pt` |
| dann | `dann.pt` |


For example, to use the checkpoint currently selected in the Dockerfile:

```dockerfile
COPY submissions/submission_cs_arm_bn.py /submission/submission.py
COPY models/cs_arm_bn/cs_arm_bn_0.pt /submission/models/cs_arm_bn.pt
```


## Making your ow submission

The built image must contain:

```text
/submission/submission.py
```

with:

```python
class Submission:
    def __init__(self):
        # Load model here.
        ...

    def adapt(self, test_images, test_metadata, negative_controls_images, negative_controls_metadata) -> None:
        # Construct any loader/sampler needed for adaptation.
        ...

    def predict(self, test_images, test_metadata, negative_controls_images, negative_controls_metadata):
        # Construct any loader needed for inference and return predictions
        # in the original image order.
        return predictions
```

The evaluation server passes the complete memory-mapped `test_images` array and
the corresponding `test_metadata` list to both methods. Images are `uint8`
arrays shaped `(N, H, W, 5)`. Each item in `test_metadata` is
`(source_id, batch_id, plate_id)`. Submissions can construct any dataset,
loader, sampler, or collation logic they need. `predict()` must return one row
per image in the original order. Labels and metric code are never exposed to
the submission.

The official runtime has no network access, so a model checkpoint must already
exist inside the built image.
