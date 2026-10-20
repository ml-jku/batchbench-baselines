FROM python:3.11-slim

WORKDIR /submission

RUN pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cu126 \
        torch==2.13.0+cu126 \
        torchvision==0.28.0+cu126

COPY requirements.txt /submission/requirements.txt
RUN pip install --no-cache-dir \
        -r /submission/requirements.txt

COPY model.py /submission/model.py
COPY data_loading.py /submission/data_loading.py
COPY image_transforms.py /submission/image_transforms.py
COPY utils.py /submission/utils.py

# To submit a different method, change these two lines to that method's
# submission_<method>.py and its checkpoint filename (see README.md):
#   resnet18 -> submission_resnet18.py  / models/resnet18.pt
#   adabn    -> submission_adabn.py     / models/resnet18_adabn_tent.pt
#   tent     -> submission_tent.py      / models/resnet18_adabn_tent.pt
#   coral    -> submission_coral.py     / models/resnet18_coral.pt
#   dann     -> submission_dann.py      / models/resnet18_dann.pt
COPY submissions/submission_adabn.py /submission/submission.py
COPY models/resnet18-acc/resnet18-acc_0.pt /submission/models/resnet18.pt

ENV PYTHONPATH=/submission
