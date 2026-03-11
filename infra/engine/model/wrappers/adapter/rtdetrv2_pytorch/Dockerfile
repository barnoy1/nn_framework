FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /model

COPY infra/engine/model/wrappers/adapter/rtdetrv2_pytorch/requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip \
    && pip install -r /tmp/requirements.txt

COPY raw_models/RT-DETR/rtdetrv2_pytorch /model/raw_models/RT-DETR/rtdetrv2_pytorch

CMD ["sleep", "infinity"]
