FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /model

COPY infra/engine/model/wrappers/adapter/rf_detr/requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip \
    && pip install -r /tmp/requirements.txt

COPY raw_models/rf-detr /model/raw_models/rf-detr

CMD ["sleep", "infinity"]
