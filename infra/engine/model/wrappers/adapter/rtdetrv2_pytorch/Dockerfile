FROM nn-framework-runner:local

WORKDIR /model

COPY infra/engine/model/wrappers/adapter/rtdetrv2_pytorch/requirements.txt /tmp/requirements.txt
RUN uv pip install --system -v -r /tmp/requirements.txt

COPY raw_models/RT-DETR/rtdetrv2_pytorch /model/raw_models/RT-DETR/rtdetrv2_pytorch

CMD ["sleep", "infinity"]
