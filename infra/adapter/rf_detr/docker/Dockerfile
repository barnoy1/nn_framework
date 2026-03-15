FROM nn-framework-runner:local

WORKDIR /model

COPY raw_models/rf-detr /model/raw_models/rf-detr

COPY infra/adapter/rf_detr/requirements.txt /tmp/requirements.txt
RUN uv pip install --system -r /tmp/requirements.txt \
	&& uv pip install --system /model/raw_models/rf-detr
