FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        xauth \
        x11-apps \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip \
    && pip install -r /app/requirements.txt \
    && pip install debugpy

COPY docker/runner_entrypoint.sh /usr/local/bin/runner_entrypoint.sh
RUN chmod +x /usr/local/bin/runner_entrypoint.sh

COPY . /app

ENTRYPOINT ["/usr/local/bin/runner_entrypoint.sh"]
CMD ["--help"]
